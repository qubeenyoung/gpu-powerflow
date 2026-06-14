#include "analyze/analyze.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <thread>
#include <utility>
#include <vector>

#include "analyze/plan/lower.hpp"
#include "analyze/reorder/metis_nd.hpp"
#include "analyze/symbolic/elimination_tree.hpp"
#include "analyze/symbolic/supernode.hpp"

namespace custom_linear_solver::plan {

namespace {

// Run a parallel-for over [lo, hi) chunked across up to 12 host threads. Falls back to a
// single-thread call for small ranges or when the system reports zero hardware concurrency.
template <typename Fn>
void par_for(int lo, int hi, Fn&& fn)
{
    unsigned hw = std::thread::hardware_concurrency();
    const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
    if (hi - lo < 32768 || nth <= 1) {
        fn(lo, hi);
        return;
    }
    std::vector<std::thread> th;
    const int chunk = (hi - lo + nth - 1) / nth;
    for (int t = 0; t < nth; ++t) {
        const int a = lo + t * chunk;
        const int b = std::min(hi, a + chunk);
        if (a < b) th.emplace_back([&fn, a, b] { fn(a, b); });
    }
    for (auto& x : th) x.join();
}

// Relabel a symmetric CSC pattern (col_ptr / row_idx) under (perm, iperm) into a new CSC.
// new_col = iperm[old_col]; new_row = iperm[old_row]. Used to prepare etree / fill_pattern in
// the post-METIS ordering.
void permute_symmetric_pattern(int n, const std::vector<int>& col_ptr,
                               const std::vector<int>& row_idx,
                               const std::vector<int>& perm,
                               const std::vector<int>& iperm,
                               std::vector<int>& out_col_ptr,
                               std::vector<int>& out_row_idx)
{
    out_col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int new_col = 0; new_col < n; ++new_col) {
        const int old_col = perm[new_col];
        out_col_ptr[new_col + 1] =
            out_col_ptr[new_col] + (col_ptr[old_col + 1] - col_ptr[old_col]);
    }
    out_row_idx.resize(static_cast<std::size_t>(out_col_ptr[n]));
    par_for(0, n, [&](int lo, int hi) {
        for (int new_col = lo; new_col < hi; ++new_col) {
            const int old_col = perm[new_col];
            int w = out_col_ptr[new_col];
            for (int p = col_ptr[old_col]; p < col_ptr[old_col + 1]; ++p) {
                out_row_idx[w++] = iperm[row_idx[p]];
            }
        }
    });
}

// Dump per-front structure to a CSV when SolverConfig.analyze_dump_fronts_path is non-empty.
// Used by offline front-distribution and parent-update analysis scripts.
void maybe_dump_fronts(const MultifrontalPlan& plan, const std::string& path)
{
    if (path.empty()) return;
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "[analyze] dump-fronts: failed to open %s\n", path.c_str());
        return;
    }
    std::fprintf(f, "q,p,fsz,nc,uc,level,parent,asm_len,extend_elems\n");
    for (int L = 0; L < plan.num_plevels; ++L) {
        for (int q = plan.panel_level_ptr[L]; q < plan.panel_level_ptr[L + 1]; ++q) {
            const int p = plan.h_plcols[q];
            const int fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
            const int nc = plan.h_ncols[p];
            const int uc = fsz - nc;
            const int parent = plan.h_panel_parent.empty() ? -1 : plan.h_panel_parent[p];
            const int asm_len = plan.h_asm_ptr.empty() ? 0 : plan.h_asm_ptr[p + 1] - plan.h_asm_ptr[p];
            const long extend_elems = (parent >= 0) ? (long)uc * uc : 0;
            std::fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%d,%ld\n",
                         q, p, fsz, nc, uc, L, parent, asm_len, extend_elems);
        }
    }
    std::fclose(f);
    std::fprintf(stderr, "[analyze] wrote %d front entries to %s\n",
                 plan.num_panels, path.c_str());
}

// --- best-of-k critical-path-aware ordering selection (CLS_ORDER_K) ----------------------
// B=1 factorize time is set by the under-filled deep elimination-tree levels: each large front
// runs ~1 block/SM, so a level with `cnt` fronts of max size `mf` costs ~ceil(cnt/SM) "waves" of
// an O(mf^3) dense panel factorization. Summing that over levels gives a fill-free, factorize-free
// proxy that tracks measured B=1 factor time across ND orderings far better than the older
// spine-work proxy (validated, docs/exp_260612). CLS_ORDER_K>1 runs deterministic serial ND on K
// seeds and keeps the plan with the lowest proxy cost; this also removes the run-to-run
// nondeterminism of parallel ND. The fill-minimizing default (parallel ND, K=1) is unchanged.
int env_int(const char* name, int dflt)
{
    const char* s = std::getenv(name);
    return s ? std::atoi(s) : dflt;
}

// "tail-cube" proxy: sum of (largest front)^3 over the UNDER-OCCUPIED large levels — those with
// fewer fronts than SMs (cnt < SM) and a non-small front (maxfsz > kSmallFrontMax). Such a level
// runs one wave of ~SM-or-fewer 1-block/SM dense panel factorizations, so its wall time is set by
// the largest front (~O(mf^3)); the well-filled upper levels are throughput-bound and contribute
// little ordering-dependent variance, so excluding them sharpens selection. This isolates the
// serial critical-path tail and tracks measured B=1 factor time across ND orderings with ~0-3%
// top-1 gap vs an oracle on the 13K/25K/70K-class cases (validated, docs/exp_260612), clearly
// beating both the old spine-work proxy and an all-levels wave model.
// TC-aware (CLS_ORDER_TC=1, for TF32/Ozaki execution, exp_260612 doc 11): on tensor cores the
// trailing GEMM of a large front runs much faster than scalar, so a big front costs LESS on the
// critical path than its scalar fsz³ — the scalar proxy over-penalizes big fronts and may pick an
// ordering that needlessly avoids them. The discount splits each front into a panel-LU part (share
// 1−s, NOT TC-accelerated) and a trailing part (share s≈uc/fsz, divided by the trailing TC speedup
// g): factor = (1−s) + s/g. So big fronts (s→1) are discounted toward 1/g, small/near-threshold
// fronts barely change. tc_aware=0 keeps the exact scalar proxy (default, byte-identical). g default
// 1.25 ≈ measured Ozaki trailing speedup (doc 11); raise (~1.5–2) for plain TF32.
double ordering_cost_model(const MultifrontalPlan& plan)
{
    const int SM = std::max(1, env_int("CLS_ORDER_SM", 82));   // under-fill threshold (RTX 3090=82)
    static const int tc_aware = env_int("CLS_ORDER_TC", 0);
    static const int tc_min = env_int("CLS_ORDER_TC_MIN", 48); // fsz where the TC trailing engages
    static const double tc_g = [] {
        const char* s = std::getenv("CLS_ORDER_TC_G"); return s ? std::atof(s) : 1.25;
    }();
    const int* lvl = plan.panel_level_ptr.data();
    double cost = 0.0;
    for (int L = 0; L < plan.num_plevels; ++L) {
        const long cnt = lvl[L + 1] - lvl[L];
        if (cnt >= SM) continue;                               // level fills the GPU: not discriminating
        long mf = 0;
        int mf_nc = 0;
        for (int q = lvl[L]; q < lvl[L + 1]; ++q) {
            const int p = plan.h_plcols[q];
            const long fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
            if (fsz > mf) { mf = fsz; mf_nc = plan.h_ncols[p]; }
        }
        if (mf <= kSmallFrontMax) continue;                   // warp-packed small fronts: cheap
        double c = (double)mf * (double)mf * (double)mf;       // one wave, dominated by largest front
        if (tc_aware && mf >= tc_min && mf_nc > 0 && tc_g > 0.0) {
            const double s = (double)(mf - mf_nc) / (double)mf;   // trailing share ~ uc/fsz
            c *= (1.0 - s) + s / tc_g;                            // panel-LU scalar + trailing/g
        }
        cost += c;
    }
    return cost;
}

}  // namespace

// Build the plan for one fixed ND seed (the original single-ordering pipeline).
static bool build_plan_seed(const CsrMatrixView& matrix,
                            const PlanBuildOptions& options,
                            int metis_seed, bool parallel_nd,
                            PlanBuildResult& out)
{
    try {
        const int n = static_cast<int>(matrix.nrows);
        const int nnz = static_cast<int>(matrix.nnz);
        const auto* d_csr_row_ptr = static_cast<const int*>(matrix.row_offsets);
        const auto* d_csr_col_idx = static_cast<const int*>(matrix.col_indices);

        // 1. CSR → CSC on device.
        matrix::DeviceCscPattern csc_device;
        if (matrix::build_csc_from_csr_device(n, nnz, d_csr_row_ptr, d_csr_col_idx, csc_device)
            != Status::kSuccess) return false;

        // 2. Symmetric adjacency graph (A + A^T) on device. Reused below for permute_metis_graph.
        std::vector<int> metis_sym_col_ptr, metis_sym_row_idx;
        if (matrix::build_symmetric_graph_device(csc_device, metis_sym_col_ptr, metis_sym_row_idx)
            != Status::kSuccess) return false;

        // 3. METIS nested-dissection.
        out.perm.assign(static_cast<std::size_t>(n), 0);
        {
            std::vector<int> nd_xadj = metis_sym_col_ptr;     // consumed (moved-from) by ND call
            std::vector<int> nd_adjncy = metis_sym_row_idx;
            if (!reordering::metis_nd_from_graph(n, nd_xadj, nd_adjncy, out.perm,
                                                 parallel_nd, metis_seed))
                return false;
        }
        out.iperm.assign(static_cast<std::size_t>(n), 0);
        for (int k = 0; k < n; ++k) out.iperm[out.perm[k]] = k;
        if (out.d_perm.upload(out.perm) != Status::kSuccess) return false;
        if (out.d_iperm.upload(out.iperm) != Status::kSuccess) return false;

        // 4. Apply permutation to CSC; capture ordered_value_to_csr mapping.
        matrix::DeviceCscPattern ordered_device;
        if (matrix::permute_csc_device(csc_device, out.d_iperm.ptr, ordered_device)
            != Status::kSuccess) return false;
        out.d_ordered_value_to_csr = std::move(ordered_device.source_pos);

        // 5. Relabel the symmetric adjacency under the permutation for etree / fill_pattern.
        std::vector<int> sym_col_ptr, sym_row_idx;
        permute_symmetric_pattern(n, metis_sym_col_ptr, metis_sym_row_idx, out.perm, out.iperm,
                                  sym_col_ptr, sym_row_idx);

        // 6. Elimination tree.
        std::vector<int> parent =
            symbolic::etree(n, sym_col_ptr.data(), sym_row_idx.data());

        // 7. Fill pattern. The METIS ordering is a postorder → fill-neutral, so we can compute
        // fill in METIS order and relabel below without a second fill_pattern pass.
        std::vector<int> Lp, Li;
        symbolic::fill_pattern(n, sym_col_ptr.data(), sym_row_idx.data(), parent, Lp, Li);

        // 8. Build the multifrontal plan.
        out.plan = analyze_multifrontal(n, nnz, ordered_device.col_ptr.ptr,
                                        ordered_device.row_idx.ptr, Lp, Li, parent,
                                        options.max_panel_width, /*forced_panels=*/nullptr,
                                        options.float_front, options.emit_analyze_info);
        if (out.plan.num_panels == 0) return false;

        maybe_dump_fronts(out.plan, options.dump_fronts_csv_path);
        return true;
    } catch (const std::bad_alloc&) {
        return false;
    } catch (const std::exception&) {
        return false;
    }
}

bool build_plan_from_csr(const CsrMatrixView& matrix,
                         const PlanBuildOptions& options,
                         PlanBuildResult& out)
{
    // Measured best-of-k drives per-seed builds itself: one serial-ND plan, no env best-of-k.
    if (options.single_seed_only)
        return build_plan_seed(matrix, options, options.metis_seed,
                               /*parallel_nd=*/false, out);

    const int K = std::max(1, env_int("CLS_ORDER_K", 1));
    if (K <= 1)
        return build_plan_seed(matrix, options, options.metis_seed,
                               options.use_parallel_nested_dissection, out);

    // Best-of-K: deterministic serial ND on seeds [metis_seed .. metis_seed+K-1]; keep the plan
    // with the lowest critical-path proxy cost. Each candidate's device buffers are freed when its
    // PlanBuildResult is destroyed at the end of the iteration (only the running best is retained).
    PlanBuildResult best;
    double best_cost = 0.0;
    int best_seed = -1;
    bool have = false;
    for (int i = 0; i < K; ++i) {
        PlanBuildResult cand;
        const int seed = options.metis_seed + i;
        if (!build_plan_seed(matrix, options, seed, /*parallel_nd=*/false, cand)) continue;
        const double c = ordering_cost_model(cand.plan);
        if (options.emit_analyze_info)
            std::fprintf(stderr, "[analyze] order-cand seed=%d cost=%.4e\n", seed, c);
        if (!have || c < best_cost) {
            best_cost = c; best_seed = seed; best = std::move(cand); have = true;
        }
    }
    if (!have) return false;
    std::fprintf(stderr, "[analyze] order-select K=%d -> seed=%d cost=%.4e (serial ND)\n",
                 K, best_seed, best_cost);
    out = std::move(best);
    return true;
}

}  // namespace custom_linear_solver::plan
