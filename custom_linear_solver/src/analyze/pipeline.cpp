#include "analyze/analyze.hpp"

#include <cuda_runtime.h>

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
#include "analyze/symbolic/multifrontal.hpp"
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

// Σ fsz² (the dense front arena size == plan.front_total) for a panel partition.
static long front_arena_fill(const symbolic::MultifrontalSymbolic& mf)
{
    long t = 0;
    for (int p = 0; p < mf.num_panels; ++p) {
        const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
        t += fsz * fsz;
    }
    return t;
}

// exp_260612 Stage 0: deep-K amalgamation (CLS_AMALG_K). When set, replace the default
// relaxed_panels chain partition with deep_k_panels (thicker nc -> higher arithmetic intensity).
// Guards before accepting it: (1) the nesting invariant — multifrontal_symbolic flags any child
// CB row that does not nest in its parent front as asm_idx==-1; (2) a fill-growth budget
// CLS_AMALG_FILL (default 1.30 = +30% dense front arena vs the baseline the solver would use).
// Returns true and fills `out` only if both pass; otherwise the default ordering is kept.
static bool maybe_amalgamate(int n, int max_panel_width,
                             const std::vector<int>& parent, const std::vector<int>& colcount,
                             const std::vector<int>& Lp, const std::vector<int>& Li,
                             bool emit_info, symbolic::PanelPartition& out)
{
    const int amalg_k = env_int("CLS_AMALG_K", 0);
    if (amalg_k <= 0 || n <= 0) return false;

    // Baseline partition the solver would otherwise use (mirrors compute_effective_panel_width).
    int eff = (n >= 16000) ? 16 : max_panel_width;
    if (eff < 1) eff = 1;
    if (eff > 64) eff = 64;
    const symbolic::PanelPartition base = symbolic::relaxed_panels(n, parent, colcount, eff);
    const symbolic::PanelPartition amalg = symbolic::deep_k_panels(n, parent, colcount, amalg_k);

    const symbolic::MultifrontalSymbolic mf = symbolic::multifrontal_symbolic(n, Lp, Li, amalg);
    long nest_bad = 0;
    for (std::size_t i = 0; i < mf.asm_idx.size(); ++i)
        if (mf.asm_idx[i] < 0) ++nest_bad;

    const symbolic::MultifrontalSymbolic mfb = symbolic::multifrontal_symbolic(n, Lp, Li, base);
    const long fill_amalg = front_arena_fill(mf);
    const long fill_base = front_arena_fill(mfb);
    const double ratio = fill_base > 0 ? static_cast<double>(fill_amalg) / fill_base : 1.0;
    const char* fs = std::getenv("CLS_AMALG_FILL");
    const double budget = fs ? std::atof(fs) : 1.30;

    if (emit_info) {
        auto nc_hist = [](const symbolic::PanelPartition& pp, const char* tag) {
            long b[6] = {0};  // nc in {1-2,3-4,5-8,9-16,17-32,>32}
            double wsum = 0; long wn = 0;
            for (int p = 0; p < pp.num_panels; ++p) {
                const int c = pp.ncols[p];
                int k = c <= 2 ? 0 : c <= 4 ? 1 : c <= 8 ? 2 : c <= 16 ? 3 : c <= 32 ? 4 : 5;
                ++b[k]; wsum += (double)c * c; wn += c;  // wsum weights by column-share of work
            }
            std::fprintf(stderr,
                "  nc-hist %s [1-2]=%ld [3-4]=%ld [5-8]=%ld [9-16]=%ld [17-32]=%ld [>32]=%ld meanw=%.1f\n",
                tag, b[0], b[1], b[2], b[3], b[4], b[5], wn ? wsum / wn : 0.0);
        };
        std::fprintf(stderr,
                     "[analyze] amalg K=%d panels %d->%d fill_ratio=%.3f (budget %.2f) nest_bad=%ld\n",
                     amalg_k, base.num_panels, amalg.num_panels, ratio, budget, nest_bad);
        nc_hist(base, "base ");
        nc_hist(amalg, "amalg");
    }
    if (nest_bad > 0) {
        std::fprintf(stderr, "[analyze] amalg REJECT: %ld nesting violations -> baseline\n", nest_bad);
        return false;
    }
    if (ratio > budget) {
        std::fprintf(stderr, "[analyze] amalg REJECT: fill_ratio %.3f > budget %.2f -> baseline\n",
                     ratio, budget);
        return false;
    }
    out = amalg;
    return true;
}

// Build the plan for one fixed ND seed / mode (the original single-ordering pipeline).
static bool build_plan_seed(const CsrMatrixView& matrix,
                            const PlanBuildOptions& options,
                            int metis_seed, bool parallel_nd,
                            PlanBuildResult& out, bool use_gpu_nd = false)
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

        // 3. Nested-dissection ordering: METIS (fill), GPU-objective ND, or ELECTRICAL-weighted ND
        // (Stage 2: cut weak |J_ij| tie-lines — needs the Jacobian values, downloaded to host once).
        out.perm.assign(static_cast<std::size_t>(n), 0);
        const bool use_ew = use_gpu_nd && env_int("CLS_GPU_ND_EW", 0) != 0;
        if (use_ew) {
            std::vector<int> h_rp(static_cast<std::size_t>(n) + 1), h_ci(static_cast<std::size_t>(nnz));
            std::vector<double> h_v(static_cast<std::size_t>(nnz));
            cudaMemcpy(h_rp.data(), d_csr_row_ptr, (static_cast<std::size_t>(n) + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ci.data(), d_csr_col_idx, static_cast<std::size_t>(nnz) * sizeof(int), cudaMemcpyDeviceToHost);
            if (matrix.value_type == ValueType::kFloat64) {
                cudaMemcpy(h_v.data(), matrix.values, static_cast<std::size_t>(nnz) * sizeof(double), cudaMemcpyDeviceToHost);
            } else {
                std::vector<float> hf(static_cast<std::size_t>(nnz));
                cudaMemcpy(hf.data(), matrix.values, static_cast<std::size_t>(nnz) * sizeof(float), cudaMemcpyDeviceToHost);
                for (long i = 0; i < nnz; ++i) h_v[i] = hf[i];
            }
            if (!reordering::gpu_nd_weighted_from_graph(n, h_rp.data(), h_ci.data(), h_v.data(), out.perm, metis_seed))
                return false;
        } else {
            std::vector<int> nd_xadj = metis_sym_col_ptr;     // consumed (moved-from) by ND call
            std::vector<int> nd_adjncy = metis_sym_row_idx;
            const bool ok = use_gpu_nd
                ? reordering::gpu_nd_from_graph(n, nd_xadj, nd_adjncy, out.perm, metis_seed)
                : reordering::metis_nd_from_graph(n, nd_xadj, nd_adjncy, out.perm, parallel_nd, metis_seed);
            if (!ok) return false;
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

        // 7b. exp_260612 Stage 0: optional deep-K amalgamation (CLS_AMALG_K), validated for
        // nesting + a fill-growth budget. Injected via analyze_multifrontal's forced_panels.
        std::vector<int> colcount(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) colcount[j] = Lp[j + 1] - Lp[j];
        symbolic::PanelPartition amalg_panels;
        const bool use_amalg = maybe_amalgamate(n, options.max_panel_width, parent, colcount,
                                                Lp, Li, options.emit_analyze_info, amalg_panels);

        // 8. Build the multifrontal plan.
        out.plan = analyze_multifrontal(n, nnz, ordered_device.col_ptr.ptr,
                                        ordered_device.row_idx.ptr, Lp, Li, parent,
                                        options.max_panel_width,
                                        use_amalg ? &amalg_panels : nullptr,
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
    // Measured best-of-k drives per-seed builds itself: one serial-ND plan, no env best-of-k / GPU-ND.
    if (options.single_seed_only)
        return build_plan_seed(matrix, options, options.metis_seed,
                               /*parallel_nd=*/false, out);

    // exp_260612: custom GPU/TC-objective ND (CLS_GPU_ND). Build the gpu_nd plan + a METIS baseline,
    // and keep gpu_nd ONLY if its dense front fill stays within budget (CLS_GPU_ND_FILL, default 1.3)
    // — a GPU objective can trade fill for structure, so gate it cheaply at analyze time (no GPU run).
    if (env_int("CLS_GPU_ND", 0) != 0) {
        PlanBuildResult gpu, base;
        const bool gok = build_plan_seed(matrix, options, options.metis_seed,
                                         /*parallel_nd=*/false, gpu, /*use_gpu_nd=*/true);
        const bool bok = build_plan_seed(matrix, options, options.metis_seed,
                                         /*parallel_nd=*/false, base, /*use_gpu_nd=*/false);
        if (!gok && !bok) return false;
        if (!gok) { out = std::move(base); return true; }
        if (!bok) { out = std::move(gpu); return true; }
        const char* fs = std::getenv("CLS_GPU_ND_FILL");
        const double budget = fs ? std::atof(fs) : 1.3;
        const double ratio = base.plan.front_total > 0
            ? (double)gpu.plan.front_total / (double)base.plan.front_total : 1.0;
        std::fprintf(stderr,
                     "[analyze] gpu_nd front_total %ld vs metis %ld  fill_ratio=%.3f (budget %.2f) -> %s\n",
                     gpu.plan.front_total, base.plan.front_total, ratio, budget,
                     ratio <= budget ? "gpu_nd" : "metis(reject)");
        out = (ratio <= budget) ? std::move(gpu) : std::move(base);
        return true;
    }

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
