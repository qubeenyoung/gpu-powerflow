#include "plan/build.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <thread>
#include <utility>
#include <vector>

#include "plan/analyze.hpp"
#include "reordering/metis_nd.hpp"
#include "symbolic/elimination_tree.hpp"
#include "symbolic/supernode.hpp"

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

// Dump per-front (q, p, fsz, nc, uc, level) to a CSV when SolverConfig.analyze_dump_fronts_path
// is non-empty. Used by the front-distribution analysis scripts in docs/04-benchmarks-profiling.
void maybe_dump_fronts(const MultifrontalPlan& plan, const std::string& path)
{
    if (path.empty()) return;
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "[analyze] dump-fronts: failed to open %s\n", path.c_str());
        return;
    }
    std::fprintf(f, "q,p,fsz,nc,uc,level\n");
    for (int L = 0; L < plan.num_plevels; ++L) {
        for (int q = plan.plptr[L]; q < plan.plptr[L + 1]; ++q) {
            const int p = plan.h_plcols[q];
            const int fsz = plan.h_front_ptr[p + 1] - plan.h_front_ptr[p];
            const int nc = plan.h_ncols[p];
            std::fprintf(f, "%d,%d,%d,%d,%d,%d\n", q, p, fsz, nc, fsz - nc, L);
        }
    }
    std::fclose(f);
    std::fprintf(stderr, "[analyze] wrote %d front entries to %s\n",
                 plan.num_panels, path.c_str());
}

}  // namespace

bool build_plan_from_csr(const CsrMatrixView& matrix,
                         const PlanBuildOptions& options,
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
            != Status::Success) return false;

        // 2. Symmetric adjacency graph (A + A^T) on device. Reused below for permute_metis_graph.
        std::vector<int> metis_sym_col_ptr, metis_sym_row_idx;
        if (matrix::build_symmetric_graph_device(csc_device, metis_sym_col_ptr, metis_sym_row_idx)
            != Status::Success) return false;

        // 3. METIS nested-dissection.
        out.perm.assign(static_cast<std::size_t>(n), 0);
        {
            std::vector<int> nd_xadj = metis_sym_col_ptr;     // consumed (moved-from) by ND call
            std::vector<int> nd_adjncy = metis_sym_row_idx;
            if (!reordering::metis_nd_from_graph(n, nd_xadj, nd_adjncy, out.perm,
                                                 options.use_parallel_nested_dissection))
                return false;
        }
        out.iperm.assign(static_cast<std::size_t>(n), 0);
        for (int k = 0; k < n; ++k) out.iperm[out.perm[k]] = k;
        if (out.d_perm.upload(out.perm) != Status::Success) return false;
        if (out.d_iperm.upload(out.iperm) != Status::Success) return false;

        // 4. Apply permutation to CSC; capture ordered_value_to_csr mapping.
        matrix::DeviceCscPattern ordered_device;
        if (matrix::permute_csc_device(csc_device, out.d_iperm.ptr, ordered_device)
            != Status::Success) return false;
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
                                        options.panel_cap, /*forced_panels=*/nullptr,
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

}  // namespace custom_linear_solver::plan
