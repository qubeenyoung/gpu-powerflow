#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "factorize/multifrontal.hpp"
#include "factorize/multifrontal_batched.hpp"
#include "matrix/pattern_kernels.hpp"
#include "symbolic/supernode.hpp"
#include "reordering/metis_nd.hpp"
#include "solve/multifrontal.hpp"
#include "symbolic/elimination_tree.hpp"

namespace custom_linear_solver {
namespace {

struct CscMatrix {
    int n = 0;
    std::vector<int> col_ptr;
    std::vector<int> row_idx;
};

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

// Correct etree-respecting amalgamation for deep-K tensor-core fronts. Merges child columns
// into their parent supernode (a child's contribution block always nests in its parent's
// front, so the multifrontal stays valid), up to `cap` columns per supernode and gated by a
// colcount-similarity `ratio` to bound the zero-padding. Emits `order` (final position ->
// current position) as a postorder of the resulting supernode tree with each supernode's
// columns contiguous (descendants before ancestors -> a valid elimination order, same fill
// class as any postorder of this etree); `panel_sizes` are the contiguous panel column counts.
void amalgamation_reorder(int n, const std::vector<int>& parent, const std::vector<int>& colcount,
                          int cap, double ratio, std::vector<int>& order,
                          std::vector<int>& panel_sizes)
{
    std::vector<int> root(n), sz(n, 1), mx(colcount), mn(colcount);
    for (int i = 0; i < n; ++i) root[i] = i;
    auto find = [&](int x) { while (root[x] != x) { root[x] = root[root[x]]; x = root[x]; } return x; };
    for (int j = 0; j < n; ++j) {
        const int p = parent[j];
        if (p < 0) continue;
        const int rj = find(j), rp = find(p);
        if (rj == rp) continue;
        if (sz[rj] + sz[rp] <= cap &&
            (double)std::max(mx[rj], mx[rp]) <= ratio * (double)std::min(mn[rj], mn[rp])) {
            const int keep = std::max(rj, rp), drop = (keep == rj ? rp : rj);
            root[drop] = keep;
            sz[keep] += sz[drop];
            mx[keep] = std::max(mx[rj], mx[rp]);
            mn[keep] = std::min(mn[rj], mn[rp]);
        }
    }
    std::vector<std::vector<int>> cols(n);
    for (int j = 0; j < n; ++j) cols[find(j)].push_back(j);  // ascending within a supernode
    std::vector<std::vector<int>> snch(n);
    std::vector<int> roots;
    for (int R = 0; R < n; ++R) {
        if (find(R) != R) continue;  // R is the top (max-index) column of its supernode
        const int p = parent[R];
        if (p < 0) roots.push_back(R);
        else snch[find(p)].push_back(R);
    }
    order.clear();
    order.reserve(n);
    panel_sizes.clear();
    std::vector<std::pair<int, int>> stk;
    for (int r : roots) {
        stk.push_back({r, 0});
        while (!stk.empty()) {
            auto& top = stk.back();
            if (top.second < static_cast<int>(snch[top.first].size())) {
                const int c = snch[top.first][top.second++];
                stk.push_back({c, 0});
            } else {
                for (int col : cols[top.first]) order.push_back(col);
                panel_sizes.push_back(static_cast<int>(cols[top.first].size()));
                stk.pop_back();
            }
        }
    }
}

}  // namespace

struct Solver::Impl {
    SolverConfig config;
    CsrMatrixView matrix;
    DenseVectorView rhs;
    DenseVectorView solution;
    bool has_matrix = false;
    bool has_rhs = false;
    bool has_solution = false;
    bool analyzed = false;

    std::vector<int> perm;
    std::vector<int> iperm;
    custom_linear_solver::matrix::IntDeviceBuffer d_ordered_value_to_csr;
    custom_linear_solver::matrix::IntDeviceBuffer d_perm;
    custom_linear_solver::matrix::IntDeviceBuffer d_iperm;
    custom_linear_solver::plan::MultifrontalPlan plan;
    custom_linear_solver::factorize::BatchedState batched;
};

Solver::Solver(const SolverConfig& config) : impl_(new Impl{config}) {}
Solver::~Solver() = default;
Solver::Solver(Solver&&) noexcept = default;
Solver& Solver::operator=(Solver&&) noexcept = default;

Status Solver::set_data(const CsrMatrixView& matrix)
{
    if (!impl_) return Status::InvalidState;
    if (matrix.nrows <= 0 || matrix.nrows != matrix.ncols || matrix.nnz < 0)
        return Status::InvalidValue;
    if (matrix.index_type != IndexType::Int32 || matrix.location != DataLocation::Device)
        return Status::InvalidValue;
    if (matrix.row_offsets == nullptr || matrix.col_indices == nullptr || matrix.values == nullptr)
        return Status::InvalidValue;
    impl_->matrix = matrix;
    impl_->has_matrix = true;
    impl_->analyzed = false;
    return Status::Success;
}

Status Solver::set_rhs(const DenseVectorView& rhs)
{
    if (!impl_) return Status::InvalidState;
    if (rhs.size <= 0 || rhs.values == nullptr || rhs.location != DataLocation::Device)
        return Status::InvalidValue;
    impl_->rhs = rhs;
    impl_->has_rhs = true;
    return Status::Success;
}

Status Solver::set_solution(const DenseVectorView& solution)
{
    if (!impl_) return Status::InvalidState;
    if (solution.size <= 0 || solution.values == nullptr ||
        solution.location != DataLocation::Device)
        return Status::InvalidValue;
    impl_->solution = solution;
    impl_->has_solution = true;
    return Status::Success;
}

Status Solver::get_data(CsrMatrixView* matrix) const
{
    if (!impl_ || !matrix) return Status::InvalidValue;
    if (!impl_->has_matrix) return Status::InvalidState;
    *matrix = impl_->matrix;
    return Status::Success;
}

Status Solver::get_rhs(DenseVectorView* rhs) const
{
    if (!impl_ || !rhs) return Status::InvalidValue;
    if (!impl_->has_rhs) return Status::InvalidState;
    *rhs = impl_->rhs;
    return Status::Success;
}

Status Solver::get_solution(DenseVectorView* solution) const
{
    if (!impl_ || !solution) return Status::InvalidValue;
    if (!impl_->has_solution) return Status::InvalidState;
    *solution = impl_->solution;
    return Status::Success;
}

Status Solver::analyze()
{
    if (!impl_ || !impl_->has_matrix) return Status::InvalidState;
    try {
        const bool tm = std::getenv("CLS_ANALYZE_TIME") != nullptr;
        auto tclk = std::chrono::steady_clock::now();
        auto lap = [&](const char* name) {
            if (!tm) return;
            cudaDeviceSynchronize();
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "  [solver-analyze] %-24s %.2f ms\n", name,
                         std::chrono::duration<double, std::milli>(now - tclk).count());
            tclk = now;
        };

        const int n = static_cast<int>(impl_->matrix.nrows);
        const int nnz = static_cast<int>(impl_->matrix.nnz);
        const auto* d_csr_row_ptr = static_cast<const int*>(impl_->matrix.row_offsets);
        const auto* d_csr_col_idx = static_cast<const int*>(impl_->matrix.col_indices);

        custom_linear_solver::matrix::DeviceCscPattern csc_device;
        Status st = custom_linear_solver::matrix::build_csc_from_csr_device(
            n, nnz, d_csr_row_ptr, d_csr_col_idx, csc_device);
        if (st != Status::Success) return st;
        lap("build_csc_device");

        // GPU-built symmetric adjacency graph (replaces the host CSC download + CPU
        // build_symmetric_adjacency). Also reused below for permute_metis_graph.
        std::vector<int> metis_sym_col_ptr, metis_sym_row_idx;
        st = custom_linear_solver::matrix::build_symmetric_graph_device(
            csc_device, metis_sym_col_ptr, metis_sym_row_idx);
        if (st != Status::Success) return st;
        lap("build_symmetric_graph_device");
        impl_->perm.assign(static_cast<std::size_t>(n), 0);
        std::vector<int> nd_xadj = metis_sym_col_ptr;     // consumed (moved-from) by the ND call
        std::vector<int> nd_adjncy = metis_sym_row_idx;
        if (!custom_linear_solver::reordering::metis_nd_from_graph(
                n, nd_xadj, nd_adjncy, impl_->perm,
                impl_->config.use_parallel_nested_dissection))
            return Status::AnalysisFailed;
        lap("metis_nd");
        impl_->iperm.assign(static_cast<std::size_t>(n), 0);
        for (int k = 0; k < n; ++k) impl_->iperm[impl_->perm[k]] = k;
        lap("build_iperm");
        st = impl_->d_perm.upload(impl_->perm);
        if (st != Status::Success) return st;
        st = impl_->d_iperm.upload(impl_->iperm);
        if (st != Status::Success) return st;
        lap("upload_perm_iperm");

        custom_linear_solver::matrix::DeviceCscPattern ordered_device;
        st = custom_linear_solver::matrix::permute_csc_device(
            csc_device, impl_->d_iperm.ptr, ordered_device);
        if (st != Status::Success) return st;
        lap("permute_csc_device");
        impl_->d_ordered_value_to_csr = std::move(ordered_device.source_pos);

        std::vector<int> sym_col_ptr, sym_row_idx;
        permute_symmetric_pattern(n, metis_sym_col_ptr, metis_sym_row_idx, impl_->perm,
                                  impl_->iperm, sym_col_ptr, sym_row_idx);
        lap("permute_metis_graph");
        std::vector<int> parent = custom_linear_solver::symbolic::etree(
            n, sym_col_ptr.data(), sym_row_idx.data());
        lap("etree");

        // Optional deep-K amalgamation reorder (env MF_AMALG="cap:ratio"): grow supernodes for
        // tensor-core fronts by merging child columns into their parent supernode, then reorder
        // columns so the supernodes are contiguous. Re-derives perm/iperm, the device ordered CSC
        // (value map), the symmetric pattern and the etree in the new order; the resulting panel
        // partition is forced into analyze_multifrontal.
        std::vector<int> amalg_panel_sizes;
        if (const char* as = std::getenv("MF_AMALG")) {
            int cap = std::atoi(as);
            double ratio = 2.0;
            if (const char* col = std::strchr(as, ':')) ratio = std::atof(col + 1);
            if (cap < 1) cap = 1;
            const std::vector<int> post = custom_linear_solver::symbolic::postorder(parent, n);
            const std::vector<int> cc = custom_linear_solver::symbolic::column_counts(
                n, sym_col_ptr.data(), sym_row_idx.data(), parent, post);
            std::vector<int> order;
            amalgamation_reorder(n, parent, cc, cap, ratio, order, amalg_panel_sizes);
            std::vector<int> newperm(static_cast<std::size_t>(n));
            for (int f = 0; f < n; ++f) newperm[f] = impl_->perm[order[f]];
            impl_->perm = std::move(newperm);
            for (int k = 0; k < n; ++k) impl_->iperm[impl_->perm[k]] = k;
            st = impl_->d_perm.upload(impl_->perm);
            if (st != Status::Success) return st;
            st = impl_->d_iperm.upload(impl_->iperm);
            if (st != Status::Success) return st;
            custom_linear_solver::matrix::DeviceCscPattern ordered2;
            st = custom_linear_solver::matrix::permute_csc_device(csc_device, impl_->d_iperm.ptr,
                                                                 ordered2);
            if (st != Status::Success) return st;
            ordered_device = std::move(ordered2);
            impl_->d_ordered_value_to_csr = std::move(ordered_device.source_pos);
            permute_symmetric_pattern(n, metis_sym_col_ptr, metis_sym_row_idx, impl_->perm,
                                      impl_->iperm, sym_col_ptr, sym_row_idx);
            parent = custom_linear_solver::symbolic::etree(n, sym_col_ptr.data(),
                                                           sym_row_idx.data());
            lap("amalgamation_reorder");
        }

        std::vector<int> Lp, Li;
        custom_linear_solver::symbolic::fill_pattern(n, sym_col_ptr.data(), sym_row_idx.data(),
                                                     parent, Lp, Li);
        lap("fill_pattern");

        custom_linear_solver::symbolic::PanelPartition amalg_panels;
        const custom_linear_solver::symbolic::PanelPartition* forced = nullptr;
        if (!amalg_panel_sizes.empty()) {
            amalg_panels.panel_of.assign(static_cast<std::size_t>(n), -1);
            int col = 0;
            for (int p = 0; p < static_cast<int>(amalg_panel_sizes.size()); ++p) {
                const int szp = amalg_panel_sizes[p];
                int w = 0;
                for (int c = col; c < col + szp; ++c) {
                    amalg_panels.panel_of[c] = p;
                    w = std::max(w, Lp[c + 1] - Lp[c]);
                }
                amalg_panels.first.push_back(col);
                amalg_panels.ncols.push_back(szp);
                amalg_panels.width.push_back(w);
                amalg_panels.padded_fill += static_cast<long>(szp) * w;
                col += szp;
            }
            amalg_panels.num_panels = static_cast<int>(amalg_panel_sizes.size());
            forced = &amalg_panels;
        }

        impl_->plan = custom_linear_solver::factorize::analyze_multifrontal(
            n, nnz, ordered_device.col_ptr.ptr, ordered_device.row_idx.ptr, Lp, Li, parent,
            impl_->config.panel_cap, false, forced);
        lap("analyze_multifrontal");
        if (impl_->plan.num_panels == 0) return Status::AnalysisFailed;
        impl_->analyzed = true;
        return Status::Success;
    } catch (const std::bad_alloc&) {
        return Status::AllocationFailed;
    } catch (const std::exception&) {
        return Status::AnalysisFailed;
    }
}

Status Solver::factorize(double* kernel_ms)
{
    if (!impl_ || !impl_->has_matrix || !impl_->analyzed) return Status::InvalidState;
    try {
        if (!custom_linear_solver::factorize::factorize_multifrontal_device(
                impl_->plan, impl_->matrix.values, impl_->d_ordered_value_to_csr.ptr, kernel_ms))
            return Status::FactorizationFailed;
        return Status::Success;
    } catch (const std::bad_alloc&) {
        return Status::AllocationFailed;
    } catch (const std::exception&) {
        return Status::FactorizationFailed;
    }
}

Status Solver::solve(double* kernel_ms)
{
    if (!impl_ || !impl_->has_rhs || !impl_->has_solution || !impl_->analyzed)
        return Status::InvalidState;
    const int n = static_cast<int>(impl_->matrix.nrows);
    if (impl_->rhs.size != n || impl_->solution.size != n) return Status::InvalidValue;
    try {
        if (!custom_linear_solver::solve::solve_multifrontal_device(
                impl_->plan, impl_->rhs.values, impl_->solution.values, impl_->d_perm.ptr, kernel_ms))
            return Status::SolveFailed;
        return Status::Success;
    } catch (const std::bad_alloc&) {
        return Status::AllocationFailed;
    } catch (const std::exception&) {
        return Status::SolveFailed;
    }
}

Status Solver::batched_setup(int batch, bool fp32)
{
    if (!impl_ || !impl_->analyzed) return Status::InvalidState;
    if (batch <= 0) return Status::InvalidValue;
    return custom_linear_solver::factorize::batched_setup(impl_->plan, batch, fp32, impl_->batched)
               ? Status::Success
               : Status::AllocationFailed;
}

Status Solver::batched_factorize(const double* d_valuesB, double* kernel_ms)
{
    if (!impl_ || !impl_->analyzed || impl_->batched.B == 0) return Status::InvalidState;
    return custom_linear_solver::factorize::batched_factorize(
               impl_->plan, impl_->batched, d_valuesB, impl_->d_ordered_value_to_csr.ptr, kernel_ms)
               ? Status::Success
               : Status::FactorizationFailed;
}

Status Solver::batched_solve(const double* d_rhsB, double* d_solB, double* kernel_ms)
{
    if (!impl_ || !impl_->analyzed || impl_->batched.B == 0) return Status::InvalidState;
    return custom_linear_solver::factorize::batched_solve(impl_->plan, impl_->batched, d_rhsB,
                                                          d_solB, impl_->d_perm.ptr, kernel_ms)
               ? Status::Success
               : Status::SolveFailed;
}

const char* status_string(Status status)
{
    switch (status) {
        case Status::Success:
            return "success";
        case Status::InvalidValue:
            return "invalid value";
        case Status::InvalidState:
            return "invalid state";
        case Status::AllocationFailed:
            return "allocation failed";
        case Status::AnalysisFailed:
            return "analysis failed";
        case Status::FactorizationFailed:
            return "factorization failed";
        case Status::SolveFailed:
            return "solve failed";
    }
    return "unknown";
}

}  // namespace custom_linear_solver
