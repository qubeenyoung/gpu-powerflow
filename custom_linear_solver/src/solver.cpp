#include "solver.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "factorize/multifrontal.hpp"
#include "batched/multifrontal_batched.hpp"
#include "tc/multifrontal_tc.hpp"
#include "matrix/pattern_kernels.hpp"
#include "symbolic/supernode.hpp"
#include "symbolic/amalgamate.hpp"
#include "reordering/metis_nd.hpp"
#include "solve/multifrontal.hpp"
#include "symbolic/elimination_tree.hpp"

#include <cstdio>
#include <cstdlib>

namespace custom_linear_solver {
namespace {

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
    custom_linear_solver::batched::BatchedState batched;
    custom_linear_solver::tc::TCState tc;
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
    // values may be null: the batched float-input path (set via batched_factorize(const float*))
    // supplies numeric values separately and only needs the pattern here. The single-case
    // factorize() rechecks values != nullptr at the point it actually reads them.
    if (matrix.row_offsets == nullptr || matrix.col_indices == nullptr)
        return Status::InvalidValue;
    impl_->matrix = matrix;
    impl_->has_matrix = true;
    impl_->analyzed = false;
    return Status::Success;
}

Status Solver::set_values(const void* values, ValueType value_type)
{
    if (!impl_) return Status::InvalidState;
    if (!impl_->has_matrix || values == nullptr) return Status::InvalidState;
    impl_->matrix.values = values;
    impl_->matrix.value_type = value_type;
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
        auto lap = [](const char*) {};

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

        // Fill pattern in METIS order. When amalgamating, the per-column counts come from here for
        // free (no separate cs_counts) and the pattern is relabeled to the reordered space below
        // (the reorder is a postorder -> fill-neutral), avoiding a second fill_pattern.
        std::vector<int> Lp, Li;
        custom_linear_solver::symbolic::fill_pattern(n, sym_col_ptr.data(), sym_row_idx.data(),
                                                     parent, Lp, Li);
        lap("fill_pattern");

        // ---- Etree-aware amalgamation + repostorder (CLS_USE_AMAL=1, opt-in) -------------
        // Default (no env): existing chain-merge `relaxed_panels` runs inside
        // analyze_multifrontal -- avg 2 cols/panel, panel-etree depth 30/39 (case8387 / USA),
        // 3-4× deeper than cuDSS. With CLS_USE_AMAL=1 we run a second amalgamation pass that
        // greedily merges chain panels with their etree parent (bottom-up, union-find) up to
        // `amal_cap` columns, then re-postorders the supernode etree so each merged panel's
        // cols are a contiguous range in the new ordering. Python prototype
        // (tests/depth_analysis_v2.py) shows case8387 depth 30->12 (cap=32), USA 39->27.
        //
        // amal_cap defaults to 32 (matches batched MF_REG_NC); CLS_AMAL_CAP overrides.
        const char* amal_env = std::getenv("CLS_USE_AMAL");
        const bool use_amal = amal_env && amal_env[0] && amal_env[0] != '0';
        custom_linear_solver::symbolic::PanelPartition amal_panels;
        custom_linear_solver::symbolic::PanelPartition* forced_panels_ptr = nullptr;
        std::vector<int> amal_Lp, amal_Li, amal_parent;
        custom_linear_solver::matrix::DeviceCscPattern amal_ordered_device;
        const int* analyze_col_ptr = ordered_device.col_ptr.ptr;
        const int* analyze_row_idx = ordered_device.row_idx.ptr;
        const std::vector<int>* analyze_Lp = &Lp;
        const std::vector<int>* analyze_Li = &Li;
        const std::vector<int>* analyze_parent = &parent;
        if (use_amal) {
            // Match the eff_cap adaptive logic in analyze_multifrontal so we get the same
            // chain panels (and thus the same starting structure to amalgamate from).
            int chain_cap = impl_->config.panel_cap;
            if (n >= 80000) chain_cap = 20;
            else if (n >= 16000) chain_cap = 12;
            if (const char* ce = std::getenv("CLS_CAP")) chain_cap = std::atoi(ce);
            if (chain_cap < 1) chain_cap = 1;
            int amal_cap = 32;
            if (const char* ac = std::getenv("CLS_AMAL_CAP")) amal_cap = std::atoi(ac);
            if (amal_cap < chain_cap) amal_cap = chain_cap;

            std::vector<int> colcount(n);
            for (int j = 0; j < n; ++j) colcount[j] = Lp[j + 1] - Lp[j];
            const auto chain_panels = custom_linear_solver::symbolic::relaxed_panels(
                n, parent, colcount, chain_cap);
            const auto amal = custom_linear_solver::symbolic::amalgamate_and_repostorder(
                n, parent, chain_panels, amal_cap);
            lap("amalgamate_and_repostorder");

            // Compose perm/iperm: final perm = METIS perm ∘ amalgamation perm
            std::vector<int> composed_perm(n);
            for (int new_idx = 0; new_idx < n; ++new_idx) {
                composed_perm[new_idx] = impl_->perm[amal.perm[new_idx]];
            }
            impl_->perm = std::move(composed_perm);
            impl_->iperm.assign(n, 0);
            for (int k = 0; k < n; ++k) impl_->iperm[impl_->perm[k]] = k;
            st = impl_->d_perm.upload(impl_->perm);
            if (st != Status::Success) return st;
            st = impl_->d_iperm.upload(impl_->iperm);
            if (st != Status::Success) return st;
            lap("amal_compose_perm");

            // Re-permute the device CSC matrix using the composed iperm.
            st = custom_linear_solver::matrix::permute_csc_device(
                csc_device, impl_->d_iperm.ptr, amal_ordered_device);
            if (st != Status::Success) return st;
            impl_->d_ordered_value_to_csr = std::move(amal_ordered_device.source_pos);
            lap("amal_permute_csc");

            // Permute Lp/Li to the new index space (avoids a second fill_pattern).
            // new_col_k = amal.perm[k] in OLD (METIS) index space.
            amal_Lp.assign(n + 1, 0);
            for (int k = 0; k < n; ++k) {
                const int old_col = amal.perm[k];
                amal_Lp[k + 1] = amal_Lp[k] + (Lp[old_col + 1] - Lp[old_col]);
            }
            amal_Li.assign(amal_Lp[n], 0);
            for (int k = 0; k < n; ++k) {
                const int old_col = amal.perm[k];
                int w = amal_Lp[k];
                for (int p = Lp[old_col]; p < Lp[old_col + 1]; ++p) {
                    amal_Li[w++] = amal.iperm[Li[p]];
                }
                std::sort(amal_Li.begin() + amal_Lp[k], amal_Li.begin() + amal_Lp[k + 1]);
            }
            lap("amal_permute_Lp_Li");

            amal_parent = amal.new_parent;
            amal_panels = amal.panels;
            // Populate panel widths from the fill colcount (in new index space).
            amal_panels.width.assign(amal_panels.num_panels, 0);
            for (int p = 0; p < amal_panels.num_panels; ++p) {
                int wmax = 0;
                const int first_col = amal_panels.first[p];
                const int ncols = amal_panels.ncols[p];
                for (int c = first_col; c < first_col + ncols; ++c) {
                    const int cc = amal_Lp[c + 1] - amal_Lp[c];
                    if (cc > wmax) wmax = cc;
                }
                amal_panels.width[p] = wmax;
            }
            forced_panels_ptr = &amal_panels;
            analyze_col_ptr = amal_ordered_device.col_ptr.ptr;
            analyze_row_idx = amal_ordered_device.row_idx.ptr;
            analyze_Lp = &amal_Lp;
            analyze_Li = &amal_Li;
            analyze_parent = &amal_parent;

            if (std::getenv("CLS_AMAL_INFO")) {
                fprintf(stderr,
                        "[CLS_AMAL_INFO] chain_cap=%d amal_cap=%d  chain_P=%d -> amal_P=%d  "
                        "(avg cols/panel %.2f -> %.2f)\n",
                        chain_cap, amal_cap, chain_panels.num_panels, amal_panels.num_panels,
                        (double)n / chain_panels.num_panels,
                        (double)n / amal_panels.num_panels);
            }
        }

        impl_->plan = custom_linear_solver::factorize::analyze_multifrontal(
            n, nnz, analyze_col_ptr, analyze_row_idx, *analyze_Lp, *analyze_Li, *analyze_parent,
            impl_->config.panel_cap,
            impl_->config.single_precision == SinglePrecision::Mixed,
            forced_panels_ptr,
            impl_->config.single_precision == SinglePrecision::FP32);
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
    if (impl_->matrix.values == nullptr) return Status::InvalidState;  // single-case needs values
    try {
        bool ok = false;
        if (impl_->config.single_precision == SinglePrecision::FP32) {
            if (impl_->matrix.value_type != ValueType::Float32) return Status::InvalidValue;
            ok = custom_linear_solver::factorize::factorize_multifrontal_device(
                impl_->plan, static_cast<const float*>(impl_->matrix.values),
                impl_->d_ordered_value_to_csr.ptr, kernel_ms);
        } else {
            if (impl_->matrix.value_type != ValueType::Float64) return Status::InvalidValue;
            ok = custom_linear_solver::factorize::factorize_multifrontal_device(
                impl_->plan, static_cast<const double*>(impl_->matrix.values),
                impl_->d_ordered_value_to_csr.ptr, kernel_ms);
        }
        if (!ok) return Status::FactorizationFailed;
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
        bool ok = false;
        if (impl_->config.single_precision == SinglePrecision::FP32) {
            if (impl_->rhs.value_type == ValueType::Float32 &&
                impl_->solution.value_type == ValueType::Float32) {
                ok = custom_linear_solver::solve::solve_multifrontal_device(
                    impl_->plan, static_cast<const float*>(impl_->rhs.values),
                    static_cast<float*>(impl_->solution.values), impl_->d_perm.ptr, kernel_ms);
            } else if (impl_->rhs.value_type == ValueType::Float64 &&
                       impl_->solution.value_type == ValueType::Float32) {
                ok = custom_linear_solver::solve::solve_multifrontal_device(
                    impl_->plan, static_cast<const double*>(impl_->rhs.values),
                    static_cast<float*>(impl_->solution.values), impl_->d_perm.ptr, kernel_ms);
            } else {
                return Status::InvalidValue;
            }
        } else {
            if (impl_->rhs.value_type != ValueType::Float64 ||
                impl_->solution.value_type != ValueType::Float64) {
                return Status::InvalidValue;
            }
            ok = custom_linear_solver::solve::solve_multifrontal_device(
                impl_->plan, static_cast<const double*>(impl_->rhs.values),
                static_cast<double*>(impl_->solution.values), impl_->d_perm.ptr, kernel_ms);
        }
        if (!ok) return Status::SolveFailed;
        return Status::Success;
    } catch (const std::bad_alloc&) {
        return Status::AllocationFailed;
    } catch (const std::exception&) {
        return Status::SolveFailed;
    }
}

Status Solver::set_stream(void* stream)
{
    if (!impl_ || !impl_->analyzed) return Status::InvalidState;
    impl_->plan.stream = stream;
    impl_->plan.owns_stream = false;
    return Status::Success;
}

Status Solver::batched_setup(int batch, custom_linear_solver::batched::BatchPrecision prec)
{
    if (!impl_ || !impl_->analyzed) return Status::InvalidState;
    if (batch <= 0) return Status::InvalidValue;
    return custom_linear_solver::batched::batched_setup(impl_->plan, batch, prec, impl_->batched)
               ? Status::Success
               : Status::AllocationFailed;
}

Status Solver::batched_set_stream(void* stream)
{
    if (!impl_ || !impl_->analyzed || impl_->batched.B == 0) return Status::InvalidState;
    custom_linear_solver::batched::batched_set_stream(impl_->batched, stream);
    return Status::Success;
}

Status Solver::batched_factorize(const double* d_valuesB, double* kernel_ms)
{
    if (!impl_ || !impl_->analyzed || impl_->batched.B == 0) return Status::InvalidState;
    return custom_linear_solver::batched::batched_factorize(
               impl_->plan, impl_->batched, d_valuesB, impl_->d_ordered_value_to_csr.ptr, kernel_ms)
               ? Status::Success
               : Status::FactorizationFailed;
}

Status Solver::batched_solve(const double* d_rhsB, double* d_solB, double* kernel_ms)
{
    if (!impl_ || !impl_->analyzed || impl_->batched.B == 0) return Status::InvalidState;
    return custom_linear_solver::batched::batched_solve(impl_->plan, impl_->batched, d_rhsB,
                                                          d_solB, impl_->d_perm.ptr, kernel_ms)
               ? Status::Success
               : Status::SolveFailed;
}

Status Solver::batched_factorize(const float* d_valuesB, double* kernel_ms)
{
    if (!impl_ || !impl_->analyzed || impl_->batched.B == 0) return Status::InvalidState;
    return custom_linear_solver::batched::batched_factorize(
               impl_->plan, impl_->batched, d_valuesB, impl_->d_ordered_value_to_csr.ptr, kernel_ms)
               ? Status::Success
               : Status::FactorizationFailed;
}

Status Solver::batched_solve(const double* d_rhsB, float* d_solB, double* kernel_ms)
{
    if (!impl_ || !impl_->analyzed || impl_->batched.B == 0) return Status::InvalidState;
    return custom_linear_solver::batched::batched_solve(impl_->plan, impl_->batched, d_rhsB,
                                                          d_solB, impl_->d_perm.ptr, kernel_ms)
               ? Status::Success
               : Status::SolveFailed;
}

Status Solver::batched_solve(const float* d_rhsB, float* d_solB, double* kernel_ms)
{
    if (!impl_ || !impl_->analyzed || impl_->batched.B == 0) return Status::InvalidState;
    return custom_linear_solver::batched::batched_solve(impl_->plan, impl_->batched, d_rhsB,
                                                          d_solB, impl_->d_perm.ptr, kernel_ms)
               ? Status::Success
               : Status::SolveFailed;
}

// ---- TC-dedicated path ------------------------------------------------------------------------

Status Solver::tc_setup(int batch)
{
    if (!impl_ || !impl_->analyzed) return Status::InvalidState;
    return custom_linear_solver::tc::tc_setup(impl_->plan, batch, impl_->tc) ? Status::Success
                                                                              : Status::InvalidState;
}

Status Solver::tc_set_stream(void* stream)
{
    if (!impl_ || !impl_->analyzed || impl_->tc.B == 0) return Status::InvalidState;
    custom_linear_solver::tc::tc_set_stream(impl_->tc, stream);
    return Status::Success;
}

Status Solver::tc_factorize(const float* d_valuesB, double* kernel_ms)
{
    if (!impl_ || !impl_->analyzed || impl_->tc.B == 0) return Status::InvalidState;
    return custom_linear_solver::tc::tc_factorize(
               impl_->plan, impl_->tc, d_valuesB, impl_->d_ordered_value_to_csr.ptr, kernel_ms)
               ? Status::Success
               : Status::FactorizationFailed;
}

Status Solver::tc_solve(const float* d_rhsB, float* d_solB, double* kernel_ms)
{
    if (!impl_ || !impl_->analyzed || impl_->tc.B == 0) return Status::InvalidState;
    return custom_linear_solver::tc::tc_solve(impl_->plan, impl_->tc, d_rhsB, d_solB,
                                              impl_->d_perm.ptr, kernel_ms)
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
