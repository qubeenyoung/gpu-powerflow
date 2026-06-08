#include "solver.hpp"

#include <utility>
#include <vector>

#include "matrix/pattern_kernels.hpp"  // IntDeviceBuffer
#include "multifrontal.hpp"             // setup, factorize, solve, set_stream, State
#include "plan/build.hpp"               // build_plan_from_csr
#include "profile/profile.hpp"          // cls::profile::init / flush + CLS_PROFILE_* macros

namespace custom_linear_solver {

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
    custom_linear_solver::State state;
};

Solver::Solver(const SolverConfig& config) : impl_(new Impl{config}) {
    cls::profile::init();
}
Solver::~Solver() { cls::profile::flush(); }
Solver::Solver(Solver&&) noexcept = default;
Solver& Solver::operator=(Solver&&) noexcept = default;

Status Solver::set_data(const CsrMatrixView& matrix)
{
    if (!impl_) return Status::InvalidState;
    if (matrix.nrows <= 0 || matrix.nrows != matrix.ncols || matrix.nnz < 0)
        return Status::InvalidValue;
    if (matrix.index_type != IndexType::Int32 || matrix.location != DataLocation::Device)
        return Status::InvalidValue;
    // values may be null: the float-input path (set via factorize(const float*))
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
    CLS_PROFILE_NVTX("Solver::analyze");
    CLS_PROFILE_CPU("Solver::analyze");
    if (!impl_ || !impl_->has_matrix) return Status::InvalidState;
    plan::PlanBuildOptions opts;
    opts.use_parallel_nested_dissection = impl_->config.use_parallel_nested_dissection;
    opts.panel_cap = impl_->config.panel_cap;
    opts.float_front = is_fp32_front(impl_->config.precision);
    opts.dump_fronts_csv_path = impl_->config.analyze_dump_fronts_path;
    opts.emit_analyze_info = impl_->config.analyze_emit_info;
    plan::PlanBuildResult result;
    if (!plan::build_plan_from_csr(impl_->matrix, opts, result))
        return Status::AnalysisFailed;
    impl_->perm                    = std::move(result.perm);
    impl_->iperm                   = std::move(result.iperm);
    impl_->d_perm                  = std::move(result.d_perm);
    impl_->d_iperm                 = std::move(result.d_iperm);
    impl_->d_ordered_value_to_csr  = std::move(result.d_ordered_value_to_csr);
    impl_->plan                    = std::move(result.plan);
    impl_->analyzed = true;
    return Status::Success;
}

Status Solver::setup(int batch_size)
{
    if (!impl_ || !impl_->analyzed) return Status::InvalidState;
    if (batch_size <= 0) return Status::InvalidValue;
    return custom_linear_solver::setup(impl_->plan, batch_size, impl_->config.precision,
                                       impl_->state, impl_->config.use_multistream_subtrees,
                                       impl_->config.tier_split)
               ? Status::Success
               : Status::AllocationFailed;
}

Status Solver::set_stream(void* stream)
{
    if (!impl_ || !impl_->analyzed || impl_->state.batch_count == 0) return Status::InvalidState;
    custom_linear_solver::set_stream(impl_->state, stream);
    return Status::Success;
}

Status Solver::factorize()
{
    CLS_PROFILE_NVTX("Solver::factorize");
    CLS_PROFILE_GPU("Solver::factorize",
                    static_cast<cudaStream_t>(impl_ ? impl_->state.stream : nullptr));
    if (!impl_ || !impl_->has_matrix || !impl_->analyzed) return Status::InvalidState;
    if (impl_->matrix.values == nullptr) return Status::InvalidState;
    // Auto-setup with batch size 1 if the caller skipped setup().
    if (impl_->state.batch_count == 0) {
        if (auto st = setup(1); st != Status::Success) return st;
    }
    const int* o2c = impl_->d_ordered_value_to_csr.ptr;
    bool ok = false;
    if (impl_->matrix.value_type == ValueType::Float64) {
        ok = custom_linear_solver::factorize(impl_->plan, impl_->state,
                                             static_cast<const double*>(impl_->matrix.values), o2c);
    } else if (impl_->matrix.value_type == ValueType::Float32) {
        ok = custom_linear_solver::factorize(impl_->plan, impl_->state,
                                             static_cast<const float*>(impl_->matrix.values), o2c);
    } else {
        return Status::InvalidValue;
    }
    return ok ? Status::Success : Status::FactorizationFailed;
}

Status Solver::solve()
{
    CLS_PROFILE_NVTX("Solver::solve");
    CLS_PROFILE_GPU("Solver::solve",
                    static_cast<cudaStream_t>(impl_ ? impl_->state.stream : nullptr));
    if (!impl_ || !impl_->has_rhs || !impl_->has_solution || !impl_->analyzed)
        return Status::InvalidState;
    if (impl_->state.batch_count == 0) return Status::InvalidState;
    const int* perm = impl_->d_perm.ptr;
    bool ok = false;
    const auto rt = impl_->rhs.value_type;
    const auto st = impl_->solution.value_type;
    if (rt == ValueType::Float64 && st == ValueType::Float64) {
        ok = custom_linear_solver::solve(impl_->plan, impl_->state,
                                         static_cast<const double*>(impl_->rhs.values),
                                         static_cast<double*>(impl_->solution.values), perm);
    } else if (rt == ValueType::Float64 && st == ValueType::Float32) {
        ok = custom_linear_solver::solve(impl_->plan, impl_->state,
                                         static_cast<const double*>(impl_->rhs.values),
                                         static_cast<float*>(impl_->solution.values), perm);
    } else if (rt == ValueType::Float32 && st == ValueType::Float32) {
        ok = custom_linear_solver::solve(impl_->plan, impl_->state,
                                         static_cast<const float*>(impl_->rhs.values),
                                         static_cast<float*>(impl_->solution.values), perm);
    } else {
        return Status::InvalidValue;
    }
    return ok ? Status::Success : Status::SolveFailed;
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
