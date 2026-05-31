#ifdef CUPF_WITH_CUDA

#include "cuda_custom_solver.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <solver.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>


namespace {

namespace cls = custom_linear_solver;

void check_status(cls::Status status, const char* where)
{
    if (status == cls::Status::Success) {
        return;
    }
    throw std::runtime_error(std::string("CudaLinearSolveCustomFp64::") +
                             where + ": custom solver " +
                             cls::status_string(status));
}

cls::CsrMatrixView make_matrix_view(CudaFp64Storage& buf)
{
    cls::CsrMatrixView matrix;
    matrix.nrows = buf.dimF;
    matrix.ncols = buf.dimF;
    matrix.nnz = static_cast<int64_t>(buf.d_J_values.size());
    matrix.index_type = cls::IndexType::Int32;
    matrix.location = cls::DataLocation::Device;
    matrix.row_offsets = buf.d_J_row_ptr.data();
    matrix.col_indices = buf.d_J_col_idx.data();
    matrix.values = buf.d_J_values.data();
    return matrix;
}

cls::DenseVectorView make_vector_view(int32_t size, double* values)
{
    cls::DenseVectorView vector;
    vector.size = size;
    vector.location = cls::DataLocation::Device;
    vector.values = values;
    return vector;
}

[[noreturn]] void throw_adjoint_unsupported()
{
    throw std::runtime_error(
        "CudaLinearSolveCustomFp64: adjoint/transpose solve is not implemented for the custom solver");
}

}  // namespace


struct CudaLinearSolveCustomFp64::State {
    cls::Solver solver;
    bool analyzed = false;
    bool factorized = false;
};


CudaLinearSolveCustomFp64::CudaLinearSolveCustomFp64() = default;


CudaLinearSolveCustomFp64::~CudaLinearSolveCustomFp64()
{
    delete state_;
}


CudaLinearSolveCustomFp64::CudaLinearSolveCustomFp64(CudaLinearSolveCustomFp64&& other) noexcept
    : state_(std::exchange(other.state_, nullptr))
{}


void CudaLinearSolveCustomFp64::initialize(CudaFp64Storage& buf, const InitializeContext& ctx)
{
    if (buf.dimF <= 0 || buf.d_J_row_ptr.empty() || buf.d_J_col_idx.empty() ||
        buf.d_J_values.empty() || buf.d_F.empty() || buf.d_dx.empty()) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::initialize: buffers are not prepared");
    }
    if (ctx.J.dim != buf.dimF ||
        ctx.J.nnz != static_cast<int32_t>(buf.d_J_values.size())) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::initialize: Jacobian pattern mismatch");
    }

    auto state = std::make_unique<State>();
    check_status(state->solver.set_data(make_matrix_view(buf)), "set_data");
    check_status(state->solver.set_rhs(make_vector_view(buf.dimF, buf.d_F.data())), "set_rhs");
    check_status(state->solver.set_solution(make_vector_view(buf.dimF, buf.d_dx.data())),
                 "set_solution");
    check_status(state->solver.analyze(), "analyze");
    sync_cuda_for_timing();
    state->analyzed = true;

    delete state_;
    state_ = state.release();
}


void CudaLinearSolveCustomFp64::prepare_rhs(CudaFp64Storage& buf, IterationContext& ctx)
{
    (void)buf;
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::prepare_rhs: initialize() must be called first");
    }
}


void CudaLinearSolveCustomFp64::factorize(CudaFp64Storage& buf, IterationContext& ctx)
{
    (void)buf;
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::factorize: initialize() must be called first");
    }
    state_->factorized = false;
    check_status(state_->solver.factorize(), "factorize");
    sync_cuda_for_timing();
    state_->factorized = true;
}


void CudaLinearSolveCustomFp64::solve(CudaFp64Storage& buf, IterationContext& ctx)
{
    (void)buf;
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::solve: initialize() must be called first");
    }
    if (!state_->factorized) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::solve: factorize() must be called first");
    }
    check_status(state_->solver.solve(), "solve");
    sync_cuda_for_timing();
}


void CudaLinearSolveCustomFp64::prepare_adjoint_explicit_transpose_cache(
    CudaFp64Storage& buf, IterationContext& ctx, double& factorization_time_ms)
{
    (void)buf;
    (void)ctx;
    (void)factorization_time_ms;
    throw_adjoint_unsupported();
}


void CudaLinearSolveCustomFp64::solve_adjoint_explicit_transpose_host(
    const double* rhs, double* solution, int32_t batch_size, double& solve_time_ms)
{
    (void)rhs;
    (void)solution;
    (void)batch_size;
    (void)solve_time_ms;
    throw_adjoint_unsupported();
}


double* CudaLinearSolveCustomFp64::adjoint_rhs_data()
{
    throw_adjoint_unsupported();
}


double* CudaLinearSolveCustomFp64::adjoint_solution_data()
{
    throw_adjoint_unsupported();
}


void CudaLinearSolveCustomFp64::solve_adjoint_explicit_transpose_cached(double& solve_time_ms)
{
    (void)solve_time_ms;
    throw_adjoint_unsupported();
}

#endif  // CUPF_WITH_CUDA
