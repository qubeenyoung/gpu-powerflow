// ---------------------------------------------------------------------------
// cuda_custom_solver.cpp
//
// Adapter that drives the external custom_linear_solver library (FP64, single
// case) behind the same analyze -> factorize -> solve interface the pipelines
// expect. The adjoint / transpose-solve methods are intentionally unsupported
// here and throw. Compiled only when CUPF_ENABLE_CUSTOM_SOLVER is set.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_custom_solver.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <solver.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>


namespace {

namespace cls = custom_linear_solver;

// Select the factor precision via SolverConfig. The library runs the
// mixed/FP32/TF32/FP16 factor internally on the registered Jacobian, so cuPF
// only picks the precision here. Env CUPF_CUSTOM_PRECISION = fp64|fp32|tf32|fp16|tc|mixed
// (default = the per-class `dflt`). "tc" maps to TF32 for back-compat; "mixed"
// maps to FP32 (the FP32 storage has no FP64 master to refine against).
cls::SolverConfig config_from_env(cls::Precision dflt)
{
    cls::SolverConfig config;
    config.precision = dflt;
    if (const char* s = std::getenv("CUPF_CUSTOM_PRECISION")) {
        if (std::strcmp(s, "fp64") == 0)      config.precision = cls::Precision::FP64;
        else if (std::strcmp(s, "fp32") == 0) config.precision = cls::Precision::FP32;
        else if (std::strcmp(s, "tf32") == 0) config.precision = cls::Precision::TF32;
        else if (std::strcmp(s, "fp16") == 0) config.precision = cls::Precision::FP16;
        else if (std::strcmp(s, "tc") == 0)   config.precision = cls::Precision::TF32; // back-compat
        else if (std::strcmp(s, "mixed") == 0) config.precision = cls::Precision::FP32;
    }
    return config;
}

// Translate a library status into an exception (no-op on success).
void check_status(cls::Status status, const char* where)
{
    if (status == cls::Status::kSuccess) {
        return;
    }
    throw std::runtime_error(std::string("CudaLinearSolveCustomFp64::") +
                             where + ": custom solver " +
                             cls::status_string(status));
}

// Wrap the device-resident CSR Jacobian as the library's matrix view (no copy).
cls::CsrMatrixView make_matrix_view(CudaFp64Storage& buf)
{
    cls::CsrMatrixView matrix;
    matrix.nrows = buf.dimF;
    matrix.ncols = buf.dimF;
    matrix.nnz = static_cast<int64_t>(buf.d_J_values.size());  // library wants int64 nnz
    matrix.index_type = cls::IndexType::kInt32;
    matrix.location = cls::DataLocation::kDevice;
    matrix.value_type = cls::ValueType::kFloat64;
    matrix.row_offsets = buf.d_J_row_ptr.data();
    matrix.col_indices = buf.d_J_col_idx.data();
    matrix.values = buf.d_J_values.data();
    return matrix;
}

// Wrap a device pointer as the library's dense vector view (no copy).
cls::DenseVectorView make_vector_view(int32_t size, double* values)
{
    cls::DenseVectorView vector;
    vector.size = size;
    vector.location = cls::DataLocation::kDevice;
    vector.value_type = cls::ValueType::kFloat64;
    vector.values = values;
    return vector;
}

cls::DenseVectorView make_vector_view(int32_t size, float* values)
{
    cls::DenseVectorView vector;
    vector.size = size;
    vector.location = cls::DataLocation::kDevice;
    vector.value_type = cls::ValueType::kFloat32;
    vector.values = values;
    return vector;
}

cls::CsrMatrixView make_float_matrix_view(int32_t dim, int32_t nnz, const int32_t* row_ptr,
                                          const int32_t* col_idx, const float* values)
{
    cls::CsrMatrixView matrix;
    matrix.nrows = dim;
    matrix.ncols = dim;
    matrix.nnz = nnz;
    matrix.index_type = cls::IndexType::kInt32;
    matrix.location = cls::DataLocation::kDevice;
    matrix.value_type = cls::ValueType::kFloat32;
    matrix.row_offsets = row_ptr;
    matrix.col_indices = col_idx;
    matrix.values = values;
    return matrix;
}

// Shared throw for the unimplemented adjoint surface ([[noreturn]]).
[[noreturn]] void throw_adjoint_unsupported()
{
    throw std::runtime_error(
        "CudaLinearSolveCustomFp64: adjoint/transpose solve is not implemented for the custom solver");
}

}  // namespace


// Owns the library solver handle plus the pipeline-progress flags.
struct CudaLinearSolveCustomFp64::State {
    cls::Solver solver;
    bool analyzed = false;
    bool factorized = false;
    int batch = 0;          // last setup() B (0 = not set up yet); B>1 uses the uniform-batch path
    bool graph_mode = false; // graph-capture mode: capture-safe (no host sync)
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

    // Bind J / F / dx views into the solver, then run symbolic analysis once.
    auto state = std::make_unique<State>();
    state->solver = cls::Solver(config_from_env(cls::Precision::FP64));
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
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::factorize: initialize() must be called first");
    }
    state_->factorized = false;
    const int batch = buf.batch_size;
    if (state_->batch != batch) {                 // skipped in graph mode (graph_prepare already set it up)
        check_status(state_->solver.setup(batch), "setup");
        state_->batch = batch;
    }
    check_status(state_->solver.set_values(buf.d_J_values.data(), cls::ValueType::kFloat64), "set_values");
    check_status(state_->solver.factorize(), "factorize");
    if (!state_->graph_mode) sync_cuda_for_timing();
    state_->factorized = true;
}


void CudaLinearSolveCustomFp64::solve(CudaFp64Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::solve: initialize() must be called first");
    }
    if (!state_->factorized) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::solve: factorize() must be called first");
    }
    check_status(state_->solver.set_rhs(make_vector_view(state_->batch * buf.dimF, buf.d_F.data())), "set_rhs");
    check_status(state_->solver.set_solution(make_vector_view(state_->batch * buf.dimF, buf.d_dx.data())), "set_solution");
    check_status(state_->solver.solve(), "solve");
    if (!state_->graph_mode) sync_cuda_for_timing();
}

#ifdef CUPF_ENABLE_CUDA_GRAPH
void CudaLinearSolveCustomFp64::graph_prepare(CudaFp64Storage& buf, void* stream)
{
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::graph_prepare: initialize() must be called first");
    }
    const int batch = buf.batch_size;
    check_status(state_->solver.setup(batch), "setup");
    state_->batch = batch;
    state_->graph_mode = true;
    state_->factorized = false;
    check_status(state_->solver.set_stream(stream), "set_stream");
}
#endif


// --- Adjoint / transpose-solve surface: not implemented for this backend ---

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

// ===========================================================================
// CudaLinearSolveCustomFp32 — FP32 storage adapter
// ===========================================================================

struct CudaLinearSolveCustomFp32::State {
    cls::Solver solver;
    bool analyzed = false;
    bool factorized = false;
    int batch = 0;
    bool graph_mode = false;
};


CudaLinearSolveCustomFp32::CudaLinearSolveCustomFp32() = default;


CudaLinearSolveCustomFp32::~CudaLinearSolveCustomFp32()
{
    delete state_;
}


CudaLinearSolveCustomFp32::CudaLinearSolveCustomFp32(CudaLinearSolveCustomFp32&& other) noexcept
    : state_(std::exchange(other.state_, nullptr))
{}


void CudaLinearSolveCustomFp32::initialize(CudaFp32Storage& buf, const InitializeContext& ctx)
{
    if (buf.dimF <= 0 || buf.d_J_row_ptr.empty() || buf.d_J_col_idx.empty() ||
        buf.d_J_values.empty() || buf.d_F.empty() || buf.d_dx.empty()) {
        throw std::runtime_error("CudaLinearSolveCustomFp32::initialize: buffers are not prepared");
    }
    if (ctx.J.dim != buf.dimF || ctx.J.nnz != buf.nnz_J) {
        throw std::runtime_error("CudaLinearSolveCustomFp32::initialize: Jacobian pattern mismatch");
    }

    auto state = std::make_unique<State>();
    state->solver = cls::Solver(config_from_env(cls::Precision::FP32));
    check_status(state->solver.set_data(make_float_matrix_view(
                     buf.dimF, buf.nnz_J, buf.d_J_row_ptr.data(), buf.d_J_col_idx.data(),
                     buf.d_J_values.data())),
                 "set_data");
    check_status(state->solver.set_rhs(make_vector_view(buf.dimF, buf.d_F.data())), "set_rhs");
    check_status(state->solver.set_solution(make_vector_view(buf.dimF, buf.d_dx.data())),
                 "set_solution");
    check_status(state->solver.analyze(), "analyze");
    sync_cuda_for_timing();
    state->analyzed = true;

    delete state_;
    state_ = state.release();
}


void CudaLinearSolveCustomFp32::prepare_rhs(CudaFp32Storage& buf, IterationContext& ctx)
{
    (void)buf;
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp32::prepare_rhs: initialize() must be called first");
    }
}


void CudaLinearSolveCustomFp32::factorize(CudaFp32Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp32::factorize: initialize() must be called first");
    }
    state_->factorized = false;
    const int batch = buf.batch_size;
    if (state_->batch != batch) {                 // skipped in graph mode (graph_prepare already set it up)
        check_status(state_->solver.setup(batch), "setup");
        state_->batch = batch;
    }
    check_status(state_->solver.set_values(buf.d_J_values.data(), cls::ValueType::kFloat32), "set_values");
    check_status(state_->solver.factorize(), "factorize");
    if (!state_->graph_mode) sync_cuda_for_timing();
    state_->factorized = true;
}


void CudaLinearSolveCustomFp32::solve(CudaFp32Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp32::solve: initialize() must be called first");
    }
    if (!state_->factorized) {
        throw std::runtime_error("CudaLinearSolveCustomFp32::solve: factorize() must be called first");
    }
    check_status(state_->solver.set_rhs(make_vector_view(state_->batch * buf.dimF, buf.d_F.data())), "set_rhs");
    check_status(state_->solver.set_solution(make_vector_view(state_->batch * buf.dimF, buf.d_dx.data())), "set_solution");
    check_status(state_->solver.solve(), "solve");
    if (!state_->graph_mode) sync_cuda_for_timing();
}

#ifdef CUPF_ENABLE_CUDA_GRAPH
void CudaLinearSolveCustomFp32::graph_prepare(CudaFp32Storage& buf, void* stream)
{
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp32::graph_prepare: initialize() must be called first");
    }
    const int batch = buf.batch_size;
    check_status(state_->solver.setup(batch), "setup");
    state_->batch = batch;
    state_->graph_mode = true;
    state_->factorized = false;
    check_status(state_->solver.set_stream(stream), "set_stream");
}
#endif

void CudaLinearSolveCustomFp32::prepare_adjoint_explicit_transpose_cache(
    CudaFp32Storage& buf, IterationContext& ctx, double& factorization_time_ms)
{
    (void)buf;
    (void)ctx;
    (void)factorization_time_ms;
    throw_adjoint_unsupported();
}


void CudaLinearSolveCustomFp32::solve_adjoint_explicit_transpose_host(
    const double* rhs, double* solution, int32_t batch_size, double& solve_time_ms)
{
    (void)rhs;
    (void)solution;
    (void)batch_size;
    (void)solve_time_ms;
    throw_adjoint_unsupported();
}


float* CudaLinearSolveCustomFp32::adjoint_rhs_data()
{
    throw_adjoint_unsupported();
}


float* CudaLinearSolveCustomFp32::adjoint_solution_data()
{
    throw_adjoint_unsupported();
}


void CudaLinearSolveCustomFp32::solve_adjoint_explicit_transpose_cached(double& solve_time_ms)
{
    (void)solve_time_ms;
    throw_adjoint_unsupported();
}


// ===========================================================================
// CudaLinearSolveCustomMixed — FP32-input (cuPF Mixed profile) adapter
// ===========================================================================

struct CudaLinearSolveCustomMixed::State {
    cls::Solver solver;
    bool analyzed = false;
    bool factorized = false;
    int batch = 0;          // last setup() B (0 = not set up yet)
    bool graph_mode = false; // graph-capture mode: capture-safe (no host sync)
};


CudaLinearSolveCustomMixed::CudaLinearSolveCustomMixed() = default;


CudaLinearSolveCustomMixed::~CudaLinearSolveCustomMixed()
{
    delete state_;
}


CudaLinearSolveCustomMixed::CudaLinearSolveCustomMixed(CudaLinearSolveCustomMixed&& other) noexcept
    : state_(std::exchange(other.state_, nullptr))
{}


void CudaLinearSolveCustomMixed::initialize(CudaMixedStorage& buf, const InitializeContext& ctx)
{
    if (buf.dimF <= 0 || buf.d_J_row_ptr.empty() || buf.d_J_col_idx.empty() ||
        buf.d_J_values.empty() || buf.d_F.empty() || buf.d_dx.empty()) {
        throw std::runtime_error("CudaLinearSolveCustomMixed::initialize: buffers are not prepared");
    }
    if (ctx.J.dim != buf.dimF || ctx.J.nnz != buf.nnz_J) {
        throw std::runtime_error("CudaLinearSolveCustomMixed::initialize: Jacobian pattern mismatch");
    }

    // The Jacobian values are FP32, so only the pattern is bound here (values = nullptr); the
    // numeric values are supplied per factorize via set_values. analyze (symbolic) reads only
    // the pattern.
    cls::CsrMatrixView matrix;
    matrix.nrows = buf.dimF;
    matrix.ncols = buf.dimF;
    matrix.nnz = buf.nnz_J;  // per-case nnz (pattern shared across the batch)
    matrix.index_type = cls::IndexType::kInt32;
    matrix.location = cls::DataLocation::kDevice;
    matrix.value_type = cls::ValueType::kFloat32;
    matrix.row_offsets = buf.d_J_row_ptr.data();
    matrix.col_indices = buf.d_J_col_idx.data();
    matrix.values = nullptr;

    auto state = std::make_unique<State>();
    state->solver = cls::Solver(config_from_env(cls::Precision::FP32));
    check_status(state->solver.set_data(matrix), "set_data");
    check_status(state->solver.analyze(), "analyze");
    sync_cuda_for_timing();
    state->analyzed = true;

    delete state_;
    state_ = state.release();
}


void CudaLinearSolveCustomMixed::prepare_rhs(CudaMixedStorage& buf, IterationContext& ctx)
{
    (void)buf;
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomMixed::prepare_rhs: initialize() must be called first");
    }
}


void CudaLinearSolveCustomMixed::factorize(CudaMixedStorage& buf, IterationContext& ctx)
{
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomMixed::factorize: initialize() must be called first");
    }
    state_->factorized = false;
    const int batch = buf.batch_size;
    if (state_->batch != batch) {                 // skipped in graph mode (graph_prepare already set it up)
        check_status(state_->solver.setup(batch), "setup");
        state_->batch = batch;
    }
    check_status(state_->solver.set_values(buf.d_J_values.data(), cls::ValueType::kFloat32), "set_values");
    check_status(state_->solver.factorize(), "factorize");
    if (!state_->graph_mode) sync_cuda_for_timing();
    state_->factorized = true;
}


void CudaLinearSolveCustomMixed::solve(CudaMixedStorage& buf, IterationContext& ctx)
{
    (void)ctx;
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomMixed::solve: initialize() must be called first");
    }
    if (!state_->factorized) {
        throw std::runtime_error("CudaLinearSolveCustomMixed::solve: factorize() must be called first");
    }
    check_status(state_->solver.set_rhs(make_vector_view(state_->batch * buf.dimF, buf.d_F.data())), "set_rhs");
    check_status(state_->solver.set_solution(make_vector_view(state_->batch * buf.dimF, buf.d_dx.data())), "set_solution");
    check_status(state_->solver.solve(), "solve");
    if (!state_->graph_mode) sync_cuda_for_timing();
}

#ifdef CUPF_ENABLE_CUDA_GRAPH
void CudaLinearSolveCustomMixed::graph_prepare(CudaMixedStorage& buf, void* stream)
{
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomMixed::graph_prepare: initialize() must be called first");
    }
    const int batch = buf.batch_size;
    check_status(state_->solver.setup(batch), "setup");
    state_->batch = batch;
    state_->graph_mode = true;
    state_->factorized = false;
    check_status(state_->solver.set_stream(stream), "set_stream");
}
#endif

// --- Adjoint / transpose-solve surface: not implemented for this backend ---

void CudaLinearSolveCustomMixed::prepare_adjoint_explicit_transpose_cache(
    CudaMixedStorage& buf, IterationContext& ctx, double& factorization_time_ms)
{
    (void)buf;
    (void)ctx;
    (void)factorization_time_ms;
    throw_adjoint_unsupported();
}


void CudaLinearSolveCustomMixed::solve_adjoint_explicit_transpose_host(
    const double* rhs, double* solution, int32_t batch_size, double& solve_time_ms)
{
    (void)rhs;
    (void)solution;
    (void)batch_size;
    (void)solve_time_ms;
    throw_adjoint_unsupported();
}


float* CudaLinearSolveCustomMixed::adjoint_rhs_data()
{
    throw_adjoint_unsupported();
}


float* CudaLinearSolveCustomMixed::adjoint_solution_data()
{
    throw_adjoint_unsupported();
}


void CudaLinearSolveCustomMixed::solve_adjoint_explicit_transpose_cached(double& solve_time_ms)
{
    (void)solve_time_ms;
    throw_adjoint_unsupported();
}

#endif  // CUPF_WITH_CUDA
