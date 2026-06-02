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
#include <batched/multifrontal_batched.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>


namespace {

namespace cls = custom_linear_solver;

// Batch precision for B>1 (the single-case B==1 path stays FP64). The standalone solver does the
// mixed/FP32/FP16-TC factor internally on the FP64 Jacobian, so cuPF keeps its FP64 storage and
// only selects the factor precision here. Env CUPF_CUSTOM_PRECISION = fp64|fp32|mixed|tc
// (default mixed: fastest factor that still passes ~1e-3 without refinement). TC additionally
// needs the deep-K amalgamation (set MF_AMALG=cap:ratio so analyze grows the supernodes).
cls::batched::BatchPrecision batch_precision_from_env()
{
    const char* s = std::getenv("CUPF_CUSTOM_PRECISION");
    if (s != nullptr) {
        if (std::strcmp(s, "fp64") == 0) return cls::batched::BatchPrecision::FP64;
        if (std::strcmp(s, "fp32") == 0) return cls::batched::BatchPrecision::FP32;
        if (std::strcmp(s, "tc") == 0) return cls::batched::BatchPrecision::TC;
    }
    return cls::batched::BatchPrecision::Mixed;
}

// Factor precision for the Mixed profile. The Jacobian arrives already in FP32, so pure-FP32 is
// the natural default (no FP64 master to gain accuracy from); CUPF_CUSTOM_PRECISION still
// overrides (e.g. tc for the FP16 trailing GEMM, which also needs MF_AMALG).
cls::batched::BatchPrecision mixed_batch_precision_from_env()
{
    const char* s = std::getenv("CUPF_CUSTOM_PRECISION");
    if (s != nullptr) {
        if (std::strcmp(s, "fp64") == 0) return cls::batched::BatchPrecision::FP64;
        if (std::strcmp(s, "mixed") == 0) return cls::batched::BatchPrecision::Mixed;
        if (std::strcmp(s, "tc") == 0) return cls::batched::BatchPrecision::TC;
    }
    return cls::batched::BatchPrecision::FP32;
}

// Translate a library status into an exception (no-op on success).
void check_status(cls::Status status, const char* where)
{
    if (status == cls::Status::Success) {
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
    matrix.index_type = cls::IndexType::Int32;
    matrix.location = cls::DataLocation::Device;
    matrix.value_type = cls::ValueType::Float64;
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
    vector.location = cls::DataLocation::Device;
    vector.value_type = cls::ValueType::Float64;
    vector.values = values;
    return vector;
}

cls::DenseVectorView make_vector_view(int32_t size, float* values)
{
    cls::DenseVectorView vector;
    vector.size = size;
    vector.location = cls::DataLocation::Device;
    vector.value_type = cls::ValueType::Float32;
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
    matrix.index_type = cls::IndexType::Int32;
    matrix.location = cls::DataLocation::Device;
    matrix.value_type = cls::ValueType::Float32;
    matrix.row_offsets = row_ptr;
    matrix.col_indices = col_idx;
    matrix.values = values;
    return matrix;
}

cls::SolverConfig fp32_single_config()
{
    cls::SolverConfig config;
    config.single_precision = cls::SinglePrecision::FP32;
    return config;
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
    int batch = 1;          // B (= buf.batch_size); B>1 uses the library's uniform-batch path
    bool batched = false;   // true iff batch > 1
    bool graph_mode = false; // graph-capture mode: always batched, capture-safe (no host sync)
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
#ifdef CUPF_ENABLE_CUDA_GRAPH
    if (state_->graph_mode) {
        if (state_->batched) {
            check_status(state_->solver.batched_factorize(buf.d_J_values.data()), "batched_factorize");
        } else {
            check_status(state_->solver.set_values(buf.d_J_values.data(), cls::ValueType::Float64),
                         "set_values");
            check_status(state_->solver.factorize(), "factorize");
        }
        state_->factorized = true;
        return;
    }
#endif
    // The batch size is known now (set on the storage at upload()), so (re)build the uniform-batch
    // state lazily for B>1; B==1 stays on the single-case path.
    if (batch > 1) {
        if (!(state_->batched && state_->batch == batch)) {
            check_status(state_->solver.batched_setup(batch, batch_precision_from_env()),
                         "batched_setup");
            state_->batch = batch;
            state_->batched = true;
        }
        check_status(state_->solver.batched_factorize(buf.d_J_values.data()), "batched_factorize");
    } else {
        state_->batched = false;
        state_->batch = 1;
        check_status(state_->solver.set_values(buf.d_J_values.data(), cls::ValueType::Float64),
                     "set_values");
        check_status(state_->solver.factorize(), "factorize");
    }
    sync_cuda_for_timing();
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
#ifdef CUPF_ENABLE_CUDA_GRAPH
    if (state_->graph_mode) {
        if (state_->batched) {
            check_status(state_->solver.batched_solve(buf.d_F.data(), buf.d_dx.data()), "batched_solve");
        } else {
            check_status(state_->solver.set_rhs(make_vector_view(buf.dimF, buf.d_F.data())),
                         "set_rhs");
            check_status(state_->solver.set_solution(make_vector_view(buf.dimF, buf.d_dx.data())),
                         "set_solution");
            check_status(state_->solver.solve(), "solve");
        }
        return;  // capture-safe: no host sync inside capture
    }
#endif
    if (state_->batched) {
        check_status(state_->solver.batched_solve(buf.d_F.data(), buf.d_dx.data()), "batched_solve");
    } else {
        check_status(state_->solver.set_rhs(make_vector_view(buf.dimF, buf.d_F.data())),
                     "set_rhs");
        check_status(state_->solver.set_solution(make_vector_view(buf.dimF, buf.d_dx.data())),
                     "set_solution");
        check_status(state_->solver.solve(), "solve");
    }
    sync_cuda_for_timing();
}

#ifdef CUPF_ENABLE_CUDA_GRAPH
void CudaLinearSolveCustomFp64::graph_prepare(CudaFp64Storage& buf, void* stream)
{
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp64::graph_prepare: initialize() must be called first");
    }
    const int batch = buf.batch_size;
    state_->batch = batch;
    state_->graph_mode = true;
    state_->factorized = false;
    if (batch > 1) {
        check_status(state_->solver.batched_setup(batch, batch_precision_from_env()), "batched_setup");
        state_->batched = true;
        check_status(state_->solver.batched_set_stream(stream), "batched_set_stream");
    } else {
        state_->batched = false;
        check_status(state_->solver.set_stream(stream), "set_stream");
    }
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
    bool batched = false;
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
    state->solver = cls::Solver(fp32_single_config());
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
    if (batch > 1) {
        if (!(state_->batched && state_->batch == batch)) {
            check_status(state_->solver.batched_setup(batch, mixed_batch_precision_from_env()),
                         "batched_setup");
            state_->batch = batch;
            state_->batched = true;
        }
        check_status(state_->solver.batched_factorize(buf.d_J_values.data()), "batched_factorize");
    } else {
        state_->batch = 1;
        state_->batched = false;
        check_status(state_->solver.set_values(buf.d_J_values.data(), cls::ValueType::Float32),
                     "set_values");
        check_status(state_->solver.factorize(), "factorize");
    }
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
    if (state_->batched) {
        check_status(state_->solver.batched_solve(buf.d_F.data(), buf.d_dx.data()), "batched_solve");
    } else {
        check_status(state_->solver.set_rhs(make_vector_view(buf.dimF, buf.d_F.data())), "set_rhs");
        check_status(state_->solver.set_solution(make_vector_view(buf.dimF, buf.d_dx.data())),
                     "set_solution");
        check_status(state_->solver.solve(), "solve");
    }
    if (!state_->graph_mode) sync_cuda_for_timing();
}

#ifdef CUPF_ENABLE_CUDA_GRAPH
void CudaLinearSolveCustomFp32::graph_prepare(CudaFp32Storage& buf, void* stream)
{
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomFp32::graph_prepare: initialize() must be called first");
    }
    const int batch = buf.batch_size;
    state_->batch = batch;
    state_->graph_mode = true;
    state_->factorized = false;
    if (batch > 1) {
        check_status(state_->solver.batched_setup(batch, mixed_batch_precision_from_env()),
                     "batched_setup");
        state_->batched = true;
        check_status(state_->solver.batched_set_stream(stream), "batched_set_stream");
    } else {
        state_->batched = false;
        check_status(state_->solver.set_stream(stream), "set_stream");
    }
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
    int batch = 0;          // last batched_setup B (0 = not set up yet)
    bool batched = false;
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
    // numeric values are supplied per factorize via the float batched_factorize entry. analyze
    // (symbolic) reads only the pattern.
    cls::CsrMatrixView matrix;
    matrix.nrows = buf.dimF;
    matrix.ncols = buf.dimF;
    matrix.nnz = buf.nnz_J;  // per-case nnz (pattern shared across the batch)
    matrix.index_type = cls::IndexType::Int32;
    matrix.location = cls::DataLocation::Device;
    matrix.value_type = cls::ValueType::Float32;
    matrix.row_offsets = buf.d_J_row_ptr.data();
    matrix.col_indices = buf.d_J_col_idx.data();
    matrix.values = nullptr;

    auto state = std::make_unique<State>();
    state->solver = cls::Solver(fp32_single_config());
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
    if (batch > 1) {
        if (!(state_->batched && state_->batch == batch)) {
            check_status(state_->solver.batched_setup(batch, mixed_batch_precision_from_env()),
                         "batched_setup");
            state_->batched = true;
        }
        state_->batch = batch;
        check_status(state_->solver.batched_factorize(buf.d_J_values.data()), "batched_factorize");
    } else {
        state_->batch = 1;
        state_->batched = false;
        check_status(state_->solver.set_values(buf.d_J_values.data(), cls::ValueType::Float32),
                     "set_values");
        check_status(state_->solver.factorize(), "factorize");
    }
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
    if (state_->batched) {
        check_status(state_->solver.batched_solve(buf.d_F.data(), buf.d_dx.data()), "batched_solve");
    } else {
        check_status(state_->solver.set_rhs(make_vector_view(buf.dimF, buf.d_F.data())),
                     "set_rhs");
        check_status(state_->solver.set_solution(make_vector_view(buf.dimF, buf.d_dx.data())),
                     "set_solution");
        check_status(state_->solver.solve(), "solve");
    }
    if (!state_->graph_mode) sync_cuda_for_timing();
}

#ifdef CUPF_ENABLE_CUDA_GRAPH
void CudaLinearSolveCustomMixed::graph_prepare(CudaMixedStorage& buf, void* stream)
{
    if (state_ == nullptr || !state_->analyzed) {
        throw std::runtime_error("CudaLinearSolveCustomMixed::graph_prepare: initialize() must be called first");
    }
    const int batch = buf.batch_size;
    state_->batch = batch;
    state_->graph_mode = true;
    state_->factorized = false;
    if (batch > 1) {
        check_status(state_->solver.batched_setup(batch, mixed_batch_precision_from_env()),
                     "batched_setup");
        state_->batched = true;
        check_status(state_->solver.batched_set_stream(stream), "batched_set_stream");
    } else {
        state_->batched = false;
        check_status(state_->solver.set_stream(stream), "set_stream");
    }
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
