// ---------------------------------------------------------------------------
// cuda_cudss.cpp
//
// cuDSS-backed sparse *direct* linear solver for the Newton step. Each NR
// iteration solves  J * dx = F  (J = Jacobian, F = mismatch residual, dx = step)
// via an LU factorization of J, which cuDSS performs in four phases:
//
//   ANALYSIS        symbolic: reordering + symbolic factorization from the
//                   sparsity pattern only (no values). Done once per pattern.
//   FACTORIZATION   numeric LU using the current J values. First solve.
//   REFACTORIZATION cheaper numeric LU reusing the symbolic structure; used on
//                   later iterations where only J's values changed.
//   SOLVE           triangular solves L,U against the RHS to produce dx.
//
// Two pipelines share one cuDSS handle/config:
//   * forward  — J dx = F          (the NR step every iteration)
//   * adjoint  — J^T x = b         (gradient / sensitivity passes). J^T is
//                materialized explicitly from J via a precomputed value-permu-
//                tation, then factorized and cached so repeat adjoint solves
//                only pay the triangular solve.
//
// Batching: a whole batch is one cuDSS uniform-batch problem (CUDSS_CONFIG_
// UBATCH_SIZE in cudss_config.hpp); the batch-major device buffers feed it with
// no per-case loop, so B == 1 and B > 1 take the same path.
//
// Precision (T) / RHS selection:
//   - double                  : solve against buf.d_F directly
//   - float + CudaFp32Storage : solve against buf.d_F directly (all-FP32)
//   - float + CudaMixedStorage: RHS is a down-cast FP32 copy of the FP64 buf.d_F
// All cuDSS handles/descriptors live in the PIMPL State (below) so the public
// header stays free of cuDSS types.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_cudss.hpp"

#include "cuda_linear_solve_kernels.hpp"
#include "newton_solver/core/csr_transpose.hpp"
#include "cudss_config.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/dump.hpp"
#include "utils/timer.hpp"

#include <cstdint>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>


// ===========================================================================
// Solver state (PIMPL)
//
// All cuDSS handles and device buffers live here so the public class header
// stays free of cuDSS types. One State instance is owned per solver object.
// ===========================================================================
template <typename T, typename Buffers>
struct CudaLinearSolveCuDSS<T, Buffers>::State {
#ifdef CUPF_ENABLE_CUDSS
    // --- cuDSS library objects (forward solve) ---
    cudssHandle_t handle           = nullptr;
    cudssConfig_t config           = nullptr;
    cudssData_t   data             = nullptr;
    cudssMatrix_t jacobian         = nullptr;  // J  (CSR)
    cudssMatrix_t rhs_matrix       = nullptr;  // F  (dense)
    cudssMatrix_t solution_matrix  = nullptr;  // dx (dense)

    // --- cuDSS library objects (adjoint solve, uses J^T) ---
    cudssMatrix_t adjoint_matrix   = nullptr;
    cudssMatrix_t adjoint_rhs_matrix = nullptr;
    cudssMatrix_t adjoint_solution_matrix = nullptr;
#endif
    // --- device buffers ---
    DeviceBuffer<T> rhs;                            // FP32 RHS copy (mixed precision only)
    DeviceBuffer<int32_t> adjoint_row_ptr;          // J^T sparsity, computed once
    DeviceBuffer<int32_t> adjoint_col_idx;
    DeviceBuffer<int32_t> adjoint_src_to_transpose_pos;  // maps J value index -> J^T value index
    DeviceBuffer<T> adjoint_values;
    DeviceBuffer<T> adjoint_rhs;
    DeviceBuffer<T> adjoint_solution;

    // --- cached descriptor dimensions; a change forces matrix re-creation ---
    int32_t descriptor_batch_size  = 0;
    int32_t descriptor_dimF        = 0;
    int32_t descriptor_nnz_J       = 0;
    int32_t adjoint_descriptor_batch_size = 0;
    int32_t adjoint_descriptor_dimF = 0;
    int32_t adjoint_descriptor_nnz_J = 0;

    // --- pipeline progress flags (skip redundant analysis/factorization) ---
    bool    analysis_done          = false;
    bool    factorized             = false;
    bool    adjoint_analysis_done  = false;
    bool    adjoint_factorized     = false;

    // Release every cuDSS object in reverse creation order.
    ~State()
    {
#ifdef CUPF_ENABLE_CUDSS
        if (jacobian)        cudssMatrixDestroy(jacobian);
        if (rhs_matrix)      cudssMatrixDestroy(rhs_matrix);
        if (solution_matrix) cudssMatrixDestroy(solution_matrix);
        if (adjoint_matrix)  cudssMatrixDestroy(adjoint_matrix);
        if (adjoint_rhs_matrix) cudssMatrixDestroy(adjoint_rhs_matrix);
        if (adjoint_solution_matrix) cudssMatrixDestroy(adjoint_solution_matrix);
        if (data)            cudssDataDestroy(handle, data);
        if (config)          cudssConfigDestroy(config);
        if (handle)          cudssDestroy(handle);
#endif
    }
};


// ===========================================================================
// Translation-unit-local helpers
// ===========================================================================
namespace {

#ifdef CUPF_ENABLE_CUDSS
// Map the C++ scalar type T to the matching cuDSS runtime data-type enum.
template <typename T> cudaDataType_t cudss_value_type();
template <> cudaDataType_t cudss_value_type<double>() { return CUDA_R_64F; }
template <> cudaDataType_t cudss_value_type<float>()  { return CUDA_R_32F; }

// Destroy a cuDSS matrix descriptor and null it so it is safe to call twice.
void destroy_matrix(cudssMatrix_t& matrix)
{
    if (matrix != nullptr) {
        cudssMatrixDestroy(matrix);
        matrix = nullptr;
    }
}

// Bind cuDSS work to the project's current CUDA stream.
void set_cudss_stream(cudssHandle_t handle)
{
    CUDSS_CHECK(cudssSetStream(handle, cupf_current_cuda_stream()));
}
#endif

}  // namespace


// ===========================================================================
// Construction / destruction
// ===========================================================================
template <typename T, typename Buffers>
CudaLinearSolveCuDSS<T, Buffers>::CudaLinearSolveCuDSS(CuDSSOptions cudss_options)
    : cudss_options_(cudss_options) {}


template <typename T, typename Buffers>
CudaLinearSolveCuDSS<T, Buffers>::~CudaLinearSolveCuDSS()
{
    delete state_;
}


// ===========================================================================
// Forward solve pipeline:  initialize -> prepare_rhs -> factorize -> solve
// ===========================================================================

// Create the cuDSS handle/config/data, build descriptors, and run symbolic
// analysis (plus the adjoint analysis when the Jacobian pattern is known).
template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::initialize(Buffers& buf, const InitializeContext& ctx)
{
    if (buf.dimF <= 0 || buf.d_J_row_ptr.empty() || buf.d_J_col_idx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS::initialize: buffers are not prepared");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS::initialize requires a cuDSS-enabled build");
#else
    // Start from a clean state, then create the cuDSS handle/config/data trio.
    delete state_;
    state_ = nullptr;

    auto state = std::make_unique<State>();
    CUDSS_CHECK(cudssCreate(&state->handle));
    cupf_cudss_detail::configure_handle(state->handle);
    CUDSS_CHECK(cudssConfigCreate(&state->config));
    CUDSS_CHECK(cudssDataCreate(state->handle, &state->data));
    state_ = state.release();

    // Build the forward J/F/dx descriptors for the current dimensions.
    ensure_descriptors(buf);

    // When the Jacobian pattern is already known, precompute the J^T sparsity
    // so adjoint solves can reuse it without re-transposing every iteration.
    if (ctx.J.dim == buf.dimF && ctx.J.nnz == cuda_storage_nnz_j(buf)) {
        const CsrTransposePattern transpose =
            build_transpose_pattern(ctx.J.row_ptr, ctx.J.col_idx, ctx.J.dim);
        state_->adjoint_row_ptr.assign(transpose.row_ptr.data(), transpose.row_ptr.size());
        state_->adjoint_col_idx.assign(transpose.col_idx.data(), transpose.col_idx.size());
        state_->adjoint_src_to_transpose_pos.assign(
            transpose.src_to_transpose_pos.data(), transpose.src_to_transpose_pos.size());
    }

    // Symbolic analysis can run now unless this reordering needs matrix values
    // (in that case it is deferred to factorize()).
    if (!cupf_cudss_detail::analysis_requires_matrix_values(cudss_options_)) {
        newton_solver::utils::ScopedTimer timer("NR.initialize.cudss_analyze");
        set_cudss_stream(state_->handle);
        CUDSS_CHECK(cudssExecute(
            state_->handle, CUDSS_PHASE_ANALYSIS,
            state_->config, state_->data,
            state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
        sync_cuda_for_timing();
        state_->analysis_done = true;
    }
    // Mirror the symbolic analysis for the adjoint (J^T) system when possible.
    if (ctx.J.dim == buf.dimF && ctx.J.nnz == cuda_storage_nnz_j(buf)) {
        ensure_adjoint_descriptors(buf);
        if (!state_->adjoint_analysis_done) {
            set_cudss_stream(state_->handle);
            CUDSS_CHECK(cudssExecute(
                state_->handle, CUDSS_PHASE_ANALYSIS,
                state_->config, state_->data,
                state_->adjoint_matrix, state_->adjoint_solution_matrix, state_->adjoint_rhs_matrix));
            sync_cuda_for_timing();
            state_->adjoint_analysis_done = true;
        }
    }
#endif
}


// In mixed precision the RHS must be down-cast from the FP64 residual into a
// dedicated FP32 buffer; all other layouts solve against buf.d_F in place and
// this step is a no-op.
template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::prepare_rhs(Buffers& buf, IterationContext& ctx)
{
    (void)ctx;

    if (buf.dimF <= 0 || buf.d_F.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS::prepare_rhs: buffers are not prepared");
    }
    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS::prepare_rhs: initialize() must be called first");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS::prepare_rhs requires a cuDSS-enabled build");
#else
    // Only the float + mixed-storage combination keeps a separate FP32 RHS.
    if constexpr (std::is_same_v<T, float> && std::is_same_v<Buffers, CudaMixedStorage>) {
        ensure_descriptors(buf);
        const int32_t rhs_count = cuda_storage_batch_size(buf) * buf.dimF;
        launch_prepare_rhs(buf.d_F.data(), state_->rhs.data(), rhs_count);
    }
#endif
}


// Numeric factorization of J. The first call does a full factorization; later
// calls (same sparsity, new values) take the cheaper refactorization path.
// If analysis was deferred in initialize(), it runs here first.
template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::factorize(Buffers& buf, IterationContext& ctx)
{
    (void)ctx;

    if (buf.dimF <= 0 || buf.d_F.empty() || buf.d_dx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS::factorize: buffers are not prepared");
    }
    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS::factorize: initialize() must be called first");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS::factorize requires a cuDSS-enabled build");
#else
    ensure_descriptors(buf);

    // Deferred symbolic analysis (value-dependent reordering path).
    if (!state_->analysis_done) {
        newton_solver::utils::ScopedTimer timer("NR.iteration.cudss_analyze");
        set_cudss_stream(state_->handle);
        CUDSS_CHECK(cudssExecute(
            state_->handle, CUDSS_PHASE_ANALYSIS,
            state_->config, state_->data,
            state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
        sync_cuda_for_timing();
        state_->analysis_done = true;
    }

    // First time -> full factorization; afterwards -> refactorization (reuses
    // the symbolic structure, only the numeric values change).
    const bool is_refactorization = state_->factorized;
    const int  phase = is_refactorization ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
    set_cudss_stream(state_->handle);
    CUDSS_CHECK(cudssExecute(
        state_->handle, phase,
        state_->config, state_->data,
        state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
    sync_cuda_for_timing();
    state_->factorized = true;
#endif
}


// Triangular solve producing dx; optionally dumps the result for debugging.
template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::solve(Buffers& buf, IterationContext& ctx)
{
    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS::solve: initialize() must be called first");
    }
    if (!state_->factorized) {
        throw std::runtime_error("CudaLinearSolveCuDSS::solve: factorize() must be called first");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS::solve requires a cuDSS-enabled build");
#else
    set_cudss_stream(state_->handle);
    CUDSS_CHECK(cudssExecute(
        state_->handle, CUDSS_PHASE_SOLVE,
        state_->config, state_->data,
        state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
    sync_cuda_for_timing();

    // Debug-only: copy dx back to host and record it.
    if (newton_solver::utils::isDumpEnabled()) {
        const int32_t count = cuda_storage_batch_size(buf) * buf.dimF;
        std::vector<T> h_dx(count);
        buf.d_dx.copyTo(h_dx.data(), h_dx.size());
        newton_solver::utils::dumpVector("dx", ctx.iter, h_dx);
    }
#endif
}


// ===========================================================================
// Adjoint solve pipeline (J^T x = b), used for gradient / sensitivity passes
//
// The implicit-function-theorem backward pass needs J^T solves. cuDSS does not
// expose a transpose-solve for this configuration (supports_transpose_solve()
// returns false), so we form J^T *explicitly*: its sparsity is the transpose of
// J's (computed once in initialize() as a value-permutation map), and each pass
// scatters the current J values through that map into adjoint_values. J^T is
// then factorized and the factorization is cached, so repeated adjoint solves
// against different right-hand sides only pay for the triangular SOLVE.
//
// Two solve entry points share the cached factorization:
//   solve_adjoint_explicit_transpose_host   - RHS/solution cross the host (cast)
//   solve_adjoint_explicit_transpose_cached  - RHS already staged on device
// ===========================================================================

// True once the adjoint system has been numerically factorized.
template <typename T, typename Buffers>
bool CudaLinearSolveCuDSS<T, Buffers>::has_adjoint_cache() const
{
    return state_ != nullptr && state_->adjoint_factorized;
}


// True once the adjoint symbolic analysis (sparsity) has been computed.
template <typename T, typename Buffers>
bool CudaLinearSolveCuDSS<T, Buffers>::has_adjoint_symbolic_analysis() const
{
    return state_ != nullptr && state_->adjoint_analysis_done;
}


// Materialize J^T (values + descriptors) and factorize it, caching the result
// so repeated adjoint solves only pay for the triangular solve. Reports the
// factorization wall-clock time via the out-parameter.
template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::prepare_adjoint_explicit_transpose_cache(
    Buffers& buf,
    IterationContext& ctx,
    double& factorization_time_ms)
{
    (void)ctx;

    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS::prepare_adjoint_explicit_transpose_cache: initialize() must be called first");
    }
    if (buf.dimF <= 0 || buf.d_J_values.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS::prepare_adjoint_explicit_transpose_cache: buffers are not prepared");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS::prepare_adjoint_explicit_transpose_cache requires a cuDSS-enabled build");
#else
    ensure_adjoint_descriptors(buf);

    // Scatter J's values into J^T order using the precomputed position map.
    const int32_t batch_size = cuda_storage_batch_size(buf);
    const int32_t nnz_J = cuda_storage_nnz_j(buf);
    launch_transpose_csr_values(
        buf.d_J_values.data(),
        state_->adjoint_values.data(),
        state_->adjoint_src_to_transpose_pos.data(),
        nnz_J,
        batch_size);

    // Analyze (once) then factorize J^T; only the factorization is timed.
    newton_solver::utils::ScopedTimer factor_timer("cuda_cudss.adjoint.factorize");
    if (!state_->adjoint_analysis_done) {
        set_cudss_stream(state_->handle);
        CUDSS_CHECK(cudssExecute(
            state_->handle, CUDSS_PHASE_ANALYSIS,
            state_->config, state_->data,
            state_->adjoint_matrix, state_->adjoint_solution_matrix, state_->adjoint_rhs_matrix));
        sync_cuda_for_timing();
        state_->adjoint_analysis_done = true;
    }

    const bool is_refactorization = state_->adjoint_factorized;
    const int phase = is_refactorization ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
    set_cudss_stream(state_->handle);
    CUDSS_CHECK(cudssExecute(
        state_->handle, phase,
        state_->config, state_->data,
        state_->adjoint_matrix, state_->adjoint_solution_matrix, state_->adjoint_rhs_matrix
    ));
    sync_cuda_for_timing();
    state_->adjoint_factorized = true;
    factor_timer.stop();
    factorization_time_ms = factor_timer.elapsedMilliseconds();
#endif
}


// One-shot adjoint solve from host memory: cast RHS in, solve J^T x = b on the
// device, cast the solution back out. Used when no cached RHS is in flight.
template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::solve_adjoint_explicit_transpose_host(
    const double* rhs,
    double* solution,
    int32_t batch_size,
    double& solve_time_ms)
{
    if (state_ == nullptr || !state_->adjoint_factorized) {
        throw std::runtime_error("CudaLinearSolveCuDSS::solve_adjoint_explicit_transpose_host: adjoint cache is not factorized");
    }
    if (rhs == nullptr || solution == nullptr || batch_size <= 0) {
        throw std::invalid_argument("CudaLinearSolveCuDSS::solve_adjoint_explicit_transpose_host: invalid arguments");
    }
    if (batch_size != state_->adjoint_descriptor_batch_size) {
        throw std::invalid_argument("CudaLinearSolveCuDSS::solve_adjoint_explicit_transpose_host: batch size does not match adjoint cache");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS::solve_adjoint_explicit_transpose_host requires a cuDSS-enabled build");
#else
    const int32_t count = batch_size * state_->adjoint_descriptor_dimF;

    // Down-cast the FP64 host RHS into the solver's working precision T.
    // (static_cast<T> is the explicit double->float narrowing for FP32 solves;
    //  for the double instantiation it is an identity conversion.)
    std::vector<T> rhs_t(count);
    for (int32_t i = 0; i < count; ++i) {
        rhs_t[i] = static_cast<T>(rhs[i]);
    }
    state_->adjoint_rhs.assign(rhs_t.data(), rhs_t.size());
    state_->adjoint_solution.memsetZero();

    // Time only the device solve.
    newton_solver::utils::ScopedTimer solve_timer("cuda_cudss.adjoint.solve_host");
    set_cudss_stream(state_->handle);
    CUDSS_CHECK(cudssExecute(
        state_->handle, CUDSS_PHASE_SOLVE,
        state_->config, state_->data,
        state_->adjoint_matrix, state_->adjoint_solution_matrix, state_->adjoint_rhs_matrix));
    sync_cuda_for_timing();
    solve_timer.stop();
    solve_time_ms = solve_timer.elapsedMilliseconds();

    // Copy the solution back and up-cast T -> FP64 for the caller's buffer.
    std::vector<T> sol_t(count);
    state_->adjoint_solution.copyTo(sol_t.data(), sol_t.size());
    for (int32_t i = 0; i < count; ++i) {
        solution[i] = static_cast<double>(sol_t[i]);
    }
#endif
}


// Direct device pointer to the cached adjoint RHS buffer (caller fills it in
// place to avoid a host round-trip before solve_..._cached()).
template <typename T, typename Buffers>
T* CudaLinearSolveCuDSS<T, Buffers>::adjoint_rhs_data()
{
    if (state_ == nullptr || state_->adjoint_rhs.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS::adjoint_rhs_data: adjoint cache is not prepared");
    }
    return state_->adjoint_rhs.data();
}


// Direct device pointer to the cached adjoint solution buffer.
template <typename T, typename Buffers>
T* CudaLinearSolveCuDSS<T, Buffers>::adjoint_solution_data()
{
    if (state_ == nullptr || state_->adjoint_solution.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS::adjoint_solution_data: adjoint cache is not prepared");
    }
    return state_->adjoint_solution.data();
}


// Adjoint solve that reuses an RHS already staged on the device (no host copy).
template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::solve_adjoint_explicit_transpose_cached(double& solve_time_ms)
{
    if (state_ == nullptr || !state_->adjoint_factorized) {
        throw std::runtime_error("CudaLinearSolveCuDSS::solve_adjoint_explicit_transpose_cached: adjoint cache is not factorized");
    }
#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS::solve_adjoint_explicit_transpose_cached requires a cuDSS-enabled build");
#else
    state_->adjoint_solution.memsetZero();
    newton_solver::utils::ScopedTimer solve_timer("cuda_cudss.adjoint.solve_cached");
    set_cudss_stream(state_->handle);
    CUDSS_CHECK(cudssExecute(
        state_->handle, CUDSS_PHASE_SOLVE,
        state_->config, state_->data,
        state_->adjoint_matrix, state_->adjoint_solution_matrix, state_->adjoint_rhs_matrix));
    sync_cuda_for_timing();
    solve_timer.stop();
    solve_time_ms = solve_timer.elapsedMilliseconds();
#endif
}


// ===========================================================================
// Descriptor management
//
// cuDSS matrix descriptors are bound to fixed dimensions, so they are cached
// and only re-created when batch_size / dimF / nnz_J change. rhs_data() picks
// the correct RHS pointer for the active precision/storage combination.
// ===========================================================================

// (Re)create the forward J/F/dx descriptors for the current buffer dimensions.
template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::ensure_descriptors(Buffers& buf)
{
#ifndef CUPF_ENABLE_CUDSS
    (void)buf;
    throw std::runtime_error("CudaLinearSolveCuDSS: requires a cuDSS-enabled build");
#else
    const int32_t batch_size = cuda_storage_batch_size(buf);
    const int32_t dimF       = buf.dimF;
    const int32_t nnz_J      = cuda_storage_nnz_j(buf);

    if (batch_size <= 0 || dimF <= 0 || nnz_J <= 0) {
        throw std::runtime_error("CudaLinearSolveCuDSS: invalid descriptor dimensions");
    }

    // Reuse existing descriptors when dimensions are unchanged.
    const bool match =
        state_->jacobian        != nullptr &&
        state_->rhs_matrix      != nullptr &&
        state_->solution_matrix != nullptr &&
        state_->descriptor_batch_size == batch_size &&
        state_->descriptor_dimF       == dimF &&
        state_->descriptor_nnz_J      == nnz_J;

    if (match) return;

    // Dimensions changed: tear down stale descriptors and rebuild.
    destroy_matrix(state_->jacobian);
    destroy_matrix(state_->rhs_matrix);
    destroy_matrix(state_->solution_matrix);

    cupf_cudss_detail::configure_solver(state_->config, cudss_options_, batch_size);

    // Mixed precision needs its own FP32 RHS buffer sized batch_size * dimF.
    if constexpr (std::is_same_v<T, float> && std::is_same_v<Buffers, CudaMixedStorage>) {
        state_->rhs.resize(batch_size * dimF);
    }

    CUDSS_CHECK(cudssMatrixCreateCsr(
        &state_->jacobian,
        // cuDSS takes nnz as int64; widen the int32 count explicitly.
        dimF, dimF, static_cast<int64_t>(nnz_J),
        buf.d_J_row_ptr.data(), nullptr, buf.d_J_col_idx.data(), buf.d_J_values.data(),
        CUDA_R_32I, cudss_value_type<T>(),
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &state_->rhs_matrix,
        dimF, 1, dimF, rhs_data(buf),
        cudss_value_type<T>(), CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &state_->solution_matrix,
        dimF, 1, dimF, buf.d_dx.data(),
        cudss_value_type<T>(), CUDSS_LAYOUT_COL_MAJOR));

    state_->descriptor_batch_size = batch_size;
    state_->descriptor_dimF       = dimF;
    state_->descriptor_nnz_J      = nnz_J;
    state_->analysis_done         = false;
    state_->factorized            = false;
#endif
}


// (Re)create the adjoint J^T/rhs/solution descriptors and their device buffers.
template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::ensure_adjoint_descriptors(Buffers& buf)
{
#ifndef CUPF_ENABLE_CUDSS
    (void)buf;
    throw std::runtime_error("CudaLinearSolveCuDSS: requires a cuDSS-enabled build");
#else
    const int32_t batch_size = cuda_storage_batch_size(buf);
    const int32_t dimF = buf.dimF;
    const int32_t nnz_J = cuda_storage_nnz_j(buf);

    if (batch_size <= 0 || dimF <= 0 || nnz_J <= 0) {
        throw std::runtime_error("CudaLinearSolveCuDSS: invalid adjoint descriptor dimensions");
    }
    if (state_->adjoint_row_ptr.empty() ||
        state_->adjoint_col_idx.empty() ||
        state_->adjoint_src_to_transpose_pos.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS: transpose pattern was not initialized");
    }

    const bool match =
        state_->adjoint_matrix != nullptr &&
        state_->adjoint_rhs_matrix != nullptr &&
        state_->adjoint_solution_matrix != nullptr &&
        state_->adjoint_descriptor_batch_size == batch_size &&
        state_->adjoint_descriptor_dimF == dimF &&
        state_->adjoint_descriptor_nnz_J == nnz_J;
    if (match) return;

    // Dimensions changed: tear down stale adjoint descriptors and rebuild.
    destroy_matrix(state_->adjoint_matrix);
    destroy_matrix(state_->adjoint_rhs_matrix);
    destroy_matrix(state_->adjoint_solution_matrix);

    cupf_cudss_detail::configure_solver(state_->config, cudss_options_, batch_size);

    // Allocate and zero the J^T value / RHS / solution buffers.
    state_->adjoint_values.resize(batch_size * nnz_J);
    state_->adjoint_rhs.resize(batch_size * dimF);
    state_->adjoint_solution.resize(batch_size * dimF);
    state_->adjoint_values.memsetZero();
    state_->adjoint_rhs.memsetZero();
    state_->adjoint_solution.memsetZero();

    CUDSS_CHECK(cudssMatrixCreateCsr(
        &state_->adjoint_matrix,
        // cuDSS takes nnz as int64; widen the int32 count explicitly.
        dimF, dimF, static_cast<int64_t>(nnz_J),
        state_->adjoint_row_ptr.data(), nullptr,
        state_->adjoint_col_idx.data(), state_->adjoint_values.data(),
        CUDA_R_32I, cudss_value_type<T>(),
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &state_->adjoint_rhs_matrix,
        dimF, 1, dimF, state_->adjoint_rhs.data(),
        cudss_value_type<T>(), CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &state_->adjoint_solution_matrix,
        dimF, 1, dimF, state_->adjoint_solution.data(),
        cudss_value_type<T>(), CUDSS_LAYOUT_COL_MAJOR));

    state_->adjoint_descriptor_batch_size = batch_size;
    state_->adjoint_descriptor_dimF = dimF;
    state_->adjoint_descriptor_nnz_J = nnz_J;
    state_->adjoint_analysis_done = false;
    state_->adjoint_factorized = false;
#endif
}


// Select the RHS device pointer for the active precision / storage:
//   - double               -> solve against buf.d_F directly
//   - float + mixed storage -> solve against the cast FP32 copy (state_->rhs)
//   - float + fp32 storage  -> solve against buf.d_F directly
template <typename T, typename Buffers>
T* CudaLinearSolveCuDSS<T, Buffers>::rhs_data(Buffers& buf)
{
    if constexpr (std::is_same_v<T, double>) {
        return buf.d_F.data();
    } else if constexpr (std::is_same_v<Buffers, CudaMixedStorage>) {
        return state_->rhs.data();
    } else {
        return buf.d_F.data();
    }
}


// ===========================================================================
// Explicit template instantiations
//
// This .cpp is the single translation unit that defines every member of the
// class template, so all supported (T, Buffers) combinations are instantiated
// here. Keep these in sync with the storage layouts the solver supports.
// ===========================================================================
template struct CudaLinearSolveCuDSS<double, CudaFp64Storage>;
template struct CudaLinearSolveCuDSS<float,  CudaFp32Storage>;
template struct CudaLinearSolveCuDSS<float,  CudaMixedStorage>;

#endif  // CUPF_WITH_CUDA
