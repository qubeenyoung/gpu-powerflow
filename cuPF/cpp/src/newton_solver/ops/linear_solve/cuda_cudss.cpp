// ---------------------------------------------------------------------------
// cuda_cudss.cpp
//
// cuDSS sparse direct linear solver. T selects precision:
//   - double: RHS = buf.d_F directly
//   - float + CudaFp32Storage: RHS = buf.d_F directly
//   - float + CudaMixedStorage: RHS is a cast copy of FP64 buf.d_F
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
#include <chrono>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>


template <typename T, typename Buffers>
struct CudaLinearSolveCuDSS<T, Buffers>::State {
#ifdef CUPF_ENABLE_CUDSS
    cudssHandle_t handle           = nullptr;
    cudssConfig_t config           = nullptr;
    cudssData_t   data             = nullptr;
    cudssMatrix_t jacobian         = nullptr;
    cudssMatrix_t rhs_matrix       = nullptr;
    cudssMatrix_t solution_matrix  = nullptr;
    cudssMatrix_t adjoint_matrix   = nullptr;
    cudssMatrix_t adjoint_rhs_matrix = nullptr;
    cudssMatrix_t adjoint_solution_matrix = nullptr;
#endif
    DeviceBuffer<T> rhs;
    DeviceBuffer<int32_t> adjoint_row_ptr;
    DeviceBuffer<int32_t> adjoint_col_idx;
    DeviceBuffer<int32_t> adjoint_src_to_transpose_pos;
    DeviceBuffer<T> adjoint_values;
    DeviceBuffer<T> adjoint_rhs;
    DeviceBuffer<T> adjoint_solution;
    int32_t descriptor_batch_size  = 0;
    int32_t descriptor_dimF        = 0;
    int32_t descriptor_nnz_J       = 0;
    int32_t adjoint_descriptor_batch_size = 0;
    int32_t adjoint_descriptor_dimF = 0;
    int32_t adjoint_descriptor_nnz_J = 0;
    bool    analysis_done          = false;
    bool    factorized             = false;
    bool    adjoint_analysis_done  = false;
    bool    adjoint_factorized     = false;

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


namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#ifdef CUPF_ENABLE_CUDSS
template <typename T> cudaDataType_t cudss_value_type();
template <> cudaDataType_t cudss_value_type<double>() { return CUDA_R_64F; }
template <> cudaDataType_t cudss_value_type<float>()  { return CUDA_R_32F; }

void destroy_matrix(cudssMatrix_t& matrix)
{
    if (matrix != nullptr) {
        cudssMatrixDestroy(matrix);
        matrix = nullptr;
    }
}

void set_cudss_stream(cudssHandle_t handle)
{
    CUDSS_CHECK(cudssSetStream(handle, cupf_current_cuda_stream()));
}
#endif

int32_t buf_batch_size(const CudaFp64Storage&)         { return 1; }
int32_t buf_batch_size(const CudaFp32Storage& b)       { return b.batch_size; }
int32_t buf_batch_size(const CudaMixedStorage& b)       { return b.batch_size; }

int32_t buf_nnz_j(const CudaFp64Storage& b)
{
    return static_cast<int32_t>(b.d_J_values.size());
}
int32_t buf_nnz_j(const CudaFp32Storage& b)            { return b.nnz_J; }
int32_t buf_nnz_j(const CudaMixedStorage& b)            { return b.nnz_J; }

}  // namespace


template <typename T, typename Buffers>
CudaLinearSolveCuDSS<T, Buffers>::CudaLinearSolveCuDSS(CuDSSOptions cudss_options)
    : cudss_options_(cudss_options) {}


template <typename T, typename Buffers>
CudaLinearSolveCuDSS<T, Buffers>::~CudaLinearSolveCuDSS()
{
    delete state_;
}


template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::initialize(Buffers& buf, const InitializeContext& ctx)
{
    if (buf.dimF <= 0 || buf.d_J_row_ptr.empty() || buf.d_J_col_idx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS::initialize: buffers are not prepared");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS::initialize requires a cuDSS-enabled build");
#else
    delete state_;
    state_ = nullptr;

    auto state = std::make_unique<State>();
    CUDSS_CHECK(cudssCreate(&state->handle));
    cupf_cudss_detail::configure_handle(state->handle);
    CUDSS_CHECK(cudssConfigCreate(&state->config));
    CUDSS_CHECK(cudssDataCreate(state->handle, &state->data));
    state_ = state.release();

    ensure_descriptors(buf);
    if (ctx.J.dim == buf.dimF && ctx.J.nnz == buf_nnz_j(buf)) {
        const CsrTransposePattern transpose =
            build_transpose_pattern(ctx.J.row_ptr, ctx.J.col_idx, ctx.J.dim);
        state_->adjoint_row_ptr.assign(transpose.row_ptr.data(), transpose.row_ptr.size());
        state_->adjoint_col_idx.assign(transpose.col_idx.data(), transpose.col_idx.size());
        state_->adjoint_src_to_transpose_pos.assign(
            transpose.src_to_transpose_pos.data(), transpose.src_to_transpose_pos.size());
    }
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
    if (ctx.J.dim == buf.dimF && ctx.J.nnz == buf_nnz_j(buf)) {
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
    if constexpr (std::is_same_v<T, float> && std::is_same_v<Buffers, CudaMixedStorage>) {
        ensure_descriptors(buf);
        const int32_t rhs_count = buf_batch_size(buf) * buf.dimF;
        launch_prepare_rhs(buf.d_F.data(), state_->rhs.data(), rhs_count);
    }
#endif
}


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
    if (newton_solver::utils::isDumpEnabled()) {
        const int32_t count = buf_batch_size(buf) * buf.dimF;
        std::vector<T> h_dx(static_cast<std::size_t>(count));
        buf.d_dx.copyTo(h_dx.data(), h_dx.size());
        newton_solver::utils::dumpVector("dx", ctx.iter, h_dx);
    }
#endif
}


template <typename T, typename Buffers>
bool CudaLinearSolveCuDSS<T, Buffers>::has_adjoint_cache() const
{
    return state_ != nullptr && state_->adjoint_factorized;
}


template <typename T, typename Buffers>
bool CudaLinearSolveCuDSS<T, Buffers>::has_adjoint_symbolic_analysis() const
{
    return state_ != nullptr && state_->adjoint_analysis_done;
}


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
    const int32_t batch_size = buf_batch_size(buf);
    const int32_t nnz_J = buf_nnz_j(buf);
    launch_transpose_csr_values(
        buf.d_J_values.data(),
        state_->adjoint_values.data(),
        state_->adjoint_src_to_transpose_pos.data(),
        nnz_J,
        batch_size);

    const auto start = Clock::now();
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
        state_->adjoint_matrix, state_->adjoint_solution_matrix, state_->adjoint_rhs_matrix));
    sync_cuda_for_timing();
    state_->adjoint_factorized = true;
    factorization_time_ms = elapsed_ms(start, Clock::now());
#endif
}


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
    std::vector<T> rhs_t(static_cast<std::size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        rhs_t[static_cast<std::size_t>(i)] = static_cast<T>(rhs[i]);
    }
    state_->adjoint_rhs.assign(rhs_t.data(), rhs_t.size());
    state_->adjoint_solution.memsetZero();

    const auto start = Clock::now();
    set_cudss_stream(state_->handle);
    CUDSS_CHECK(cudssExecute(
        state_->handle, CUDSS_PHASE_SOLVE,
        state_->config, state_->data,
        state_->adjoint_matrix, state_->adjoint_solution_matrix, state_->adjoint_rhs_matrix));
    sync_cuda_for_timing();
    solve_time_ms = elapsed_ms(start, Clock::now());

    std::vector<T> sol_t(static_cast<std::size_t>(count));
    state_->adjoint_solution.copyTo(sol_t.data(), sol_t.size());
    for (int32_t i = 0; i < count; ++i) {
        solution[i] = static_cast<double>(sol_t[static_cast<std::size_t>(i)]);
    }
#endif
}


template <typename T, typename Buffers>
T* CudaLinearSolveCuDSS<T, Buffers>::adjoint_rhs_data()
{
    if (state_ == nullptr || state_->adjoint_rhs.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS::adjoint_rhs_data: adjoint cache is not prepared");
    }
    return state_->adjoint_rhs.data();
}


template <typename T, typename Buffers>
T* CudaLinearSolveCuDSS<T, Buffers>::adjoint_solution_data()
{
    if (state_ == nullptr || state_->adjoint_solution.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS::adjoint_solution_data: adjoint cache is not prepared");
    }
    return state_->adjoint_solution.data();
}


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
    const auto start = Clock::now();
    set_cudss_stream(state_->handle);
    CUDSS_CHECK(cudssExecute(
        state_->handle, CUDSS_PHASE_SOLVE,
        state_->config, state_->data,
        state_->adjoint_matrix, state_->adjoint_solution_matrix, state_->adjoint_rhs_matrix));
    sync_cuda_for_timing();
    solve_time_ms = elapsed_ms(start, Clock::now());
#endif
}


template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::ensure_descriptors(Buffers& buf)
{
#ifndef CUPF_ENABLE_CUDSS
    (void)buf;
    throw std::runtime_error("CudaLinearSolveCuDSS: requires a cuDSS-enabled build");
#else
    const int32_t batch_size = buf_batch_size(buf);
    const int32_t dimF       = buf.dimF;
    const int32_t nnz_J      = buf_nnz_j(buf);

    if (batch_size <= 0 || dimF <= 0 || nnz_J <= 0) {
        throw std::runtime_error("CudaLinearSolveCuDSS: invalid descriptor dimensions");
    }

    const bool match =
        state_->jacobian        != nullptr &&
        state_->rhs_matrix      != nullptr &&
        state_->solution_matrix != nullptr &&
        state_->descriptor_batch_size == batch_size &&
        state_->descriptor_dimF       == dimF &&
        state_->descriptor_nnz_J      == nnz_J;

    if (match) return;

    destroy_matrix(state_->jacobian);
    destroy_matrix(state_->rhs_matrix);
    destroy_matrix(state_->solution_matrix);

    cupf_cudss_detail::configure_solver(state_->config, cudss_options_, batch_size);

    if constexpr (std::is_same_v<T, float> && std::is_same_v<Buffers, CudaMixedStorage>) {
        state_->rhs.resize(static_cast<std::size_t>(batch_size) *
                           static_cast<std::size_t>(dimF));
    }

    CUDSS_CHECK(cudssMatrixCreateCsr(
        &state_->jacobian,
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


template <typename T, typename Buffers>
void CudaLinearSolveCuDSS<T, Buffers>::ensure_adjoint_descriptors(Buffers& buf)
{
#ifndef CUPF_ENABLE_CUDSS
    (void)buf;
    throw std::runtime_error("CudaLinearSolveCuDSS: requires a cuDSS-enabled build");
#else
    const int32_t batch_size = buf_batch_size(buf);
    const int32_t dimF = buf.dimF;
    const int32_t nnz_J = buf_nnz_j(buf);

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

    destroy_matrix(state_->adjoint_matrix);
    destroy_matrix(state_->adjoint_rhs_matrix);
    destroy_matrix(state_->adjoint_solution_matrix);

    cupf_cudss_detail::configure_solver(state_->config, cudss_options_, batch_size);

    state_->adjoint_values.resize(static_cast<std::size_t>(batch_size) *
                                  static_cast<std::size_t>(nnz_J));
    state_->adjoint_rhs.resize(static_cast<std::size_t>(batch_size) *
                               static_cast<std::size_t>(dimF));
    state_->adjoint_solution.resize(static_cast<std::size_t>(batch_size) *
                                    static_cast<std::size_t>(dimF));
    state_->adjoint_values.memsetZero();
    state_->adjoint_rhs.memsetZero();
    state_->adjoint_solution.memsetZero();

    CUDSS_CHECK(cudssMatrixCreateCsr(
        &state_->adjoint_matrix,
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


template struct CudaLinearSolveCuDSS<double, CudaFp64Storage>;
template struct CudaLinearSolveCuDSS<float,  CudaFp32Storage>;
template struct CudaLinearSolveCuDSS<float,  CudaMixedStorage>;

#endif  // CUPF_WITH_CUDA
