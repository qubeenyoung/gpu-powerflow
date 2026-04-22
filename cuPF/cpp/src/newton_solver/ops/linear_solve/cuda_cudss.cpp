// ---------------------------------------------------------------------------
// cuda_cudss.cpp
//
// cuDSS sparse direct linear solver. T selects precision:
//   - double: RHS = buf.d_F directly
//   - float : RHS is a cast copy of buf.d_F (prepare_rhs)
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_cudss.hpp"

#include "cuda_linear_solve_kernels.hpp"
#include "cudss_config.hpp"
#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>


template <typename T, typename Buffers>
struct CudaLinearSolveCuDSS<T, Buffers>::State {
#ifdef CUPF_ENABLE_CUDSS
    cudssHandle_t handle           = nullptr;
    cudssConfig_t config           = nullptr;
    cudssData_t   data             = nullptr;
    cudssMatrix_t jacobian         = nullptr;
    cudssMatrix_t rhs_matrix       = nullptr;
    cudssMatrix_t solution_matrix  = nullptr;
#endif
    DeviceBuffer<T> rhs;
    int32_t descriptor_batch_size  = 0;
    int32_t descriptor_dimF        = 0;
    int32_t descriptor_nnz_J       = 0;
    bool    analysis_done          = false;
    bool    factorized             = false;

    ~State()
    {
#ifdef CUPF_ENABLE_CUDSS
        if (jacobian)        cudssMatrixDestroy(jacobian);
        if (rhs_matrix)      cudssMatrixDestroy(rhs_matrix);
        if (solution_matrix) cudssMatrixDestroy(solution_matrix);
        if (data)            cudssDataDestroy(handle, data);
        if (config)          cudssConfigDestroy(config);
        if (handle)          cudssDestroy(handle);
#endif
    }
};


namespace {

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
#endif

int32_t buf_batch_size(const CudaFp64Buffers&)         { return 1; }
int32_t buf_batch_size(const CudaMixedBuffers& b)       { return b.batch_size; }

int32_t buf_nnz_j(const CudaFp64Buffers& b)
{
    return static_cast<int32_t>(b.d_J_values.size());
}
int32_t buf_nnz_j(const CudaMixedBuffers& b)            { return b.nnz_J; }

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
    (void)ctx;

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

    if constexpr (std::is_same_v<Buffers, CudaFp64Buffers>) {
        ensure_descriptors(buf);
        if (!cupf_cudss_detail::analysis_requires_matrix_values(cudss_options_)) {
            CUDSS_CHECK(cudssExecute(
                state_->handle, CUDSS_PHASE_ANALYSIS,
                state_->config, state_->data,
                state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
            sync_cuda_for_timing();
            state_->analysis_done = true;
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
    if constexpr (std::is_same_v<T, float>) {
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
        CUDSS_CHECK(cudssExecute(
            state_->handle, CUDSS_PHASE_ANALYSIS,
            state_->config, state_->data,
            state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
        sync_cuda_for_timing();
        state_->analysis_done = true;
    }

    const bool is_refactorization = state_->factorized;
    const int  phase = is_refactorization ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
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
    (void)buf;
    (void)ctx;

    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS::solve: initialize() must be called first");
    }
    if (!state_->factorized) {
        throw std::runtime_error("CudaLinearSolveCuDSS::solve: factorize() must be called first");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS::solve requires a cuDSS-enabled build");
#else
    CUDSS_CHECK(cudssExecute(
        state_->handle, CUDSS_PHASE_SOLVE,
        state_->config, state_->data,
        state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
    sync_cuda_for_timing();
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

    if constexpr (std::is_same_v<T, float>) {
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
T* CudaLinearSolveCuDSS<T, Buffers>::rhs_data(Buffers& buf)
{
    if constexpr (std::is_same_v<T, double>) {
        return buf.d_F.data();
    } else {
        return state_->rhs.data();
    }
}


template struct CudaLinearSolveCuDSS<double, CudaFp64Buffers>;
template struct CudaLinearSolveCuDSS<float,  CudaMixedBuffers>;

#endif  // CUPF_WITH_CUDA
