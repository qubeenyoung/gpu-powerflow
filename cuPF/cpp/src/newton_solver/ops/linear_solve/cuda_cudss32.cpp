// ---------------------------------------------------------------------------
// cuda_cudss32.cpp — CUDA FP32 선형 솔버 구현 (cuDSS FP32, Mixed 전용)
//
// cuda_cudss64.cpp 와 동일한 구조이며 정밀도만 FP32 (float).
// Mixed 정밀도 경로에서 Jacobian(FP32)·dx(FP32)을 처리한다.
//
// FP64 버전과의 차이:
//   - cudssMatrixCreateCsr: CUDA_R_32F (J_values = float*)
//   - cudssMatrixCreateDn:  rhs = float*, solution = d_dx(float*)
//   - RHS 준비: d_F(FP64) → h_F → -static_cast<float>(h_F[i]) → d_rhs(float)
//     Mixed 모드에서 F는 FP64이므로 float으로 다운캐스트 후 부호 반전.
//
// 생명주기: cuda_cudss64.cpp 와 동일 (analyze → run×N).
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_cudss32.hpp"

#include "cudss_config.hpp"
#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/timer.hpp"

#include <memory>
#include <stdexcept>
#include <vector>


// FP32 경로 cuDSS 핸들·디스크립터 pimpl.
struct CudaLinearSolveCuDSS32::CuDSS32State {
#ifdef CUPF_ENABLE_CUDSS
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t jacobian = nullptr;
    cudssMatrix_t rhs_matrix = nullptr;
    cudssMatrix_t solution_matrix = nullptr;
#endif
    DeviceBuffer<float> rhs;
    bool factorized = false;

    ~CuDSS32State()
    {
#ifdef CUPF_ENABLE_CUDSS
        if (jacobian) {
            cudssMatrixDestroy(jacobian);
        }
        if (rhs_matrix) {
            cudssMatrixDestroy(rhs_matrix);
        }
        if (solution_matrix) {
            cudssMatrixDestroy(solution_matrix);
        }
        if (data) {
            cudssDataDestroy(handle, data);
        }
        if (config) {
            cudssConfigDestroy(config);
        }
        if (handle) {
            cudssDestroy(handle);
        }
#endif
    }
};

namespace {

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

}  // namespace

CudaLinearSolveCuDSS32::CudaLinearSolveCuDSS32(IStorage& storage)
    : storage_(storage) {}

CudaLinearSolveCuDSS32::~CudaLinearSolveCuDSS32()
{
    delete state_;
}


void CudaLinearSolveCuDSS32::analyze(const AnalyzeContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaMixedStorage&>(storage_);

    if (storage.dimF <= 0 || storage.d_J_row_ptr.empty() || storage.d_J_col_idx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS32::analyze: storage is not prepared");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS32::analyze requires a cuDSS-enabled build");
#else
    delete state_;
    state_ = nullptr;

    auto state = std::make_unique<CuDSS32State>();
    state->rhs.resize(static_cast<std::size_t>(storage.dimF));

    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.cudss32.setup");
        CUDSS_CHECK(cudssCreate(&state->handle));
        cupf_cudss_detail::configure_handle(state->handle);
        CUDSS_CHECK(cudssConfigCreate(&state->config));
        cupf_cudss_detail::configure_solver(state->config);
        CUDSS_CHECK(cudssDataCreate(state->handle, &state->data));

        CUDSS_CHECK(cudssMatrixCreateCsr(
            &state->jacobian,
            storage.dimF, storage.dimF, static_cast<int64_t>(storage.d_J_values.size()),
            storage.d_J_row_ptr.data(), nullptr, storage.d_J_col_idx.data(), storage.d_J_values.data(),
            CUDA_R_32I, CUDA_R_32F,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &state->rhs_matrix,
            storage.dimF, 1, storage.dimF, state->rhs.data(),
            CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &state->solution_matrix,
            storage.dimF, 1, storage.dimF, storage.d_dx.data(),
            CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
    }

    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.cudss32.analysis");
        CUDSS_CHECK(cudssExecute(
            state->handle, CUDSS_PHASE_ANALYSIS,
            state->config, state->data,
            state->jacobian, state->solution_matrix, state->rhs_matrix));
        sync_cuda_for_timing();
    }

    state_ = state.release();
#endif
}


void CudaLinearSolveCuDSS32::run(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaMixedStorage&>(storage_);

    if (storage.dimF <= 0 || storage.d_F.empty() || storage.d_dx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS32::run: storage is not prepared");
    }
    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS32::run: analyze() must be called first");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS32::run requires a cuDSS-enabled build");
#else
    std::vector<double> h_F(static_cast<std::size_t>(storage.dimF));
    std::vector<float> h_rhs(static_cast<std::size_t>(storage.dimF));

    {
        // RHS = -F (FP64 → FP32 다운캐스트 후 부호 반전):
        //   d_F(double*) → h_F → -static_cast<float>(h_F[i]) → d_rhs(float*)
        // Mixed 모드에서 d_F 는 FP64이므로 float으로 변환한 후 upload.
        newton_solver::utils::ScopedTimer timer("CUDA.solve.rhsPrepare");
        storage.d_F.copyTo(h_F.data(), h_F.size());
        for (std::size_t i = 0; i < h_F.size(); ++i) {
            h_rhs[i] = -static_cast<float>(h_F[i]);
        }
        state_->rhs.assign(h_rhs.data(), h_rhs.size());
    }

    {
        // 첫 번째: FACTORIZATION, 이후: REFACTORIZATION (symbolic 재사용)
        newton_solver::utils::ScopedTimer timer(
            state_->factorized ? "CUDA.solve.refactorization32" : "CUDA.solve.factorization32");
        const int phase = state_->factorized ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
        CUDSS_CHECK(cudssExecute(
            state_->handle, phase,
            state_->config, state_->data,
            state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
        sync_cuda_for_timing();
        state_->factorized = true;
    }

    {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.solve32");
        CUDSS_CHECK(cudssExecute(
            state_->handle, CUDSS_PHASE_SOLVE,
            state_->config, state_->data,
            state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
        sync_cuda_for_timing();
    }
#endif
}

#endif  // CUPF_WITH_CUDA
