// ---------------------------------------------------------------------------
// cuda_cudss32.cpp — CUDA FP32 선형 솔버 구현 (cuDSS FP32, Mixed 전용)
//
// cuda_cudss64.cpp 와 동일한 구조이며 정밀도만 FP32 (float).
// Mixed 정밀도 경로에서 Jacobian(FP32)·dx(FP32)을 처리한다.
//
// FP64 버전과의 차이:
//   - cudssMatrixCreateBatchCsr: CUDA_R_32F (J_values = float*)
//   - cudssMatrixCreateBatchDn:  rhs = float*, solution = d_dx(float*)
//   - RHS 준비: d_F(FP64) → device cast → d_rhs(FP32)
//     Mixed 모드에서 F는 FP64이므로 device에서 float으로 다운캐스트한다.
//     update stage가 state -= dx를 적용하므로 선형계는 J * dx = F다.
//
// 생명주기:
//   analyze()는 handle/config/data만 만든다.
//   batch size와 batch-major value buffer는 solve upload 이후 확정되므로
//   UBATCH_SIZE 설정과 matrix descriptors는 첫 factorize()에서 만든다.
//
// cuDSS uniform batch는 v1 코드와 같은 방식으로 붙인다:
//   - cudssMatrixCreateBatchCsr가 아니라 cudssMatrixCreateCsr를 사용
//   - CUDSS_CONFIG_UBATCH_SIZE = B
//   - J/rhs/dx buffer는 flat batch-major [B * nnz] / [B * dimF]
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_cudss32.hpp"

#include "cuda_linear_solve_kernels.hpp"
#include "cudss_config.hpp"
#include "linear_diagnostics.hpp"
#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/dump.hpp"
#include "utils/timer.hpp"

#include <cstddef>
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
    int32_t descriptor_batch_size = 0;
    int32_t descriptor_dimF = 0;
    int32_t descriptor_nnz_J = 0;
    bool analyzed = false;
    bool factorized = false;
    const char* pending_solve_phase = "solve_only";

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

#ifdef CUPF_ENABLE_CUDSS
void set_ubatch_size(cudssConfig_t config, int32_t batch_size)
{
    int value = batch_size;
    CUDSS_CHECK(cudssConfigSet(
        config,
        CUDSS_CONFIG_UBATCH_SIZE,
        &value,
        sizeof(value)));
}

void destroy_matrix(cudssMatrix_t& matrix)
{
    if (matrix != nullptr) {
        cudssMatrixDestroy(matrix);
        matrix = nullptr;
    }
}

void ensure_batch_descriptors(CudaLinearSolveCuDSS32::CuDSS32State& state,
                              CudaMixedStorage& storage)
{
    const int32_t batch_size = storage.batch_size;
    const int32_t dimF = storage.dimF;
    const int32_t nnz_J = storage.nnz_J;

    if (batch_size <= 0 || dimF <= 0 || nnz_J <= 0) {
        throw std::runtime_error("CudaLinearSolveCuDSS32: invalid batch descriptor dimensions");
    }

    const bool descriptors_match =
        state.jacobian != nullptr &&
        state.rhs_matrix != nullptr &&
        state.solution_matrix != nullptr &&
        state.descriptor_batch_size == batch_size &&
        state.descriptor_dimF == dimF &&
        state.descriptor_nnz_J == nnz_J;

    if (descriptors_match) {
        return;
    }

    destroy_matrix(state.jacobian);
    destroy_matrix(state.rhs_matrix);
    destroy_matrix(state.solution_matrix);

    state.rhs.resize(static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(dimF));

    set_ubatch_size(state.config, batch_size);

    CUDSS_CHECK(cudssMatrixCreateCsr(
        &state.jacobian,
        dimF, dimF, static_cast<int64_t>(nnz_J),
        storage.d_J_row_ptr.data(), nullptr, storage.d_J_col_idx.data(), storage.d_J_values.data(),
        CUDA_R_32I, CUDA_R_32F,
        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &state.rhs_matrix,
        dimF, 1, dimF, state.rhs.data(),
        CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &state.solution_matrix,
        dimF, 1, dimF, storage.d_dx.data(),
        CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));

    state.descriptor_batch_size = batch_size;
    state.descriptor_dimF = dimF;
    state.descriptor_nnz_J = nnz_J;
    state.analyzed = false;
    state.factorized = false;
}
#endif

}  // namespace

CudaLinearSolveCuDSS32::CudaLinearSolveCuDSS32(IStorage& storage,
                                               CuDSSOptions cudss_options)
    : storage_(storage),
      cudss_options_(cudss_options) {}

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

    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.cudss32.setup");
        CUDSS_CHECK(cudssCreate(&state->handle));
        cupf_cudss_detail::configure_handle(state->handle);
        CUDSS_CHECK(cudssConfigCreate(&state->config));
        cupf_cudss_detail::configure_solver(state->config, cudss_options_);
        CUDSS_CHECK(cudssDataCreate(state->handle, &state->data));
    }

    state_ = state.release();
#endif
}


void CudaLinearSolveCuDSS32::run(IterationContext& ctx)
{
    factorize_and_solve(ctx);
}

void CudaLinearSolveCuDSS32::factorize_and_solve(IterationContext& ctx)
{
    factorize(ctx);
    solve(ctx);
}

void CudaLinearSolveCuDSS32::factorize(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaMixedStorage&>(storage_);

    if (storage.dimF <= 0 || storage.d_F.empty() || storage.d_dx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS32::factorize: storage is not prepared");
    }
    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS32::factorize: analyze() must be called first");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS32::factorize requires a cuDSS-enabled build");
#else
    ensure_batch_descriptors(*state_, storage);
    set_ubatch_size(state_->config, storage.batch_size);

    if (!state_->analyzed) {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.cudss32.analysis");
        CUDSS_CHECK(cudssExecute(
            state_->handle, CUDSS_PHASE_ANALYSIS,
            state_->config, state_->data,
            state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
        sync_cuda_for_timing();
        state_->analyzed = true;
    }

    const bool is_refactorization = state_->factorized;
    state_->pending_solve_phase = is_refactorization ? "refactorization" : "factorization";

    {
        // 첫 번째: FACTORIZATION, 이후: REFACTORIZATION (symbolic 재사용)
        newton_solver::utils::ScopedTimer timer(
            is_refactorization ? "CUDA.solve.refactorization32" : "CUDA.solve.factorization32");
        const int phase = is_refactorization ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
        CUDSS_CHECK(cudssExecute(
            state_->handle, phase,
            state_->config, state_->data,
            state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
        sync_cuda_for_timing();
        state_->factorized = true;
    }
#endif
}

void CudaLinearSolveCuDSS32::solve(IterationContext& ctx)
{
    auto& storage = static_cast<CudaMixedStorage&>(storage_);

    if (storage.dimF <= 0 || storage.d_F.empty() || storage.d_dx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS32::solve: storage is not prepared");
    }
    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS32::solve: analyze() must be called first");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS32::solve requires a cuDSS-enabled build");
#else
    if (!state_->factorized) {
        throw std::runtime_error("CudaLinearSolveCuDSS32::solve: factorize() must be called first");
    }
    set_ubatch_size(state_->config, storage.batch_size);

    {
        // RHS = F (FP64 → FP32 device cast):
        //   d_F(double*) → d_rhs(float*)
        // update stage가 state -= dx를 적용한다.
        newton_solver::utils::ScopedTimer timer("CUDA.solve.rhsPrepare");
        const int32_t rhs_count = storage.batch_size * storage.dimF;
        launch_cast_rhs_f64_to_f32(storage.d_F.data(), state_->rhs.data(), rhs_count);
    }

    {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.solve32");
        CUDSS_CHECK(cudssExecute(
            state_->handle, CUDSS_PHASE_SOLVE,
            state_->config, state_->data,
            state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
        sync_cuda_for_timing();
    }

    if (newton_solver::utils::isDumpEnabled()) {
        std::vector<int32_t> h_row_ptr(static_cast<std::size_t>(storage.dimF + 1));
        std::vector<int32_t> h_col_idx(storage.d_J_col_idx.size());
        std::vector<float> h_jacobian_values(storage.d_J_values.size());
        std::vector<double> h_F(storage.d_F.size());
        std::vector<float> h_dx(storage.d_dx.size());

        storage.d_J_row_ptr.copyTo(h_row_ptr.data(), h_row_ptr.size());
        storage.d_J_col_idx.copyTo(h_col_idx.data(), h_col_idx.size());
        storage.d_J_values.copyTo(h_jacobian_values.data(), h_jacobian_values.size());
        storage.d_F.copyTo(h_F.data(), h_F.size());
        storage.d_dx.copyTo(h_dx.data(), h_dx.size());

        const std::vector<int64_t> npivots =
            newton_solver::linear_diagnostics::try_get_cudss_int_values(
                state_->handle, state_->data, CUDSS_DATA_NPIVOTS);
        if (storage.batch_size == 1) {
            newton_solver::linear_diagnostics::dump_linear_system(
                ctx,
                "cuda",
                "fp32",
                state_->pending_solve_phase,
                h_row_ptr,
                h_col_idx,
                h_jacobian_values,
                h_F,
                h_dx,
                npivots);
        } else {
            const auto j_begin = h_jacobian_values.begin();
            const auto f_begin = h_F.begin();
            const auto dx_begin = h_dx.begin();
            std::vector<float> h_jacobian_values_b0(
                j_begin,
                j_begin + static_cast<std::ptrdiff_t>(storage.nnz_J));
            std::vector<double> h_F_b0(
                f_begin,
                f_begin + static_cast<std::ptrdiff_t>(storage.dimF));
            std::vector<float> h_dx_b0(
                dx_begin,
                dx_begin + static_cast<std::ptrdiff_t>(storage.dimF));

            newton_solver::linear_diagnostics::dump_linear_system(
                ctx,
                "cuda",
                "fp32_batch0",
                state_->pending_solve_phase,
                h_row_ptr,
                h_col_idx,
                h_jacobian_values_b0,
                h_F_b0,
                h_dx_b0,
                npivots);
        }
    }
    state_->pending_solve_phase = "solve_only";
#endif
}

#endif  // CUPF_WITH_CUDA
