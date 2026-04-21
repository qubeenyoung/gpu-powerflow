// ---------------------------------------------------------------------------
// cuda_cudss64.cpp — CUDA FP64 선형 솔버 구현 (cuDSS FP64)
//
// cuDSS (CUDA Direct Sparse Solver) FP64 경로.
// Jacobian CSR(FP64) 을 직접 cuDSS 에 등록하고 LU 분해·역대입을 GPU에서 수행.
//
// 생명주기:
//   analyze() — 한 번만 호출:
//     1. cudssCreate / cudssConfigCreate / cudssDataCreate
//     2. cudssMatrixCreateCsr (J, int32 인덱스 / FP64 값)
//     3. cudssMatrixCreateDn  (rhs: FP64 dense, solution: d_dx FP64 dense)
//     4. CUDSS_PHASE_ANALYSIS (symbolic: 재순서화, 비영 패턴 분석)
//
//   run() — 매 NR 반복 호출:
//     0. matching을 켠 런타임 옵션에서는 첫 run()에서 값 기반 CUDSS_PHASE_ANALYSIS 수행
//     1. RHS는 d_F를 직접 사용한다 (rhs = F).
//        update stage가 state -= dx를 적용하므로 선형계는 J * dx = F다.
//     2. CUDSS_PHASE_FACTORIZATION (첫 번째) 또는 CUDSS_PHASE_REFACTORIZATION (이후)
//        REFACTORIZATION은 symbolic 자료구조를 재사용하므로 첫 번째보다 빠름.
//     3. CUDSS_PHASE_SOLVE → d_dx = J⁻¹·F
//
// CuDSS64State: cuDSS 핸들·디스크립터를 소유하는 pimpl 구조체.
//   소멸자에서 순서대로 해제한다 (matrix → data → config → handle).
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_cudss64.hpp"

#include "cudss_config.hpp"
#include "linear_diagnostics.hpp"
#include "newton_solver/core/contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/dump.hpp"
#include "utils/timer.hpp"

#include <memory>
#include <stdexcept>
#include <vector>


// cuDSS 핸들·디스크립터를 소유하는 pimpl 구조체.
// 소멸자에서 역순으로 해제한다.
struct CudaLinearSolveCuDSS64::CuDSS64State {
#ifdef CUPF_ENABLE_CUDSS
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t jacobian = nullptr;
    cudssMatrix_t rhs_matrix = nullptr;
    cudssMatrix_t solution_matrix = nullptr;
#endif
    bool analyzed = false;
    bool factorized = false;
    const char* pending_solve_phase = "solve_only";

    ~CuDSS64State()
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

CudaLinearSolveCuDSS64::CudaLinearSolveCuDSS64(IStorage& storage,
                                               CuDSSOptions cudss_options)
    : storage_(storage),
      cudss_options_(cudss_options) {}

CudaLinearSolveCuDSS64::~CudaLinearSolveCuDSS64()
{
    delete state_;
}


void CudaLinearSolveCuDSS64::analyze(const AnalyzeContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaFp64Storage&>(storage_);

    if (storage.dimF <= 0 || storage.d_J_row_ptr.empty() || storage.d_J_col_idx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS64::analyze: storage is not prepared");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS64::analyze requires a cuDSS-enabled build");
#else
    delete state_;
    state_ = nullptr;

    auto state = std::make_unique<CuDSS64State>();

    {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.cudss64.setup");
        CUDSS_CHECK(cudssCreate(&state->handle));
        cupf_cudss_detail::configure_handle(state->handle);
        CUDSS_CHECK(cudssConfigCreate(&state->config));
        cupf_cudss_detail::configure_solver(state->config, cudss_options_);
        CUDSS_CHECK(cudssDataCreate(state->handle, &state->data));

        CUDSS_CHECK(cudssMatrixCreateCsr(
            &state->jacobian,
            storage.dimF, storage.dimF, static_cast<int64_t>(storage.d_J_values.size()),
            storage.d_J_row_ptr.data(), nullptr, storage.d_J_col_idx.data(), storage.d_J_values.data(),
            CUDA_R_32I, CUDA_R_64F,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &state->rhs_matrix,
            storage.dimF, 1, storage.dimF, storage.d_F.data(),
            CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(
            &state->solution_matrix,
            storage.dimF, 1, storage.dimF, storage.d_dx.data(),
            CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    }

    if (!cupf_cudss_detail::analysis_requires_matrix_values(cudss_options_)) {
        newton_solver::utils::ScopedTimer timer("CUDA.analyze.cudss64.analysis");
        CUDSS_CHECK(cudssExecute(
            state->handle, CUDSS_PHASE_ANALYSIS,
            state->config, state->data,
            state->jacobian, state->solution_matrix, state->rhs_matrix));
        sync_cuda_for_timing();
        state->analyzed = true;
    }

    state_ = state.release();
#endif
}


void CudaLinearSolveCuDSS64::run(IterationContext& ctx)
{
    factorize_and_solve(ctx);
}

void CudaLinearSolveCuDSS64::factorize_and_solve(IterationContext& ctx)
{
    factorize(ctx);
    solve(ctx);
}

void CudaLinearSolveCuDSS64::factorize(IterationContext& ctx)
{
    (void)ctx;
    auto& storage = static_cast<CudaFp64Storage&>(storage_);

    if (storage.dimF <= 0 || storage.d_F.empty() || storage.d_dx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS64::factorize: storage is not prepared");
    }
    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS64::factorize: analyze() must be called first");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS64::factorize requires a cuDSS-enabled build");
#else
    if (!state_->analyzed) {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.cudss64.analysis");
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
        // 첫 번째 호출: FACTORIZATION (symbolic 자료구조 포함 완전 분해)
        // 이후 호출:   REFACTORIZATION (symbolic 재사용, 수치 재분해만 수행 → 더 빠름)
        newton_solver::utils::ScopedTimer timer(
            is_refactorization ? "CUDA.solve.refactorization64" : "CUDA.solve.factorization64");
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

void CudaLinearSolveCuDSS64::solve(IterationContext& ctx)
{
    auto& storage = static_cast<CudaFp64Storage&>(storage_);

    if (storage.dimF <= 0 || storage.d_F.empty() || storage.d_dx.empty()) {
        throw std::runtime_error("CudaLinearSolveCuDSS64::solve: storage is not prepared");
    }
    if (state_ == nullptr) {
        throw std::runtime_error("CudaLinearSolveCuDSS64::solve: analyze() must be called first");
    }

#ifndef CUPF_ENABLE_CUDSS
    throw std::runtime_error("CudaLinearSolveCuDSS64::solve requires a cuDSS-enabled build");
#else
    if (!state_->factorized) {
        throw std::runtime_error("CudaLinearSolveCuDSS64::solve: factorize() must be called first");
    }

    {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.solve64");
        CUDSS_CHECK(cudssExecute(
            state_->handle, CUDSS_PHASE_SOLVE,
            state_->config, state_->data,
            state_->jacobian, state_->solution_matrix, state_->rhs_matrix));
        sync_cuda_for_timing();
    }

    if (newton_solver::utils::isDumpEnabled()) {
        std::vector<int32_t> h_row_ptr(static_cast<std::size_t>(storage.dimF + 1));
        std::vector<int32_t> h_col_idx(storage.d_J_col_idx.size());
        std::vector<double> h_jacobian_values(storage.d_J_values.size());
        std::vector<double> h_F(static_cast<std::size_t>(storage.dimF));
        std::vector<double> h_dx(static_cast<std::size_t>(storage.dimF));

        storage.d_J_row_ptr.copyTo(h_row_ptr.data(), h_row_ptr.size());
        storage.d_J_col_idx.copyTo(h_col_idx.data(), h_col_idx.size());
        storage.d_J_values.copyTo(h_jacobian_values.data(), h_jacobian_values.size());
        storage.d_F.copyTo(h_F.data(), h_F.size());
        storage.d_dx.copyTo(h_dx.data(), h_dx.size());

        const std::vector<int64_t> npivots =
            newton_solver::linear_diagnostics::try_get_cudss_int_values(
                state_->handle, state_->data, CUDSS_DATA_NPIVOTS);
        newton_solver::linear_diagnostics::dump_linear_system(
            ctx,
            "cuda",
            "fp64",
            state_->pending_solve_phase,
            h_row_ptr,
            h_col_idx,
            h_jacobian_values,
            h_F,
            h_dx,
            npivots);
    }
    state_->pending_solve_phase = "solve_only";
#endif
}

#endif  // CUPF_WITH_CUDA
