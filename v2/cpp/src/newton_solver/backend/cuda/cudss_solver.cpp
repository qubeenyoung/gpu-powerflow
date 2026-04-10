#include "cuda_backend_impl.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/timer.hpp"

#include <cuda_runtime.h>
#include <cudss.h>

#include <vector>

namespace {

void sync_cuda_for_timing()
{
#ifdef CUPF_ENABLE_TIMING
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

}  // namespace
// ---------------------------------------------------------------------------
// solveLinearSystem
//
// Numerically factorizes the Jacobian and solves J·dx = -F.
//
// cuDSS pipeline per NR iteration:
//   REFACTORIZATION — numeric LU reusing symbolic ordering from analyze()
//   SOLVE           — triangular solve J·x = b
//
// Mixed — Jacobian FP32 (d_J_csr_f); RHS: d_F (FP64) → d_b_f (FP32) cast
// FP64  — Jacobian FP64 (d_J_csr_d); RHS: d_F (FP64) → d_b_d (FP64) negate
//
// Host pointers F/dx are intentionally ignored — all data stays on GPU.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::solveLinearSystem(const double* /*F*/, double* /*dx*/)
{
    auto& im = *impl_;

    if (im.precision_mode == PrecisionMode::FP64) {
        // FP64: negate d_F → d_b_d, then FP64 cuDSS solve
        {
            newton_solver::utils::ScopedTimer timer("CUDA.solve.rhsPrepare");
            cuda_negate_f64(im.d_F, im.d_b_d, im.dimF);
            sync_cuda_for_timing();
        }
        {
            newton_solver::utils::ScopedTimer timer("CUDA.solve.refactorization");
            CUDSS_CHECK(cudssExecute(
                im.dss_handle, CUDSS_PHASE_REFACTORIZATION,
                im.dss_config_d64, im.dss_data_d64,
                im.dss_J_d64, im.dss_x_d64, im.dss_b_d64));
            sync_cuda_for_timing();
        }
        {
            newton_solver::utils::ScopedTimer timer("CUDA.solve.solve");
            CUDSS_CHECK(cudssExecute(
                im.dss_handle, CUDSS_PHASE_SOLVE,
                im.dss_config_d64, im.dss_data_d64,
                im.dss_J_d64, im.dss_x_d64, im.dss_b_d64));
            sync_cuda_for_timing();
        }
        // d_x_d stays on GPU; updateVoltage() reads it directly.
        return;
    }

    // Mixed: cast FP64 d_F → FP32 d_b_f, then FP32 cuDSS solve
    {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.rhsPrepare");
        cuda_negate_cast(im.d_F, im.d_b_f, im.dimF);
        sync_cuda_for_timing();
    }
    {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.refactorization");
        CUDSS_CHECK(cudssExecute(
            im.dss_handle, CUDSS_PHASE_REFACTORIZATION,
            im.dss_config, im.dss_data,
            im.dss_J, im.dss_x, im.dss_b));
        sync_cuda_for_timing();
    }
    {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.solve");
        CUDSS_CHECK(cudssExecute(
            im.dss_handle, CUDSS_PHASE_SOLVE,
            im.dss_config, im.dss_data,
            im.dss_J, im.dss_x, im.dss_b));
        sync_cuda_for_timing();
    }
    // d_x_f stays on GPU; updateVoltage() consumes it directly.
}


// ---------------------------------------------------------------------------
// solveLinearSystem_f32
//
// FP32 end-to-end solve. d_b_f already holds -F (written by computeMismatch_f32).
// Runs REFACTORIZATION + SOLVE with FP32 cuDSS matrices.
// Host pointers F/dx are intentionally ignored.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::solveLinearSystem_f32(const float* /*F*/, float* /*dx*/)
{
    auto& im = *impl_;

    // d_b_f = -F already prepared by computeMismatch_f32 (no extra cast needed)
    {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.refactorization");
        CUDSS_CHECK(cudssExecute(
            im.dss_handle, CUDSS_PHASE_REFACTORIZATION,
            im.dss_config, im.dss_data,
            im.dss_J, im.dss_x, im.dss_b));
        sync_cuda_for_timing();
    }
    {
        newton_solver::utils::ScopedTimer timer("CUDA.solve.solve");
        CUDSS_CHECK(cudssExecute(
            im.dss_handle, CUDSS_PHASE_SOLVE,
            im.dss_config, im.dss_data,
            im.dss_J, im.dss_x, im.dss_b));
        sync_cuda_for_timing();
    }
    // d_x_f stays on GPU; updateVoltage_f32() consumes it directly.
}


// ---------------------------------------------------------------------------
// solveLinearSystem_batch
//
// cuDSS UBATCH pipeline per NR iteration:
//   1. Upload -F_batch (FP64→FP32) into d_b_f_batch
//   2. REFACTORIZATION: numeric LU for all n_batch Jacobians simultaneously
//   3. SOLVE: triangular solve for all n_batch RHS simultaneously
//
// The solution d_x_f_batch is consumed directly by updateVoltage_batch() via
// the GPU buffer — no host download needed between these two calls.
// F_batch is [n_batch * dimF] on the host (FP64).
// dx_batch output is [n_batch * dimF] (FP64), converted from d_x_f_batch.
// ---------------------------------------------------------------------------
void CudaNewtonSolverBackend::solveLinearSystem_batch(
    const double* F_batch,
    double*       dx_batch,
    int32_t       n_batch)
{
    auto& im = *impl_;
    const int32_t dimF  = im.dimF;
    const int64_t total = (int64_t)n_batch * dimF;

    {
        newton_solver::utils::ScopedTimer timer("CUDA.batch.solve.rhsPrepare");
        cuda_negate_cast(im.d_F_batch, im.d_b_f_batch, static_cast<int32_t>(total));
        sync_cuda_for_timing();
    }

    {
        newton_solver::utils::ScopedTimer timer("CUDA.batch.solve.refactorization");
        CUDSS_CHECK(cudssExecute(
            im.dss_handle, CUDSS_PHASE_REFACTORIZATION,
            im.dss_config_batch, im.dss_data_batch,
            im.dss_J_batch, im.dss_x_batch, im.dss_b_batch));
        sync_cuda_for_timing();
    }

    {
        newton_solver::utils::ScopedTimer timer("CUDA.batch.solve.solve");
        CUDSS_CHECK(cudssExecute(
            im.dss_handle, CUDSS_PHASE_SOLVE,
            im.dss_config_batch, im.dss_data_batch,
            im.dss_J_batch, im.dss_x_batch, im.dss_b_batch));
        sync_cuda_for_timing();
    }
    // d_x_f_batch stays on GPU; updateVoltage_batch() reads it directly.
    // dx_batch (host) is intentionally not populated.
    (void)dx_batch;
}
