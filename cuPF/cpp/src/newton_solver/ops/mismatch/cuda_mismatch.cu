// ---------------------------------------------------------------------------
// cuda_mismatch.cu
//
// CUDA dispatch for the two mismatch NR stages. The device kernels live in
// compute_mismatch_from_ibus.cu (residual F) and reduce_mismatch_norm.cu
// (per-case max|F|); these Op::run overloads validate state, launch them, and
// for the norm pull the per-case results to host, take the worst case as the
// batch convergence norm, and set the converged flag.
//
// All three precision profiles share one templated helper each; the profiles
// differ only in the device scalar (FP32 -> float, FP64/Mixed -> double state).
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include "cuda_mismatch.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_fp32_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/dump.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>


// Defined in compute_mismatch_from_ibus.cu / reduce_mismatch_norm.cu.
void launch_compute_mismatch_from_ibus(CudaFp64Storage& buf);
void launch_compute_mismatch_from_ibus(CudaFp32Storage& buf);
void launch_compute_mismatch_from_ibus(CudaMixedStorage& buf);
void launch_reduce_mismatch_norm(CudaFp64Storage& buf);
void launch_reduce_mismatch_norm(CudaFp32Storage& buf);
void launch_reduce_mismatch_norm(CudaMixedStorage& buf);


namespace {

// Shared body of CudaMismatchOp::run: validate, then compute the residual F.
template <typename Storage>
void run_compute_mismatch(Storage& buf)
{
    if (buf.n_bus <= 0 || buf.dimF <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("CudaMismatchOp::run: buffers are not prepared");
    }
    launch_compute_mismatch_from_ibus(buf);
}

// Shared body of CudaMismatchNormOp::run. NormScalar is the storage's residual/
// norm element type (float for FP32, double for FP64/Mixed). Reduces max|F| per
// case on device, takes the worst case as the batch norm, checks finiteness,
// optionally dumps the residual, and sets the converged flag.
template <typename NormScalar, typename Storage>
void reduce_norm_into_ctx(Storage& buf, IterationContext& ctx)
{
    if (buf.dimF <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("CudaMismatchNormOp::run: buffers are not prepared");
    }

    launch_reduce_mismatch_norm(buf);

    // Pull the per-case norms and take the worst case as the batch norm (so the
    // whole batch is "converged" only when every case is). NormScalar may be
    // float; widen to double for the host-side comparison.
    std::vector<NormScalar> h_norm(buf.batch_size);
    buf.d_normF.copyTo(h_norm.data(), h_norm.size());

    ctx.normF = 0.0;
    for (NormScalar v : h_norm) {
        ctx.normF = std::max(ctx.normF, static_cast<double>(v));
    }

    // A non-finite norm means the iteration blew up (e.g. FP32 on a stiff case);
    // surface it instead of silently looping to the iteration cap.
    if (!std::isfinite(ctx.normF)) {
        throw std::runtime_error("CudaMismatchNormOp::run: mismatch norm is not finite");
    }

    if (newton_solver::utils::isDumpEnabled()) {
        std::vector<NormScalar> h_F(buf.batch_size * buf.dimF);
        buf.d_F.copyTo(h_F.data(), h_F.size());
        newton_solver::utils::dumpVector("residual", ctx.iter, h_F);
        newton_solver::utils::dumpVector("residual_before_update", ctx.iter, h_F);
    }

    ctx.converged = (ctx.normF <= ctx.config.tolerance);
}

}  // namespace


// --- CudaMismatchOp: F = V * conj(Ibus) - Sbus ------------------------------

template <>
void CudaMismatchOp<CudaFp64Storage>::run(CudaFp64Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    run_compute_mismatch(buf);
}

template <>
void CudaMismatchOp<CudaFp32Storage>::run(CudaFp32Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    run_compute_mismatch(buf);
}

template <>
void CudaMismatchOp<CudaMixedStorage>::run(CudaMixedStorage& buf, IterationContext& ctx)
{
    (void)ctx;
    run_compute_mismatch(buf);
}


// --- CudaMismatchNormOp: convergence norm = max over cases of max|F| --------

template <>
void CudaMismatchNormOp<CudaFp64Storage>::run(CudaFp64Storage& buf, IterationContext& ctx)
{
    reduce_norm_into_ctx<double>(buf, ctx);
}

template <>
void CudaMismatchNormOp<CudaFp32Storage>::run(CudaFp32Storage& buf, IterationContext& ctx)
{
    reduce_norm_into_ctx<float>(buf, ctx);
}

template <>
void CudaMismatchNormOp<CudaMixedStorage>::run(CudaMixedStorage& buf, IterationContext& ctx)
{
    reduce_norm_into_ctx<double>(buf, ctx);
}


#endif  // CUPF_WITH_CUDA
