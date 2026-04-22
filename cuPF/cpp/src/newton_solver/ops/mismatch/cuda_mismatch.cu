#ifdef CUPF_WITH_CUDA

#include "cuda_mismatch.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#include "utils/dump.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>


void launch_compute_mismatch_from_ibus(CudaFp64Buffers& buf);
void launch_compute_mismatch_from_ibus(CudaMixedBuffers& buf);
void launch_reduce_mismatch_norm(CudaFp64Buffers& buf);
void launch_reduce_mismatch_norm(CudaMixedBuffers& buf);


// CudaMismatchOp

template <>
void CudaMismatchOp<CudaFp64Buffers>::run(CudaFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;
    if (buf.n_bus <= 0 || buf.dimF <= 0) {
        throw std::runtime_error("CudaMismatchOp::run: buffers are not prepared");
    }
    launch_compute_mismatch_from_ibus(buf);
}

template <>
void CudaMismatchOp<CudaMixedBuffers>::run(CudaMixedBuffers& buf, IterationContext& ctx)
{
    (void)ctx;
    if (buf.n_bus <= 0 || buf.dimF <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("CudaMismatchOp::run: buffers are not prepared");
    }
    launch_compute_mismatch_from_ibus(buf);
}


// CudaMismatchNormOp

template <>
void CudaMismatchNormOp<CudaFp64Buffers>::run(CudaFp64Buffers& buf, IterationContext& ctx)
{
    if (buf.dimF <= 0) {
        throw std::runtime_error("CudaMismatchNormOp::run: buffers are not prepared");
    }

    launch_reduce_mismatch_norm(buf);
    buf.d_normF.copyTo(&ctx.normF, 1);

    if (!std::isfinite(ctx.normF)) {
        throw std::runtime_error("CudaMismatchNormOp::run: mismatch norm is not finite");
    }

    if (newton_solver::utils::isDumpEnabled()) {
        std::vector<double> h_F(buf.dimF);
        buf.d_F.copyTo(h_F.data(), h_F.size());
        newton_solver::utils::dumpVector("residual", ctx.iter, h_F);
        newton_solver::utils::dumpVector("residual_before_update", ctx.iter, h_F);
    }

    ctx.converged = (ctx.normF <= ctx.config.tolerance);
}

template <>
void CudaMismatchNormOp<CudaMixedBuffers>::run(CudaMixedBuffers& buf, IterationContext& ctx)
{
    if (buf.dimF <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("CudaMismatchNormOp::run: buffers are not prepared");
    }

    launch_reduce_mismatch_norm(buf);

    std::vector<double> h_norm(buf.batch_size);
    buf.d_normF.copyTo(h_norm.data(), h_norm.size());

    ctx.normF = 0.0;
    for (double v : h_norm) {
        ctx.normF = std::max(ctx.normF, v);
    }

    if (!std::isfinite(ctx.normF)) {
        throw std::runtime_error("CudaMismatchNormOp::run: mismatch norm is not finite");
    }

    if (newton_solver::utils::isDumpEnabled()) {
        std::vector<double> h_F(buf.batch_size * buf.dimF);
        buf.d_F.copyTo(h_F.data(), h_F.size());
        newton_solver::utils::dumpVector("residual", ctx.iter, h_F);
        newton_solver::utils::dumpVector("residual_before_update", ctx.iter, h_F);
    }

    ctx.converged = (ctx.normF <= ctx.config.tolerance);
}


#endif  // CUPF_WITH_CUDA
