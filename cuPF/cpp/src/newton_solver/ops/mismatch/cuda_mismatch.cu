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


void launch_compute_mismatch_from_ibus(CudaFp64Storage& buf);
void launch_compute_mismatch_from_ibus(CudaFp32Storage& buf);
void launch_compute_mismatch_from_ibus(CudaMixedStorage& buf);
void launch_reduce_mismatch_norm(CudaFp64Storage& buf);
void launch_reduce_mismatch_norm(CudaFp32Storage& buf);
void launch_reduce_mismatch_norm(CudaMixedStorage& buf);


// CudaMismatchOp

template <>
void CudaMismatchOp<CudaFp64Storage>::run(CudaFp64Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    if (buf.n_bus <= 0 || buf.dimF <= 0) {
        throw std::runtime_error("CudaMismatchOp::run: buffers are not prepared");
    }
    launch_compute_mismatch_from_ibus(buf);
}

template <>
void CudaMismatchOp<CudaFp32Storage>::run(CudaFp32Storage& buf, IterationContext& ctx)
{
    (void)ctx;
    if (buf.n_bus <= 0 || buf.dimF <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("CudaMismatchOp::run: buffers are not prepared");
    }
    launch_compute_mismatch_from_ibus(buf);
}

template <>
void CudaMismatchOp<CudaMixedStorage>::run(CudaMixedStorage& buf, IterationContext& ctx)
{
    (void)ctx;
    if (buf.n_bus <= 0 || buf.dimF <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("CudaMismatchOp::run: buffers are not prepared");
    }
    launch_compute_mismatch_from_ibus(buf);
}


// CudaMismatchNormOp

template <>
void CudaMismatchNormOp<CudaFp64Storage>::run(CudaFp64Storage& buf, IterationContext& ctx)
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
void CudaMismatchNormOp<CudaFp32Storage>::run(CudaFp32Storage& buf, IterationContext& ctx)
{
    if (buf.dimF <= 0 || buf.batch_size <= 0) {
        throw std::runtime_error("CudaMismatchNormOp::run: buffers are not prepared");
    }

    launch_reduce_mismatch_norm(buf);

    std::vector<float> h_norm(buf.batch_size);
    buf.d_normF.copyTo(h_norm.data(), h_norm.size());

    ctx.normF = 0.0;
    for (float v : h_norm) {
        ctx.normF = std::max(ctx.normF, static_cast<double>(v));
    }

    if (!std::isfinite(ctx.normF)) {
        throw std::runtime_error("CudaMismatchNormOp::run: mismatch norm is not finite");
    }

    if (newton_solver::utils::isDumpEnabled()) {
        std::vector<float> h_F(buf.batch_size * buf.dimF);
        buf.d_F.copyTo(h_F.data(), h_F.size());
        newton_solver::utils::dumpVector("residual", ctx.iter, h_F);
        newton_solver::utils::dumpVector("residual_before_update", ctx.iter, h_F);
    }

    ctx.converged = (ctx.normF <= ctx.config.tolerance);
}

template <>
void CudaMismatchNormOp<CudaMixedStorage>::run(CudaMixedStorage& buf, IterationContext& ctx)
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
