#include "cpu_mismatch.hpp"

#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/dump.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>


namespace {
using CpuComplexVectorF64 = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
}


void CpuMismatchOp::run(CpuFp64Buffers& buf, IterationContext& ctx)
{
    if (buf.n_bus <= 0 || buf.dimF <= 0) {
        throw std::runtime_error("CpuMismatchOp::run: buffers are not prepared");
    }
    if ((ctx.n_pv > 0 && ctx.pv == nullptr) || (ctx.n_pq > 0 && ctx.pq == nullptr)) {
        throw std::invalid_argument("CpuMismatchOp::run: pv/pq pointers must not be null");
    }
    if (!buf.has_cached_Ibus) {
        throw std::runtime_error("CpuMismatchOp::run: Ibus stage has not run");
    }

    Eigen::Map<const CpuComplexVectorF64> V(buf.V.data(), buf.n_bus);
    Eigen::Map<const CpuComplexVectorF64> Sbus(buf.Sbus.data(), buf.n_bus);
    Eigen::Map<const CpuComplexVectorF64> Ibus(buf.Ibus.data(), buf.n_bus);

    const CpuComplexVectorF64 mis =
        V.array() * Ibus.array().conjugate() - Sbus.array();

    int32_t k = 0;
    for (int32_t i = 0; i < ctx.n_pv; ++i) buf.F[k++] = mis[ctx.pv[i]].real();
    for (int32_t i = 0; i < ctx.n_pq; ++i) buf.F[k++] = mis[ctx.pq[i]].real();
    for (int32_t i = 0; i < ctx.n_pq; ++i) buf.F[k++] = mis[ctx.pq[i]].imag();
}


void CpuMismatchNormOp::run(CpuFp64Buffers& buf, IterationContext& ctx)
{
    if (buf.dimF <= 0 || buf.F.empty()) {
        throw std::runtime_error("CpuMismatchNormOp::run: buffers are not prepared");
    }

    ctx.normF = 0.0;
    for (double value : buf.F) {
        ctx.normF = std::max(ctx.normF, std::abs(value));
    }

    if (!std::isfinite(ctx.normF)) {
        throw std::runtime_error("CpuMismatchNormOp::run: mismatch norm is not finite");
    }

    newton_solver::utils::dumpArray("residual", ctx.iter,
                                    buf.F.data(), static_cast<int32_t>(buf.F.size()));
    newton_solver::utils::dumpArray("residual_before_update", ctx.iter,
                                    buf.F.data(), static_cast<int32_t>(buf.F.size()));
    ctx.converged = (ctx.normF <= ctx.config.tolerance);
}
