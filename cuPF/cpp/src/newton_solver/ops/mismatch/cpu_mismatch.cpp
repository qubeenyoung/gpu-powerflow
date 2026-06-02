#include "cpu_mismatch.hpp"

#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"
#include "utils/dump.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>


// Power-flow residual F from the complex power mismatch S(V) - Sbus. Packed in
// the dimF order [ dP@pv | dP@pq | dQ@pq ]: real parts (active power) for pv+pq
// buses, then imaginary parts (reactive power) for pq buses.
void CpuMismatchOp::run(CpuFp64Storage& buf, IterationContext& ctx)
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

    // mis[i] = V[i] * conj(Ibus[i]) - Sbus[i]
    auto mis_at = [&buf](int32_t i) -> std::complex<double> {
        const std::size_t idx = i;
        return buf.V[idx] * std::conj(buf.Ibus[idx]) - buf.Sbus[idx];
    };

    int32_t k = 0;
    for (int32_t i = 0; i < ctx.n_pv; ++i) buf.F[k++] = mis_at(ctx.pv[i]).real();
    for (int32_t i = 0; i < ctx.n_pq; ++i) buf.F[k++] = mis_at(ctx.pq[i]).real();
    for (int32_t i = 0; i < ctx.n_pq; ++i) buf.F[k++] = mis_at(ctx.pq[i]).imag();
}


// Infinity-norm of the residual; sets the convergence flag against tolerance.
void CpuMismatchNormOp::run(CpuFp64Storage& buf, IterationContext& ctx)
{
    if (buf.dimF <= 0 || buf.F.empty()) {
        throw std::runtime_error("CpuMismatchNormOp::run: buffers are not prepared");
    }

    // max |F_i| over all residual entries.
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
