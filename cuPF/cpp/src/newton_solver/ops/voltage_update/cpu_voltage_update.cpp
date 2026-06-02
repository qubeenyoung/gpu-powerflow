#include "cpu_voltage_update.hpp"

#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <cmath>
#include <stdexcept>


// Apply the Newton step dx to the voltage state: V <- V - dx, in the dimF
// ordering [ dVa@pv | dVa@pq | dVm@pq ], then rebuild the rectangular voltage
// V = Vm * (cos Va + i sin Va). Marks the cached Ibus stale for the next iter.
void CpuVoltageUpdateOp::run(CpuFp64Storage& buf, IterationContext& ctx)
{
    if (buf.n_bus <= 0 || buf.dimF <= 0) {
        throw std::runtime_error("CpuVoltageUpdateOp::run: buffers are not prepared");
    }
    if ((ctx.n_pv > 0 && ctx.pv == nullptr) || (ctx.n_pq > 0 && ctx.pq == nullptr)) {
        throw std::invalid_argument("CpuVoltageUpdateOp::run: pv/pq pointers must not be null");
    }
    if (ctx.n_pv + 2 * ctx.n_pq != buf.dimF) {
        throw std::invalid_argument("CpuVoltageUpdateOp::run: pv/pq dimensions do not match buffers");
    }

    // Angle update at pv buses, then angle and magnitude updates at pq buses.
    for (int32_t i = 0; i < ctx.n_pv; ++i) {
        buf.Va[ctx.pv[i]] -= buf.dx[i];
    }
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        buf.Va[ctx.pq[i]] -= buf.dx[ctx.n_pv + i];
    }
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        buf.Vm[ctx.pq[i]] -= buf.dx[ctx.n_pv + ctx.n_pq + i];
    }

    // Refresh the rectangular voltage from the updated polar (Vm, Va).
    for (int32_t bus = 0; bus < buf.n_bus; ++bus) {
        const double vm = buf.Vm[bus];
        const double va = buf.Va[bus];
        buf.V[bus] = std::complex<double>(vm * std::cos(va), vm * std::sin(va));
    }

    buf.has_cached_Ibus = false;
}
