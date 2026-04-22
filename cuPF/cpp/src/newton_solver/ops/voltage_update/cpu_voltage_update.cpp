#include "cpu_voltage_update.hpp"

#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <cmath>
#include <stdexcept>


void CpuVoltageUpdateOp::run(CpuFp64Buffers& buf, IterationContext& ctx)
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

    for (int32_t i = 0; i < ctx.n_pv; ++i) {
        buf.Va[static_cast<std::size_t>(ctx.pv[i])] -= buf.dx[static_cast<std::size_t>(i)];
    }
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        buf.Va[static_cast<std::size_t>(ctx.pq[i])] -=
            buf.dx[static_cast<std::size_t>(ctx.n_pv + i)];
    }
    for (int32_t i = 0; i < ctx.n_pq; ++i) {
        buf.Vm[static_cast<std::size_t>(ctx.pq[i])] -=
            buf.dx[static_cast<std::size_t>(ctx.n_pv + ctx.n_pq + i)];
    }

    for (int32_t bus = 0; bus < buf.n_bus; ++bus) {
        const double vm = buf.Vm[static_cast<std::size_t>(bus)];
        const double va = buf.Va[static_cast<std::size_t>(bus)];
        buf.V[static_cast<std::size_t>(bus)] =
            std::complex<double>(vm * std::cos(va), vm * std::sin(va));
    }

    buf.has_cached_Ibus = false;
}
