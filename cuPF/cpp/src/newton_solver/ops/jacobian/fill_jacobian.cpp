#include "fill_jacobian.hpp"

#include "newton_solver/core/solver_contexts.hpp"
#include "newton_solver/ops/ibus/compute_ibus.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>


namespace {
static constexpr std::complex<double> kImaginaryUnit(0.0, 1.0);
}


void CpuJacobianOpF64::run(CpuFp64Buffers& buf, IterationContext& ctx)
{
    (void)ctx;

    if (buf.n_bus <= 0 || buf.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuJacobianOpF64::run: buffers are not prepared");
    }

    double* J_values = buf.J.valuePtr();
    std::fill(J_values, J_values + buf.J.nonZeros(), 0.0);

    if (!buf.has_cached_Ibus) {
        compute_ibus(buf);
    }

    // Vm_clamped[i] = max(|V[i]|, 1e-8); Vnorm[i] = V[i] / Vm_clamped[i].
    std::vector<std::complex<double>> Vnorm(static_cast<std::size_t>(buf.n_bus));
    for (int32_t i = 0; i < buf.n_bus; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const double vm = std::max(std::abs(buf.V[idx]), 1e-8);
        Vnorm[idx] = buf.V[idx] / vm;
    }

    // 오프 대각 기여
    int32_t t = 0;
    for (int32_t row = 0; row < buf.n_bus; ++row) {
        for (int32_t k = buf.Ybus_indptr[static_cast<std::size_t>(row)];
             k < buf.Ybus_indptr[static_cast<std::size_t>(row + 1)];
             ++k, ++t) {
            const int32_t y_i = row;
            const int32_t y_j = buf.Ybus_indices[static_cast<std::size_t>(k)];
            const std::complex<double> y = buf.Ybus_data[static_cast<std::size_t>(k)];

            const std::complex<double> va =
                -kImaginaryUnit * buf.V[static_cast<std::size_t>(y_i)] *
                std::conj(y * buf.V[static_cast<std::size_t>(y_j)]);

            const std::complex<double> vm =
                buf.V[static_cast<std::size_t>(y_i)] *
                std::conj(y * Vnorm[static_cast<std::size_t>(y_j)]);

            const int32_t p11 = buf.maps.mapJ11[static_cast<std::size_t>(t)];
            const int32_t p21 = buf.maps.mapJ21[static_cast<std::size_t>(t)];
            const int32_t p12 = buf.maps.mapJ12[static_cast<std::size_t>(t)];
            const int32_t p22 = buf.maps.mapJ22[static_cast<std::size_t>(t)];

            if (p11 >= 0) J_values[p11] = va.real();
            if (p21 >= 0) J_values[p21] = va.imag();
            if (p12 >= 0) J_values[p12] = vm.real();
            if (p22 >= 0) J_values[p22] = vm.imag();
        }
    }

    // 대각 기여
    for (int32_t bus = 0; bus < buf.n_bus; ++bus) {
        const std::complex<double> va =
            kImaginaryUnit *
            (buf.V[static_cast<std::size_t>(bus)] *
             std::conj(buf.Ibus[static_cast<std::size_t>(bus)]));

        const std::complex<double> vm =
            std::conj(buf.Ibus[static_cast<std::size_t>(bus)]) *
            Vnorm[static_cast<std::size_t>(bus)];

        const int32_t q11 = buf.maps.diagJ11[static_cast<std::size_t>(bus)];
        const int32_t q21 = buf.maps.diagJ21[static_cast<std::size_t>(bus)];
        const int32_t q12 = buf.maps.diagJ12[static_cast<std::size_t>(bus)];
        const int32_t q22 = buf.maps.diagJ22[static_cast<std::size_t>(bus)];

        if (q11 >= 0) J_values[q11] += va.real();
        if (q21 >= 0) J_values[q21] += va.imag();
        if (q12 >= 0) J_values[q12] += vm.real();
        if (q22 >= 0) J_values[q22] += vm.imag();
    }
}
