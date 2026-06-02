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


// Assemble the CPU FP64 power-flow Jacobian values in place. The four real
// sub-blocks (dP/dVa, dQ/dVa, dP/dVm, dQ/dVm) come from the real/imaginary
// parts of two complex sensitivities per Ybus entry: an angle term (d/dVa,
// factor i*V) and a magnitude term (d/dVm, via the normalized voltage Vnorm).
// Scatter maps precomputed in jacobian_analysis place each contribution into
// the right CSR slot. Off-diagonal entries are assignments; the diagonal adds
// the self-term involving the bus current injection Ibus.
void CpuJacobianOpF64::run(CpuFp64Storage& buf, IterationContext& ctx)
{
    (void)ctx;

    if (buf.n_bus <= 0 || buf.J.nonZeros() <= 0) {
        throw std::runtime_error("CpuJacobianOpF64::run: buffers are not prepared");
    }

    // Reset values; recompute Ibus if the cached injection is stale.
    double* J_values = buf.J.valuePtr();
    std::fill(J_values, J_values + buf.J.nonZeros(), 0.0);

    if (!buf.has_cached_Ibus) {
        compute_ibus(buf);
    }

    // Vm_clamped[i] = max(|V[i]|, 1e-8); Vnorm[i] = V[i] / Vm_clamped[i].
    std::vector<std::complex<double>> Vnorm(buf.n_bus);
    for (int32_t i = 0; i < buf.n_bus; ++i) {
        const std::size_t idx = i;
        const double vm = std::max(std::abs(buf.V[idx]), 1e-8);
        Vnorm[idx] = buf.V[idx] / vm;
    }

    // 오프 대각 기여
    int32_t t = 0;
    for (int32_t row = 0; row < buf.n_bus; ++row) {
        for (int32_t k = buf.Ybus_indptr[row];
             k < buf.Ybus_indptr[row + 1];
             ++k, ++t) {
            const int32_t y_i = row;
            const int32_t y_j = buf.Ybus_indices[k];
            const std::complex<double> y = buf.Ybus_data[k];

            const std::complex<double> va = -kImaginaryUnit * buf.V[y_i] * std::conj(y * buf.V[y_j]);

            const std::complex<double> vm = buf.V[y_i] * std::conj(y * Vnorm[y_j]);

            const int32_t p11 = buf.maps.mapJ11[t];
            const int32_t p21 = buf.maps.mapJ21[t];
            const int32_t p12 = buf.maps.mapJ12[t];
            const int32_t p22 = buf.maps.mapJ22[t];

            if (p11 >= 0) J_values[p11] = va.real();
            if (p21 >= 0) J_values[p21] = va.imag();
            if (p12 >= 0) J_values[p12] = vm.real();
            if (p22 >= 0) J_values[p22] = vm.imag();
        }
    }

    // 대각 기여
    for (int32_t bus = 0; bus < buf.n_bus; ++bus) {
        const std::complex<double> va = kImaginaryUnit * (buf.V[bus] * std::conj(buf.Ibus[bus]));

        const std::complex<double> vm = std::conj(buf.Ibus[bus]) * Vnorm[bus];

        const int32_t q11 = buf.maps.diagJ11[bus];
        const int32_t q21 = buf.maps.diagJ21[bus];
        const int32_t q12 = buf.maps.diagJ12[bus];
        const int32_t q22 = buf.maps.diagJ22[bus];

        if (q11 >= 0) J_values[q11] += va.real();
        if (q21 >= 0) J_values[q21] += va.imag();
        if (q12 >= 0) J_values[q12] += vm.real();
        if (q22 >= 0) J_values[q22] += vm.imag();
    }
}
