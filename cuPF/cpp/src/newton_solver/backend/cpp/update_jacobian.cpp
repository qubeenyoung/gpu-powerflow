#include "cpu_backend_impl.hpp"

#include <Eigen/Sparse>

#include <complex>
#include <cmath>


// ---------------------------------------------------------------------------
// updateJacobian
//
// Fills Jacobian values using the pre-computed JacobianMaps from analyze().
//
// Jacobian block structure:
//   J = [ J11  J12 ] = [ dP/dθ    dP/d|V| ]
//       [ J21  J22 ]   [ dQ/dθ    dQ/d|V| ]
//
// For each off-diagonal Ybus entry k at (Y_i, Y_j):
//   va = -j * V[Y_i] * conj(Y[k] * V[Y_j])   — phase angle sensitivity
//   vm =  V[Y_i] * conj(Y[k] * Vnorm[Y_j])   — voltage magnitude sensitivity
//
//   J_val[mapJ11[t]] = Re(va),  J_val[mapJ21[t]] = Im(va)
//   J_val[mapJ12[t]] = Re(vm),  J_val[mapJ22[t]] = Im(vm)
//
// Diagonal contribution of bus i (accumulated from all connected edges):
//   va_diag = j * (V[i] * conj(Ibus[i]))
//   vm_diag = conj(Ibus[i]) * Vnorm[i]
//
// Iteration is in CSR order (row-major over ybus_indptr/ybus_indices).
// JacobianMaps were remapped to CSC positions in analyze(), so writes
// go directly into J.valuePtr() at the correct CSC location.
// ---------------------------------------------------------------------------
void CpuNewtonSolverBackend::updateJacobian()
{
    using cxd = std::complex<double>;
    static constexpr cxd j(0.0, 1.0);

    auto& im = *impl_;
    const auto& maps = im.maps;

    double* J_val = im.J.valuePtr();

    // Clear all values before filling (structure stays fixed)
    std::fill(J_val, J_val + im.J.nonZeros(), 0.0);

    // Reuse the Ibus SpMV from computeMismatch() within the same NR iteration.
    if (!im.has_cached_Ibus) {
        im.Ibus = im.Ybus * im.V;
        im.has_cached_Ibus = true;
    }

    // Match v1 semantics: normalize from the current complex voltage magnitude.
    // im.Vm can temporarily hold signed magnitudes after applying dx to PQ buses.
    Eigen::Matrix<double, Eigen::Dynamic, 1> Vm_clamped = im.V.cwiseAbs().cwiseMax(1e-8);
    Eigen::Matrix<cxd, Eigen::Dynamic, 1>   Vnorm =
        im.V.array() / Vm_clamped.cast<cxd>().array();

    // ------------------------------------------------------------------
    // Off-diagonal entries: iterate Ybus in CSR order to match JacobianMaps.
    // maps.mapJxx[t] is CSC-remapped, so J_val[pos] writes to the right slot.
    // ------------------------------------------------------------------
    int32_t t = 0;
    for (int32_t row = 0; row < im.n_bus; ++row) {
        for (int32_t k = im.ybus_indptr[row]; k < im.ybus_indptr[row + 1]; ++k, ++t) {
            const int32_t Y_i = row;
            const int32_t Y_j = im.ybus_indices[k];
            const cxd     y   = im.ybus_data[k];

            const cxd va = -j * im.V[Y_i] * std::conj(y * im.V[Y_j]);
            const cxd vm =      im.V[Y_i] * std::conj(y * Vnorm[Y_j]);

            const int32_t p11 = maps.mapJ11[t];
            const int32_t p21 = maps.mapJ21[t];
            const int32_t p12 = maps.mapJ12[t];
            const int32_t p22 = maps.mapJ22[t];

            if (p11 >= 0) J_val[p11] = va.real();
            if (p21 >= 0) J_val[p21] = va.imag();
            if (p12 >= 0) J_val[p12] = vm.real();
            if (p22 >= 0) J_val[p22] = vm.imag();
        }
    }

    // ------------------------------------------------------------------
    // Diagonal entries: one contribution per bus
    // ------------------------------------------------------------------
    for (int32_t bus = 0; bus < im.n_bus; ++bus) {
        const cxd va = j * (im.V[bus] * std::conj(im.Ibus[bus]));
        const cxd vm = std::conj(im.Ibus[bus]) * Vnorm[bus];

        const int32_t q11 = maps.diagJ11[bus];
        const int32_t q21 = maps.diagJ21[bus];
        const int32_t q12 = maps.diagJ12[bus];
        const int32_t q22 = maps.diagJ22[bus];

        if (q11 >= 0) J_val[q11] += va.real();
        if (q21 >= 0) J_val[q21] += va.imag();
        if (q12 >= 0) J_val[q12] += vm.real();
        if (q22 >= 0) J_val[q22] += vm.imag();
    }
}


// ---------------------------------------------------------------------------
// updateVoltage
//
// Apply correction dx to internal Va/Vm state, then reconstruct V.
//
// dx layout (dimension = n_pv + 2*n_pq):
//   dx[0       : n_pv]       → Va[pv]  (phase angle correction at PV buses)
//   dx[n_pv    : n_pv+n_pq]  → Va[pq]  (phase angle correction at PQ buses)
//   dx[n_pv+n_pq : dimF]     → Vm[pq]  (voltage magnitude correction at PQ buses)
// ---------------------------------------------------------------------------
void CpuNewtonSolverBackend::updateVoltage(
    const double*  dx,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    auto& im = *impl_;

    // Match v1 semantics: re-decompose the current complex voltage every
    // iteration before applying dx, rather than accumulating Va/Vm state.
    im.Va = im.V.unaryExpr([](const std::complex<double>& v) { return std::arg(v); });
    im.Vm = im.V.cwiseAbs();

    // Apply angle corrections to PV and PQ buses
    for (int32_t i = 0; i < n_pv; ++i) im.Va[pv[i]] += dx[i];
    for (int32_t i = 0; i < n_pq; ++i) im.Va[pq[i]] += dx[n_pv + i];

    // Apply magnitude corrections to PQ buses
    for (int32_t i = 0; i < n_pq; ++i) im.Vm[pq[i]] += dx[n_pv + n_pq + i];

    im.V.real() = (im.Vm.array() * im.Va.array().cos()).matrix();
    im.V.imag() = (im.Vm.array() * im.Va.array().sin()).matrix();

    im.has_cached_Ibus = false;
}


// ---------------------------------------------------------------------------
// downloadV: copy final voltage to caller's buffer.
// ---------------------------------------------------------------------------
void CpuNewtonSolverBackend::downloadV(std::complex<double>* V_out, int32_t n_bus)
{
    for (int32_t i = 0; i < n_bus; ++i) V_out[i] = impl_->V[i];
}
