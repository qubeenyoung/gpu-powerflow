#include "cpu_backend_impl.hpp"

#include <Eigen/Sparse>

#include <cmath>
#include <algorithm>
#include <complex>


// ---------------------------------------------------------------------------
// computeMismatch
//
// Power flow mismatch at current voltage V:
//   Ibus = Ybus * V        (sparse matrix-vector product)
//   mis  = V .* conj(Ibus) - Sbus
//
// F layout (dimension = n_pv + 2*n_pq):
//   F[0   : n_pv]         = Re(mis[pv])   — active power mismatch at PV buses
//   F[n_pv : n_pv+n_pq]   = Re(mis[pq])   — active power mismatch at PQ buses
//   F[n_pv+n_pq : dimF]   = Im(mis[pq])   — reactive power mismatch at PQ buses
//
// normF = max|F_i|  (infinity norm, used as convergence criterion)
// ---------------------------------------------------------------------------
void CpuNewtonSolverBackend::computeMismatch(
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq,
    double* F, double& normF)
{
    using cxd = std::complex<double>;

    auto& im = *impl_;

    // Ibus = Ybus * V  (Eigen sparse-dense product)
    // Cache it so updateJacobian() can reuse the same SpMV result.
    im.Ibus = im.Ybus * im.V;
    im.has_cached_Ibus = true;

    // mis[i] = V[i] * conj(Ibus[i]) - Sbus[i]
    Eigen::Matrix<cxd, Eigen::Dynamic, 1> mis =
        im.V.array() * im.Ibus.array().conjugate() - im.Sbus.array();

    // Pack F: [Re(pv), Re(pq), Im(pq)]
    int32_t k = 0;
    for (int32_t i = 0; i < n_pv; ++i) F[k++] = mis[pv[i]].real();
    for (int32_t i = 0; i < n_pq; ++i) F[k++] = mis[pq[i]].real();
    for (int32_t i = 0; i < n_pq; ++i) F[k++] = mis[pq[i]].imag();

    // normF = max|F_i|
    const int32_t dimF = n_pv + 2 * n_pq;
    normF = *std::max_element(F, F + dimF, [](double a, double b) {
        return std::abs(a) < std::abs(b);
    });
    normF = std::abs(normF);
}
