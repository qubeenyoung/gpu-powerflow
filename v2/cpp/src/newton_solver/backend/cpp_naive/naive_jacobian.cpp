#include "naive_cpu_backend_impl.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <complex>
#include <cmath>
#include <vector>


// ---------------------------------------------------------------------------
// computeMismatch
//
// Identical math to the Optimized backend (and to Python's newtonpf):
//   Ibus = Ybus * V
//   mis  = V .* conj(Ibus) - Sbus
//   F    = [Re(mis[pv]), Re(mis[pq]), Im(mis[pq])]
//
// Also caches pv/pq/pvpq for use by updateJacobian(), which is called
// immediately after in the NR loop and has no pv/pq arguments of its own.
// ---------------------------------------------------------------------------
void NaiveCpuNewtonSolverBackend::computeMismatch(
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq,
    double* F, double& normF)
{
    using cxd = std::complex<double>;
    auto& im = *impl_;

    // Cache pv/pq for the upcoming updateJacobian() call.
    im.pv  .assign(pv, pv + n_pv);
    im.pq  .assign(pq, pq + n_pq);
    im.pvpq.resize(n_pv + n_pq);
    std::copy(pv, pv + n_pv, im.pvpq.begin());
    std::copy(pq, pq + n_pq, im.pvpq.begin() + n_pv);

    Eigen::Matrix<cxd, Eigen::Dynamic, 1> Ibus = im.Ybus * im.V;
    Eigen::Matrix<cxd, Eigen::Dynamic, 1> mis  =
        im.V.array() * Ibus.array().conjugate() - im.Sbus.array();

    int32_t k = 0;
    for (int32_t i = 0; i < n_pv; ++i) F[k++] = mis[pv[i]].real();
    for (int32_t i = 0; i < n_pq; ++i) F[k++] = mis[pq[i]].real();
    for (int32_t i = 0; i < n_pq; ++i) F[k++] = mis[pq[i]].imag();

    const int32_t dimF = n_pv + 2 * n_pq;
    normF = *std::max_element(F, F + dimF,
                [](double a, double b) { return std::abs(a) < std::abs(b); });
    normF = std::abs(normF);
}


// ---------------------------------------------------------------------------
// updateJacobian
//
// Mirrors Python's dSbus_dV() + J11/J12/J21/J22 slicing + vstack/hstack.
//
// Python reference (newtonpf.py:72 / dSbus_dV.py):
//
//   Ibus     = Ybus * V
//   diagV    = diag(V)
//   diagIbus = diag(Ibus)
//   diagVnorm= diag(V / |V|)
//
//   dS_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
//   dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV)
//
//   J = [ Re(dS_dVa)[pvpq,pvpq]   Re(dS_dVm)[pvpq,pq] ]
//       [ Im(dS_dVa)[pq,  pvpq]   Im(dS_dVm)[pq,  pq] ]
//
// C++ differences from the Optimized backend:
//   - No pre-computed JacobianMaps; sparsity pattern derived each call.
//   - Builds two full n×n complex sparse matrices (dS_dVa, dS_dVm) as
//     intermediate results, then slices into 4 blocks — same as Python.
//   - J is rebuilt via triplets; no structure reuse.
// ---------------------------------------------------------------------------
void NaiveCpuNewtonSolverBackend::updateJacobian()
{
    using cxd     = std::complex<double>;
    using SpCx    = Eigen::SparseMatrix<cxd, Eigen::ColMajor, int32_t>;
    using TripCx  = Eigen::Triplet<cxd>;
    using TripD   = Eigen::Triplet<double>;
    static constexpr cxd j_unit(0.0, 1.0);

    auto& im = *impl_;
    const int32_t n      = im.n_bus;
    const int32_t n_pvpq = static_cast<int32_t>(im.pvpq.size());
    const int32_t n_pq   = static_cast<int32_t>(im.pq.size());
    const int32_t dimF   = n_pvpq + n_pq;

    // ------------------------------------------------------------------
    // Index maps: bus → position in pvpq or pq list (-1 if absent).
    // Used to map (row_bus, col_bus) → (J_row, J_col) during slicing.
    // ------------------------------------------------------------------
    std::vector<int32_t> bus_to_pvpq(n, -1);
    std::vector<int32_t> bus_to_pq  (n, -1);
    for (int32_t i = 0; i < n_pvpq; ++i) bus_to_pvpq[im.pvpq[i]] = i;
    for (int32_t j = 0; j < n_pq;   ++j) bus_to_pq  [im.pq[j]]   = j;

    // ------------------------------------------------------------------
    // Intermediate quantities
    // ------------------------------------------------------------------
    Eigen::Matrix<cxd, Eigen::Dynamic, 1> Ibus = im.Ybus * im.V;

    // Vnorm[i] = V[i] / |V[i]|  (unit phasor; guard against |V|=0)
    Eigen::Matrix<double, Eigen::Dynamic, 1> Vm_safe = im.Vm.cwiseMax(1e-8);
    Eigen::Matrix<cxd,    Eigen::Dynamic, 1> Vnorm   =
        im.V.array() / Vm_safe.cast<cxd>().array();

    // ------------------------------------------------------------------
    // Build dS_dVm  (same sparsity as Ybus ∪ diagonal)
    //
    //   dS_dVm(i,j) = V[i] * conj(Ybus(i,j) * Vnorm[j])   [off-diag & diag]
    //               + conj(Ibus[i]) * Vnorm[i]              [diagonal only]
    //
    // Construct via triplets; setFromTriplets sums duplicate diagonal entries.
    // ------------------------------------------------------------------
    SpCx dS_dVm(n, n);
    {
        std::vector<TripCx> trips;
        trips.reserve(im.Ybus.nonZeros() + n);

        // Term 1: diagV * conj(Ybus * diagVnorm) — iterate Ybus in CSC order
        for (int32_t col = 0; col < n; ++col) {
            const cxd vn_col = Vnorm[col];
            for (SpCx::InnerIterator it(im.Ybus, col); it; ++it) {
                const int32_t row = static_cast<int32_t>(it.row());
                trips.emplace_back(row, col,
                    im.V[row] * std::conj(it.value() * vn_col));
            }
        }

        // Term 2: conj(diagIbus) * diagVnorm — diagonal only
        for (int32_t i = 0; i < n; ++i) {
            trips.emplace_back(i, i, std::conj(Ibus[i]) * Vnorm[i]);
        }

        dS_dVm.setFromTriplets(trips.begin(), trips.end());
    }

    // ------------------------------------------------------------------
    // Build dS_dVa  (same sparsity as Ybus ∪ diagonal)
    //
    //   dS_dVa = j * diagV * conj(diagIbus - Ybus * diagV)
    //
    //   entry (i,j):
    //     off-diagonal (i≠j): j * V[i] * conj(-Ybus(i,j) * V[j])
    //     diagonal     (i=j): j * V[i] * conj( Ibus[i]   - Ybus(i,i) * V[i])
    //
    // Build (diagIbus - Ybus*diagV) first via triplets, then scale.
    // ------------------------------------------------------------------
    SpCx dS_dVa(n, n);
    {
        std::vector<TripCx> trips;
        trips.reserve(im.Ybus.nonZeros() + n);

        // -(Ybus * diagV): entry (i,j) = -Ybus(i,j)*V[j]
        for (int32_t col = 0; col < n; ++col) {
            const cxd v_col = im.V[col];
            for (SpCx::InnerIterator it(im.Ybus, col); it; ++it) {
                const int32_t row = static_cast<int32_t>(it.row());
                trips.emplace_back(row, col, -it.value() * v_col);
            }
        }

        // +diagIbus: add Ibus[i] to diagonal
        for (int32_t i = 0; i < n; ++i) {
            trips.emplace_back(i, i, Ibus[i]);
        }

        // Accumulate into a temp sparse, then scale rows by j*V[i]
        SpCx tmp(n, n);
        tmp.setFromTriplets(trips.begin(), trips.end());

        // dS_dVa(i,j) = j * V[i] * conj(tmp(i,j))
        std::vector<TripCx> trips2;
        trips2.reserve(tmp.nonZeros());
        for (int32_t col = 0; col < n; ++col) {
            for (SpCx::InnerIterator it(tmp, col); it; ++it) {
                const int32_t row = static_cast<int32_t>(it.row());
                trips2.emplace_back(row, col,
                    j_unit * im.V[row] * std::conj(it.value()));
            }
        }
        dS_dVa.setFromTriplets(trips2.begin(), trips2.end());
    }

    // ------------------------------------------------------------------
    // Slice and assemble J
    //
    // J layout (dimF × dimF):
    //   cols 0..n_pvpq-1    : angle (θ) variables  → from dS_dVa
    //   cols n_pvpq..dimF-1 : magnitude (|V|) vars → from dS_dVm (pq only)
    //
    //   rows 0..n_pvpq-1    : pvpq buses (P equations)
    //   rows n_pvpq..dimF-1 : pq buses   (Q equations)
    //
    //   J11 = Re(dS_dVa)[pvpq, pvpq]  top-left
    //   J12 = Re(dS_dVm)[pvpq, pq  ]  top-right
    //   J21 = Im(dS_dVa)[pq,   pvpq]  bottom-left
    //   J22 = Im(dS_dVm)[pq,   pq  ]  bottom-right
    // ------------------------------------------------------------------
    std::vector<TripD> jtrips;
    jtrips.reserve(dS_dVa.nonZeros() + dS_dVm.nonZeros());

    // From dS_dVa  →  J11 (real) and J21 (imag)
    for (int32_t col = 0; col < n; ++col) {
        const int32_t jcol = bus_to_pvpq[col];  // θ column
        if (jcol < 0) continue;                 // col not in pvpq → skip

        for (SpCx::InnerIterator it(dS_dVa, col); it; ++it) {
            const int32_t bus_i = static_cast<int32_t>(it.row());

            const int32_t irow_pvpq = bus_to_pvpq[bus_i];
            if (irow_pvpq >= 0)  // J11: Re, rows pvpq
                jtrips.emplace_back(irow_pvpq, jcol, it.value().real());

            const int32_t irow_pq = bus_to_pq[bus_i];
            if (irow_pq >= 0)    // J21: Im, rows pq
                jtrips.emplace_back(n_pvpq + irow_pq, jcol, it.value().imag());
        }
    }

    // From dS_dVm  →  J12 (real) and J22 (imag)  [only pq columns matter]
    for (int32_t col = 0; col < n; ++col) {
        const int32_t jcol_pq = bus_to_pq[col];      // |V| column (pq only)
        if (jcol_pq < 0) continue;
        const int32_t jcol = n_pvpq + jcol_pq;

        for (SpCx::InnerIterator it(dS_dVm, col); it; ++it) {
            const int32_t bus_i = static_cast<int32_t>(it.row());

            const int32_t irow_pvpq = bus_to_pvpq[bus_i];
            if (irow_pvpq >= 0)  // J12: Re, rows pvpq
                jtrips.emplace_back(irow_pvpq, jcol, it.value().real());

            const int32_t irow_pq = bus_to_pq[bus_i];
            if (irow_pq >= 0)    // J22: Im, rows pq
                jtrips.emplace_back(n_pvpq + irow_pq, jcol, it.value().imag());
        }
    }

    im.J.resize(dimF, dimF);
    im.J.setFromTriplets(jtrips.begin(), jtrips.end());
    im.J.makeCompressed();
}


// ---------------------------------------------------------------------------
// updateVoltage
//
// Apply dx correction to Va/Vm state, then reconstruct V = Vm * exp(j*Va).
// Same as the Optimized backend — no algorithmic difference here.
// ---------------------------------------------------------------------------
void NaiveCpuNewtonSolverBackend::updateVoltage(
    const double*  dx,
    const int32_t* pv, int32_t n_pv,
    const int32_t* pq, int32_t n_pq)
{
    auto& im = *impl_;

    for (int32_t i = 0; i < n_pv; ++i) im.Va[pv[i]] += dx[i];
    for (int32_t i = 0; i < n_pq; ++i) im.Va[pq[i]] += dx[n_pv + i];
    for (int32_t i = 0; i < n_pq; ++i) im.Vm[pq[i]] += dx[n_pv + n_pq + i];

    for (int32_t bus = 0; bus < im.n_bus; ++bus) {
        im.V[bus] = std::polar(im.Vm[bus], im.Va[bus]);
    }
}
