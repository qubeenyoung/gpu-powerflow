#include "cpu_backend_impl.hpp"
#include "utils/timer.hpp"

#include <complex>
#include <vector>


// ---------------------------------------------------------------------------
// Constructor / Destructor / Move
// ---------------------------------------------------------------------------
CpuNewtonSolverBackend::CpuNewtonSolverBackend()
    : impl_(std::make_unique<Impl>()) {}

CpuNewtonSolverBackend::~CpuNewtonSolverBackend() = default;

CpuNewtonSolverBackend::CpuNewtonSolverBackend(CpuNewtonSolverBackend&&) noexcept = default;
CpuNewtonSolverBackend& CpuNewtonSolverBackend::operator=(CpuNewtonSolverBackend&&) noexcept = default;


// ---------------------------------------------------------------------------
// analyze: build Eigen CSC Jacobian from JacobianStructure (CSR), remap
//          JacobianMaps positions from CSR to CSC, and perform KLU
//          symbolic factorization.
//
// JacobianBuilder produces positions in CSR order. KLU requires CSC.
// We convert once here so updateJacobian can write directly into J.valuePtr()
// without any per-iteration remapping.
// ---------------------------------------------------------------------------
void CpuNewtonSolverBackend::analyze(
    const YbusView&          ybus,
    const JacobianMaps&       maps,
    const JacobianStructure&  J_csr,
    int32_t                   n_bus)
{
    impl_->n_bus = n_bus;

    // ------------------------------------------------------------------
    // 1. Store Ybus CSR arrays for CSR-order iteration in updateJacobian.
    //    (The Eigen CSC Ybus is built in initialize() for SpMV.)
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CPU.analyze.storeYbus");
        impl_->ybus_indptr.assign(ybus.indptr, ybus.indptr + ybus.rows + 1);
        impl_->ybus_indices.assign(ybus.indices, ybus.indices + ybus.nnz);
        impl_->ybus_data.assign(ybus.data, ybus.data + ybus.nnz);
    }

    // ------------------------------------------------------------------
    // 2. Build Eigen CSC Jacobian from JacobianStructure (CSR).
    //    Use triplets so Eigen handles ordering.
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CPU.analyze.buildJacobianCSC");
        using Triplet = Eigen::Triplet<double>;
        std::vector<Triplet> trips;
        trips.reserve(J_csr.nnz);

        for (int32_t row = 0; row < J_csr.dim; ++row) {
            for (int32_t k = J_csr.row_ptr[row]; k < J_csr.row_ptr[row + 1]; ++k) {
                trips.emplace_back(row, J_csr.col_idx[k], 1.0);
            }
        }

        impl_->J.resize(J_csr.dim, J_csr.dim);
        impl_->J.setFromTriplets(trips.begin(), trips.end());
        impl_->J.makeCompressed();
    }

    // ------------------------------------------------------------------
    // 3. Build CSR→CSC position mapping for J.
    //
    //    JacobianMaps from the builder index into J's CSR value array.
    //    KLU (and updateJacobian) need CSC positions (J.valuePtr()).
    //
    //    We build: csr_to_csc[csr_k] = csc_k
    //    by inverting the CSC→CSR permutation that Eigen's storage implies.
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CPU.analyze.remapJacobianMaps");
        const int32_t  j_nnz  = impl_->J.nonZeros();
        const int32_t* csc_cp = impl_->J.outerIndexPtr();   // column pointers
        const int32_t* csc_ri = impl_->J.innerIndexPtr();   // row indices

        // Build CSR row_ptr from CSC structure
        std::vector<int32_t> csr_rp(J_csr.dim + 1, 0);
        for (int32_t k = 0; k < j_nnz; ++k) csr_rp[csc_ri[k] + 1]++;
        for (int32_t r = 0; r < J_csr.dim; ++r) csr_rp[r + 1] += csr_rp[r];

        // For each CSC entry k: find its CSR position csr_pos, then set csr_to_csc[csr_pos] = k
        std::vector<int32_t> csr_to_csc(j_nnz);
        std::vector<int32_t> row_cursor(csr_rp.begin(), csr_rp.end());

        for (int32_t col = 0; col < J_csr.dim; ++col) {
            for (int32_t k = csc_cp[col]; k < csc_cp[col + 1]; ++k) {
                int32_t row     = csc_ri[k];
                int32_t csr_pos = row_cursor[row]++;
                csr_to_csc[csr_pos] = k;
            }
        }

        // ------------------------------------------------------------------
        // 4. Remap all JacobianMaps entries from CSR positions to CSC positions.
        //    After this, maps.mapJ11[t] gives a CSC index into J.valuePtr().
        // ------------------------------------------------------------------
        auto remap = [&csr_to_csc](int32_t csr_pos) -> int32_t {
            return (csr_pos >= 0) ? csr_to_csc[csr_pos] : -1;
        };

        impl_->maps = maps;

        for (auto& p : impl_->maps.mapJ11)  p = remap(p);
        for (auto& p : impl_->maps.mapJ12)  p = remap(p);
        for (auto& p : impl_->maps.mapJ21)  p = remap(p);
        for (auto& p : impl_->maps.mapJ22)  p = remap(p);
        for (auto& p : impl_->maps.diagJ11) p = remap(p);
        for (auto& p : impl_->maps.diagJ12) p = remap(p);
        for (auto& p : impl_->maps.diagJ21) p = remap(p);
        for (auto& p : impl_->maps.diagJ22) p = remap(p);
    }

    // ------------------------------------------------------------------
    // 5. KLU symbolic factorization — determines ordering once.
    // ------------------------------------------------------------------
    {
        newton_solver::utils::ScopedTimer timer("CPU.analyze.kluSymbolic");
        impl_->lu.analyzePattern(impl_->J);
    }
}


// ---------------------------------------------------------------------------
// initialize: convert CSRView Ybus to Eigen CSC and copy V0, Sbus.
//
// Called once per solve (not once per iteration) because V0 and Sbus change
// between cases in a batch.
// ---------------------------------------------------------------------------
void CpuNewtonSolverBackend::initialize(
    const YbusView&             ybus,
    const std::complex<double>* sbus,
    const std::complex<double>* V0)
{
    using cxd    = std::complex<double>;
    const int32_t n = impl_->n_bus;

    // Build Eigen CSC from the CSRView (row-major input).
    // Eigen ColMajor needs CSC — build via COO triplets.
    using Triplet = Eigen::Triplet<cxd>;
    std::vector<Triplet> trips;
    trips.reserve(ybus.nnz);

    for (int32_t row = 0; row < ybus.rows; ++row) {
        for (int32_t k = ybus.indptr[row]; k < ybus.indptr[row + 1]; ++k) {
            trips.emplace_back(row, ybus.indices[k], ybus.data[k]);
        }
    }

    impl_->Ybus.resize(ybus.rows, ybus.cols);
    impl_->Ybus.setFromTriplets(trips.begin(), trips.end());
    impl_->Ybus.makeCompressed();

    // Copy V0 and Sbus into Eigen column vectors.
    impl_->V    = Eigen::Map<const Eigen::Matrix<cxd, Eigen::Dynamic, 1>>(V0,   n);
    impl_->Sbus = Eigen::Map<const Eigen::Matrix<cxd, Eigen::Dynamic, 1>>(sbus, n);
    impl_->Ibus.resize(n);
    impl_->has_cached_Ibus = false;

    // Decompose initial voltage into separate magnitude and angle vectors.
    // These are updated in-place by updateVoltage() each NR iteration.
    impl_->Vm = impl_->V.cwiseAbs();
    impl_->Va = impl_->V.unaryExpr([](const cxd& v) { return std::arg(v); });
}
