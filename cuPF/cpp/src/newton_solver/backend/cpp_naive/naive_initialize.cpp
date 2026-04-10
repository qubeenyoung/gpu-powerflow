#include "naive_cpu_backend_impl.hpp"

#include <complex>
#include <vector>


// ---------------------------------------------------------------------------
// Constructor / Destructor / Move
// ---------------------------------------------------------------------------
NaiveCpuNewtonSolverBackend::NaiveCpuNewtonSolverBackend()
    : impl_(std::make_unique<Impl>()) {}

NaiveCpuNewtonSolverBackend::~NaiveCpuNewtonSolverBackend() = default;

NaiveCpuNewtonSolverBackend::NaiveCpuNewtonSolverBackend(NaiveCpuNewtonSolverBackend&&) noexcept = default;
NaiveCpuNewtonSolverBackend& NaiveCpuNewtonSolverBackend::operator=(NaiveCpuNewtonSolverBackend&&) noexcept = default;


// ---------------------------------------------------------------------------
// analyze: store Ybus as Eigen CSC. No map computation, no symbolic factor.
//
// Python's newtonpf() has no equivalent of this phase. Jacobian sparsity is
// derived implicitly every iteration inside dSbus_dV(). We only build the
// Eigen CSC Ybus here so SpMV (Ybus*V) is available in computeMismatch and
// updateJacobian without rebuilding each call.
//
// JacobianMaps and JacobianStructure arguments are intentionally ignored.
// ---------------------------------------------------------------------------
void NaiveCpuNewtonSolverBackend::analyze(
    const YbusView&          ybus,
    const JacobianMaps&      /*maps*/,
    const JacobianStructure& /*J*/,
    int32_t                   n_bus)
{
    using cxd    = std::complex<double>;
    using Triplet = Eigen::Triplet<cxd>;

    impl_->n_bus = n_bus;

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
}


// ---------------------------------------------------------------------------
// initialize: upload V0 and Sbus for a new solve.
//
// Called once per solve() invocation; V0 and Sbus change between cases.
// Decomposes V0 into magnitude (Vm) and angle (Va) vectors — these are
// updated in-place by updateVoltage() each NR iteration.
// ---------------------------------------------------------------------------
void NaiveCpuNewtonSolverBackend::initialize(
    const YbusView&             /*ybus*/,
    const std::complex<double>* sbus,
    const std::complex<double>* V0)
{
    using cxd = std::complex<double>;
    const int32_t n = impl_->n_bus;

    impl_->V    = Eigen::Map<const Eigen::Matrix<cxd, Eigen::Dynamic, 1>>(V0,   n);
    impl_->Sbus = Eigen::Map<const Eigen::Matrix<cxd, Eigen::Dynamic, 1>>(sbus, n);

    impl_->Vm = impl_->V.cwiseAbs();
    impl_->Va = impl_->V.unaryExpr([](const cxd& v) { return std::arg(v); });
}


// ---------------------------------------------------------------------------
// downloadV: copy final voltage to caller's buffer.
// ---------------------------------------------------------------------------
void NaiveCpuNewtonSolverBackend::downloadV(std::complex<double>* V_out, int32_t n_bus)
{
    for (int32_t i = 0; i < n_bus; ++i) V_out[i] = impl_->V[i];
}
