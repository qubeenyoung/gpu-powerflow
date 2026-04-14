#pragma once

#include "newton_solver/ops/op_interfaces.hpp"
#include "newton_solver/core/contexts.hpp"

#include <Eigen/KLUSupport>
#include <Eigen/Sparse>

#include <vector>
#include <complex>

using CpuYbusMatrixF64 = Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor, int32_t>;
using CpuJacobianMatrixF64 = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>;


// ---------------------------------------------------------------------------
// CpuFp64Storage: host-side buffers + KLU/Eigen handles for FP64 CPU path.
// ---------------------------------------------------------------------------
class CpuFp64Storage final : public IStorage {
public:
    CpuFp64Storage()  = default;
    ~CpuFp64Storage() = default;

    BackendKind   backend() const override { return BackendKind::CPU; }
    ComputePolicy compute()  const override { return ComputePolicy::FP64; }

    void prepare(const AnalyzeContext& ctx) override;
    void upload(const SolveContext&   ctx) override;
    void download_result(NRResultF64& result) const override;

    // -----------------------------------------------------------------------
    // Buffers exposed to CPU Ops (package-private convention: accessed by
    // ops in the same directory, not through IStorage interface).
    // -----------------------------------------------------------------------

    // Host-side sparse matrices / solver state
    CpuYbusMatrixF64           Ybus;
    CpuJacobianMatrixF64       J;
    Eigen::KLU<CpuJacobianMatrixF64> lu;
    bool                       has_klu_symbolic = false;

    // Jacobian maps / structure
    JacobianMaps      maps;
    JacobianStructure J_pattern;

    // Mismatch / solve vectors (host)
    std::vector<double> F;      // mismatch [dimF]
    std::vector<double> dx;     // solution  [dimF]

    // Voltage state (host)
    std::vector<double>               Va;   // [n_bus]
    std::vector<double>               Vm;   // [n_bus]
    std::vector<std::complex<double>> V;    // [n_bus]

    // Ybus copy (host, FP64)
    std::vector<int32_t>              Ybus_indptr;
    std::vector<int32_t>              Ybus_indices;
    std::vector<std::complex<double>> Ybus_data;
    std::vector<std::complex<double>> Ibus;
    bool                              has_cached_Ibus = false;

    // Sbus (host, FP64)
    std::vector<std::complex<double>> Sbus;

    // Topology (populated by prepare())
    int32_t n_bus  = 0;
    int32_t n_pvpq = 0;
    int32_t n_pq   = 0;
    int32_t dimF   = 0;
};
