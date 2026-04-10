#pragma once

#include "i_backend.hpp"

#include <memory>


// ---------------------------------------------------------------------------
// NaiveCpuNewtonSolverBackend: PyPower-equivalent CPU backend.
//
// Algorithmically mirrors Python PyPower's newtonpf():
//   - No pre-computed JacobianMaps or symbolic factorization.
//   - updateJacobian(): dSbus_dV sparse-matrix formulation, full J rebuilt
//     from scratch every NR iteration (same math as dSbus_dV() + vstack/hstack).
//   - solveLinearSystem(): SparseLU one-shot (analyzePattern + factorize on
//     every call), equivalent to scipy.sparse.linalg.spsolve().
//
// Purpose: isolate the pure language (Python → C++) speedup by holding the
// algorithm constant. Compare against CpuNewtonSolverBackend (Optimized) to
// measure the additional gain from the algorithmic improvements in cuPF.
//
// Enable at build time with -DBUILD_NAIVE_CPU=ON.
// ---------------------------------------------------------------------------
class NaiveCpuNewtonSolverBackend final : public INewtonSolverBackend {
public:
    NaiveCpuNewtonSolverBackend();
    ~NaiveCpuNewtonSolverBackend() override;

    NaiveCpuNewtonSolverBackend(const NaiveCpuNewtonSolverBackend&) = delete;
    NaiveCpuNewtonSolverBackend& operator=(const NaiveCpuNewtonSolverBackend&) = delete;
    NaiveCpuNewtonSolverBackend(NaiveCpuNewtonSolverBackend&&) noexcept;
    NaiveCpuNewtonSolverBackend& operator=(NaiveCpuNewtonSolverBackend&&) noexcept;

    // analyze: stores Ybus as Eigen CSC for SpMV. No map/symbolic computation.
    void analyze(const YbusView&          ybus,
                 const JacobianMaps&       maps,   // ignored
                 const JacobianStructure&  J,      // ignored
                 int32_t                   n_bus) override;

    void initialize(const YbusView&             ybus,
                    const std::complex<double>* sbus,
                    const std::complex<double>* V0) override;

    // Stores pv/pq indices for use in updateJacobian().
    void computeMismatch(const int32_t* pv, int32_t n_pv,
                         const int32_t* pq, int32_t n_pq,
                         double* F, double& normF) override;

    // Rebuilds J from scratch using dSbus_dV sparse-matrix formulation.
    void updateJacobian() override;

    // One-shot SparseLU: analyzePattern + factorize + solve every call.
    void solveLinearSystem(const double* F, double* dx) override;

    void updateVoltage(const double*  dx,
                       const int32_t* pv, int32_t n_pv,
                       const int32_t* pq, int32_t n_pq) override;

    void downloadV(std::complex<double>* V_out, int32_t n_bus) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
