#pragma once

#include "i_backend.hpp"

#include <memory>


// ---------------------------------------------------------------------------
// CpuNewtonSolverBackend: CPU implementation of INewtonSolverBackend.
//
// Internally uses Eigen sparse matrices and Eigen's KLU solver.
// CSRView inputs are converted to Eigen sparse format in analyze/initialize.
//
// Pimpl pattern hides Eigen headers from users of this header —
// keeping compile times down and CUDA/CPU compilation separate.
// ---------------------------------------------------------------------------
class CpuNewtonSolverBackend final : public INewtonSolverBackend {
public:
    CpuNewtonSolverBackend();
    ~CpuNewtonSolverBackend() override;

    // Non-copyable; movable.
    CpuNewtonSolverBackend(const CpuNewtonSolverBackend&) = delete;
    CpuNewtonSolverBackend& operator=(const CpuNewtonSolverBackend&) = delete;
    CpuNewtonSolverBackend(CpuNewtonSolverBackend&&) noexcept;
    CpuNewtonSolverBackend& operator=(CpuNewtonSolverBackend&&) noexcept;

    void analyze(const YbusView&          ybus,
                 const JacobianMaps&       maps,
                 const JacobianStructure&  J,
                 int32_t                   n_bus) override;

    void initialize(const YbusView&             ybus,
                    const std::complex<double>* sbus,
                    const std::complex<double>* V0) override;

    void computeMismatch(const int32_t* pv, int32_t n_pv,
                         const int32_t* pq, int32_t n_pq,
                         double* F, double& normF) override;

    void updateJacobian() override;

    void solveLinearSystem(const double* F, double* dx) override;

    void updateVoltage(const double*  dx,
                       const int32_t* pv, int32_t n_pv,
                       const int32_t* pq, int32_t n_pq) override;

    void downloadV(std::complex<double>* V_out, int32_t n_bus) override;

private:
    // Impl holds all Eigen types — insulates callers from Eigen headers.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
