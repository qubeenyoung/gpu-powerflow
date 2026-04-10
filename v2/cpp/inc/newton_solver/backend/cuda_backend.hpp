#pragma once

#include "i_backend.hpp"

#include <cstdint>
#include <memory>
#include <vector>


// ---------------------------------------------------------------------------
// CudaNewtonSolverBackend: CUDA implementation of INewtonSolverBackend.
//
// GPU pipeline per NR iteration:
//
//   PrecisionMode::Mixed (default, n_batch >= 1):
//     computeMismatch:   cuSPARSE SpMV FP64 complex + mismatch kernel FP64
//     updateJacobian:    FP32 CUDA kernel
//     solveLinearSystem: cuDSS FP32 (Jacobian FP32, RHS FP64→FP32 cast)
//     updateVoltage:     FP64 CUDA kernel (Va, Vm update + V reconstruct)
//
//   PrecisionMode::FP64 (n_batch == 1 only):
//     computeMismatch:   cuSPARSE SpMV FP64 complex + mismatch kernel FP64
//     updateJacobian:    FP64 CUDA kernel
//     solveLinearSystem: cuDSS FP64 (all FP64)
//     updateVoltage:     FP64 CUDA kernel
//
//   PrecisionMode::FP32 (n_batch == 1 only):
//     computeMismatch:   cuSPARSE SpMV FP32 complex + mismatch kernel FP32
//     updateJacobian:    FP32 CUDA kernel
//     solveLinearSystem: cuDSS FP32 (all FP32)
//     updateVoltage:     FP32 CUDA kernel
//
// CUDA headers are NOT included here — they are confined to the .cu/.cpp
// implementation files so this header is safe to include from CPU-only code.
// ---------------------------------------------------------------------------
class CudaNewtonSolverBackend final : public INewtonSolverBackend {
public:
    explicit CudaNewtonSolverBackend(int n_batch = 1,
                                     PrecisionMode precision = PrecisionMode::Mixed);
    ~CudaNewtonSolverBackend() override;

    // Non-copyable; movable.
    CudaNewtonSolverBackend(const CudaNewtonSolverBackend&) = delete;
    CudaNewtonSolverBackend& operator=(const CudaNewtonSolverBackend&) = delete;
    CudaNewtonSolverBackend(CudaNewtonSolverBackend&&) noexcept;
    CudaNewtonSolverBackend& operator=(CudaNewtonSolverBackend&&) noexcept;

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

    void synchronizeForTiming() override;

    // -----------------------------------------------------------------------
    // FP32 single-case interface (PrecisionMode::FP32 only)
    // -----------------------------------------------------------------------
    void analyze_f32(const YbusViewF32&       ybus,
                     const JacobianMaps&      maps,
                     const JacobianStructure& J,
                     int32_t                  n_bus) override;

    void initialize_f32(const YbusViewF32&        ybus,
                         const std::complex<float>* sbus,
                         const std::complex<float>* V0) override;

    void computeMismatch_f32(const int32_t* pv, int32_t n_pv,
                              const int32_t* pq, int32_t n_pq,
                              float* F, float& normF) override;

    void solveLinearSystem_f32(const float* F, float* dx) override;

    void updateVoltage_f32(const float*   dx,
                            const int32_t* pv, int32_t n_pv,
                            const int32_t* pq, int32_t n_pq) override;

    void downloadV_f32(std::complex<float>* V_out, int32_t n_bus) override;

    // -----------------------------------------------------------------------
    // Batch interface (cuDSS UBATCH + SpMM)
    // -----------------------------------------------------------------------
    bool supportsBatch() const override { return n_batch_ > 1; }

    void initialize_batch(const YbusView&             ybus,
                          const std::complex<double>* sbus_batch,
                          const std::complex<double>* V0_batch,
                          int32_t                     n_batch) override;

    void computeMismatch_batch(const int32_t* pv, int32_t n_pv,
                               const int32_t* pq, int32_t n_pq,
                               double* F_batch,
                               double* normF_batch,
                               int32_t n_batch) override;

    void updateJacobian_batch(int32_t n_batch) override;

    void solveLinearSystem_batch(const double* F_batch,
                                 double*       dx_batch,
                                 int32_t       n_batch) override;

    void updateVoltage_batch(const double*  dx_batch,
                             const int32_t* pv, int32_t n_pv,
                             const int32_t* pq, int32_t n_pq,
                             int32_t        n_batch) override;

    void downloadV_batch(std::complex<double>* V_out,
                         int32_t               n_bus,
                         int32_t               n_batch) override;

private:
    // All CUDA/cuDSS/cuSPARSE state is hidden behind Impl.
    // This prevents CUDA headers from leaking into CPU-only translation units.
    struct Impl;
    std::unique_ptr<Impl> impl_;

    int           n_batch_;
    PrecisionMode precision_mode_;
};
