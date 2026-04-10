#pragma once

#include "newton_solver/core/newton_solver_types.hpp"
#include "newton_solver/core/jacobian_types.hpp"

#include <complex>
#include <cstdint>


// ---------------------------------------------------------------------------
// INewtonSolverBackend: pure interface that every backend must implement.
//
// The NR loop in NewtonSolver calls these methods in order each iteration:
//
//   analyze()          — called once before the first solve
//   initialize()       — called once per solve (uploads V0, Sbus)
//   computeMismatch()  — computes F and ||F||_inf
//   updateJacobian()   — fills J.values using JacobianMaps
//   solveLinearSystem()— solves J·dx = -F
//   updateVoltage()    — applies dx to V (Va, Vm)
//   downloadV()        — copies final V to caller's buffer
//
// Batch extension (optional — CUDA backend only):
//   supportsBatch()         — returns true if native batch NR is available
//   initialize_batch()      — upload V0/Sbus for all n_batch cases
//   computeMismatch_batch() — F and normF for each case simultaneously
//   updateJacobian_batch()  — fill J values for all cases simultaneously
//   solveLinearSystem_batch()— cuDSS UBATCH: solve n_batch systems at once
//   updateVoltage_batch()   — apply dx corrections for all cases
//   downloadV_batch()       — copy final voltages for all cases
// ---------------------------------------------------------------------------
class INewtonSolverBackend {
public:
    virtual ~INewtonSolverBackend() = default;

    // -----------------------------------------------------------------------
    // Single-case interface (required)
    // -----------------------------------------------------------------------

    virtual void analyze(const YbusView&          ybus,
                         const JacobianMaps&       maps,
                         const JacobianStructure&  J,
                         int32_t                   n_bus) = 0;

    virtual void initialize(const YbusView&             ybus,
                            const std::complex<double>* sbus,
                            const std::complex<double>* V0) = 0;

    // F layout: [Re(mis[pv]), Re(mis[pq]), Im(mis[pq])],  normF = max|F_i|
    virtual void computeMismatch(const int32_t* pv, int32_t n_pv,
                                 const int32_t* pq, int32_t n_pq,
                                 double* F, double& normF) = 0;

    virtual void updateJacobian() = 0;

    virtual void solveLinearSystem(const double* F, double* dx) = 0;

    // dx layout: [Va[pv], Va[pq], Vm[pq]]
    virtual void updateVoltage(const double*  dx,
                               const int32_t* pv, int32_t n_pv,
                               const int32_t* pq, int32_t n_pq) = 0;

    virtual void downloadV(std::complex<double>* V_out, int32_t n_bus) = 0;

    // Optional hook for accurate wall-clock timing at the NewtonSolver layer.
    // CPU backends can keep the default no-op; CUDA overrides this to flush
    // outstanding asynchronous work before a timer scope ends.
    virtual void synchronizeForTiming() {}

    // -----------------------------------------------------------------------
    // FP32 single-case interface (optional — CUDA FP32 mode only)
    //
    // Default implementations are no-ops. CPU backend does not override these.
    // NewtonSolver calls _f32 variants when PrecisionMode::FP32 is active.
    // updateJacobian() is shared — it dispatches internally on precision_mode.
    // -----------------------------------------------------------------------

    virtual void analyze_f32(const YbusViewF32&       ybus,
                              const JacobianMaps&      maps,
                              const JacobianStructure& J,
                              int32_t                  n_bus) {}

    virtual void initialize_f32(const YbusViewF32&        ybus,
                                 const std::complex<float>* sbus,
                                 const std::complex<float>* V0) {}

    // F layout: [Re(mis[pv]), Re(mis[pq]), Im(mis[pq])],  normF = max|F_i|
    virtual void computeMismatch_f32(const int32_t* pv, int32_t n_pv,
                                      const int32_t* pq, int32_t n_pq,
                                      float* F, float& normF) {}

    // Host F/dx are ignored by CUDA (GPU-resident); called for interface symmetry.
    virtual void solveLinearSystem_f32(const float* F, float* dx) {}

    // dx layout: [Va[pv], Va[pq], Vm[pq]]
    virtual void updateVoltage_f32(const float*   dx,
                                    const int32_t* pv, int32_t n_pv,
                                    const int32_t* pq, int32_t n_pq) {}

    virtual void downloadV_f32(std::complex<float>* V_out, int32_t n_bus) {}

    // -----------------------------------------------------------------------
    // Batch interface (optional — default: not supported)
    // -----------------------------------------------------------------------

    virtual bool supportsBatch() const { return false; }

    // Upload V0_batch [n_batch × n_bus] and sbus_batch [n_batch × n_bus]
    virtual void initialize_batch(const YbusView&             ybus,
                                  const std::complex<double>* sbus_batch,
                                  const std::complex<double>* V0_batch,
                                  int32_t                     n_batch) {}

    // F_batch [n_batch × dimF], normF_batch [n_batch]
    virtual void computeMismatch_batch(const int32_t* pv, int32_t n_pv,
                                       const int32_t* pq, int32_t n_pq,
                                       double* F_batch,
                                       double* normF_batch,
                                       int32_t n_batch) {}

    virtual void updateJacobian_batch(int32_t n_batch) {}

    // F_batch [n_batch × dimF] in, dx_batch [n_batch × dimF] out
    virtual void solveLinearSystem_batch(const double* F_batch,
                                         double*       dx_batch,
                                         int32_t       n_batch) {}

    virtual void updateVoltage_batch(const double*  dx_batch,
                                     const int32_t* pv, int32_t n_pv,
                                     const int32_t* pq, int32_t n_pq,
                                     int32_t        n_batch) {}

    // V_out [n_batch × n_bus]
    virtual void downloadV_batch(std::complex<double>* V_out,
                                 int32_t               n_bus,
                                 int32_t               n_batch) {}
};
