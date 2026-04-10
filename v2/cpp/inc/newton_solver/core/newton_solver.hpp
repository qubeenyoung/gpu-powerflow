#pragma once

#include "newton_solver_types.hpp"
#include "jacobian_builder.hpp"

#include <memory>
#include <vector>


class INewtonSolverBackend;


// ---------------------------------------------------------------------------
// NewtonSolver: Newton-Raphson power flow solver.
//
// Precision is selected via NewtonOptions::precision at construction time.
// There is no compile-time template parameter.
//
//   CPU backend:
//     PrecisionMode::FP64 only — FP64 throughout (Eigen SparseLU).
//     FP32 / Mixed combinations throw at construction.
//
//   CUDA backend (n_batch == 1):
//     PrecisionMode::FP32  — end-to-end FP32 public API and CUDA pipeline.
//     PrecisionMode::Mixed — FP64 public API, FP32 Jacobian/solve + FP64 voltage.
//     PrecisionMode::FP64  — end-to-end FP64 public API and CUDA pipeline.
//
//   n_batch > 1:
//     currently out of scope for the precision-selection refactor.
//     Solver construction rejects n_batch != 1 for now.
//
// Typical usage (FP64):
//   NewtonSolver solver({.backend = BackendKind::CUDA, .precision = PrecisionMode::FP64});
//   solver.analyze(ybus_f64, pv, n_pv, pq, n_pq);
//   solver.solve(ybus_f64, sbus, V0, pv, n_pv, pq, n_pq, config, result_f64);
//
// Typical usage (FP32):
//   NewtonSolver solver({.backend = BackendKind::CUDA, .precision = PrecisionMode::FP32});
//   solver.analyze(ybus_f32, pv, n_pv, pq, n_pq);
//   solver.solve(ybus_f32, sbus_f, V0_f, pv, n_pv, pq, n_pq, config, result_f32);
//
// analyze() must be called once before any solve(). It builds the Jacobian
// sparsity pattern and mapping tables which are reused every iteration.
// ---------------------------------------------------------------------------
class NewtonSolver {
public:
    explicit NewtonSolver(const NewtonOptions& options = {});
    ~NewtonSolver();

    // -----------------------------------------------------------------------
    // FP64 / Mixed path (PrecisionMode::FP64 or PrecisionMode::Mixed)
    // -----------------------------------------------------------------------

    // Analyze Ybus sparsity. Must be called before the first solve().
    void analyze(const YbusViewF64& ybus,
                 const int32_t* pv, int32_t n_pv,
                 const int32_t* pq, int32_t n_pq);

    // Single solve. Result V is complex<double>.
    void solve(const YbusViewF64&          ybus,
               const std::complex<double>* sbus,
               const std::complex<double>* V0,
               const int32_t*              pv, int32_t n_pv,
               const int32_t*              pq, int32_t n_pq,
               const NRConfig&             config,
               NRResultF64&                result);

    // -----------------------------------------------------------------------
    // FP32 path (PrecisionMode::FP32, CUDA + n_batch==1 only)
    // -----------------------------------------------------------------------

    // Analyze with FP32 Ybus. JacobianBuilder uses only structure (not values).
    void analyze(const YbusViewF32& ybus,
                 const int32_t* pv, int32_t n_pv,
                 const int32_t* pq, int32_t n_pq);

    // Single solve. Result V is complex<float>.
    void solve(const YbusViewF32&         ybus,
               const std::complex<float>* sbus,
               const std::complex<float>* V0,
               const int32_t*             pv, int32_t n_pv,
               const int32_t*             pq, int32_t n_pq,
               const NRConfig&            config,
               NRResultF32&               result);

    // Batch solve is temporarily disabled while precision-selection is being
    // refactored. The implementation throws if called.
    void solve_batch(const YbusView&             ybus,
                     const std::complex<double>* sbus_batch,   // [n_batch * n_bus]
                     const std::complex<double>* V0_batch,     // [n_batch * n_bus]
                     const int32_t*              pv, int32_t n_pv,
                     const int32_t*              pq, int32_t n_pq,
                     int32_t                     n_batch,
                     const NRConfig&             config,
                     NRResult*                   results);     // [n_batch]

private:
    NewtonOptions    options_;
    JacobianBuilder  jac_builder_;
    JacobianMaps     jac_maps_;    // populated by analyze(), reused by solve()
    JacobianStructure J_;          // Jacobian CSR sparsity structure

    std::unique_ptr<INewtonSolverBackend> backend_;

    bool analyzed_ = false;
};
