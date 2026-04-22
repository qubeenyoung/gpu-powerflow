#pragma once

#include "newton_solver_types.hpp"

#include <memory>


// Opaque pipeline state — defined in src/newton_solver/core/pipeline.hpp
struct SolverPipeline;
struct IterationContext;
struct InitializeContext;


// ---------------------------------------------------------------------------
// NewtonSolver: Newton-Raphson 전력조류 solver.
//
// 역할:
//   1. NewtonOptions를 받아 backend별 pipeline을 조립한다.
//   2. initialize / solve / 반복 루프 생명주기를 오케스트레이션한다.
//   3. backend·compute policy별 구체 로직은 Pipeline이 위임한다.
//
// 사용 예 (CPU FP64):
//   NewtonSolver solver;  // 기본값: CPU, FP64
//   solver.initialize(ybus, pv, n_pv, pq, n_pq);
//   solver.solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, result);
//
// 사용 예 (CUDA Mixed):
//   NewtonSolver solver({.backend = BackendKind::CUDA,
//                        .compute = ComputePolicy::Mixed});
//   solver.initialize(ybus, pv, n_pv, pq, n_pq);
//   solver.solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, result);
//
// public I/O는 항상 FP64다.
// ---------------------------------------------------------------------------
class NewtonSolver {
public:
    explicit NewtonSolver(const NewtonOptions& options = {});
    ~NewtonSolver();

    void initialize(const YbusView& ybus,
                    const int32_t* pv, int32_t n_pv,
                    const int32_t* pq, int32_t n_pq);

    void solve(const YbusView&          ybus,
               const std::complex<double>* sbus,
               const std::complex<double>* V0,
               const int32_t*              pv, int32_t n_pv,
               const int32_t*              pq, int32_t n_pq,
               const NRConfig&             config,
               NRResult&                result);

    void solve_batch(const YbusView&          ybus,
                     const std::complex<double>* sbus,
                     int64_t                     sbus_stride,
                     const std::complex<double>* V0,
                     int64_t                     V0_stride,
                     int32_t                     batch_size,
                     const int32_t*              pv, int32_t n_pv,
                     const int32_t*              pq, int32_t n_pq,
                     const NRConfig&             config,
                     NRBatchResult&           result);

private:
    int32_t run_iteration_stages(IterationContext& ctx);

    std::unique_ptr<SolverPipeline> pipeline_;
    bool initialized_ = false;
};
