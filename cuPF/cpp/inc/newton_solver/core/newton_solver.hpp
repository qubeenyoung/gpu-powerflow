#pragma once

#include "newton_solver_types.hpp"
#include "execution_plan.hpp"
#include "jacobian_builder.hpp"

#include <memory>

struct AnalyzeContext;
struct IterationContext;


// ---------------------------------------------------------------------------
// NewtonSolver: Newton-Raphson 전력조류 solver.
//
// 역할:
//   1. NewtonOptions를 받아 ExecutionPlan을 조립한다.
//   2. analyze / solve / 반복 루프 생명주기를 오케스트레이션한다.
//   3. backend·compute policy별 구체 로직은 ExecutionPlan과 Op에 위임한다.
//
// 사용 예 (CPU FP64):
//   NewtonSolver solver;  // 기본값: CPU, FP64
//   solver.analyze(ybus, pv, n_pv, pq, n_pq);
//   solver.solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, result);
//
// 사용 예 (CUDA Mixed):
//   NewtonSolver solver({.backend = BackendKind::CUDA,
//                        .compute = ComputePolicy::Mixed});
//   solver.analyze(ybus, pv, n_pv, pq, n_pq);
//   solver.solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, result);
//
// public I/O는 항상 FP64다. Mixed 모드에서 내부 FP32 연산이 일어나더라도
// 사용자는 이를 직접 알 필요가 없다.
// ---------------------------------------------------------------------------
class NewtonSolver {
public:
    explicit NewtonSolver(const NewtonOptions& options = {});
    ~NewtonSolver();

    // Ybus 희소 구조를 분석하고 solver 내부 상태를 초기화한다.
    // solve() 호출 전 반드시 한 번 실행해야 한다.
    void analyze(const YbusViewF64& ybus,
                 const int32_t* pv, int32_t n_pv,
                 const int32_t* pq, int32_t n_pq);

    // Newton-Raphson 반복을 실행하고 결과를 result에 채운다.
    // analyze()가 먼저 호출되어 있어야 한다.
    void solve(const YbusViewF64&          ybus,
               const std::complex<double>* sbus,
               const std::complex<double>* V0,
               const int32_t*              pv, int32_t n_pv,
               const int32_t*              pq, int32_t n_pq,
               const NRConfig&             config,
               NRResultF64&                result);

    // Batch solve 진입점. 기본 실행 모델은 batch이며, single-case solve는
    // 내부적으로 batch_size=1인 이 경로를 사용한다.
    void solve_batch(const YbusViewF64&          ybus,
                     const std::complex<double>* sbus,
                     int64_t                     sbus_stride,
                     const std::complex<double>* V0,
                     int64_t                     V0_stride,
                     int32_t                     batch_size,
                     const int32_t*              pv, int32_t n_pv,
                     const int32_t*              pq, int32_t n_pq,
                     const NRConfig&             config,
                     NRBatchResultF64&           result);

private:
    // analyze stage 실행: storage 준비 → linear_solve symbolic 분석
    void run_analyze_stages(const AnalyzeContext& ctx);

    // NR 반복 루프: mismatch → jacobian → linear_solve → voltage_update
    // 반환값: 실제 수행한 반복 횟수
    int32_t run_iteration_stages(IterationContext& ctx);

    NewtonOptions   options_;
    ExecutionPlan   plan_;
    JacobianBuilder jac_builder_;

    bool analyzed_ = false;
};
