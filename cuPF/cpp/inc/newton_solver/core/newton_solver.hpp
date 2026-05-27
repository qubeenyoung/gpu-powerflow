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
// public I/O는 항상 FP64다. CUDA FP32는 upload/download boundary에서
// FP64 public data와 FP32 device data를 변환한다.
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
               const SolveOptions&         solve_options,
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
                     const SolveOptions&         solve_options,
                     NRBatchResult&           result);

    void solve(const YbusView&          ybus,
               const std::complex<double>* sbus,
               const std::complex<double>* V0,
               const int32_t*              pv, int32_t n_pv,
               const int32_t*              pq, int32_t n_pq,
               const NRConfig&             config,
               NRResult&                result)
    {
        solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, SolveOptions{}, result);
    }

    void solve_batch(const YbusView&          ybus,
                     const std::complex<double>* sbus,
                     int64_t                     sbus_stride,
                     const std::complex<double>* V0,
                     int64_t                     V0_stride,
                     int32_t                     batch_size,
                     const int32_t*              pv, int32_t n_pv,
                     const int32_t*              pq, int32_t n_pq,
                     const NRConfig&             config,
                     NRBatchResult&           result)
    {
        solve_batch(ybus, sbus, sbus_stride, V0, V0_stride, batch_size,
                    pv, n_pv, pq, n_pq, config, SolveOptions{}, result);
    }

    void solve_adjoint(const double*        grad_va,
                       int64_t              grad_va_stride,
                       const double*        grad_vm,
                       int64_t              grad_vm_stride,
                       int32_t              batch_size,
                       const int32_t*       pv, int32_t n_pv,
                       const int32_t*       pq, int32_t n_pq,
                       const AdjointOptions& options,
                       AdjointResult&       result);

    void solve_adjoint_cuda_raw(uintptr_t grad_va_device_ptr,
                                uintptr_t grad_vm_device_ptr,
                                uintptr_t grad_load_p_device_ptr,
                                uintptr_t grad_load_q_device_ptr,
                                int32_t batch_size,
                                int32_t n_bus,
                                const char* dtype,
                                const AdjointOptions& options,
                                AdjointResult& result);

    void solve_adjoint_cuda_raw_unsafe(uintptr_t grad_va_device_ptr,
                                       uintptr_t grad_vm_device_ptr,
                                       uintptr_t grad_load_p_device_ptr,
                                       uintptr_t grad_load_q_device_ptr,
                                       int32_t batch_size,
                                       int32_t n_bus,
                                       const char* dtype,
                                       const AdjointOptions& options,
                                       AdjointResult& result);

    void solve_cuda_load_pq_device(const void* sbus_base_re_device_ptr,
                                   const void* sbus_base_im_device_ptr,
                                   const void* load_p_device_ptr,
                                   const void* load_q_device_ptr,
                                   const void* v0_va_device_ptr,
                                   const void* v0_vm_device_ptr,
                                   void* va_out_device_ptr,
                                   void* vm_out_device_ptr,
                                   int32_t batch_size,
                                   int32_t n_bus,
                                   const char* dtype,
                                   const NRConfig& config,
                                   const SolveOptions& solve_options,
                                   AdjointResult& result);

    void solve_adjoint_cuda_device(const void* grad_va_device_ptr,
                                   const void* grad_vm_device_ptr,
                                   void* grad_load_p_device_ptr,
                                   void* grad_load_q_device_ptr,
                                   int32_t batch_size,
                                   int32_t n_bus,
                                   const char* dtype,
                                   const AdjointOptions& options,
                                   bool raw_pointer_api_used,
                                   AdjointResult& result);

private:
    int32_t run_iteration_stages(IterationContext& ctx);
    void prepare_adjoint_cache(IterationContext& ctx,
                               const SolveOptions& solve_options,
                               double final_mismatch_norm);

    std::unique_ptr<SolverPipeline> pipeline_;
    NewtonOptions options_;
    bool initialized_ = false;
};
