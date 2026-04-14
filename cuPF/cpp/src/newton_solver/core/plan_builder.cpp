// ---------------------------------------------------------------------------
// plan_builder.cpp
//
// NewtonOptions를 받아 ExecutionPlan을 조립한다.
//
// 지원 프로파일별로 builder 함수를 하나씩 두어, 어떤 Storage·Op 조합이
// 사용되는지 코드를 읽는 것만으로 바로 알 수 있게 한다.
// 별도 validation / registry / factory 레이어 없이 직접 조립한다.
// ---------------------------------------------------------------------------

#include "newton_solver/core/plan_builder.hpp"

// CPU FP64 ops
#include "newton_solver/ops/mismatch/cpu_f64.hpp"
#include "newton_solver/ops/jacobian/cpu_f64.hpp"
#include "newton_solver/ops/linear_solve/cpu_klu.hpp"
#include "newton_solver/ops/voltage_update/cpu_f64.hpp"
#include "newton_solver/storage/cpu/cpu_fp64_storage.hpp"

#ifdef CUPF_WITH_CUDA
// CUDA ops
#include "newton_solver/ops/mismatch/cuda_f64.hpp"
#include "newton_solver/ops/jacobian/cuda_edge_fp64.hpp"
#include "newton_solver/ops/jacobian/cuda_vertex_fp64.hpp"
#include "newton_solver/ops/linear_solve/cuda_cudss64.hpp"
#include "newton_solver/ops/voltage_update/cuda_fp64.hpp"
#include "newton_solver/storage/cuda/cuda_fp64_storage.hpp"

// CUDA Mixed ops
#include "newton_solver/ops/jacobian/cuda_edge_fp32.hpp"
#include "newton_solver/ops/jacobian/cuda_vertex_fp32.hpp"
#include "newton_solver/ops/linear_solve/cuda_cudss32.hpp"
#include "newton_solver/ops/voltage_update/cuda_mixed.hpp"
#include "newton_solver/storage/cuda/cuda_mixed_storage.hpp"
#endif

#include <stdexcept>


ExecutionPlan PlanBuilder::build(const NewtonOptions& options)
{
    if (options.backend == BackendKind::CPU) {
        // CPU는 FP64만 지원한다.
        return build_cpu_fp64_plan();
    }

#ifdef CUPF_WITH_CUDA
    if (options.backend == BackendKind::CUDA) {
        if (options.compute == ComputePolicy::FP64)
            return build_cuda_fp64_plan(options.jacobian_builder);

        if (options.compute == ComputePolicy::Mixed)
            return build_cuda_mixed_plan(options.jacobian_builder);
    }
#else
    if (options.backend == BackendKind::CUDA) {
        throw std::invalid_argument(
            "PlanBuilder: CUDA backend를 요청했지만 cuPF가 CUDA 없이 빌드되었습니다.");
    }
#endif

    throw std::invalid_argument(
        "PlanBuilder: 지원하지 않는 backend/compute 조합입니다.");
}


// ---------------------------------------------------------------------------
// CPU FP64 프로파일
//
//   mismatch       — Eigen SpMV, FP64
//   jacobian       — edge-based fill, FP64
//   linear_solve   — KLU sparse direct, FP64
//   voltage_update — dx → Va/Vm → V 재구성, FP64
// ---------------------------------------------------------------------------
ExecutionPlan PlanBuilder::build_cpu_fp64_plan()
{
    ExecutionPlan plan;

    plan.storage        = std::make_unique<CpuFp64Storage>();
    plan.mismatch       = std::make_unique<CpuMismatchOpF64>(*plan.storage);
    plan.jacobian       = std::make_unique<CpuJacobianOpF64>(*plan.storage);
    plan.linear_solve   = std::make_unique<CpuLinearSolveKLU>(*plan.storage);
    plan.voltage_update = std::make_unique<CpuVoltageUpdateF64>(*plan.storage);
    plan.ready          = true;

    return plan;
}


#ifdef CUPF_WITH_CUDA

// ---------------------------------------------------------------------------
// CUDA FP64 프로파일
//
//   mismatch       — cuSPARSE SpMV, FP64
//   jacobian       — edge-based fill kernel, FP64
//   linear_solve   — cuDSS FP64
//   voltage_update — device kernel, FP64
// ---------------------------------------------------------------------------
ExecutionPlan PlanBuilder::build_cuda_fp64_plan(JacobianBuilderType jacobian_builder)
{
    ExecutionPlan plan;

    plan.storage        = std::make_unique<CudaFp64Storage>();
    plan.mismatch       = std::make_unique<CudaMismatchOpF64>(*plan.storage);
    if (jacobian_builder == JacobianBuilderType::VertexBased) {
        plan.jacobian = std::make_unique<CudaJacobianOpVertexFp64>(*plan.storage);
    } else {
        plan.jacobian = std::make_unique<CudaJacobianOpEdgeFp64>(*plan.storage);
    }
    plan.linear_solve   = std::make_unique<CudaLinearSolveCuDSS64>(*plan.storage);
    plan.voltage_update = std::make_unique<CudaVoltageUpdateFp64>(*plan.storage);
    plan.ready          = true;

    return plan;
}


// ---------------------------------------------------------------------------
// CUDA Mixed 프로파일 (고정 프로파일)
//
//   mismatch       — cuSPARSE SpMV, FP64 (mismatch는 FP64 정밀도 유지)
//   jacobian       — edge-based fill kernel, FP32
//   linear_solve   — cuDSS FP32
//   voltage_update — dx(FP32) → Va/Vm(FP64) → V(FP64) 재구성
//
// Mixed는 stage별 자유 조합이 아니라 이 고정된 프로파일이다.
// ---------------------------------------------------------------------------
ExecutionPlan PlanBuilder::build_cuda_mixed_plan(JacobianBuilderType jacobian_builder)
{
    ExecutionPlan plan;

    plan.storage        = std::make_unique<CudaMixedStorage>();
    plan.mismatch       = std::make_unique<CudaMismatchOpF64>(*plan.storage);
    if (jacobian_builder == JacobianBuilderType::VertexBased) {
        plan.jacobian = std::make_unique<CudaJacobianOpVertexFp32>(*plan.storage);
    } else {
        plan.jacobian = std::make_unique<CudaJacobianOpEdgeFp32>(*plan.storage);
    }
    plan.linear_solve   = std::make_unique<CudaLinearSolveCuDSS32>(*plan.storage);
    plan.voltage_update = std::make_unique<CudaVoltageUpdateMixed>(*plan.storage);
    plan.ready          = true;

    return plan;
}

#endif  // CUPF_WITH_CUDA
