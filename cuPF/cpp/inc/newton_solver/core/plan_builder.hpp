#pragma once

#include "newton_solver/core/newton_solver_types.hpp"
#include "execution_plan.hpp"


// ---------------------------------------------------------------------------
// PlanBuilder: NewtonOptions를 받아 ExecutionPlan을 조립한다.
//
// 지원하는 프로파일은 아래 세 가지다. 코드 구조 자체가 지원 조합을 제한한다.
//
//   CPU  + FP64  → build_cpu_fp64_plan()
//   CUDA + FP64  → build_cuda_fp64_plan()
//   CUDA + Mixed → build_cuda_mixed_plan()
//
// 각 builder 함수는 해당 프로파일에 맞는 Storage와 Op를 직접 생성한다.
// 복잡한 validation 레이어나 registry 조회 없이 한눈에 읽힌다.
// ---------------------------------------------------------------------------
class PlanBuilder {
public:
    // NewtonOptions로부터 ExecutionPlan을 조립해 반환한다.
    // 지원하지 않는 조합이면 std::invalid_argument를 던진다.
    static ExecutionPlan build(const NewtonOptions& options);

private:
    static ExecutionPlan build_cpu_fp64_plan();

#ifdef CUPF_WITH_CUDA
    static ExecutionPlan build_cuda_fp64_plan(JacobianBuilderType jacobian_builder);
    static ExecutionPlan build_cuda_mixed_plan(JacobianBuilderType jacobian_builder);
#endif
};
