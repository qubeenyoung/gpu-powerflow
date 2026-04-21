# Phase 01 — Batch-Aware API and Core Types

이 단계의 목적은 public/core layer가 batch 입력을 표현할 수 있게 하되, 기존 single-case API를 wrapper로 유지하는 것이다.

---

## 목표

1. CUDA 기본 실행 모델이 batch라는 사실을 type level에서 반영한다.
2. single-case solve는 `B=1` wrapper로 유지한다.
3. `PlanBuilder`는 여전히 backend/compute/jacobian_builder 조합만 선택한다.

---

## 대상 파일

- [cpp/inc/newton_solver/core/newton_solver_types.hpp](../../cpp/inc/newton_solver/core/newton_solver_types.hpp)
- [cpp/inc/newton_solver/core/contexts.hpp](../../cpp/inc/newton_solver/core/contexts.hpp)
- [cpp/inc/newton_solver/core/newton_solver.hpp](../../cpp/inc/newton_solver/core/newton_solver.hpp)
- [cpp/src/newton_solver/core/newton_solver.cpp](../../cpp/src/newton_solver/core/newton_solver.cpp)
- [bindings/pybind_cupf.cpp](../../bindings/pybind_cupf.cpp)

---

## 작업 내용

### 1. batch-aware solve context

`SolveContext`는 최소한 아래 정보를 표현할 수 있어야 한다.

```text
batch_size
sbus stride
V0 stride
optional batched Ybus values
```

single-case `solve()`는 내부에서 `batch_size = 1`인 `SolveContext`를 만든다.

### 2. batch result type

single-case 결과 타입은 유지하되, 내부 batch path를 표현할 수 있는 result 타입을 추가한다.

예:

```text
NRBatchResultF64
  V[B * n_bus]
  iterations[B]
  final_mismatch[B]
  converged[B]
```

single-case `NRResultF64`는 `B=1` 결과를 unwrap한 wrapper다.

### 3. analyze API는 구조 중심 유지

`AnalyzeContext`는 계속 sparse structure와 maps 중심으로 유지한다.
값 dtype/layout 문제는 `SolveContext`와 storage upload 단계에서 처리한다.

### 4. profile 선택은 그대로 유지

`PlanBuilder`는 아래 축만 계속 본다.

```text
backend
compute
jacobian_builder
cudss options
```

batch는 profile 선택 축이 아니라 runtime dimension이다.

---

## 주의사항

- 이 단계에서 `CudaBatch*` class를 추가하지 않는다.
- `BatchExecutionPlan`을 만들지 않는다.
- batch-aware public API를 추가하더라도 기존 single-case call site는 깨지지 않아야 한다.

---

## 완료 조건

- core type이 batch 입력/출력을 표현할 수 있다.
- single-case API는 여전히 동작하며 내부적으로 `B=1`을 사용한다.
- `PlanBuilder`의 의미가 바뀌지 않는다.
