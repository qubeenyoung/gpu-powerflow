# Operator Pipeline Refactor Plan

## 1. 목적

현재 solver는 `backend`, `precision`, `jacobian builder`, CUDA kernel 선택이 여러 파일에 분산된 `if`/오버로드/버퍼 분기로 섞여 있다.

이 문서의 목표는 다음과 같다.

- `MismatchOp`
- `JacobianOp`
- `LinearSolveOp`
- `VoltageUpdateOp`

를 독립적인 교체 가능 단위로 정의한다.

그리고 다음 선택 축을 런타임에 조합할 수 있는 구조를 만든다.

- `backend`: CPU / CUDA
- `precision`: FP32 / Mixed / FP64
- `jacobian builder`: EdgeBased / VertexBased
- `kernel variant`: 예를 들어 mismatch v1/v2, jacobian edge v1/v2, voltage update v1/v2

핵심 요구는 "실험할 때 코어 구조를 다시 뜯지 않고 조합만 바꿀 수 있어야 한다"는 것이다.

---

## 2. 현재 구조의 문제

현재 문제는 크게 네 가지다.

### 2.1 오케스트레이션과 구현이 강하게 결합되어 있다

`NewtonSolver`와 backend 구현이 다음 역할을 동시에 갖고 있다.

- solver loop orchestration
- precision validation
- device/host buffer layout 결정
- kernel 선택
- library descriptor 생성
- 각 단계 실행

이 때문에 새로운 precision 또는 커널 variant를 추가할 때 수정 지점이 너무 많아진다.

### 2.2 선택 축이 서로 다른 레벨에 섞여 있다

예를 들어:

- `backend` 선택은 `NewtonSolver` 생성 시점
- `precision` 선택은 public API overload와 CUDA impl 내부 분기
- `jacobian builder` 선택은 Jacobian 단계 안에서만 의미 있음
- mismatch kernel variant는 별도 개념조차 없음

즉, "어떤 조합이 유효한지"를 한 곳에서 설명하거나 검증하기 어렵다.

### 2.3 실험용 kernel 추가가 구조를 계속 더럽힌다

예를 들어 새로운 mismatch kernel을 시험하려면 보통 다음 셋 중 하나가 필요해진다.

- 기존 함수에 `if` 추가
- 기존 backend impl에 새로운 버퍼 추가
- 기존 public path를 따라가며 분기 삽입

이 방식은 실험을 계속할수록 구조적 부채가 쌓인다.

### 2.4 batch와 single-case가 같은 구조 안에 섞여 있다

현재 precision refactor의 범위는 `n_batch == 1` single-case인데, 코드베이스에는 여전히 batch 전용 구현과 설명이 섞여 있다.

문제는 batch가 single-case의 단순 반복이 아니라는 점이다.

- buffer layout이 다르다
- cuSPARSE/cuDSS descriptor 구성이 다르다
- kernel launch shape가 다르다
- 성능 실험 축도 따로 생긴다

따라서 CUDA multi-batch kernel은 single-case kernel과 같은 파일/같은 op 안에서 관리하지 않는 편이 낫다.

이번 구조 개편의 1차 범위는 single-case only로 고정하고, CUDA multi-batch는 별도 TODO 트랙으로 분리하는 편이 안전하다.

---

## 3. 설계 원칙

새 구조는 아래 원칙을 따른다.

### 3.1 solver는 "계획 실행기"가 되어야 한다

`NewtonSolver`는 더 이상 precision별 구현을 직접 알지 않는다.
대신 `ExecutionPlan`을 들고 있고, iteration마다 정해진 `Op`를 호출만 한다.

### 3.2 버퍼/descriptor와 계산 단계를 분리한다

precision과 backend에 따라 달라지는 것은 크게 두 가지다.

- 어떤 버퍼와 descriptor를 가지는가
- 각 단계를 어떤 구현으로 실행하는가

이 둘은 분리되어야 한다.

### 3.3 런타임 조합 가능해야 한다

실험을 위해 다음을 런타임에 바꿀 수 있어야 한다.

- `backend`
- `precision`
- `jacobian builder`
- stage별 kernel variant

### 3.4 unsupported 조합은 plan build 시점에 막는다

`FP32 + CPU`처럼 유효하지 않은 조합은 solve 도중이 아니라 `ExecutionPlan` 생성 시점에 즉시 거부한다.

### 3.5 hot path는 단순해야 한다

NR iteration hot path 안에는 "큰 if-else 트리" 대신 다음 정도만 남는 구조가 이상적이다.

```cpp
plan.mismatch->run(...);
plan.jacobian->run(...);
plan.linear_solve->run(...);
plan.voltage_update->run(...);
```

### 3.6 single-case와 multi-batch CUDA 경로는 물리적으로 분리한다

single-case와 multi-batch는 같은 추상 개념을 공유할 수는 있어도, 구현 파일과 registry 항목은 분리하는 것이 좋다.

즉:

- single-case CUDA op
- multi-batch CUDA op

는 같은 `.cu` 파일 안에서 `if (n_batch > 1)`로 갈라지지 않아야 한다.

multi-batch는 나중에 별도 TODO로 다시 붙이되, single-case 실험 경로를 오염시키지 않는 것이 우선이다.

---

## 4. 목표 구조

### 4.1 최상위 개념

```cpp
struct SolverSpec {
    BackendKind         backend;
    PrecisionMode       precision;
    JacobianBuilderType jacobian_builder;
    KernelChoice        kernels;
};

struct ExecutionPlan {
    std::unique_ptr<IStorage>          storage;
    std::unique_ptr<IMismatchOp>       mismatch;
    std::unique_ptr<IJacobianOp>       jacobian;
    std::unique_ptr<ILinearSolveOp>    linear_solve;
    std::unique_ptr<IVoltageUpdateOp>  voltage_update;
};
```

이때 `ExecutionPlan`은 "이 solver 인스턴스가 어떤 구현 조합으로 동작하는가"를 표현한다.

### 4.2 선택 축을 담는 `KernelChoice`

```cpp
enum class MismatchKernel {
    Auto,
    CpuEigen,
    CudaSpmvF64Basic,
    CudaSpmvF32Basic,
    CudaSpmvF64FastNorm,
};

enum class JacobianKernel {
    Auto,
    CpuEdge,
    CudaEdgeF32V1,
    CudaEdgeF32V2,
    CudaVertexF32V1,
    CudaEdgeF64V1,
    CudaVertexF64V1,
};

enum class LinearSolveKernel {
    Auto,
    CpuKLU,
    CudaCuDSS32,
    CudaCuDSS64,
};

enum class VoltageKernel {
    Auto,
    CpuVoltageF64,
    CudaVoltageMixed,
    CudaVoltageF32,
    CudaVoltageF64,
};

struct KernelChoice {
    MismatchKernel    mismatch       = MismatchKernel::Auto;
    JacobianKernel    jacobian       = JacobianKernel::Auto;
    LinearSolveKernel linear_solve   = LinearSolveKernel::Auto;
    VoltageKernel     voltage_update = VoltageKernel::Auto;
};
```

`Auto`는 registry가 `backend/precision/jacobian_builder`를 보고 기본값을 고르게 한다.

---

## 5. 핵심 구성요소

## 5.1 `Storage`

`Storage`는 버퍼와 library handle/descriptors를 소유한다.
각 `Op`는 이 `Storage`를 사용하지만, 직접 할당 정책을 결정하지는 않는다.

예시:

```cpp
class IStorage {
public:
    virtual ~IStorage() = default;
    virtual BackendKind backend() const = 0;
    virtual PrecisionMode precision() const = 0;
};

class CpuFp64Storage final : public IStorage { ... };
class CudaMixedStorage final : public IStorage { ... };
class CudaFp32Storage final : public IStorage { ... };
class CudaFp64Storage final : public IStorage { ... };
```

`Storage`의 책임:

- host/device buffer 소유
- cuSPARSE/cuDSS/Eigen/KLU 핸들 및 descriptor 소유
- analyze 단계에서 필요한 메모리 layout 준비
- solve 초기화에 필요한 입력 업로드

`Storage`는 "메모리/라이브러리 상태"를 담당하고, "계산 로직"은 담당하지 않는다.

## 5.2 `Op` 인터페이스

각 단계는 독립 인터페이스로 분리한다.

```cpp
class IMismatchOp {
public:
    virtual ~IMismatchOp() = default;
    virtual void run(IterationContext& ctx) = 0;
};

class IJacobianOp {
public:
    virtual ~IJacobianOp() = default;
    virtual void run(IterationContext& ctx) = 0;
};

class ILinearSolveOp {
public:
    virtual ~ILinearSolveOp() = default;
    virtual void run(IterationContext& ctx) = 0;
};

class IVoltageUpdateOp {
public:
    virtual ~IVoltageUpdateOp() = default;
    virtual void run(IterationContext& ctx) = 0;
};
```

각 concrete op는 특정 storage/layout 조합에 묶인다.

예시:

- `CpuMismatchOpF64`
- `CudaMismatchOpMixedSpmvF64`
- `CudaMismatchOpFp32SpmvF32`
- `CudaJacobianOpEdgeF32V1`
- `CudaJacobianOpVertexF32V2`
- `CudaJacobianOpEdgeF64V1`
- `CudaLinearSolveCuDSS32`
- `CudaLinearSolveCuDSS64`
- `CudaVoltageUpdateMixed`
- `CudaVoltageUpdateFp32`

## 5.3 `IterationContext`

각 단계가 공통으로 접근하는 실행 컨텍스트를 둔다.

```cpp
struct IterationContext {
    IStorage&        storage;
    const NRConfig&  config;
    const int32_t*   pv;
    int32_t          n_pv;
    const int32_t*   pq;
    int32_t          n_pq;
    bool             converged = false;
};
```

필요하다면 `AnalyzeContext`와 `SolveContext`를 따로 둘 수도 있다.

예시:

- `AnalyzeContext`: `Ybus`, `JacobianMaps`, `JacobianStructure`, `n_bus`
- `SolveContext`: `sbus`, `V0`
- `IterationContext`: `F`, `normF`, `dx`, bus indices

---

## 6. 조합 방식

### 6.1 `PlanBuilder`

`SolverSpec`을 받아 실제 실행 계획을 만든다.

```cpp
class PlanBuilder {
public:
    static ExecutionPlan build(const SolverSpec& spec);
};
```

순서는 다음과 같다.

1. spec validation
2. storage 생성
3. stage별 op 선택
4. op와 storage를 묶어 `ExecutionPlan` 완성

### 6.2 validation은 중앙집중식으로

예시:

```cpp
void validate_spec(const SolverSpec& spec) {
    if (spec.backend == BackendKind::CPU && spec.precision != PrecisionMode::FP64)
        throw std::invalid_argument("CPU backend supports only FP64");

    if (spec.precision == PrecisionMode::FP32 && spec.backend != BackendKind::CUDA)
        throw std::invalid_argument("FP32 is currently CUDA-only");
}
```

유효하지 않은 조합은 여기서 한 번만 막는다.

### 6.3 registry 기반 op 선택

```cpp
std::unique_ptr<IMismatchOp> build_mismatch_op(
    const SolverSpec& spec,
    IStorage& storage);

std::unique_ptr<IJacobianOp> build_jacobian_op(
    const SolverSpec& spec,
    IStorage& storage);
```

registry는 다음 입력을 본다.

- `backend`
- `precision`
- `jacobian_builder`
- 사용자가 지정한 kernel variant

그리고 concrete op를 하나 고른다.

---

## 7. 연산 단계별 책임

## 7.1 `MismatchOp`

책임:

- `Ibus = Ybus * V`
- mismatch packing
- `normF` 계산
- 필요하면 `-F` 준비

예시 조합:

- CPU FP64: Eigen 기반 mismatch
- CUDA Mixed: FP64 SpMV + FP64 mismatch pack
- CUDA FP32: FP32 SpMV + FP32 mismatch pack
- CUDA FP64: FP64 SpMV + FP64 mismatch pack

이 단계는 Jacobian builder 종류와는 무관해야 한다.

## 7.2 `JacobianOp`

책임:

- Jacobian value buffer zeroing
- 필요하면 전압 표현 변환
- edge 또는 vertex kernel 실행

이 단계는 다음 두 축을 모두 본다.

- `precision`
- `jacobian_builder`

즉, `JacobianOp`은 축이 가장 많은 단계다.

## 7.3 `LinearSolveOp`

책임:

- `b = -F` 준비
- factorization / refactorization
- solve 실행

예시:

- CPU FP64: KLU
- CUDA Mixed: cuDSS32
- CUDA FP32: cuDSS32
- CUDA FP64: cuDSS64

`Mixed`와 `FP32`가 같은 cuDSS32를 쓰더라도, 입력 buffer layout이 다를 수 있으므로 op 또는 storage에서 그 차이를 흡수해야 한다.

## 7.4 `VoltageUpdateOp`

책임:

- `dx` 적용
- `Va/Vm` 갱신
- 복소 전압 재구성
- 다음 Jacobian 또는 mismatch 단계가 요구하는 전압 표현 동기화

예시:

- CPU FP64 voltage update
- CUDA Mixed voltage update
- CUDA FP32 voltage update
- CUDA FP64 voltage update

---

## 8. precision은 scalar가 아니라 layout 정책이다

이 설계에서 중요한 포인트는 `precision`을 단순히 `float`/`double` 선택으로 보면 안 된다는 것이다.

실제로는 다음이 함께 묶인다.

- public API dtype
- voltage state dtype
- Jacobian value dtype
- linear solve dtype
- mismatch dtype
- library descriptor dtype

따라서 내부적으로는 아래처럼 layout policy 개념으로 보는 편이 낫다.

```cpp
struct Fp32Layout {
    using PublicScalar   = float;
    using VoltageScalar  = float;
    using JacobianScalar = float;
    using SolveScalar    = float;
};

struct MixedLayout {
    using PublicScalar   = double;
    using VoltageScalar  = double;
    using JacobianScalar = float;
    using SolveScalar    = float;
};

struct Fp64Layout {
    using PublicScalar   = double;
    using VoltageScalar  = double;
    using JacobianScalar = double;
    using SolveScalar    = double;
};
```

이 개념을 쓰면 구현 코드를 조금 더 재사용할 수 있다.

---

## 9. 추천 디렉터리 구조

```text
cpp/inc/newton_solver/pipeline/
  solver_spec.hpp
  kernel_choice.hpp
  execution_plan.hpp
  op_interfaces.hpp
  storage.hpp
  contexts.hpp

cpp/src/newton_solver/pipeline/
  plan_builder.cpp
  spec_validation.cpp
  storage_factory.cpp
  op_registry.cpp

cpp/src/newton_solver/storage/cpu/
  cpu_fp64_storage.cpp

cpp/src/newton_solver/storage/cuda/
  cuda_fp32_storage.cpp
  cuda_mixed_storage.cpp
  cuda_fp64_storage.cpp

cpp/src/newton_solver/ops/cpu/mismatch/
cpp/src/newton_solver/ops/cpu/jacobian/
cpp/src/newton_solver/ops/cpu/linear_solve/
cpp/src/newton_solver/ops/cpu/voltage/

cpp/src/newton_solver/ops/cuda_single/mismatch/
cpp/src/newton_solver/ops/cuda_single/jacobian/
cpp/src/newton_solver/ops/cuda_single/linear_solve/
cpp/src/newton_solver/ops/cuda_single/voltage/

cpp/src/newton_solver/ops/cuda_batch/
  TODO.md
```

이 구조의 장점:

- 선택 축이 명확해진다
- 파일 책임이 작아진다
- 실험용 kernel variant를 추가하기 쉽다
- single-case 실험과 future batch 실험이 서로를 오염시키지 않는다

---

## 10. `NewtonSolver`의 역할 변화

최종적으로 `NewtonSolver`는 다음 역할만 맡아야 한다.

1. `SolverSpec` 구성
2. `ExecutionPlan` 생성
3. analyze/initialize/iteration/download 순서 orchestration

예시:

```cpp
class NewtonSolver {
public:
    explicit NewtonSolver(const NewtonOptions& options);

    void analyze(...);
    void solve(...);

private:
    SolverSpec     spec_;
    ExecutionPlan  plan_;
};
```

iteration loop는 대략 이렇게 단순해진다.

```cpp
while (iter < config.max_iter) {
    plan_.mismatch->run(ctx);
    if (ctx.converged) break;

    plan_.jacobian->run(ctx);
    plan_.linear_solve->run(ctx);
    plan_.voltage_update->run(ctx);
}
```

---

## 11. 실험 친화성을 위한 추가 규칙

실험을 계속할 계획이라면 아래 규칙을 지키는 것이 중요하다.

### 11.1 새로운 실험은 새 enum + 새 op로 추가한다

기존 op에 실험용 `if (experimental)`를 넣지 않는다.

좋은 예:

- `CudaJacobianOpEdgeF32V3`
- `CudaMismatchOpSpmvF64FastNorm`

나쁜 예:

- 기존 `CudaBackend::updateJacobian()` 안에 `if (use_v3)` 추가

### 11.2 성능 실험 선택은 `KernelChoice`로만 노출한다

CLI, Python binding, benchmark harness는 결국 `KernelChoice`만 바꾸게 한다.
그러면 실험 레이어와 구현 레이어가 깔끔하게 분리된다.

### 11.3 unsupported 조합은 registry에서 명확히 거부한다

예를 들어:

- `MismatchKernel::CudaSpmvF32Basic` + `PrecisionMode::FP64`
- `JacobianKernel::CudaVertexF64V1` + `BackendKind::CPU`

같은 조합은 명확한 에러 메시지로 막는다.

---

## 12. 이번 refactor의 범위

1차 범위는 다음으로 제한한다.

- single-case only (`n_batch == 1`)
- CPU FP64
- CUDA Mixed
- CUDA FP32
- CUDA FP64
- `MismatchOp`, `JacobianOp`, `LinearSolveOp`, `VoltageUpdateOp`

이번 범위에서 제외:

- multi-batch
- CUDA multi-batch kernel/operator/storage 구현
- Python binding의 실험용 kernel enum 노출
- autotuning

이유는 "구조 재정렬"과 "실험 속도 개선"을 먼저 달성하는 것이 우선이기 때문이다.

### 12.1 future TODO: CUDA multi-batch 분리 트랙

multi-batch는 "나중에 같은 구조 위에 덧붙이는 기능"으로 남긴다.

원칙:

- single-case와 같은 interface 개념은 재사용 가능
- 하지만 concrete storage/op/kernel 구현은 별도 디렉터리로 분리
- 같은 `.cu` 파일에서 single/batch를 동시에 관리하지 않는다

예시 TODO:

- `CudaBatchMixedStorage`
- `CudaBatchMismatchOpMixed`
- `CudaBatchJacobianOpEdgeF32`
- `CudaBatchLinearSolveCuDSS32`
- `CudaBatchVoltageUpdateMixed`

즉, batch는 "지금 구현할 대상"이 아니라 "나중에 독립 트랙으로 붙일 대상"으로 정의한다.

---

## 13. 단계별 이행 계획

### Phase A. skeleton 도입

- `SolverSpec`
- `KernelChoice`
- `ExecutionPlan`
- `IStorage`
- `IMismatchOp`, `IJacobianOp`, `ILinearSolveOp`, `IVoltageUpdateOp`
- `PlanBuilder`

이 단계에서는 기존 구현을 그대로 호출하는 thin wrapper op를 만들어도 된다.

### Phase B. CUDA Mixed를 첫 번째 concrete plan으로 추출

현재 가장 안정적인 경로를 기준 구현으로 삼는다.

- `CudaMixedStorage`
- `CudaMismatchOpMixed`
- `CudaJacobianOpEdgeF32V1` / `CudaJacobianOpVertexF32V1`
- `CudaLinearSolveCuDSS32`
- `CudaVoltageUpdateMixed`

이 단계가 끝나면 "기존 mixed CUDA path"가 더 이상 giant backend impl 안에 직접 박혀 있지 않게 된다.

### Phase C. CPU FP64를 같은 구조로 이전

- `CpuFp64Storage`
- `CpuMismatchOpF64`
- `CpuJacobianOpF64`
- `CpuLinearSolveKLU`
- `CpuVoltageUpdateF64`

### Phase D. CUDA FP32, FP64 추가

- `CudaFp32Storage`
- `CudaFp64Storage`
- FP32/FP64 mismatch/jacobian/solve/voltage ops

### Phase E. kernel variant 실험 체계 추가

- `KernelChoice`를 benchmark/CLI에서 설정 가능하게 만듦
- 새 커널은 registry 항목만 추가

### Future TODO. CUDA multi-batch 별도 트랙

- `ops/cuda_batch/`와 `storage/cuda_batch/`를 새로 만든다
- single-case와 별도 registry를 둔다
- batch 전용 `ExecutionPlan` 또는 `BatchExecutionPlan` 필요 여부를 검토한다
- single-case path와 소스 파일을 공유하지 않는 방향을 기본 원칙으로 삼는다

---

## 14. 추천 구현 순서

실제 작업 순서는 아래가 안전하다.

1. `NewtonSolver` 안에 `SolverSpec`과 `ExecutionPlan` 개념 도입
2. 기존 CUDA Mixed 경로를 wrapper op로 감싼다
3. 기존 CPU FP64 경로를 wrapper op로 감싼다
4. backend 내부 giant switch를 줄이고 registry로 이동시킨다
5. CUDA FP32를 새 storage/op 조합으로 추가한다
6. CUDA FP64를 추가한다
7. 실험용 kernel variant를 도입한다

이 순서를 지키면 "구조 개편"과 "기능 추가"를 동시에 하면서도 매 단계마다 동작하는 상태를 유지하기 쉽다.

---

## 15. 요약

원하는 구조는 "backend 하나가 모든 경우를 다 처리하는 구조"가 아니라,
"storage + stage op + registry"를 조합하는 구조다.

핵심 아이디어는 다음 한 줄로 정리된다.

> solver는 계산을 직접 구현하지 않고, 선택된 `ExecutionPlan`을 실행한다.

이 구조를 쓰면 다음이 가능해진다.

- precision 실험
- edge/vertex 실험
- 커널 버전 실험
- backend 간 비교

그리고 새 실험을 추가할 때 기존 giant backend implementation을 계속 더럽히지 않아도 된다.
