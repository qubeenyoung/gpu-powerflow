# core — 핵심 컴포넌트 설계

`cpp/inc/newton_solver/core/` 및 `cpp/src/newton_solver/core/` 에 위치한다.

---

## 컴포넌트 목록

| 파일 | 역할 |
|------|------|
| `newton_solver_types.hpp` | 공개 타입: CSRView, NRConfig, BackendKind, ComputePolicy, NewtonOptions, NRResultF64, NRBatchResultF64 |
| `jacobian_types.hpp` | JacobianBuilderType, JacobianMaps, JacobianStructure |
| `jacobian_builder.hpp/.cpp` | Ybus 희소 구조 분석 → 산포 맵 생성 |
| `contexts.hpp` | AnalyzeContext, SolveContext, IterationContext |
| `execution_plan.hpp` | Storage + 4-Op 조합을 소유하는 구조체 |
| `plan_builder.hpp/.cpp` | NewtonOptions → ExecutionPlan 조립 |
| `newton_solver.hpp/.cpp` | 공개 API: analyze(), solve() 오케스트레이션 |

---

## NewtonSolver

**파일:** [newton_solver.hpp](../../cpp/inc/newton_solver/core/newton_solver.hpp)

### 책임

1. 생성자에서 `PlanBuilder::build(options)`로 `ExecutionPlan`을 조립한다.
2. `analyze()`로 희소 구조를 분석하고 device-side 상태를 초기화한다.
3. `solve()`/`solve_batch()`로 NR 반복 루프를 실행하고 결과를 반환한다.

### 생명주기

```
NewtonSolver(options)    → PlanBuilder::build()  [ExecutionPlan 조립]
  analyze(ybus, ...)     → JacobianBuilder::analyze()
                         → IStorage::prepare()
                         → ILinearSolveOp::analyze()
  solve(ybus, sbus, V0, ...) 또는 solve_batch(...)
                         → IStorage::upload()
                         → 반복 루프 (mismatch → jacobian → linear_solve → voltage_update)
                         → IStorage::download_result() / download_batch_result()
```

### 주의사항

- `analyze()`는 Ybus의 **희소 구조**가 바뀌지 않는 한 한 번만 호출하면 된다.
- `solve()`는 Ybus 값(원소값)이 바뀌어도 호출할 수 있다. 구조가 같으면 된다.
- 기본 실행 모델은 batch이고, 기존 `solve()`는 내부적으로 `solve_batch(B=1)` wrapper다.
- 현재 `solve_batch(B>1)` 실제 실행은 CUDA Mixed path에서 지원한다.
- `analyze()` 전에 `solve()`를 호출하면 `std::runtime_error`가 발생한다.

---

## JacobianBuilder

**파일:** [jacobian_builder.hpp](../../cpp/inc/newton_solver/core/jacobian_builder.hpp)

### 책임

`analyze()` 호출 시 Ybus 희소 구조를 분석해 다음 두 가지를 반환한다.

1. **`JacobianMaps`** — Ybus 원소 k → J.values 위치를 기록한 산포 맵 테이블
2. **`JacobianStructure`** — Jacobian CSR 희소 패턴 (row_ptr, col_idx)

### 알고리즘 선택

| 타입 | 방식 | CUDA fill 방식 |
|------|------|---------------|
| `EdgeBased` | Ybus 비영 원소(엣지) 순회 | 스레드 1개 = 원소 1개 |
| `VertexBased` | 버스(정점) 기준 순회 | warp 1개 = 버스 1개 |

두 알고리즘의 희소 패턴 분석 결과(`JacobianMaps`, `JacobianStructure`)는 동일하다.
`maps.builder_type` 필드만 다르며, 이를 통해 `PlanBuilder`가 커널을 선택한다.

### 내부 구현 (EdgeBased)

```
1. pvpq = pv ∥ pq 구성, 역방향 인덱스 맵(rmap, cmap) 생성
2. Ybus 비영 원소마다 최대 4개 J triplet 등록 (Eigen)
3. Eigen CSC 조립 → CSC→CSR 변환 → JacobianStructure
4. Ybus 원소 k마다 find_coeff_index()로 J CSR 위치 기록 → JacobianMaps
```

자세한 설명은 [math.md](../math.md#4-jacobian-희소-구조) 참조.

---

## JacobianMaps

**파일:** [jacobian_types.hpp](../../cpp/inc/newton_solver/core/jacobian_types.hpp)

Jacobian은 2×2 블록 구조다.

```
J = [ J11  J12 ] = [ ∂P/∂θ     ∂P/∂|V| ]
    [ J21  J22 ]   [ ∂Q/∂θ     ∂Q/∂|V| ]

행: J11/J12 → pvpq 버스 (인덱스 0..n_pvpq-1)
    J21/J22 → pq   버스 (인덱스 n_pvpq..dimF-1)
열: J11/J21 → pvpq 버스 (인덱스 0..n_pvpq-1)
    J12/J22 → pq   버스 (인덱스 n_pvpq..dimF-1)
```

| 필드 | 의미 | 크기 |
|------|------|------|
| `mapJ11[k]` | Ybus k번째 원소 → J11 블록 내 CSR 위치 (-1: 비기여) | [nnz_Y] |
| `mapJ12[k]` | Ybus k번째 원소 → J12 블록 내 CSR 위치 | [nnz_Y] |
| `mapJ21[k]` | Ybus k번째 원소 → J21 블록 내 CSR 위치 | [nnz_Y] |
| `mapJ22[k]` | Ybus k번째 원소 → J22 블록 내 CSR 위치 | [nnz_Y] |
| `diagJ11[bus]` | 버스 bus → J11 대각 CSR 위치 | [n_bus] |
| `diagJ12[bus]` | 버스 bus → J12 대각 CSR 위치 | [n_bus] |
| `diagJ21[bus]` | 버스 bus → J21 대각 CSR 위치 | [n_bus] |
| `diagJ22[bus]` | 버스 bus → J22 대각 CSR 위치 | [n_bus] |
| `pvpq` | pv ∥ pq 연접 인덱스 배열 | [n_pvpq] |

---

## Context 구조체

**파일:** [contexts.hpp](../../cpp/inc/newton_solver/core/contexts.hpp)

### AnalyzeContext

`IStorage::prepare()` 와 `ILinearSolveOp::analyze()`에 전달된다.
`analyze()` 단계에서 **한 번만** 생성된다.

| 필드 | 의미 |
|------|------|
| `ybus` | Ybus CSR view (구조 정보 사용, 값은 무시될 수 있음) |
| `maps` | JacobianBuilder가 생성한 산포 맵 |
| `J` | Jacobian CSR 희소 패턴 |
| `n_bus, pv, pq` | 버스 개수와 버스 유형 인덱스 |

### SolveContext

`IStorage::upload()`에 전달된다. `solve()` 시작 시마다 생성된다.

| 필드 | 의미 |
|------|------|
| `ybus` | 현재 Ybus 값 (FP64) |
| `sbus` | 복소 전력 주입 벡터 (FP64) |
| `V0` | 초기 전압 벡터 (FP64) |
| `config` | NR 수렴 설정 (tolerance, max_iter) |
| `batch_size` | batch 개수. single-case는 1 |
| `sbus_stride`, `V0_stride` | batch-major 입력 stride |
| `ybus_values_batched`, `ybus_value_stride` | batch별 Ybus 값 경로 예약 필드 |

### IterationContext

NR 반복 루프 내 모든 Op가 공유하는 상태다. Op가 읽고 쓴다.

| 필드 | 의미 |
|------|------|
| `storage` | IStorage 참조 (Op가 버퍼에 접근하는 창구) |
| `config` | NR 설정 참조 |
| `iter` | 현재 반복 횟수 (0-based) |
| `normF` | MismatchOp이 계산한 L∞ 노름 |
| `converged` | MismatchOp이 설정하는 수렴 플래그 |

---

## ExecutionPlan

**파일:** [execution_plan.hpp](../../cpp/inc/newton_solver/core/execution_plan.hpp)

한 `NewtonOptions` 프로파일에 대응하는 Storage + Op 집합을 소유한다.

```cpp
struct ExecutionPlan {
    std::unique_ptr<IStorage>         storage;
    std::unique_ptr<IMismatchOp>      mismatch;
    std::unique_ptr<IJacobianOp>      jacobian;
    std::unique_ptr<ILinearSolveOp>   linear_solve;
    std::unique_ptr<IVoltageUpdateOp> voltage_update;
    bool ready = false;
};
```

`PlanBuilder::build()`에서만 생성되며 직접 생성하지 않는다.

---

## PlanBuilder

**파일:** [plan_builder.cpp](../../cpp/src/newton_solver/core/plan_builder.cpp)

`NewtonOptions`를 받아 지원 프로파일 중 하나를 선택하고 `ExecutionPlan`을 조립한다.

| options.backend | options.compute | 빌드 조건 | 선택 함수 |
|-----------------|-----------------|-----------|-----------|
| CPU | FP64 | 항상 | `build_cpu_fp64_plan()` |
| CUDA | FP64 | `WITH_CUDA` | `build_cuda_fp64_plan()` |
| CUDA | Mixed | `WITH_CUDA` | `build_cuda_mixed_plan()` |

지원하지 않는 조합이면 `std::invalid_argument` 예외를 던진다.
