# core — 핵심 컴포넌트 설계

`cpp/inc/newton_solver/core/` 및 `cpp/src/newton_solver/core/` 에 위치한다.

---

## 컴포넌트 목록

| 파일 | 역할 |
|------|------|
| `cpp/inc/newton_solver/core/newton_solver_types.hpp` | 공개 타입: CSRView/YbusView, NRConfig, SolveOptions, BackendKind, ComputePolicy, CuDSSOptions, NewtonOptions, NRResult, NRBatchResult, AdjointOptions, AdjointResult |
| `cpp/inc/newton_solver/core/newton_solver.hpp` | 공개 API: NewtonSolver(initialize/solve/solve_batch/solve_adjoint) |
| `cpp/src/newton_solver/core/newton_solver.cpp` | NewtonSolver 구현: pipeline 조립 + 생명주기 오케스트레이션 |
| `cpp/src/newton_solver/core/pipeline.hpp` | 프로파일별 pipeline struct와 `SolverPipeline` variant 정의 |
| `cpp/src/newton_solver/core/solver_contexts.hpp` | InitializeContext, SolveContext, IterationContext, AdjointCache |
| `cpp/src/newton_solver/ops/jacobian/jacobian_analysis.hpp/.cpp` | Ybus 희소 구조 분석 → Jacobian 패턴/산포 맵 생성 |
| `cpp/src/newton_solver/core/newton_solver_adjoint*.{hpp,cpp}` | adjoint(역전파) 경로 및 수치 헬퍼 |
| `cpp/src/newton_solver/core/csr_transpose.{hpp,cpp}` | CSR→CSC(전치) 희소 패턴 유틸 (adjoint·cuDSS 공유) |

> stage 조립은 별도 `solver_stages.cpp`가 아니라 `newton_solver.cpp`의 생성자에서
> 직접 수행한다(과거의 tombstone `solver_stages.cpp`는 제거됨).

---

## NewtonSolver

**파일:** [newton_solver.hpp](../../cpp/inc/newton_solver/core/newton_solver.hpp)

### 책임

1. 생성자에서 `NewtonOptions`를 보고 backend/precision에 맞는 pipeline을 조립한다.
2. `initialize()`로 희소 구조를 분석하고 device-side 상태를 초기화한다.
3. `solve()`/`solve_batch()`로 NR 반복 루프를 실행하고 결과를 반환한다.
4. `solve_adjoint()`로 정착점 기준 gradient를 계산한다.

`NewtonSolver`는 backend별 구체 로직을 직접 갖지 않는다. opaque `SolverPipeline`
(forward-declare)에 `std::visit`로 위임한다.

### 생명주기

```
NewtonSolver(options)    → options.backend/compute 분기로 pipeline_ = SolverPipeline{...Pipeline{}}
  initialize(ybus, ...)  → make_jacobian_indexing()
                         → JacobianPatternGenerator::generate()
                         → JacobianMapBuilder::build()
                         → pipeline.buf.prepare(InitializeContext)
                         → pipeline.linear_solve.initialize(buf, ...)
  solve(...) / solve_batch(...)
                         → pipeline.buf.upload(SolveContext)
                         → 반복 루프 (ibus → mismatch → mismatch_norm → jacobian
                                       → prepare_rhs → factorize → solve → voltage_update)
                         → pipeline.download_batch(NRBatchResult)
```

### 주의사항

- `initialize()`는 Ybus의 **희소 구조**가 바뀌지 않는 한 한 번만 호출하면 된다.
- `solve()`는 Ybus 값(원소값)이 바뀌어도 호출할 수 있다. 구조가 같으면 된다.
- 기존 `solve()`는 내부적으로 `batch_size=1` 경로다.
- `solve_batch(B>1)` 실제 실행은 `batch_supported == true`인 pipeline
  (CUDA FP32, CUDA Mixed)에서 지원한다. 그 외 pipeline은 `B=1` 호환 경로다.
- `initialize()` 전에 `solve()`를 호출하면 예외가 발생한다.

---

## Jacobian Analysis

**파일:** [jacobian_analysis.hpp](../../cpp/src/newton_solver/ops/jacobian/jacobian_analysis.hpp)

### 책임

`initialize()` 호출 시 Ybus 희소 구조를 분석해 다음 두 가지를 만든다.

1. **`JacobianPattern`** — Jacobian CSR 희소 패턴 (row_ptr, col_idx)
2. **`JacobianScatterMap`** — Ybus 원소 k → J.values 위치를 기록한 산포 맵 테이블

CPU/GPU fill은 같은 패턴과 맵을 공유한다.

### 내부 구현

```
1. pvpq = pv ∥ pq 구성, 역방향 인덱스 맵 생성
2. Ybus 비영 원소마다 최대 4개 J triplet 등록 (Eigen)
3. Eigen CSC 조립 → CSC→CSR 변환 → JacobianPattern
4. Ybus 원소 k마다 find_coeff_index()로 J CSR 위치 기록 → JacobianScatterMap
```

자세한 설명은 [math.md](../math.md#4-jacobian-희소-구조) 참조.

---

## JacobianScatterMap / JacobianPattern

**파일:** [jacobian_analysis.hpp](../../cpp/src/newton_solver/ops/jacobian/jacobian_analysis.hpp)

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

**파일:** [solver_contexts.hpp](../../cpp/src/newton_solver/core/solver_contexts.hpp)

### InitializeContext

pipeline의 `buf.prepare()`와 `linear_solve.initialize()`에 전달된다.
`initialize()` 단계에서 **한 번만** 생성된다.

| 필드 | 의미 |
|------|------|
| `ybus` | Ybus CSR view (`YbusView`, 구조 정보 사용) |
| `maps` | JacobianScatterMap 산포 맵 |
| `J` | JacobianPattern (Jacobian CSR 희소 패턴) |
| `n_bus, pv, n_pv, pq, n_pq` | 버스 개수와 버스 유형 인덱스 |

### SolveContext

pipeline의 `buf.upload()`에 전달된다. `solve()` 시작 시마다 생성된다.

| 필드 | 의미 |
|------|------|
| `ybus` | 현재 Ybus 값 view 포인터 (FP64) |
| `sbus` | 복소 전력 주입 벡터 (FP64) |
| `V0` | 초기 전압 벡터 (FP64) |
| `config` | NR 수렴 설정 (tolerance, max_iter) |
| `batch_size` | batch 개수. single-case는 1 |
| `sbus_stride`, `V0_stride` | batch-major 입력 stride |
| `ybus_values_batched`, `ybus_value_stride` | batch별 Ybus 값 경로 예약 필드 |

### IterationContext

NR 반복 루프 내 모든 stage Op가 공유하는 상태다. Op가 읽고 쓴다.
Storage 참조는 들어 있지 않다 — Storage는 각 stage 호출에 별도 인자로 전달되고,
`IterationContext`는 pipeline과 독립적인 순수 상태만 보유한다.

| 필드 | 의미 |
|------|------|
| `config` | NR 설정 참조 |
| `pv, n_pv, pq, n_pq` | 버스 유형 인덱스 |
| `iter` | 현재 반복 횟수 (0-based) |
| `normF` | mismatch_norm stage가 계산한 L∞ 노름 |
| `converged` | mismatch_norm stage가 설정하는 수렴 플래그 |
| `jacobian_updated_this_iter`, `jacobian_age` | Jacobian 재사용/갱신 추적 |

---

## SolverPipeline (stage 소유 구조)

**파일:** [pipeline.hpp](../../cpp/src/newton_solver/core/pipeline.hpp)

별도 가상 인터페이스나 실행 계획 구조체를 두지 않는다. 각 프로파일은 자기 Storage와
stage Op를 **값으로** 소유하는 한 개의 pipeline struct이고, 전체는 `std::variant`로 묶인다.

```cpp
struct CpuFp64Pipeline {
    CpuFp64Storage    buf;
    CpuLinearSolveKLU linear_solve;
    AdjointCache      adjoint_cache;
    // ibus()/mismatch()/mismatch_norm()/jacobian()/prepare_rhs()/
    // factorize()/solve()/voltage_update() 멤버로 stage Op를 호출
    static constexpr bool batch_supported = false;
};

struct SolverPipeline {
    std::variant<
        CpuFp64Pipeline
#ifdef CUPF_WITH_CUDA
        , CudaFp64Pipeline
#ifdef CUPF_ENABLE_CUSTOM_SOLVER
        , CudaFp64CustomPipeline
#endif
        , CudaFp32Pipeline
        , CudaMixedPipeline
#endif
    > v;
};
```

`NewtonSolver`는 `SolverPipeline`을 `std::unique_ptr`로 들고 `std::visit`로 실행한다.
`batch_supported`는 pipeline별 compile-time 플래그로, `solve_batch(B>1)` 허용 여부를 결정한다.

---

## Pipeline 선택 (stage 조립)

**파일:** [newton_solver.cpp](../../cpp/src/newton_solver/core/newton_solver.cpp) (생성자)

생성자가 `NewtonOptions`를 받아 지원 프로파일 중 하나를 골라 `pipeline_`를 채운다.

| options.backend | options.compute | 빌드 조건 | 선택되는 pipeline |
|-----------------|-----------------|-----------|-------------------|
| CPU | FP64 | 항상 | `CpuFp64Pipeline` |
| CUDA | FP64 | `WITH_CUDA` | `CudaFp64Pipeline` (또는 custom solver 지정 시 `CudaFp64CustomPipeline`) |
| CUDA | FP32 | `WITH_CUDA` | `CudaFp32Pipeline` |
| CUDA | Mixed | `WITH_CUDA` | `CudaMixedPipeline` |

`CudaFp64CustomPipeline`은 `CUPF_ENABLE_CUSTOM_SOLVER`로 빌드된 경우에만 선택 가능하다.
CUDA backend를 요청했지만 CUDA 없이 빌드된 경우, 또는 지원하지 않는 조합이면
`std::invalid_argument` 예외를 던진다.
