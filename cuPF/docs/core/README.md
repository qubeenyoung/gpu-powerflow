# core — 핵심 컴포넌트 설계

`cpp/inc/newton_solver/core/` 및 `cpp/src/newton_solver/core/` 에 위치한다.

---

## 컴포넌트 목록

| 파일 | 역할 |
|------|------|
| `newton_solver_types.hpp` | 공개 타입: CSRView, NRConfig, BackendKind, ComputePolicy, NewtonOptions, NRResult, NRBatchResult |
| `cpp/src/newton_solver/ops/jacobian/jacobian_analysis.hpp/.cpp` | Ybus 희소 구조 분석 → Jacobian 패턴/산포 맵 생성 |
| `cpp/src/newton_solver/core/solver_contexts.hpp` | InitializeContext, SolveContext, IterationContext |
| `cpp/src/newton_solver/core/solver_stages.cpp` | NewtonOptions → NewtonSolver stage 조립 |
| `newton_solver.hpp/.cpp` | 공개 API: initialize(), solve() 오케스트레이션 |

---

## NewtonSolver

**파일:** [newton_solver.hpp](../../cpp/inc/newton_solver/core/newton_solver.hpp)

### 책임

1. 생성자에서 backend별 Storage/Op stage를 직접 조립한다.
2. `initialize()`로 희소 구조를 분석하고 device-side 상태를 초기화한다.
3. `solve()`/`solve_batch()`로 NR 반복 루프를 실행하고 결과를 반환한다.

### 생명주기

```
NewtonSolver(options)    → configure_stages()
  initialize(ybus, ...)  → make_jacobian_indexing()
                         → JacobianPatternGenerator::generate()
                         → JacobianMapBuilder::build()
                         → IStorage::prepare()
                         → ILinearSolveOp::initialize()
  solve(ybus, sbus, V0, ...) 또는 solve_batch(...)
                         → IStorage::upload()
                         → 반복 루프 (mismatch → jacobian → linear_solve → voltage_update)
                         → IStorage::download_result() / download_batch_result()
```

### 주의사항

- `initialize()`는 Ybus의 **희소 구조**가 바뀌지 않는 한 한 번만 호출하면 된다.
- `solve()`는 Ybus 값(원소값)이 바뀌어도 호출할 수 있다. 구조가 같으면 된다.
- 기본 실행 모델은 batch이고, 기존 `solve()`는 내부적으로 `solve_batch(B=1)` wrapper다.
- 현재 `solve_batch(B>1)` 실제 실행은 CUDA Mixed path에서 지원한다.
- `initialize()` 전에 `solve()`를 호출하면 `std::runtime_error`가 발생한다.

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

`IStorage::prepare()` 와 `ILinearSolveOp::initialize()`에 전달된다.
`initialize()` 단계에서 **한 번만** 생성된다.

| 필드 | 의미 |
|------|------|
| `ybus` | Ybus CSR view (구조 정보 사용, 값은 무시될 수 있음) |
| `maps` | JacobianMapBuilder가 생성한 산포 맵 |
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

## Stage Ownership

**파일:** [newton_solver.hpp](../../cpp/inc/newton_solver/core/newton_solver.hpp)

별도 실행 계획 구조체는 두지 않는다. `NewtonSolver`가 한 `NewtonOptions` 프로파일에
대응하는 Storage + Op 집합을 직접 소유한다.

```cpp
std::unique_ptr<IStorage>         storage_;
std::unique_ptr<IIbusOp>          ibus_;
std::unique_ptr<IMismatchOp>      mismatch_;
std::unique_ptr<IMismatchNormOp>  mismatch_norm_;
std::unique_ptr<IJacobianOp>      jacobian_;
std::unique_ptr<ILinearSolveOp>   linear_solve_;
std::unique_ptr<IVoltageUpdateOp> voltage_update_;
```

---

## Stage Configuration

**파일:** [solver_stages.cpp](../../cpp/src/newton_solver/core/solver_stages.cpp)

`NewtonOptions`를 받아 지원 프로파일 중 하나를 선택하고 `NewtonSolver`의 stage 멤버를 채운다.

| options.backend | options.compute | 빌드 조건 | 선택 함수 |
|-----------------|-----------------|-----------|-----------|
| CPU | FP64 | 항상 | `configure_cpu_fp64_stages()` |
| CUDA | FP64 | `WITH_CUDA` | `configure_cuda_fp64_stages()` |
| CUDA | Mixed | `WITH_CUDA` | `configure_cuda_mixed_stages()` |

지원하지 않는 조합이면 `std::invalid_argument` 예외를 던진다.
