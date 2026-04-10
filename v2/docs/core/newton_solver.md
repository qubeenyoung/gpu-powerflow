# NewtonSolver

**파일**: `inc/newton_solver/core/newton_solver.hpp`  
**구현**: `src/newton_solver/core/newton_solver.cpp`

공개 API 진입점. `JacobianBuilder`와 `INewtonSolverBackend`를 조율해 NR 루프를 실행한다. 백엔드와 Jacobian 빌더 선택만 담당하고, 수치 연산은 전부 백엔드에 위임한다.

---

## 기본 사용법

```cpp
NewtonSolver solver({.backend = BackendKind::CUDA});

solver.analyze(ybus, pv, n_pv, pq, n_pq);   // 한 번만

NRResult result;
solver.solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, result);
```

---

## 메서드

### `analyze()`

```cpp
void analyze(const YbusView& ybus,
             const int32_t* pv, int32_t n_pv,
             const int32_t* pq, int32_t n_pq);
```

- `JacobianBuilder::analyze()` 호출 → `JacobianMaps`, `JacobianStructure` 생성
- `backend_->analyze()` 호출 → 백엔드 메모리/핸들 초기화
- 이후 같은 위상 구조(`pv`/`pq` 집합)에서는 재호출 불필요

---

### `solve()`

```cpp
void solve(const YbusView& ybus,
           const std::complex<double>* sbus,
           const std::complex<double>* V0,
           const int32_t* pv, int32_t n_pv,
           const int32_t* pq, int32_t n_pq,
           const NRConfig& config,
           NRResult& result);
```

내부 NR 루프:

```
initialize(ybus, sbus, V0)
while iter < config.max_iter:
    computeMismatch(pv, n_pv, pq, n_pq, F, normF)
    if normF < config.tolerance → 수렴, break
    updateJacobian()
    solveLinearSystem(F, dx)
    updateVoltage(dx, pv, n_pv, pq, n_pq)
    iter++
downloadV(result.V)
```

`F`, `dx`는 `solve()` 로컬 버퍼 (`double`, `dimF = n_pv + 2*n_pq`). 실제 연산은 백엔드 내부에서 이루어진다.

---

### `solve_batch()`

```cpp
void solve_batch(const YbusView& ybus,
                 const std::complex<double>* sbus_batch,  // [n_batch × n_bus]
                 const std::complex<double>* V0_batch,    // [n_batch × n_bus]
                 const int32_t* pv, int32_t n_pv,
                 const int32_t* pq, int32_t n_pq,
                 int32_t n_batch,
                 const NRConfig& config,
                 NRResult* results);                       // [n_batch]
```

같은 위상 구조에서 `n_batch`개의 독립 케이스를 순차적으로 `solve()` 호출. CUDA UBATCH 병렬화는 향후 작업.

---

## 내부 멤버

| 멤버 | 타입 | 역할 |
|------|------|------|
| `jac_builder_` | `JacobianBuilder` | 희소 패턴 분석 |
| `jac_maps_` | `JacobianMaps` | analyze 결과, solve에서 재사용 |
| `J_` | `JacobianStructure` | Jacobian CSR 구조 |
| `backend_` | `unique_ptr<INewtonSolverBackend>` | CPU 또는 CUDA 백엔드 |
| `analyzed_` | `bool` | analyze 호출 여부 가드 |
