# newton_solver — 공개 API와 생애주기

`NewtonSolver`는 cuPF의 진입점이자 백엔드-무관 드라이버다. 옵션으로 파이프라인
프로파일을 고르고, 초기화 → solve → (선택) adjoint의 생애주기를 오케스트레이션한다.
백엔드별 수학은 `ops/`에, 파이프라인 묶음은 `core/`에 있다([core.md](core.md)).

**소스**: `cpp/inc/newton_solver/core/newton_solver.hpp` (API),
`cpp/src/newton_solver/core/newton_solver.cpp` (드라이버),
`cpp/inc/newton_solver/core/newton_solver_types.hpp` (옵션/결과 타입).

---

## 1. 생애주기

```cpp
NewtonSolver solver(options);                 // 프로파일 선택 (생성자)
solver.initialize(ybus, pv, n_pv, pq, n_pq);  // 심볼릭 분석 + 버퍼 준비 (1회)
solver.solve(...) / solver.solve_batch(...);  // NR 풀이
solver.solve_adjoint(...);                    // (선택) 역전파 gradient
```

- **생성자** `NewtonSolver(const NewtonOptions&)`: `backend`/`compute`/solver kind로
  파이프라인 variant를 만든다(아래 §4). 잘못된 조합은 던진다.
- **`initialize(ybus, pv, pq)`**: Jacobian 심볼릭 분석(패턴 + scatter map,
  [ops/jacobian.md](ops/jacobian.md))을 1회 수행하고 `pipeline.initialize()`로 토폴로지를
  업로드한다. Ybus 희소 구조가 고정되는 시점.
- **`solve` / `solve_batch`**: 아래 §2.
- **`solve_adjoint`**: 아래 §3.

`pv`/`pq`(PV·PQ 버스 인덱스)는 initialize에 준 것과 이후 호출에서 일치해야 한다.

---

## 2. Forward solve

```cpp
void solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, solve_options, NRResult&);
void solve_batch(ybus, sbus, sbus_stride, V0, V0_stride, batch_size,
                 pv, n_pv, pq, n_pq, config, solve_options, NRBatchResult&);
```

- `solve`는 `solve_batch(batch_size=1)`의 단일-케이스 래퍼다.
- `solve_batch`: 입력 업로드(정밀도 변환·배치 resize) → NR 루프 → (옵션)adjoint 캐시
  → 결과 다운로드. 배치 레이아웃은 batch-major([system_architecture](system_architecture.md) §4).
- 입력 `sbus`/`V0`는 호스트 FP64 complex. 케이스별 stride로 더 넓은 텐서의 행을 가리킬
  수 있다(`*_stride ≥ n_bus`).
- `batch_size > 1`은 CUDA FP32/Mixed/FP64에서 지원(CPU는 B=1, 위반 시 던짐).

`NRConfig`: `tolerance`(수렴 기준, 기본 1e-8), `max_iter`(기본 50).
`SolveOptions`: `prepare_adjoint_cache`(종료 시 adjoint factorization 캐시),
`adjoint_cache_mode`, `allow_explicit_transpose_fallback`(cuDSS는 transpose solve가
없어 explicit J^T 캐시가 필요).

### 결과

- `NRResult`: `V[n_bus]`, `iterations`, `final_mismatch`, `converged`.
- `NRBatchResult`: `V[batch*n_bus]`(batch-major), `n_bus`, `batch_size`,
  케이스별 `iterations[]` / `final_mismatch[]` / `converged[]`.

---

## 3. Adjoint (backward) solve

```cpp
void solve_adjoint(grad_va, grad_va_stride, grad_vm, grad_vm_stride, batch_size,
                   pv, n_pv, pq, n_pq, AdjointOptions&, AdjointResult&);
```

수렴 상태에서 `J^T λ = dL/dx`를 풀고 λ를 부하 gradient로 사영한다(부호: load
gradient = −λ). `batch_size`는 직전 forward와 일치해야 한다. cuDSS 백엔드는 native
transpose solve가 없어 forward 때 캐시한(또는 fallback으로 재구성한) **explicit J^T
factorization**을 쓴다. CPU/KLU·UMFPACK은 같은 LU로 J^T를 직접 푼다.

`AdjointOptions`: `compute_load_gradients`, `check_residual`(J^T λ 상대 잔차),
`allow_refactorize`/`allow_refactorize_for_backward`(캐시 미스 시 재인수분해 허용),
`require_cached_factorization`, `allow_explicit_transpose_fallback`.

`AdjointResult`: `lambda[batch*dimF]`(파이썬에선 예약어라 `lambda_`/`lambda_numpy`로
접근), `grad_load_p`/`grad_load_q[batch*n_bus]`, 그리고 backend·캐시·타이밍 등 다수의
provenance 플래그(어떻게 J^T를 얻었는지 진단용 — [core.md](core.md)의 `AdjointCache`).

자세한 수학·캐시 동작은 [core.md](core.md) §adjoint, [ops/linear_solve.md](ops/linear_solve.md).

---

## 4. 파이프라인 선택 (생성자 로직)

`NewtonOptions`:

| 필드 | 값 |
|------|----|
| `backend` | `CPU` / `CUDA` |
| `compute` | `FP64` / `FP32` / `Mixed` |
| `cpu_jacobian` | `Native` / `Pandapower` |
| `cpu_linear_solver` | `KLU` / `UMFPACK` |
| `cuda_jacobian` | `Edge` / `EdgeAtomic` / `VertexWarp` ([ops/jacobian.md](ops/jacobian.md)) |
| `cuda_linear_solver` | `CuDSS` / `Custom` |
| `cudss` | `CuDSSOptions`(matching·pivot 등) |

생성자는 (backend, compute)로 5개 파이프라인 중 하나를 선택한다: `CpuFp64Pipeline`,
`CudaFp64Pipeline`, `CudaFp32Pipeline`, `CudaMixedPipeline`(+빌드 시 `CudaFp64CustomPipeline`).
조합이 맞지 않으면(예: CUDA 미빌드인데 CUDA 요청, custom인데 비-FP64) 예외.

---

## 5. torch zero-copy 경로

`solve_torch_forward` / `solve_torch_backward`는 torch CUDA 텐서의 device 포인터를
받아 호스트 복사 없이 현재 CUDA 스트림에서 푼다. 파이썬 바인딩은
`solve_torch`(=forward, alias) / `solve_adjoint_torch`이며 `CUPF_WITH_TORCH=ON`에서만
노출된다. 입력은 `[batch, n_bus]`(load/출력)와 `[n_bus]`(base power·V0, 배치 공통),
텐서 dtype은 프로파일과 일치(FP64→float64, FP32/Mixed→float32). FP32/Mixed/FP64 모두
batch>1 지원. 구현은 [core.md](core.md)의 torch bridge, 바인딩은 [bindings.md](bindings.md).
