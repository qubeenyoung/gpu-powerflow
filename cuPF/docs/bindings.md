# bindings — Python 바인딩 (_cupf)

cuPF를 파이썬 모듈 `_cupf`로 노출한다. 일반 경로는 numpy 입출력, 선택적 torch 경로는
device 텐서 zero-copy.

**소스**: `bindings/pybind_cupf.cpp` (pybind11 모듈), `bindings/torch_cupf_extension.cpp`
(torch zero-copy 확장, `CUPF_WITH_TORCH`에서만 빌드/노출).

---

## 1. 모듈 표면 (`pybind_cupf.cpp`)

- **enum/구조체**: `BackendKind`, `ComputePolicy`, `CudaLinearSolverKind`,
  `Cpu/CudaJacobianKind`, `CpuLinearSolverKind`, `AdjointCacheMode`, `CuDSSAlgorithm`;
  옵션 `NRConfig`/`CuDSSOptions`/`NewtonOptions`/`SolveOptions`/`AdjointOptions`;
  결과 `NRResult`/`NRBatchResult`/`AdjointResult`. 의미는 [newton_solver.md](newton_solver.md).
- **`NewtonSolver`** 메서드: `initialize`, `solve`, `solve_batch`, `solve_adjoint`
  (+`CUPF_WITH_TORCH` 시 `solve_torch`/`solve_with_adjoint_cache_torch`/`solve_adjoint_torch`).

### 메모리·소유권 계약

- 배열 인자는 `py::array_t<..., c_style | forcecast>` — 임의 numpy를 받되 dtype/레이아웃이
  다르면 호출 직전 연속 복사본 생성(`.data()` 포인터는 호출 동안만 유효).
- CSR Ybus는 `make_ybus_view`로 **borrowing `YbusView`**(zero-copy). caller 배열이 살아
  있는 동안만 유효한데, 동기 solve가 내부에서 즉시 업로드/복사하므로 안전.
- 결과는 `*_to_numpy` 헬퍼가 솔버의 `std::vector`를 **owning numpy로 memcpy**해 반환 —
  결과 객체보다 오래 생존. `*_numpy` 프로퍼티가 형상까지 맞춘 배열을 준다(배치 결과는
  batch-major `[batch, dim]`로 reshape).

### `lambda` 주의

`AdjointResult.lambda`는 파이썬 예약어라 `result.lambda`로 직접 접근 불가. **`lambda_`**
(키워드-안전 별칭) 또는 `lambda_numpy`, 혹은 `getattr(result, "lambda")`를 쓴다.

---

## 2. torch zero-copy 확장 (`torch_cupf_extension.cpp`)

torch CUDA 텐서를 호스트 복사 없이 처리한다. 각 바인딩은:

1. 텐서 검증 — `TORCH_CHECK`로 CUDA 상주·contiguous·dtype·shape·device 일치.
   `[batch, n_bus]`(load/grad/출력)와 `[n_bus]`(base power·V0, 배치 공통).
2. torch의 **현재 CUDA 스트림**에 cuPF를 바인딩(`ScopedCudaStream`) — 같은 스트림에서 실행.
3. raw device 포인터를 `cupf::torch_api` forward/backward로 전달([core.md](core.md) torch bridge),
   결과를 zero-copy로 태깅.

dtype은 compute policy와 일치해야 한다: **FP64 → `torch.float64`, FP32/Mixed →
`torch.float32`** (Mixed는 비-torch 경로의 FP64 complex 입력과 달리 float32 텐서를 받는다).
FP32/Mixed/FP64 모두 batch>1 지원.

`solve_torch`(=forward) → (옵션) `solve_options.prepare_adjoint_cache`로 backward용 캐시
준비 → `solve_adjoint_torch`로 load gradient를 출력 텐서에 직접 기록.
