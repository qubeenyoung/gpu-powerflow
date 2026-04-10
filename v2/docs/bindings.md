# Python 바인딩

**파일**: `bindings/pybind_module.cpp`  
**모듈명**: `cupf`

pybind11로 `NewtonSolver`를 Python에 노출한다. numpy 배열을 `YbusView`로 zero-copy 변환한 뒤 C++ API를 그대로 호출한다.

---

## 설치 후 사용

```python
import cupf
import numpy as np

solver = cupf.NewtonSolver(backend="cpu", jacobian="edge_based")

solver.analyze(indptr, indices, data, rows, cols, pv, pq)

result = solver.solve(
    indptr, indices, data, rows, cols,
    sbus, V0, pv, pq,
    tolerance=1e-8, max_iter=50
)
# result["V"]              → np.ndarray[complex128], shape (n_bus,)
# result["iterations"]     → int
# result["final_mismatch"] → float
# result["converged"]      → bool

results = solver.solve_batch(
    indptr, indices, data, rows, cols,
    sbus_batch, V0_batch, pv, pq, n_batch
)
# results → list of dict (위와 동일 구조)
```

---

## 입력 배열 타입

| 인자 | dtype | shape |
|------|-------|-------|
| `indptr` | `int32` | `(rows+1,)` |
| `indices` | `int32` | `(nnz,)` |
| `data` | `complex128` | `(nnz,)` |
| `sbus` | `complex128` | `(n_bus,)` |
| `V0` | `complex128` | `(n_bus,)` |
| `pv`, `pq` | `int32` | `(n_pv,)`, `(n_pq,)` |
| `sbus_batch` | `complex128` | `(n_batch * n_bus,)` |

---

## 백엔드 / Jacobian 옵션

```python
# backend
"cpu"   → CpuNewtonSolverBackend
"cuda"  → CudaNewtonSolverBackend  (WITH_CUDA 빌드 필요)

# jacobian
"edge_based"   → JacobianBuilderType::EdgeBased
"vertex_based" → JacobianBuilderType::VertexBased

현재 동작:
- CPU backend: edge-based와 동일한 분석 결과를 사용
- CUDA backend: vertex-based Jacobian fill kernel 사용
```

---

## `PyNewtonSolver` 래퍼

바인딩 내부에서 `PyNewtonSolver` 구조체가 `NewtonSolver`를 소유하고, numpy 배열 → `YbusView` 변환을 담당한다.

```
PyNewtonSolver
    └── NewtonSolver solver
    └── make_ybus_view(indptr, indices, data, rows, cols)
          → YbusView (포인터만 참조, zero-copy)
```

numpy 배열은 pybind11이 함수 호출 동안 alive 보장.
