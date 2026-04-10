# Precision Selection Plan

## 1. 현황 점검

### 1.1 초기 요구사항

```
Precision is backend-internal, not a compile-time template parameter.
  - CPU backend  — FP64 throughout
  - CUDA backend — FP32 Jacobian + FP64 voltage (mixed precision, ~2× speedup)
```

현재 요구사항은 "CUDA mixed 고정"에 맞춰져 있다.
사용자가 `fp32 / mixed / fp64`를 고르는 public API는 아직 없다.

### 1.2 현재 구현 상태

| 항목 | 상태 |
|------|------|
| `PrecisionMode` enum | 없음 |
| `NewtonOptions.precision` | 없음 |
| `NRConfig` 정밀도 필드 | 없음 |
| Public API FP32 overload | 없음 |
| CPU backend | FP64 전용 |
| CUDA backend | fixed mixed |
| Python binding dtype 선택 | `complex128`만 지원 |

### 1.3 현재 CUDA mixed 파이프라인

```
computeMismatch   : cuSPARSE SpMV                      [FP64 complex]
updateJacobian    : CUDA kernel                        [FP32 float]
solveLinearSystem : cuDSS                              [FP32 J, FP64->FP32 RHS]
updateVoltage     : CUDA kernel                        [FP64 double]
downloadV         : host copy                          [complex<double>]
```

핵심 제약은 "public API가 전부 `double` / `complex<double>`에 고정"되어 있다는 점이다.
따라서 `FP32`를 진짜 end-to-end FP32로 만들려면 CUDA 내부만 바꾸는 것으로는 부족하고,
`NewtonSolver`, `INewtonSolverBackend`, `NRResult`, pybind binding까지 함께 바뀌어야 한다.

---

## 2. 설계 결정

### 2.1 정밀도 source of truth

정밀도 선택은 `NRConfig`가 아니라 `NewtonOptions::precision`으로 고정한다.

이유:

- 메모리 할당과 cuDSS/cuSPARSE descriptor type은 `analyze()` 이전에 결정되어야 한다.
- `NRConfig`는 `solve()` 시점에만 전달되므로 정밀도 선택 시점이 너무 늦다.
- 같은 `NewtonSolver` 인스턴스는 하나의 정밀도 모드로 생성되고 그 모드를 유지한다.

```cpp
enum class PrecisionMode {
    FP32,
    Mixed,
    FP64,
};

struct NewtonOptions {
    BackendKind         backend       = BackendKind::CPU;
    JacobianBuilderType jacobian      = JacobianBuilderType::EdgeBased;
    int32_t             n_batch       = 1;
    CpuAlgorithm        cpu_algorithm = CpuAlgorithm::Optimized;
    PrecisionMode       precision     = PrecisionMode::FP64;
};
```

### 2.2 각 모드의 의미

| 모드 | Public API 입력/출력 | CUDA 내부 의미 |
|------|----------------------|----------------|
| `FP32` | `float`, `complex<float>` | end-to-end FP32 |
| `Mixed` | `double`, `complex<double>` | Jacobian/solve FP32 + voltage FP64 |
| `FP64` | `double`, `complex<double>` | end-to-end FP64 |

중요한 점:

- `FP32`는 "진짜 end-to-end FP32"를 의미한다.
- `Mixed`는 현재 동작과 동일한 CUDA mixed path를 의미한다.
- `FP64`는 CUDA 내부 버퍼, Jacobian, RHS, solve, voltage state까지 모두 FP64를 의미한다.
- silent cast는 하지 않는다. 모드와 API dtype이 맞지 않으면 에러로 처리한다.

### 2.3 backend 지원 정책

| Backend | 허용 precision | 정책 |
|---------|----------------|------|
| CPU | `FP64`만 허용 | `Mixed`, `FP32`는 생성자에서 예외 |
| CUDA | `FP32`, `Mixed`, `FP64` | 정상 지원 |

CPU는 이번 계획 범위에서 FP32/mixed를 지원하지 않는다.
따라서 `BackendKind::CPU`와 `PrecisionMode::FP32` 또는 `PrecisionMode::Mixed` 조합은
`std::invalid_argument`로 거부한다.

### 2.4 범위 제한: single-case only

이번 구현 범위는 `n_batch == 1`인 single-case solve로 한정한다.

정책:

- precision selection 기능은 multi-batch까지 확장하지 않는다.
- `solve_batch()`와 CUDA batch backend 경로는 이번 작업에서 수정하지 않는다.
- `NewtonOptions.n_batch != 1`이면 precision-aware path에서는 fail-fast로 예외를 던진다.

즉, 이번 문서는 "정밀도 선택 + single-case" 리팩터링 계획이다.

---

## 3. Public API 변경

### 3.1 타입 alias와 결과 타입 분리

현재 `YbusView`와 `NRResult`가 FP64로 고정되어 있으므로 FP32 경로를 위해 별도 타입이 필요하다.

```cpp
using YbusViewF32 = CSRView<std::complex<float>>;
using YbusViewF64 = CSRView<std::complex<double>>;

struct NRResultF32 {
    std::vector<std::complex<float>> V;
    int32_t iterations = 0;
    float   final_mismatch = 0.0f;
    bool    converged = false;
};

struct NRResultF64 {
    std::vector<std::complex<double>> V;
    int32_t iterations = 0;
    double  final_mismatch = 0.0;
    bool    converged = false;
};
```

`NRConfig`는 solver hyperparameter만 유지한다.

```cpp
struct NRConfig {
    double  tolerance = 1e-8;
    int32_t max_iter  = 50;
};
```

### 3.2 `NewtonSolver` overload

public API는 템플릿 public class로 바꾸지 않고, precision별 overload를 둔다.

```cpp
class NewtonSolver {
public:
    explicit NewtonSolver(const NewtonOptions& options = {});

    void analyze(const YbusViewF32& ybus,
                 const int32_t* pv, int32_t n_pv,
                 const int32_t* pq, int32_t n_pq);

    void analyze(const YbusViewF64& ybus,
                 const int32_t* pv, int32_t n_pv,
                 const int32_t* pq, int32_t n_pq);

    void solve(const YbusViewF32& ybus,
               const std::complex<float>* sbus,
               const std::complex<float>* V0,
               const int32_t* pv, int32_t n_pv,
               const int32_t* pq, int32_t n_pq,
               const NRConfig& config,
               NRResultF32& result);

    void solve(const YbusViewF64& ybus,
               const std::complex<double>* sbus,
               const std::complex<double>* V0,
               const int32_t* pv, int32_t n_pv,
               const int32_t* pq, int32_t n_pq,
               const NRConfig& config,
               NRResultF64& result);
};
```

호출 규칙:

- `PrecisionMode::FP32`인 solver는 `F32` overload만 허용한다.
- `PrecisionMode::Mixed`와 `PrecisionMode::FP64`인 solver는 `F64` overload만 허용한다.
- 잘못된 overload를 호출하면 즉시 예외를 던진다.

### 3.3 backend interface 변경

현재 `INewtonSolverBackend`는 `double*` 기반이라 FP32 end-to-end를 표현할 수 없다.
따라서 interface 변경은 필수다.

권장 방식은 precision별 method를 명시적으로 분리하는 것이다.

```cpp
class INewtonSolverBackend {
public:
    virtual ~INewtonSolverBackend() = default;

    virtual void analyze_f32(const YbusViewF32& ybus,
                             const JacobianMaps& maps,
                             const JacobianStructure& J,
                             int32_t n_bus) {}

    virtual void analyze_f64(const YbusViewF64& ybus,
                             const JacobianMaps& maps,
                             const JacobianStructure& J,
                             int32_t n_bus) {}

    virtual void initialize_f32(const YbusViewF32& ybus,
                                const std::complex<float>* sbus,
                                const std::complex<float>* V0) {}

    virtual void initialize_f64(const YbusViewF64& ybus,
                                const std::complex<double>* sbus,
                                const std::complex<double>* V0) {}

    virtual void computeMismatch_f32(const int32_t* pv, int32_t n_pv,
                                     const int32_t* pq, int32_t n_pq,
                                     float* F, float& normF) {}

    virtual void computeMismatch_f64(const int32_t* pv, int32_t n_pv,
                                     const int32_t* pq, int32_t n_pq,
                                     double* F, double& normF) {}

    virtual void updateJacobian() = 0;

    virtual void solveLinearSystem_f32(const float* F, float* dx) {}
    virtual void solveLinearSystem_f64(const double* F, double* dx) {}

    virtual void updateVoltage_f32(const float* dx,
                                   const int32_t* pv, int32_t n_pv,
                                   const int32_t* pq, int32_t n_pq) {}

    virtual void updateVoltage_f64(const double* dx,
                                   const int32_t* pv, int32_t n_pv,
                                   const int32_t* pq, int32_t n_pq) {}

    virtual void downloadV_f32(std::complex<float>* V_out, int32_t n_bus) {}
    virtual void downloadV_f64(std::complex<double>* V_out, int32_t n_bus) {}
};
```

설명:

- CPU backend는 `_f64` 계열만 구현한다.
- CUDA backend는 `_f32`, `_f64` 둘 다 구현한다.
- `Mixed`는 public API 기준으로 `_f64` 계열을 사용하되 내부 구현만 mixed로 간다.

이 구조를 쓰면 `void*` 나 `std::variant`로 interface를 흐리지 않아도 된다.

---

## 4. CUDA 구현 변경

### 4.1 `cuda_backend_impl.hpp`

현재 Impl은 mixed 전용 포인터 집합만 갖고 있다.
정밀도별 버퍼를 명시적으로 분리해야 한다.

추가 방향:

- `PrecisionMode precision_mode`
- FP32 path 전용:
  - `cuFloatComplex* d_Ybus_val_f`
  - `cuFloatComplex* d_V_cf`
  - `float* d_Va_f`, `float* d_Vm_f`
  - `float* d_F_f`, `float* d_dx_f`
  - `float* d_J_csr_f`, `float* d_b_f`, `float* d_x_f`
- FP64 path 전용:
  - `cuDoubleComplex* d_Ybus_val_d`
  - `cuDoubleComplex* d_V_cd`
  - `double* d_Va_d`, `double* d_Vm_d`
  - `double* d_F_d`, `double* d_dx_d`
  - `double* d_J_csr_d`, `double* d_b_d`, `double* d_x_d`
- Mixed path 전용:
  - FP64 voltage/state buffers
  - FP32 Jacobian/solve buffers

중요:

- `FP32`, `Mixed`, `FP64`는 메모리 layout이 다르므로 `analyze()`에서 한 번 결정한 뒤 고정한다.
- 사용하지 않는 포인터는 `nullptr`로 둔다.
- batch 전용 버퍼 설계는 이번 범위에서 제외한다.

### 4.2 `initialize.cpp`

해야 할 일:

- `options_.precision`을 Impl에 저장
- precision별 메모리 할당
- precision별 cuSPARSE descriptor type 설정
- precision별 cuDSS matrix type 설정
- FP32 path에서는 host `complex<float>` 데이터를 그대로 upload
- FP64/mixed path에서는 host `complex<double>` 데이터를 사용

예시:

```cpp
if (precision_mode == PrecisionMode::FP32) {
    // cuSPARSE: CUDA_C_32F
    // cuDSS   : CUDSS_R_32F
}

if (precision_mode == PrecisionMode::FP64) {
    // cuSPARSE: CUDA_C_64F
    // cuDSS   : CUDSS_R_64F
}

if (precision_mode == PrecisionMode::Mixed) {
    // Ybus/V/state: FP64
    // Jacobian/RHS/solve: FP32
}
```

### 4.3 `compute_mismatch.cu`

현재 구현은 FP64 complex SpMV + FP64 mismatch pack만 있다.

추가 필요 경로:

- `computeMismatch_f32()`
  - `cusparseSpMV(... CUDA_C_32F ...)`
  - `mismatch_pack_kernel_fp32(...)`
  - `max_abs_kernel_fp32(...)`
- `computeMismatch_f64()`
  - 기존 FP64 경로 유지
- `Mixed`
  - public API는 FP64이므로 기존 FP64 mismatch 경로 유지

즉, `FP32`는 mismatch 벡터 `F`와 `normF`도 `float`로 계산되고 반환되어야 한다.

### 4.4 `update_jacobian.cu`

현재 파일에는 `updateJacobian()`과 `updateVoltage()`가 함께 들어 있다.
새 파일 `update_voltage.cu`를 만드는 것이 아니라 이 파일 안에서 같이 정리하는 것이 현재 트리와 맞다.

추가 필요 경로:

- FP32 Jacobian kernel
  - existing edge-based FP32
  - existing vertex-based FP32
- FP64 Jacobian kernel
  - edge-based FP64
  - vertex-based FP64
- `Mixed`
  - 현재와 동일하게 FP32 Jacobian 사용

즉, precision case와 `JacobianBuilderType` case가 동시에 존재한다.

| Precision | JacobianBuilderType | 구현 필요 |
|-----------|---------------------|-----------|
| `FP32` | EdgeBased | FP32 edge kernel |
| `FP32` | VertexBased | FP32 vertex kernel |
| `Mixed` | EdgeBased | FP32 edge kernel |
| `Mixed` | VertexBased | FP32 vertex kernel |
| `FP64` | EdgeBased | FP64 edge kernel |
| `FP64` | VertexBased | FP64 vertex kernel |

### 4.5 `updateVoltage` 경로

현재는 FP32 `dx`를 FP64 `Va/Vm`에 적용하는 mixed 전용 구현이다.

새 설계에서는 세 경로가 필요하다.

- `updateVoltage_f32()`
  - `float* Va`, `float* Vm`
  - `float* dx`
  - `cuFloatComplex* V`
- `updateVoltage_f64()`
  - `double* Va`, `double* Vm`
  - `double* dx`
  - `cuDoubleComplex* V`
- `Mixed`
  - 현재 구현 유지

`FP32`는 최종 `downloadV_f32()`까지 `complex<float>`를 내려야 하므로 전압 state 자체를 FP32로 유지해야 한다.

### 4.6 `cudss_solver.cpp`

현재는 `double d_F` -> `float d_b_f` 변환 후 FP32 solve만 지원한다.

추가 필요 경로:

- `solveLinearSystem_f32()`
  - `float* d_F_f` -> `float* d_b_f`
  - cuDSS `CUDSS_R_32F`
  - solution `float* d_x_f`
- `solveLinearSystem_f64()`
  - `double* d_F_d` -> `double* d_b_d`
  - cuDSS `CUDSS_R_64F`
  - solution `double* d_x_d`
- `Mixed`
  - 기존 경로 유지

### 4.7 batch path 제외

multi-batch는 이번 구현 범위에서 제외한다.

정책:

- `initialize_batch`
- `computeMismatch_batch`
- `updateJacobian_batch`
- `solveLinearSystem_batch`
- `updateVoltage_batch`
- `downloadV_batch`

위 함수들은 이번 리팩터링에서 precision-aware하게 확장하지 않는다.

대신:

- `NewtonOptions.n_batch != 1`이면 precision selection 경로를 타지 않도록 막는다.
- 필요하면 후속 문서에서 batch 전용 precision 계획을 별도로 작성한다.

---

## 5. CPU backend 정책

CPU backend는 이번 계획에서 FP64 only로 유지한다.

해야 할 일:

- `cpu_backend.hpp`와 구현 파일에서 `_f64` 계열만 구현
- `_f32` 계열 호출은 `std::logic_error` 또는 `std::invalid_argument`
- `NewtonSolver` 생성 시 `BackendKind::CPU`와 `precision != FP64` 조합을 미리 거부

이렇게 하면 CPU 경로가 조용히 cast하거나 fallback하는 일을 막을 수 있다.

---

## 6. Python binding 변경

현재 pybind binding은 `complex128`만 받는다.
`FP32` end-to-end를 위해 dtype dispatch가 필요하다.
단, 이번 범위는 single-case API만 포함한다.

추가 방향:

- 생성자에서 `precision` 인자 노출
- `analyze()`와 `solve()`에서 `complex64` / `complex128` 둘 다 허용
- `precision`과 dtype 조합 검증
- `solve_batch()`는 이번 범위에서 명시적으로 예외 처리

정책:

- `precision="fp32"`이면 `data`, `sbus`, `V0`는 `complex64`여야 함
- `precision="mixed"` 또는 `precision="fp64"`이면 `complex128`이어야 함
- mismatch와 결과 전압도 precision에 맞는 dtype으로 반환

예시:

```python
solver = cupf.NewtonSolver(backend="cuda", jacobian="edge_based", precision="fp32")
result = solver.solve(... complex64 arrays ...)
assert result["V"].dtype == np.complex64
```

---

## 7. 파일별 변경 목록

| 파일 | 변경 종류 |
|------|-----------|
| `cpp/inc/newton_solver/core/newton_solver_types.hpp` | `PrecisionMode`, `NewtonOptions.precision`, `YbusViewF32/F64`, `NRResultF32/F64` 추가 |
| `cpp/inc/newton_solver/core/newton_solver.hpp` | single-case `analyze/solve` FP32/FP64 overload 추가 |
| `cpp/src/newton_solver/core/newton_solver.cpp` | overload dispatch, precision validation, CPU unsupported 조합 및 `n_batch != 1` 거부 |
| `cpp/inc/newton_solver/backend/i_backend.hpp` | `_f32`, `_f64` interface 추가 |
| `cpp/inc/newton_solver/backend/cpu_backend.hpp` | FP64 계열 interface 구현 선언 |
| `cpp/inc/newton_solver/backend/cuda_backend.hpp` | FP32/FP64 계열 interface 구현 선언 |
| `cpp/src/newton_solver/backend/cuda/cuda_backend_impl.hpp` | precision mode + precision별 device buffer 추가 |
| `cpp/src/newton_solver/backend/cuda/initialize.cpp` | precision별 allocation, descriptor type, upload path |
| `cpp/src/newton_solver/backend/cuda/compute_mismatch.cu` | FP32/FP64/mixed mismatch 경로 |
| `cpp/src/newton_solver/backend/cuda/update_jacobian.cu` | FP32/FP64 Jacobian kernel + updateVoltage FP32/FP64/mixed |
| `cpp/src/newton_solver/backend/cuda/cudss_solver.cpp` | FP32/FP64/mixed solve 경로 |
| `cpp/src/newton_solver/backend/cpp/*` | CPU FP64 interface 적응 + unsupported precision 방어 |
| `bindings/pybind_module.cpp` | `precision` 노출, complex64/complex128 dtype dispatch |
| `README.md` | public API precision 설명 업데이트 |
| `docs/core/types.md` | `YbusViewF32/F64`, `NRResultF32/F64`, `NewtonOptions.precision` 반영 |
| `docs/core/newton_solver.md` | overload와 precision 규칙 반영 |
| `docs/backend/interface.md` | backend `_f32`/`_f64` interface 반영 |
| `docs/backend/cuda.md` | 세 모드 설명 반영 |

주의:

- `src/newton_solver/backend/cuda/cuda_backend.cpp`는 현재 존재하지 않는다.
- `updateVoltage()` 구현은 `src/newton_solver/backend/cuda/update_jacobian.cu` 안에 있다.
- 따라서 기존 문서의 `cuda_backend.cpp`, `update_voltage.cu` 지시는 잘못된 파일명을 사용한 것이다.

---

## 8. 구현 우선순위

### Phase 1: public API와 validation

1. `newton_solver_types.hpp`
2. `newton_solver.hpp`
3. `i_backend.hpp`
4. `newton_solver.cpp`
5. CPU unsupported precision validation
6. `n_batch != 1` fail-fast validation

목표:

- `NewtonOptions::precision` 확정
- FP32/F64 public type 도입
- 잘못된 precision/dtype/backend 조합 즉시 거부
- multi-batch 경로를 이번 범위에서 명시적으로 제외

### Phase 2: CUDA single-case

1. `cuda_backend_impl.hpp`
2. `initialize.cpp`
3. `compute_mismatch.cu`
4. `update_jacobian.cu`
5. `cudss_solver.cpp`

목표:

- `FP32`, `Mixed`, `FP64` single-case solve 구현 완료

### Phase 3: binding, tests, docs

1. `pybind_module.cpp`
2. precision별 Python tests
3. CPU unsupported precision tests
4. `n_batch != 1` rejection tests
5. CUDA numerical regression tests
6. README / docs refresh

---

## 9. requirements.md 업데이트 문안

현재 문구:

```
Precision is backend-internal, not a compile-time template parameter.
  - CPU  — FP64 throughout
  - CUDA — FP32 Jacobian + FP64 voltage (mixed precision, ~2× speedup)
```

변경 후 문구:

```
Precision is selected at solver construction time via NewtonOptions::precision.
There is no compile-time precision parameter on NewtonSolver itself.

  CPU:
    - PrecisionMode::FP64 only
    - PrecisionMode::Mixed / PrecisionMode::FP32 are rejected

  CUDA:
    - PrecisionMode::FP32  : end-to-end FP32 public API and internal pipeline
    - PrecisionMode::Mixed : FP64 public API, FP32 Jacobian/solve + FP64 voltage
    - PrecisionMode::FP64  : end-to-end FP64 public API and internal pipeline

The solver does not silently cast between FP32 and FP64 public APIs.
The requested precision mode must match the input/output dtype used by the caller.
Precision selection is currently defined for single-case solves only (`n_batch == 1`).
```

---

## 10. 요약

이 계획의 핵심 변경점은 두 가지다.

- precision 선택은 `NewtonOptions::precision`에서 고정한다.
- `FP32`는 CUDA 내부 옵션이 아니라 public API까지 포함한 end-to-end FP32로 정의한다.

따라서 이번 작업은 CUDA kernel 몇 개 추가로 끝나는 일이 아니라,
`NewtonSolver`, backend interface, pybind, docs까지 함께 바꾸는 API 리팩터링으로 본다.
