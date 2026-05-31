# variants — 실행 프로파일 설계

cuPF는 `NewtonOptions`를 통해 실행 프로파일을 선택한다.
각 프로파일은 Storage + stage Op의 고정된 조합인 pipeline struct이고,
`NewtonSolver` 생성자가 `NewtonOptions`(backend·compute)를 보고 `SolverPipeline`
variant에 해당 pipeline을 채워 조립한다([core/README.md](../core/README.md) 참조).

빌드에 항상 포함되는 프로파일은 CPU FP64, CUDA FP64, CUDA FP32, CUDA Mixed이고,
`CUPF_ENABLE_CUSTOM_SOLVER=ON`이면 CUDA FP64(custom solver) 프로파일이 추가된다.

---

## 프로파일 선택 방법

```cpp
// CPU FP64 (기본값)
NewtonSolver solver;

// CUDA FP64
NewtonSolver solver({
    .backend         = BackendKind::CUDA,
    .compute         = ComputePolicy::FP64,
});

// CUDA FP32
NewtonSolver solver({
    .backend         = BackendKind::CUDA,
    .compute         = ComputePolicy::FP32,
});

// CUDA Mixed
NewtonSolver solver({
    .backend         = BackendKind::CUDA,
    .compute         = ComputePolicy::Mixed,
});
```

---

## 프로파일 1: CPU FP64 (`CpuFp64Pipeline`)

### 구성

| Stage | 구현체 | 라이브러리 |
|-------|--------|-----------|
| Storage | `CpuFp64Storage` | Eigen, SuiteSparse KLU |
| Ibus | `CpuIbusOp` | Eigen SpMV |
| Mismatch | `CpuMismatchOp` / `CpuMismatchNormOp` | Eigen |
| Jacobian | `CpuJacobianOpF64` | 직접 계산 (edge-based) |
| Linear Solve | `CpuLinearSolveKLU` | Eigen::KLU |
| Voltage Update | `CpuVoltageUpdateOp` | std::complex |

### 특징

- 모든 연산이 host CPU에서 실행된다.
- CUDA 의존성이 없으므로 CUDA 없이 빌드 가능하다.
- 개발·디버깅·정확도 검증의 기준선으로 사용한다.
- Ibus 캐시를 활용해 Jacobian에서 SpMV를 생략할 수 있다.
- `batch_supported == false` — single-case(B=1) 경로다.

### Jacobian 분석/Fill

CPU는 `JacobianPatternGenerator`/`JacobianMapBuilder`가 만든 edge-based 산포 맵을
사용해 Jacobian을 채운다. 버스별 루프이지만 CUDA 커널처럼 warp 단위 최적화가 없다.

---

## 프로파일 2: CUDA FP64 (`CudaFp64Pipeline`)

### 구성

| Stage | 구현체 | 라이브러리 |
|-------|--------|-----------|
| Storage | `CudaFp64Storage` | — |
| Ibus | `CudaIbusOp<double>` | CUDA 커널 |
| Mismatch | `CudaMismatchOp` | CUDA 커널 |
| Jacobian | `CudaJacobianOp<double>` | CUDA 커널 |
| Linear Solve | `CudaLinearSolveCuDSS<double, CudaFp64Storage>` | cuDSS (FP64) |
| Voltage Update | `CudaVoltageUpdateOp<double>` | CUDA 커널 |

### 특징

- Ybus, J, F, dx, V 모두 device 메모리에 상주한다.
- mismatch norm은 device reduction 후 norm vector만 host로 내린다.
- cuDSS RHS는 `d_F`를 직접 가리킨다. 공통 부호 convention은 `J * dx = F`, `state -= dx`다.
- Jacobian은 edge one-pass CUDA 커널 하나를 사용한다.
- 수렴 안정성이 CPU FP64와 동일하다.
- `batch_supported == false`.

> `CUPF_ENABLE_CUSTOM_SOLVER=ON` 빌드에서는 cuDSS 대신 자체 CUDA solver를 쓰는
> `CudaFp64CustomPipeline`을 선택할 수 있다. 그 외 stage 구성은 동일하다.

---

## 프로파일 3: CUDA FP32 (`CudaFp32Pipeline`)

`Ybus/V/Ibus/Sbus/F`까지 포함해 storage 값 버퍼를 전부 FP32로 둔 full-FP32 경로다.
Jacobian·linear solve도 FP32(`CudaJacobianOp<float>`,
`CudaLinearSolveCuDSS<float, CudaFp32Storage>`)이며, `batch_supported == true`로
`solve_batch(B>1)`을 지원한다. public I/O는 여전히 FP64이며 upload/download 경계에서 변환한다.
정밀도가 가장 낮으므로 수렴 안정성이 떨어질 수 있고, 처리량/메모리 우선 ablation 용도다.

---

## 프로파일 4: CUDA Mixed (`CudaMixedPipeline`)

### 구성

| Stage | 구현체 | 정밀도 |
|-------|--------|--------|
| Storage | `CudaMixedStorage` | 혼합 |
| Ibus | `CudaIbusOp<double>` | FP64 Ybus/V → FP64 Ibus |
| Mismatch | `CudaMismatchOp` | FP64 Ybus/V/Ibus/Sbus/F |
| Jacobian | `CudaJacobianOp<float>` | **FP32** |
| Linear Solve | `CudaLinearSolveCuDSS<float, CudaMixedStorage>` | **FP32 uniform batch** |
| Voltage Update | `CudaVoltageUpdateOp<float>` | FP64 Va/Vm / FP64 mismatch V cache / FP32 dx |

Mixed 프로파일은 **고정 구성**이다. stage별 자유 조합이 아니다.
`batch_supported == true`로 `solve_batch(B>1)`을 지원한다.

### 정밀도별 상세

```
d_Ybus       (FP64) → Ibus custom CSR kernel 입력
d_Sbus       (FP64) → mismatch 입력
d_Ibus       (FP64) → mismatch 결과, Jacobian diagonal correction 입력
d_Va/Vm      (FP64) → authoritative 전압 상태
d_V_re/im    (FP64) → mismatch 입력, Jacobian voltage 입력
d_F          (FP64) → normF 수렴 판정 정밀도 유지
d_J_values   (FP32) → Jacobian fill + cuDSS FP32 입력
d_dx         (FP32) → cuDSS FP32 출력
```

### FP32 Jacobian 커널

Mixed FP32 Jacobian은 FP64 `d_Ybus_*`, `d_V_re/im`, `d_Vm`을 읽어 커널 안에서
`float`로 변환한다. edge path는 one-pass로 동작한다. Ybus entry 담당 thread가
off-diagonal/self term을 direct write하고, diagonal Ybus entry thread가 같은 kernel에서
FP64 `d_Ibus_*`를 읽어 FP32로 변환한 diagonal correction을 더한다.

`d_J_values.memsetZero()`는 아직 유지하며, 모든 entry write coverage를 검증한 뒤 제거한다.

### Mixed 사용 시 고려사항

- FP32 Jacobian은 수치적으로 ill-conditioned한 계통에서 수렴이 느려지거나 실패할 수 있다.
- 표준 전력계통 계산(normal operating conditions)에서는 FP64와 동일한 반복 횟수에 수렴한다.
- `F`와 `Va/Vm`이 FP64이므로 수렴 판정과 public output은 FP64 기준을 유지한다.
- `J/dx`가 FP32이므로 민감한 계통은 별도 실패 케이스로 추적한다.

---

## 프로파일 비교 요약

| 항목 | CPU FP64 | CUDA FP64 | CUDA FP32 | CUDA Mixed |
|------|----------|-----------|-----------|------------|
| 최소 의존성 | Eigen, KLU | + CUDA, cuDSS | + CUDA, cuDSS | + CUDA, cuDSS |
| Jacobian 정밀도 | FP64 | FP64 | FP32 | FP32 |
| Solve 정밀도 | FP64 | FP64 | FP32 | FP32 |
| 전압 상태 | FP64 | FP64 | FP32 | Va/Vm·V cache FP64 |
| batch (B>1) | 불가 | 불가 | 가능 | 가능 |
| 수렴 안정성 | 기준 | CPU FP64와 동일 | 낮음(ablation) | 대부분 동일, ill-cond 주의 |
| 적합한 용도 | 개발·검증 | 정확도 우선 GPU | 처리량/메모리 ablation | 처리량 우선 GPU |

---

## 빌드 옵션과 프로파일의 관계

| CMake 옵션 | 의미 | 영향 |
|-----------|------|------|
| `WITH_CUDA=ON` | CUDA backend 활성화 | CUDA FP64/FP32/Mixed 사용 가능 |
| `CUPF_ENABLE_CUSTOM_SOLVER=ON` | 자체 CUDA FP64 solver | `CudaFp64CustomPipeline` 추가 (기본 OFF) |
| `CUPF_CUDSS_REORDERING_ALG` | cuDSS 재순서화 알고리즘 | CUDA solve 성능에 영향 |
| `CUPF_CUDSS_ENABLE_MT` | cuDSS 멀티스레드 | CPU 병렬 인수분해 |
| `CUPF_CUDSS_ND_NLEVELS` | Nested Dissection 레벨 | 대형 계통에서 fill-in 감소 |
| `ENABLE_TIMING=ON` | ScopedTimer 활성화 | 타이밍 수집, 각 stage별 wall-clock |
| `ENABLE_NVTX=ON` | NVTX range 활성화 | Nsight Systems 프로파일링 |

cuDSS matching과 pivot epsilon은 빌드 옵션이 아니라 `NewtonOptions.cudss` 런타임 설정으로 전달한다.
