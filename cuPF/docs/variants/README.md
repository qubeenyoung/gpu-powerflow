# variants — 실행 프로파일 설계

cuPF는 `NewtonOptions`를 통해 세 가지 실행 프로파일을 지원한다.
각 프로파일은 Storage + 4 Op의 고정된 조합이며, `solver stage configuration::build()`에서 조립된다.

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

// CUDA Mixed
NewtonSolver solver({
    .backend         = BackendKind::CUDA,
    .compute         = ComputePolicy::Mixed,
});
```

---

## 프로파일 1: CPU FP64

### 구성

| Stage | 구현체 | 라이브러리 |
|-------|--------|-----------|
| Storage | `CpuFp64Storage` | Eigen, SuiteSparse KLU |
| Mismatch | `CpuMismatchOp` | Eigen SpMV |
| Jacobian | `CpuJacobianOpF64` | 직접 계산 (edge-based) |
| Linear Solve | `CpuLinearSolveKLU` | Eigen::KLU |
| Voltage Update | `CpuVoltageUpdateOp` | std::complex |

### 특징

- 모든 연산이 host CPU에서 실행된다.
- CUDA 의존성이 없으므로 CUDA 없이 빌드 가능하다.
- 개발·디버깅·정확도 검증의 기준선으로 사용한다.
- Ibus 캐시를 활용해 MismatchOp 이후 JacobianOp에서 SpMV를 생략할 수 있다.

### Jacobian 분석/Fill

CPU는 `JacobianPatternGenerator`/`JacobianMapBuilder`가 만든 edge-based 산포 맵을
사용해 Jacobian을 채운다. 버스별 루프이지만 CUDA 커널처럼 warp 단위 최적화가 없다.

### 데이터 흐름

```
host memory → (모든 연산 host) → host memory
```

---

## 프로파일 2: CUDA FP64

### 구성

| Stage | 구현체 | 라이브러리 |
|-------|--------|-----------|
| Storage | `CudaFp64Storage` | — |
| Mismatch | `CudaMismatchOp` | CUDA 커널 |
| Jacobian | `CudaJacobianOp<double>` | CUDA 커널 |
| Linear Solve | `CudaLinearSolveCuDSS<double, CudaFp64Storage>` | cuDSS (FP64) |
| Voltage Update | `CudaVoltageUpdateOp<double>` | CUDA 커널 |

### 특징

- Ybus, J, F, dx, V 모두 device 메모리에 상주한다.
- Mismatch normF 계산을 위해 F 벡터를 host로 내려야 한다.
- cuDSS RHS는 `d_F`를 직접 가리킨다. 공통 부호 convention은 `J * dx = F`,
  `state -= dx`다.
- Jacobian은 edge one-pass CUDA 커널 하나를 사용한다.
- 수렴 안정성이 CPU FP64와 동일하다.

---

## 프로파일 3: CUDA Mixed

### 구성

| Stage | 구현체 | 정밀도 |
|-------|--------|--------|
| Storage | `CudaMixedStorage` | 혼합 |
| Mismatch | `CudaMismatchOp` | FP64 Ybus/V/Ibus/Sbus/F |
| Jacobian | `CudaJacobianOp<float>` | **FP32** |
| Linear Solve | `CudaLinearSolveCuDSS<float, CudaMixedStorage>` | **FP32 uniform batch** |
| Voltage Update | `CudaVoltageUpdateOp<float>` | FP64 Va/Vm / FP64 mismatch V cache / FP32 dx |

Mixed 프로파일은 **고정 구성**이다. stage별 자유 조합이 아니다.

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

`/workspace/gpu-powerflow/exp/20260420/jac_asm` 실험에서는 edge fill이 더 빠르게 측정되어
현재 구현은 edge one-pass만 유지한다.

### Mixed 사용 시 고려사항

- FP32 Jacobian은 수치적으로 ill-conditioned한 계통에서 수렴이 느려지거나 실패할 수 있다.
- 표준 전력계통 계산(normal operating conditions)에서는 FP64와 동일한 반복 횟수에 수렴한다.
- `F`와 `Va/Vm`이 FP64이므로 수렴 판정과 public output은 FP64 기준을 유지한다.
- `J/dx`가 FP32이므로 민감한 계통은 별도 실패 케이스로 추적한다.

---

## 프로파일 비교 요약

| 항목 | CPU FP64 | CUDA FP64 | CUDA Mixed |
|------|----------|-----------|------------|
| 최소 의존성 | Eigen, KLU | + CUDA, cuDSS | + CUDA, cuDSS |
| Jacobian 정밀도 | FP64 | FP64 | FP32 |
| Solve 정밀도 | FP64 | FP64 | FP32 |
| 전압 상태 | FP64 | FP64 | Va/Vm FP64, mismatch/Jacobian V cache FP64 |
| atomic 필요 여부 | 없음 | 없음 | 없음, one-pass diag correction |
| 수렴 안정성 | 기준 | CPU FP64와 동일 | 대부분 동일, ill-cond는 주의 |
| 적합한 용도 | 개발·검증 | 정확도 우선 GPU | 처리량 우선 GPU |

---

## 빌드 옵션과 프로파일의 관계

| CMake 옵션 | 의미 | 영향 |
|-----------|------|------|
| `WITH_CUDA=ON` | CUDA backend 활성화 | CUDA FP64, Mixed 사용 가능 |
| `CUPF_CUDSS_REORDERING_ALG` | cuDSS 재순서화 알고리즘 | CUDA solve 성능에 영향 |
| `CUPF_CUDSS_ENABLE_MT` | cuDSS 멀티스레드 | CPU 병렬 인수분해 |
| `CUPF_CUDSS_ND_NLEVELS` | Nested Dissection 레벨 | 대형 계통에서 fill-in 감소 |
| `ENABLE_TIMING=ON` | ScopedTimer 활성화 | 타이밍 수집, 각 stage별 wall-clock |
| `ENABLE_NVTX=ON` | NVTX range 활성화 | Nsight Systems 프로파일링 |

cuDSS matching과 pivot epsilon은 빌드 옵션이 아니라 `NewtonOptions.cudss` 런타임 설정으로 전달한다.
