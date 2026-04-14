# variants — 실행 프로파일 설계

cuPF는 `NewtonOptions`를 통해 세 가지 실행 프로파일을 지원한다.
각 프로파일은 Storage + 4 Op의 고정된 조합이며, `PlanBuilder::build()`에서 조립된다.

---

## 프로파일 선택 방법

```cpp
// CPU FP64 (기본값)
NewtonSolver solver;

// CUDA FP64
NewtonSolver solver({
    .backend         = BackendKind::CUDA,
    .compute         = ComputePolicy::FP64,
    .jacobian_builder = JacobianBuilderType::EdgeBased,  // 또는 VertexBased
});

// CUDA Mixed
NewtonSolver solver({
    .backend         = BackendKind::CUDA,
    .compute         = ComputePolicy::Mixed,
    .jacobian_builder = JacobianBuilderType::EdgeBased,
});
```

---

## 프로파일 1: CPU FP64

### 구성

| Stage | 구현체 | 라이브러리 |
|-------|--------|-----------|
| Storage | `CpuFp64Storage` | Eigen, SuiteSparse KLU |
| Mismatch | `CpuMismatchOpF64` | Eigen SpMV |
| Jacobian | `CpuJacobianOpF64` | 직접 계산 (edge-based) |
| Linear Solve | `CpuLinearSolveKLU` | Eigen::KLU |
| Voltage Update | `CpuVoltageUpdateF64` | std::complex |

### 특징

- 모든 연산이 host CPU에서 실행된다.
- CUDA 의존성이 없으므로 CUDA 없이 빌드 가능하다.
- 개발·디버깅·정확도 검증의 기준선으로 사용한다.
- Ibus 캐시를 활용해 MismatchOp 이후 JacobianOp에서 SpMV를 생략할 수 있다.

### Jacobian 빌더

CPU는 `JacobianBuilderType` 설정에 무관하게 항상 edge-based 방식으로 Jacobian을 채운다.
버스별 루프이지만 CUDA 커널처럼 warp 단위 최적화가 없다.

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
| Mismatch | `CudaMismatchOpF64` | CUDA 커널 |
| Jacobian | `CudaJacobianOpEdgeFp64` / `CudaJacobianOpVertexFp64` | CUDA 커널 |
| Linear Solve | `CudaLinearSolveCuDSS64` | cuDSS (FP64) |
| Voltage Update | `CudaVoltageUpdateFp64` | CUDA 커널 |

### 특징

- Ybus, J, F, dx, V 모두 device 메모리에 상주한다.
- Mismatch normF 계산을 위해 F 벡터를 host로 내려야 한다.
  (cuDSS RHS 준비 시 F도 host 경유로 처리됨)
- edge-based vs vertex-based Jacobian 커널을 선택할 수 있다.
- 수렴 안정성이 CPU FP64와 동일하다.

### Jacobian 빌더 선택 기준

| 특성 | EdgeBased | VertexBased |
|------|-----------|-------------|
| 스레드 = 버스 당 | 원소 1개 | 행 전체 (warp) |
| 대각 처리 | atomic add | warp_sum → 단일 write |
| 희박 계통 (저차수 버스 많음) | 적합 | warp 낭비 가능 |
| 조밀 계통 (고차수 버스 많음) | 경쟁 심화 | 더 효율적 |

---

## 프로파일 3: CUDA Mixed

### 구성

| Stage | 구현체 | 정밀도 |
|-------|--------|--------|
| Storage | `CudaMixedStorage` | 혼합 |
| Mismatch | `CudaMismatchOpF64` | FP64 |
| Jacobian | `CudaJacobianOpEdgeFp32` / `CudaJacobianOpVertexFp32` | **FP32** |
| Linear Solve | `CudaLinearSolveCuDSS32` | **FP32** |
| Voltage Update | `CudaVoltageUpdateMixed` | FP64 상태 / FP32 dx |

Mixed 프로파일은 **고정 구성**이다. stage별 자유 조합이 아니다.

### 정밀도별 상세

```
d_Ybus       (FP64) → Mismatch에서 I_bus 계산에 사용
d_F          (FP64) → normF 수렴 판정 정밀도 유지
d_J_values   (FP32) → Jacobian fill + cuDSS FP32 입력
d_dx         (FP32) → cuDSS FP32 출력
d_Va/Vm/V    (FP64) → 전압 상태 누산 정밀도 유지
```

### FP32 Jacobian 커널

FP64 커널과 완전히 동일한 알고리즘이며 연산 타입만 `float`로 바뀐다.
- `atomicAdd(float*, float)`: sm_20+에서 하드웨어 지원, CAS 에뮬레이션 불필요.

### Mixed 사용 시 고려사항

- FP32 Jacobian은 수치적으로 ill-conditioned한 계통에서 수렴이 느려지거나 실패할 수 있다.
- 표준 전력계통 계산(normal operating conditions)에서는 FP64와 동일한 반복 횟수에 수렴한다.
- Mismatch와 Voltage가 FP64이므로 최종 해의 정밀도는 FP64 수준이다.

---

## 프로파일 비교 요약

| 항목 | CPU FP64 | CUDA FP64 | CUDA Mixed |
|------|----------|-----------|------------|
| 최소 의존성 | Eigen, KLU | + CUDA, cuDSS | + CUDA, cuDSS |
| Jacobian 정밀도 | FP64 | FP64 | FP32 |
| Solve 정밀도 | FP64 | FP64 | FP32 |
| 전압 상태 | FP64 | FP64 | FP64 |
| atomic 필요 여부 | 없음 | Edge: 있음, Vertex: 없음 | Edge: 있음, Vertex: 없음 |
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
