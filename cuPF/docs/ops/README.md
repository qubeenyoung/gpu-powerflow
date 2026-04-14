# ops — 연산자(Operator) 설계

NR 반복 루프는 4개의 stage로 구성된다. 각 stage는 대응하는 Op 인터페이스와 구현체를 갖는다.

```
매 반복:
  IMismatchOp::run()      → F = S_calc - S_spec, normF, converged
  IJacobianOp::run()      → J.values ← scatter(Ybus, V, maps)
  ILinearSolveOp::run()   → dx = J⁻¹·(-F)
  IVoltageUpdateOp::run() → Va, Vm += dx → V 재구성
```

---

## 인터페이스

**파일:** [op_interfaces.hpp](../../cpp/inc/newton_solver/ops/op_interfaces.hpp)

### IStorage

Storage는 Op가 아니지만 같은 파일에 정의된다. 버퍼·라이브러리 핸들·디스크립터를 소유한다.

```cpp
virtual void prepare(const AnalyzeContext& ctx);   // analyze 단계 초기화
virtual void upload(const SolveContext& ctx);       // solve 시작 시 데이터 업로드
virtual void download_result(NRResultF64& result);  // 최종 결과 다운로드
```

### IMismatchOp

```cpp
virtual void run(IterationContext& ctx);
// 완료 후: ctx.normF = max|F_i|, ctx.converged = (normF <= tolerance)
```

### IJacobianOp

```cpp
virtual void run(IterationContext& ctx);
// 완료 후: storage의 J.values가 현재 V 기준으로 채워짐
```

### ILinearSolveOp

```cpp
virtual void analyze(const AnalyzeContext& ctx);  // symbolic 분석 (한 번만)
virtual void run(IterationContext& ctx);           // factorize + solve
```

### IVoltageUpdateOp

```cpp
virtual void run(IterationContext& ctx);
// 완료 후: storage의 Va, Vm, V(복소)가 갱신됨
```

---

## Op 구현체 목록

### Mismatch

| 클래스 | 파일 | backend | precision |
|--------|------|---------|-----------|
| `CpuMismatchOpF64` | `ops/mismatch/cpu_f64.cpp` | CPU | FP64 |
| `CudaMismatchOpF64` | `ops/mismatch/cuda_f64.cu` | CUDA | FP64 |

**CUDA Mismatch** 는 Storage 유형(FP64 vs Mixed)에 따라 내부에서 분기한다.
mismatch 커널 자체는 항상 FP64이며, Mixed 모드에서도 `d_F`(FP64)에 기록한다.

커널 방식: 스레드 tid가 F 벡터의 tid번째 원소를 담당.
```
tid < n_pv       → bus = pv[tid],        F[tid] = ΔP
n_pv ≤ tid < n_pvpq → bus = pq[tid-n_pv], F[tid] = ΔP
n_pvpq ≤ tid    → bus = pq[tid-n_pvpq], F[tid] = ΔQ
```
각 스레드가 해당 버스의 행 전체를 순회해 I_bus를 직접 계산한다 (SpMV inline).

---

### Jacobian

#### Edge-based (기본값)

| 클래스 | 파일 | precision |
|--------|------|-----------|
| `CudaJacobianOpEdgeFp64` | `ops/jacobian/cuda_edge_fp64.cu` | FP64 |
| `CudaJacobianOpEdgeFp32` | `ops/jacobian/cuda_edge_fp32.cu` | FP32 |
| `CpuJacobianOpF64` | `ops/jacobian/cpu_f64.cpp` | FP64 |

**스레드 할당:** 스레드 하나가 Ybus 비영 원소 하나(엣지)를 처리.

**대각 처리:** 여러 스레드가 같은 diag 위치에 기여하므로 `atomicAdd`가 필요.
- FP64: `atomicAdd` (sm_60+) 또는 CAS 에뮬레이션
- FP32: 하드웨어 `atomicAdd` 직접 사용

#### Vertex-based (대안)

| 클래스 | 파일 | precision |
|--------|------|-----------|
| `CudaJacobianOpVertexFp64` | `ops/jacobian/cuda_vertex_fp64.cu` | FP64 |
| `CudaJacobianOpVertexFp32` | `ops/jacobian/cuda_vertex_fp32.cu` | FP32 |

**스레드 할당:** warp 하나(32 스레드)가 버스 하나(정점)를 처리.
warp 내 레인이 버스의 행 원소를 스트라이드로 분담한다.

**대각 처리:** 레인별 레지스터 누산 → `warp_sum()` → lane 0이 단일 write.
atomic 불필요. 고차수(degree) 버스에서 edge-based보다 효율적.

#### CPU

CPU Jacobian은 edge-based 방식만 제공하며, Ibus 캐시를 재사용해 SpMV를 생략할 수 있다.

---

### Linear Solve

| 클래스 | 파일 | 라이브러리 | precision |
|--------|------|-----------|-----------|
| `CpuLinearSolveKLU` | `ops/linear_solve/cpu_klu.cpp` | Eigen::KLU | FP64 |
| `CudaLinearSolveCuDSS64` | `ops/linear_solve/cuda_cudss64.cpp` | cuDSS | FP64 |
| `CudaLinearSolveCuDSS32` | `ops/linear_solve/cuda_cudss32.cpp` | cuDSS | FP32 |

#### KLU (CPU)

Eigen의 SuiteSparse KLU 래퍼를 사용한다.
- `analyze()`: `lu.analyzePattern(J)` — symbolic 분석
- `run()`: `lu.factorize(J)` → `dx = lu.solve(-F)` — 수치 인수분해 + 역대입

#### cuDSS (CUDA)

NVIDIA cuDSS direct sparse solver를 사용한다.
- `analyze()`: handle/config/data 생성 → `CUDSS_PHASE_ANALYSIS` (symbolic)
- `run()`: RHS 준비(`-F`) → `CUDSS_PHASE_FACTORIZATION` (또는 REFACTORIZATION) → `CUDSS_PHASE_SOLVE`

첫 번째 `run()` 이후에는 `REFACTORIZATION`을 사용해 reuse 가능한 자료구조를 유지한다.

**cuDSS 설정** (`ops/linear_solve/cudss_config.hpp`):
- `CUPF_CUDSS_REORDERING_ALG`: 재순서화 알고리즘 (DEFAULT, ALG_1, ALG_2)
- `CUPF_CUDSS_ENABLE_MT`: 멀티스레드 활성화
- `CUPF_CUDSS_HOST_NTHREADS`: 호스트 스레드 수
- `CUPF_CUDSS_ND_NLEVELS`: Nested Dissection 레벨 수

---

### Voltage Update

| 클래스 | 파일 | precision |
|--------|------|-----------|
| `CpuVoltageUpdateF64` | `ops/voltage_update/cpu_f64.cpp` | FP64 |
| `CudaVoltageUpdateFp64` | `ops/voltage_update/cuda_fp64.cu` | FP64 |
| `CudaVoltageUpdateMixed` | `ops/voltage_update/cuda_mixed.cu` | FP64 상태 / FP32 dx |

CUDA 구현은 3개의 커널을 순차 실행한다.
1. `decompose_voltage_kernel`: V(복소) → Va, Vm 분해
2. `update_voltage_*_kernel`: dx 보정 적용 (FP64 또는 FP32 dx → FP64 Va/Vm)
3. `reconstruct_voltage_kernel`: Va, Vm → V(복소) 재구성

Mixed 커널에서 `dx`는 `float*`이며, FP64 Va/Vm에 더할 때 `static_cast<double>`로 변환된다.

---

## Op 간 데이터 흐름

```
               ┌──────────┐
 upload()  ──→ │ IStorage │ ←── 모든 Op가 참조
               └──────────┘
                    │
         ┌──────────┼───────────┐
         ▼          ▼           ▼
   d_V, d_Ybus   d_J_values  d_F, d_dx
         │          │           │
  MismatchOp  JacobianOp  LinearSolveOp
         │                      │
         └──── d_F ────────────►┘
                                │
                           d_dx (풀이 결과)
                                │
                       VoltageUpdateOp
                                │
                         d_Va, d_Vm, d_V
                                │
                        download_result()
```
