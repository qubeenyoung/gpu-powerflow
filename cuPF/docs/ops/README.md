# ops — 연산자(Operator) 설계

NR 반복 루프는 실제 계산 순서가 드러나는 stage로 구성된다.

```
매 반복:
  IIbusOp::run()          → Ibus = Ybus * V
  IMismatchOp::run()      → F = S_calc - S_spec
  IMismatchNormOp::run()  → normF, converged
  IJacobianOp::run()      → J.values ← scatter(Ybus, V, maps)
  ILinearSolveOp::prepare_rhs()
  ILinearSolveOp::factorize()
  ILinearSolveOp::solve() → dx 계산
  IVoltageUpdateOp::run() → Va, Vm 갱신 → V cache 재구성
```

---

## 인터페이스

**파일:** [op_interfaces.hpp](../../cpp/src/newton_solver/ops/op_interfaces.hpp)

### IStorage

Storage는 Op가 아니지만 같은 파일에 정의된다. 버퍼·라이브러리 핸들·디스크립터를 소유한다.

```cpp
virtual void prepare(const InitializeContext& ctx);   // initialize 단계 초기화
virtual void upload(const SolveContext& ctx);       // solve 시작 시 데이터 업로드
virtual void download_result(NRResult& result);  // 최종 결과 다운로드
virtual void download_batch_result(NRBatchResult& result);  // batch 결과 다운로드
```

### IMismatchOp

```cpp
virtual void run(IterationContext& ctx);
```

### IMismatchNormOp

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
virtual void initialize(const InitializeContext& ctx);  // solver 초기화
virtual void prepare_rhs(IterationContext& ctx);        // RHS dtype 준비
virtual void factorize(IterationContext& ctx);       // numeric factorization
virtual void solve(IterationContext& ctx);           // J·dx = F
```

### IVoltageUpdateOp

```cpp
virtual void run(IterationContext& ctx);
// 완료 후: storage의 Va, Vm과 V cache가 갱신됨
```

---

## Op 구현체 목록

### Mismatch

| 클래스 | 파일 | backend | precision |
|--------|------|---------|-----------|
| `CpuMismatchOp` | `ops/mismatch/cpu_mismatch.cpp` | CPU | FP64 |
| `CudaMismatchOp` | `ops/mismatch/cuda_mismatch.cu` | CUDA | FP64 path / Mixed FP32 input + FP64 F |

CPU와 CUDA mismatch는 모두 `Ibus`를 먼저 만들고 residual을 계산한다.

```
compute_ibus() / launch_compute_ibus()             // Ibus = Ybus * V
launch_compute_mismatch_from_ibus()                // CUDA: F = V * conj(Ibus) - Sbus
launch_reduce_mismatch_norm()                      // CUDA: normF
```

FP64 경로는 `Ybus`, `V_re/V_im`, `Ibus`, `Sbus`, `d_F`, `d_normF`가 모두 FP64다.
Mixed 경로도 `Ybus`, `V_re/V_im`, `Ibus`, `Sbus`, 수렴 판정에 쓰는
`d_F`와 `d_normF`는 FP64다. Mixed Jacobian stage는 FP64 `Ybus`, `d_V_re/im`,
`d_Vm`, `Ibus`를 읽어 커널 안에서 FP32로 변환해 `d_J_values`를 채운다.

커널 방식: 스레드 tid가 F 벡터의 tid번째 원소를 담당.
```
tid < n_pv       → bus = pv[tid],        F[tid] = ΔP
n_pv ≤ tid < n_pvpq → bus = pq[tid-n_pv], F[tid] = ΔP
n_pvpq ≤ tid    → bus = pq[tid-n_pvpq], F[tid] = ΔQ
```
CUDA FP64와 Mixed 모두 앞 단계의 `Ibus` 버퍼를 재사용한다.

---

### Jacobian

#### Edge One-Pass

| 클래스 | 파일 | precision |
|--------|------|-----------|
| `CudaJacobianOp<double>` | `ops/jacobian/fill_jacobian_gpu.cu` | FP64 |
| `CudaJacobianOp<float>` | `ops/jacobian/fill_jacobian_gpu.cu` | FP32 |
| `CpuJacobianOpF64` | `ops/jacobian/fill_jacobian.cpp` | FP64 |

**스레드 할당:** 스레드 하나가 Ybus 비영 원소 하나(엣지)를 처리.

**대각 처리:** Ybus entry 담당 thread가 off-diagonal/self term을 direct write하고,
`i == j`인 diagonal Ybus entry thread가 diagonal correction까지 같은 kernel에서 더한다.
Mixed FP32 Jacobian은 FP64 `d_Ybus_*`, `d_V_re/im`, `d_Vm`을 읽어 커널 안에서 FP32로 변환하고,
diagonal correction도 FP64 `d_Ibus_re/im`을 읽어 FP32로 변환한다.

#### CPU

CPU Jacobian은 edge-based 방식만 제공하며, `compute_ibus()`가 만든 캐시를 재사용해 SpMV를 생략할 수 있다.

---

### Linear Solve

| 클래스 | 파일 | 라이브러리 | precision |
|--------|------|-----------|-----------|
| `CpuLinearSolveKLU` | `ops/linear_solve/cpu_klu.cpp` | Eigen::KLU | FP64 |
| `CudaLinearSolveCuDSS<double, CudaFp64Storage>` | `ops/linear_solve/cuda_cudss.cpp` | cuDSS | FP64 |
| `CudaLinearSolveCuDSS<float, CudaMixedStorage>` | `ops/linear_solve/cuda_cudss.cpp` | cuDSS | FP32 |

#### KLU (CPU)

Eigen의 SuiteSparse KLU 래퍼를 사용한다.
- `initialize()`: `lu.analyzePattern(J)` — symbolic 분석
- `factorize()`: `lu.factorize(J)` — 수치 인수분해
- `solve()`: `dx = lu.solve(F)` — 역대입

#### cuDSS (CUDA)

NVIDIA cuDSS direct sparse solver를 사용한다.
- 공통 부호 convention은 `F = S_calc - S_spec`, `J * dx = F`, `state -= dx`다.
- FP64 cuDSS는 RHS dense matrix가 `d_F`를 직접 가리킨다.
- Mixed FP32 cuDSS는 `d_F(double)`을 device cast kernel로 FP32 RHS에 변환한다.
- Mixed FP32 cuDSS는 `CUDSS_CONFIG_UBATCH_SIZE`와 flat batch-major buffer를 사용하는
  uniform batch path이며, `B=1`도 같은 path다.
- `factorize()`: 필요 시 analysis → factorization/refactorization
- `solve()`: RHS 준비 → solve

첫 번째 `factorize()` 이후에는 `REFACTORIZATION`을 사용해 reuse 가능한 자료구조를 유지한다.

**cuDSS 설정** (`ops/linear_solve/cudss_config.hpp`):
- `CUPF_CUDSS_REORDERING_ALG`: 재순서화 알고리즘 (DEFAULT, ALG_1, ALG_2)
- `CUPF_CUDSS_ENABLE_MT`: 멀티스레드 활성화
- `CUPF_CUDSS_HOST_NTHREADS`: 호스트 스레드 수
- `CUPF_CUDSS_ND_NLEVELS`: Nested Dissection 레벨 수

`NewtonOptions.cudss` 런타임 설정:
- `use_matching`: matching 전처리 활성화
- `matching_alg`: matching 알고리즘 (DEFAULT, ALG_1..ALG_5)
- `auto_pivot_epsilon` / `pivot_epsilon`: pivot epsilon 자동/수동 설정

matching이 활성화된 경우에는 첫 Jacobian 값이 채워진 뒤 `CUDSS_PHASE_ANALYSIS`를 수행한다.
`use_matching=true`는 현재 `CUPF_CUDSS_REORDERING_ALG=DEFAULT`와 함께 사용한다.

Mixed FP32 path에서는 batch size와 batch-major value buffer가 `upload()` 이후 확정되므로,
cuDSS matrix descriptor와 `CUDSS_PHASE_ANALYSIS`를 첫 `factorize()` 시점에 만든다.
현재 구현은 v1 브랜치와 같은 방식으로 `cudssMatrixCreateBatchCsr`가 아니라
단일 `cudssMatrixCreateCsr`/`cudssMatrixCreateDn` descriptor를 만들고,
`CUDSS_CONFIG_UBATCH_SIZE`로 batch count를 전달한다.

덤프 빌드(`ENABLE_DUMP=ON`)에서 benchmark의 `--dump-residuals` 또는
`--dump-newton-diagnostics`를 켜면 nonlinear residual dump를 남긴다.

---

### Voltage Update

| 클래스 | 파일 | precision |
|--------|------|-----------|
| `CpuVoltageUpdateOp` | `ops/voltage_update/cpu_voltage_update.cpp` | FP64 |
| `CudaVoltageUpdateOp<double>` | `ops/voltage_update/cuda_voltage_update.cu` | FP64 dx |
| `CudaVoltageUpdateOp<float>` | `ops/voltage_update/cuda_voltage_update.cu` | FP64 상태 / FP32 dx |

CPU/CUDA 구현은 `Va/Vm`을 authoritative state로 유지한다.
1. `apply_voltage_update_kernel` 또는 CPU loop: 공통 convention에 따라 `state -= dx`
2. `reconstruct_voltage_kernel`: `Va/Vm` → `V_re/V_im` cache 재구성

Mixed 커널에서 `dx`는 `float*`이며, FP64 Va/Vm에 적용할 때 `static_cast<double>`로 변환된다.
Mixed의 mismatch/Jacobian voltage 입력용 `V_re/V_im` cache도 FP64다.
최종 public result는 FP64 `Va/Vm`에서 재구성한다.

---

## Op 간 데이터 흐름

```
               ┌──────────┐
 upload()  ──→ │ IStorage │ ←── 모든 Op가 참조
               └──────────┘
                    │
         ┌──────────┼───────────┐
         ▼          ▼           ▼
   d_V cache, d_Ybus   d_J_values  d_F, d_dx
         │          │           │
  MismatchOp  JacobianOp  LinearSolveOp
         │                      │
         └──── d_F ────────────►┘
                                │
                           d_dx (풀이 결과)
                                │
                       VoltageUpdateOp
                                │
                       d_Va, d_Vm, d_V cache
                                │
                        download_result()
```
