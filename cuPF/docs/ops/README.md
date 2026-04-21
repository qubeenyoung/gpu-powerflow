# ops — 연산자(Operator) 설계

NR 반복 루프는 4개의 stage로 구성된다. 각 stage는 대응하는 Op 인터페이스와 구현체를 갖는다.

```
매 반복:
  IMismatchOp::run()      → F = S_calc - S_spec, normF, converged
  IJacobianOp::run()      → J.values ← scatter(Ybus, V, maps)
  ILinearSolveOp::run()   → dx 계산
  IVoltageUpdateOp::run() → Va, Vm 갱신 → V cache 재구성
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
virtual void download_batch_result(NRBatchResultF64& result);  // batch 결과 다운로드
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
// 완료 후: storage의 Va, Vm과 V cache가 갱신됨
```

---

## Op 구현체 목록

### Mismatch

| 클래스 | 파일 | backend | precision |
|--------|------|---------|-----------|
| `CpuMismatchOpF64` | `ops/mismatch/cpu_f64.cpp` | CPU | FP64 |
| `CudaMismatchOpF64` | `ops/mismatch/cuda_f64.cu` | CUDA | FP64 path / Mixed FP32 input + FP64 F |

**CUDA Mismatch** 는 Storage 유형(FP64 vs Mixed)에 따라 내부에서 분기한다.
FP64 storage는 기존 typed kernel이 각 residual entry에서 필요한 Ybus row를 직접 순회한다.
Mixed storage는 substage schedule을 사용한다.

```
launch_compute_ibus()      // Ibus64 = Ybus32 * V64, custom batch CSR
launch_compute_mismatch()  // F64 = V64 * conj(Ibus64) - Sbus64
launch_reduce_norm()       // batch별 normF
```

Mixed의 `Ybus`는 FP32이고, `V_re/V_im`, `Ibus`, `Sbus`, 수렴 판정에 쓰는
`d_F`와 `d_normF`는 FP64다.

커널 방식: 스레드 tid가 F 벡터의 tid번째 원소를 담당.
```
tid < n_pv       → bus = pv[tid],        F[tid] = ΔP
n_pv ≤ tid < n_pvpq → bus = pq[tid-n_pv], F[tid] = ΔP
n_pvpq ≤ tid    → bus = pq[tid-n_pvpq], F[tid] = ΔQ
```
FP64 path에서는 각 스레드가 해당 버스의 행 전체를 순회해 I_bus를 직접 계산한다.
Mixed path에서는 앞 단계의 `Ibus` 버퍼를 재사용한다.

---

### Jacobian

#### Edge-based (기본값)

| 클래스 | 파일 | precision |
|--------|------|-----------|
| `CudaJacobianOpEdgeFp64` | `ops/jacobian/cuda_edge_fp64.cu` | FP64 |
| `CudaJacobianOpEdgeFp32` | `ops/jacobian/cuda_edge_fp32.cu` | FP32 |
| `CpuJacobianOpF64` | `ops/jacobian/cpu_f64.cpp` | FP64 |

**스레드 할당:** 스레드 하나가 Ybus 비영 원소 하나(엣지)를 처리.

**대각 처리:**
- FP64 edge path는 기존 방식대로 diag 위치에 `atomicAdd`를 사용한다.
- Mixed FP32 edge path는 off-diagonal/self term을 direct write하고,
  `Ibus` 기반 diagonal correction kernel을 별도로 실행한다.
- Mixed FP32 Jacobian은 storage 경계의 `V_re/V_im`, `Vm`, `Ibus`를 FP64로 읽되,
  커널 내부에서 FP32로 cast한 뒤 FP32 산술로 `d_J_values`를 채운다.
- Mixed FP32 path는 아직 `d_J_values.memsetZero()`를 유지한다. 모든 entry write
  coverage를 poison 검사로 확인한 뒤 제거한다.

#### Vertex-based (대안)

| 클래스 | 파일 | precision |
|--------|------|-----------|
| `CudaJacobianOpVertexFp64` | `ops/jacobian/cuda_vertex_fp64.cu` | FP64 |
| `CudaJacobianOpVertexFp32` | `ops/jacobian/cuda_vertex_fp32.cu` | FP32 |

**스레드 할당:** warp 하나(32 스레드)가 버스 하나(정점)를 처리.
warp 내 레인이 버스의 행 원소를 스트라이드로 분담한다.

**대각 처리:** FP64 path는 레인별 레지스터 누산 → `warp_sum()` → lane 0 단일 write다.
Mixed FP32 path는 edge path와 같은 diagonal correction kernel을 공유한다.

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
- CUDA 부호 convention은 `F = S_calc - S_spec`, `J * dx = F`, `state -= dx`다.
- FP64 cuDSS는 RHS dense matrix가 `d_F`를 직접 가리킨다.
- Mixed FP32 cuDSS는 `d_F(double)`을 device cast kernel로 FP32 RHS에 변환한다.
- Mixed FP32 cuDSS는 `CUDSS_CONFIG_UBATCH_SIZE`와 flat batch-major buffer를 사용하는
  uniform batch path이며, `B=1`도 같은 path다.
- `run()`: 필요 시 analysis → factorization/refactorization → solve

첫 번째 `run()` 이후에는 `REFACTORIZATION`을 사용해 reuse 가능한 자료구조를 유지한다.

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
`--dump-newton-diagnostics`를 켜면 cuDSS solve마다 `F_k`, `J_used`, `dx`,
`J_used dx - F_k`와 `linear_diagnostics.csv`가 남는다. 다음 iteration의
`residual_before_update`가 이전 update 후의 실제 nonlinear residual이다.
CUDA Mixed `B>1` dump에서는 현재 batch 0 slice의 linear residual을 기록한다.

---

### Voltage Update

| 클래스 | 파일 | precision |
|--------|------|-----------|
| `CpuVoltageUpdateF64` | `ops/voltage_update/cpu_f64.cpp` | FP64 |
| `CudaVoltageUpdateFp64` | `ops/voltage_update/cuda_fp64.cu` | FP64 |
| `CudaVoltageUpdateMixed` | `ops/voltage_update/cuda_mixed.cu` | FP64 상태 / FP32 dx |

CUDA 구현은 `Va/Vm`을 authoritative state로 유지한다.
1. `update_voltage_*_kernel`: CUDA convention에 따라 `state -= dx`
2. `reconstruct_voltage_kernel`: `Va/Vm` → `V_re/V_im` cache 재구성

Mixed 커널에서 `dx`는 `float*`이며, FP64 Va/Vm에 적용할 때 `static_cast<double>`로 변환된다.
Mixed의 `V_re/V_im` cache도 FP64다. 최종 public result는 FP64 `Va/Vm`에서 재구성한다.

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
