# Block ILU(0) and Tensor Core Acceleration Rationale

## Current implementation status

현재 코드베이스에는 **numeric GPU block ILU(0) 구현이 없다**.

있는 것은 두 가지다.

| component | file / output | role |
|---|---|---|
| symbolic analyzer | `src/tools/block_ilu_symbolic_analyzer.cpp` | block graph, L/U dependency levels, estimated work 분석 |
| numeric pilot | `include/cuiter/solver/cpu_block_ilu0_pilot.hpp` | CPU dense FP32 block ILU(0) 품질 확인 |

따라서 아래 주장은 “GPU block ILU 구현의 실제 trace 결과”가 아니라, 현재 가능한 **symbolic work breakdown + CPU numeric pilot 품질 결과**에 기반한 연구 논리다.

정직하게 표현하면:

> block ILU(0)는 NR middle correction 품질을 개선할 가능성이 확인됐다.  
> 하지만 현재 CPU pilot은 느리다.  
> 예상 병목은 dense block update/apply이고, 이 중 상당 부분은 Tensor Core 친화적인 small dense matrix 연산이다.  
> 따라서 다음 연구는 GPU block ILU(0)를 구현하고, dense-block phase가 실제 병목인지 계측한 뒤 Tensor Core 가속을 적용하는 것이다.

## Why block ILU is still worth discussing

METIS block-Jacobi는 각 block의 diagonal block만 사용한다. 이 때문에 block 사이 coupling을 preconditioner에서 버린다.

Block ILU(0)는 새 fill block을 만들지는 않지만, 기존 block graph 안에서 `L/U` update를 수행하므로 block-Jacobi보다 off-diagonal coupling을 더 반영한다.

Standalone J1/F1 품질 gate에서 block ILU(0)는 block-Jacobi보다 더 좋은 correction을 만들었다.

| block size | quality gate | mean dx norm ratio gain | mean linear residual ratio | mean cosine gain |
|---:|---:|---:|---:|---:|
| 16 | 5 / 5 pass | 3.88x | 0.44x | +0.341 |
| 32 | 3 / 5 pass | 2.57x | 0.52x | +0.158 |

여기서 `linear residual ratio`는 `block_ilu0 residual / block_jacobi residual`이다. 낮을수록 좋다.

Case-level bs16 standalone examples:

| case | BJ rel res | block ILU rel res | BJ dx ratio | block ILU dx ratio | BJ cosine | block ILU cosine |
|---|---:|---:|---:|---:|---:|---:|
| case2383wp | 1.92e-02 | 1.58e-02 | 0.017 | 0.127 | 0.427 | 0.868 |
| case3120sp | 1.66e-01 | 7.92e-02 | 0.021 | 0.108 | 0.293 | 0.757 |
| case9241pegase | 3.40e-02 | 7.80e-03 | 0.223 | 0.261 | 0.256 | 0.271 |
| case13659pegase | 3.02e-01 | 9.71e-02 | 0.006 | 0.012 | 0.020 | 0.025 |
| case6468rte | 6.63e-01 | 2.33e-01 | 0.145 | 0.530 | -0.137 | 0.644 |

Hybrid NR에서도 fallback-first 기준은 일부 개선됐다.

| block size | fallback result |
|---:|---|
| 16 | fallback decreased on 3 / 5 cases |
| 32 | fallback decreased on 3 / 5 cases |

즉, block ILU는 “NR 반복에서 쓸만한 correction을 만든다”는 근거가 있다. 다만 현재 pilot은 구현 경로가 느리다.

## Why the current block ILU pilot is slow

현재 numeric pilot은 CPU dense FP32 구현이다. GPU block-Jacobi middle solver와 직접 속도 비교하면 당연히 불리하다.

| preconditioner | avg setup ms | avg solve ms | avg middle total ms | avg ILU factor ms | avg ILU apply ms |
|---|---:|---:|---:|---:|---:|
| block-Jacobi bs16 | 1.548 | 0.710 | 0.311 | 0.000 | 0.000 |
| block ILU(0) bs16 CPU pilot | 26.843 | 4.045 | 30.563 | 23.406 | 3.338 |
| block-Jacobi bs32 | 1.436 | 0.677 | 0.309 | 0.000 | 0.000 |
| block ILU(0) bs32 CPU pilot | 80.788 | 6.490 | 87.058 | 77.119 | 5.852 |

이 숫자는 “block ILU가 이론적으로 느리다”가 아니라, “현재 CPU pilot path는 느리다”로 해석해야 한다.

그래도 병목 후보는 명확하다.

1. block ILU numeric factorization
2. block triangular apply
3. dense block update / dense block multiply

## Symbolic work breakdown

Symbolic analyzer는 block ILU(0)의 work를 다음처럼 분해했다.

Factorization:

```text
diag_factor_work    = sum_i block_dim(i)^3
offdiag_update_work = existing ILU0 block updates targeting B(i,j)
total_factor_work   = diag_factor_work + offdiag_update_work
```

Apply:

```text
diag_apply_work     = sum_i block_dim(i)^2
offdiag_gemv_work   = L/U offdiag dense block multiply work
total_apply_work    = diag_apply_work + L_offdiag + U_offdiag
```

For block coloring candidates:

| block size | factor offdiag update share | factor work / BJ setup | apply offdiag multiply share | apply work / BJ apply |
|---:|---:|---:|---:|---:|
| 8 | 88.6% | 9.85x | 85.7% | 7.09x |
| 16 | 89.2% | 9.84x | 86.4% | 7.40x |
| 32 | 89.0% | 9.45x | 86.4% | 7.39x |

이 표가 핵심이다.

Block ILU가 block-Jacobi보다 비싼 이유는 단순 diagonal block solve가 아니라, **off-diagonal block update/apply가 전체 work의 대부분**이기 때문이다.

## Tensor Core friendliness

Tensor Core 친화도는 phase별로 다르다.

### Strong Tensor Core candidate

Block ILU factorization update:

```text
B(i,j) -= L(i,k) * U(k,j)
```

이 연산은 small dense GEMM이다. block size 16/32는 Tensor Core tile과 잘 맞는다.

Symbolic work 기준으로 factorization work의 약 89%가 offdiag update 쪽이다. 따라서 GPU block ILU factorization 병목이 이 update phase로 확인되면 Tensor Core 가속 논리가 강하다.

### Moderate Tensor Core candidate

Block triangular apply의 offdiag accumulation:

```text
rhs_i -= B(i,j) * x_j
```

단일 RHS 기준으로는 GEMV이므로 Tensor Core 친화도가 GEMM보다 낮다. 하지만 여러 block row를 grouped/batched 처리하거나, 여러 Krylov vector / multi-RHS 형태로 묶으면 GEMM-like kernel로 바꿀 여지가 있다.

Apply work에서도 offdiag dense multiply가 약 86%를 차지한다. 따라서 GPU apply 병목이 offdiag multiply에 있으면 Tensor Core 또는 tensor-op 기반 batched dense kernel 연구 대상이 된다.

### Weak Tensor Core candidate

Diagonal block LU factorization, pivoting, dependency scheduling, level synchronization은 Tensor Core 친화도가 낮다.

이 부분은 Tensor Core보다 다음이 더 중요할 수 있다.

- pivoting 최소화 또는 no-pivot shifted block factor
- block coloring / level scheduling
- grouped block-size layout
- stream/block-level occupancy
- memory layout and padding

따라서 “block ILU 전체가 Tensor Core 친화적이다”라고 쓰면 과장이다.

정확한 주장은:

> block ILU 병목 중 큰 비중을 차지하는 dense block update/apply가 Tensor Core 친화적이다.

## Why block coloring matters

Symbolic analysis에서 current METIS block order는 triangular solve level이 깊고 폭이 좁았다. block coloring order는 level을 훨씬 얕고 넓게 만들었다.

| block size | ordering | mean L levels | mean U levels | mean avg width | apply / BJ | factor / BJ setup |
|---:|---|---:|---:|---:|---:|---:|
| 16 | current METIS | 74.2 | 74.2 | 10.47 | 7.40 | 9.84 |
| 16 | block coloring | 11.2 | 11.2 | 67.16 | 7.40 | 9.84 |
| 32 | current METIS | 61.8 | 61.8 | 6.14 | 7.39 | 9.45 |
| 32 | block coloring | 8.0 | 8.0 | 46.81 | 7.39 | 9.45 |

Ordering은 total work를 크게 바꾸지는 않지만, GPU 병렬 실행 가능성을 크게 바꾼다.

따라서 GPU block ILU pilot은 current order가 아니라 **block coloring order**로 시작해야 한다.

## Proposed GPU bottleneck experiment

다음 연구에서 구현해야 할 것은 production solver가 아니라 phase timing이 가능한 GPU block ILU(0) pilot이다.

Recommended fixed candidates:

| candidate | reason |
|---|---|
| block size 16 + block coloring | standalone quality 5/5 pass, lower cost than bs32 |
| block size 32 + block coloring | shallower levels, Tensor Core tile alignment stronger |

Required timing fields:

| phase | expected operation | Tensor Core relevance |
|---|---|---|
| dense block extraction | CSR to dense scatter | low |
| diagonal block factor/inverse | small dense LU/inverse | medium/low |
| ILU update | `B(i,j) -= L(i,k) U(k,j)` | high |
| forward offdiag apply | dense block matvec | medium |
| backward offdiag apply | dense block matvec | medium |
| diagonal apply | inverse block times local rhs | medium |
| level waits/sync | dependency overhead | low |

Required derived metrics:

| metric | purpose |
|---|---|
| update_ms / factor_ms | whether factorization bottleneck is TC-friendly |
| offdiag_apply_ms / apply_ms | whether apply bottleneck is dense block multiply |
| dense_math_ms / total_ms | total Tensor Core acceleration ceiling |
| level_wait_ms / total_ms | whether dependency, not math, is the real limiter |
| achieved TFLOP/s by phase | whether kernels are compute-bound or launch/memory-bound |
| block size distribution | whether padding/grouping overhead is large |

## Claim for report

Safe wording:

> Block ILU(0) is promising as a Newton middle preconditioner because the CPU numeric pilot improves standalone dx quality and reduces fallback on part of the test set. However, the pilot is far too slow. Symbolic work analysis shows that roughly 89% of factorization work is off-diagonal dense block update and roughly 86% of apply work is off-diagonal dense block multiply. These are dense small-block operations, especially at block sizes 16 and 32, and are plausible Tensor Core acceleration targets. A GPU pilot should therefore first identify whether factor update/apply dense-block phases dominate actual runtime; if they do, Tensor Core batched dense kernels are a justified next research direction.

Avoid this wording:

> Block ILU is already GPU-proven.

That is not true yet.

Also avoid:

> Tensor Cores will accelerate all of block ILU.

The dependency scheduling, pivoting, scatter/gather, and level waits may still dominate unless measured otherwise.

## Current decision

Block ILU is not the final solver path today. The final working path remains hybrid stale cuDSS-preconditioned GMRES.

Block ILU is a good **next-year acceleration research story**:

1. It has better NR correction quality than block-Jacobi.
2. It is currently too slow.
3. Its expensive symbolic work is mostly dense block math.
4. Dense block math at size 16/32 is a natural Tensor Core target.
5. The next proof step is a GPU phase-timing pilot, not another policy sweep.

## Source files

- `results/block_ilu_symbolic_report.md`
- `results/block_ilu_symbolic_apply_work.csv`
- `results/block_ilu_symbolic_factor_work.csv`
- `results/block_ilu0_report.md`
- `results/block_ilu0_standalone_quality.csv`
- `results/block_ilu0_timing.csv`
