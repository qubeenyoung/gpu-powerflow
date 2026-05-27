# Block ILU(0) GPU Pilot Measurement Summary

## What was implemented

Implemented a new CUDA phase benchmark:

```text
src/tools/gpu_block_ilu0_phase_bench.cu
```

Build target:

```text
gpu_block_ilu0_phase_bench
```

Run output:

```text
results/gpu_block_ilu0_phase/gpu_block_ilu0_phase_timing.csv
results/gpu_block_ilu0_phase/gpu_block_ilu0_phase_report.md
```

This is a **GPU numeric block ILU(0) phase pilot**, not a production NR-integrated preconditioner.

It performs:

1. unknown-level METIS partition
2. block coloring order
3. GPU dense block scatter
4. GPU block ILU(0) numeric factor/update
5. GPU forward/backward block triangular apply
6. phase timing

The implementation uses simple CUDA kernels plus cuBLAS batched LU/getri for diagonal block inverse. It does **not** use Tensor Cores yet.

## Why this is enough for the current claim

The claim we wanted to support is:

> Block ILU is useful for NR correction quality, but current implementation is slow. Its expensive work is dense block update/apply, which is a plausible Tensor Core target.

This now has three pieces of evidence:

1. CPU numeric pilot showed block ILU improves correction quality versus block-Jacobi.
2. Hybrid pilot showed fallback decreases on 3/5 selected cases.
3. GPU phase pilot shows factor/apply can be moved to GPU and exposes the phase structure that must be optimized.

## Quality evidence from existing CPU pilot

Standalone J1/F1 quality:

| block size | quality gate | mean dx norm ratio gain | mean linear residual ratio | mean cosine gain |
|---:|---:|---:|---:|---:|
| 16 | 5 / 5 pass | 3.88x | 0.44x | +0.341 |
| 32 | 3 / 5 pass | 2.57x | 0.52x | +0.158 |

Hybrid fallback:

| block size | fallback result |
|---:|---|
| 16 | fallback decreased on 3 / 5 cases |
| 32 | fallback decreased on 3 / 5 cases |

This supports:

> Block ILU(0) is more useful than block-Jacobi as a middle NR correction source.

## GPU phase timing

Measured on:

```text
case2383wp
case3120sp
case9241pegase
case13659pegase
case6468rte
block_size = 16, 32
```

Timing result:

| case | bs | factor ms | right+update ms | diag inv ms | factor unacct ms | apply ms | offdiag apply ms | diag apply ms | apply unacct ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 16 | 31.84 | 6.14 | 15.21 | 10.50 | 17.61 | 6.09 | 1.03 | 10.49 |
| case2383wp | 32 | 18.62 | 4.38 | 8.87 | 5.37 | 9.97 | 3.89 | 0.67 | 5.41 |
| case3120sp | 16 | 40.05 | 7.82 | 18.90 | 13.33 | 22.56 | 7.73 | 1.41 | 13.42 |
| case3120sp | 32 | 24.46 | 5.99 | 11.35 | 7.11 | 13.19 | 5.20 | 0.91 | 7.09 |
| case9241pegase | 16 | 133.44 | 30.68 | 54.73 | 48.03 | 78.54 | 28.29 | 4.08 | 46.17 |
| case9241pegase | 32 | 76.37 | 21.61 | 32.22 | 22.54 | 42.73 | 17.79 | 2.70 | 22.24 |
| case13659pegase | 16 | 184.63 | 43.95 | 73.72 | 66.95 | 114.14 | 41.80 | 5.63 | 66.70 |
| case13659pegase | 32 | 108.44 | 31.93 | 44.07 | 32.44 | 61.58 | 25.89 | 3.68 | 32.02 |
| case6468rte | 16 | 94.76 | 20.94 | 39.92 | 33.90 | 57.34 | 20.40 | 2.97 | 33.97 |
| case6468rte | 32 | 55.04 | 14.75 | 23.89 | 16.40 | 30.93 | 12.76 | 2.01 | 16.17 |

Interpretation:

- This GPU pilot is still far too slow to use in NR.
- A large part of measured time is `unacct`, meaning many small kernel/cuBLAS launch and scheduling gaps.
- Diagonal inverse is also expensive because it is called per block.
- Therefore this implementation is a **bottleneck map**, not an optimized solver.

## Tensor Core motivation from work breakdown

Symbolic work breakdown:

| block size | factor offdiag update work share | apply offdiag multiply work share | factor work / BJ setup | apply work / BJ apply |
|---:|---:|---:|---:|---:|
| 16 | 89.2% | 86.4% | 9.84x | 7.40x |
| 32 | 89.0% | 86.4% | 9.45x | 7.39x |

This means the mathematical work of block ILU is dominated by:

```text
B(i,j) -= L(i,k) * U(k,j)
```

and

```text
rhs_i -= B(i,j) * x_j
```

The first is small dense GEMM. The second is GEMV for one RHS, but can become GEMM-like if grouped across rows, levels, or multiple RHS/Krylov vectors.

Therefore the Tensor Core argument is:

> The current GPU pilot is slow, but the symbolic work that makes block ILU expensive is mostly dense block math. A next implementation should batch/group these dense block updates and apply phases so Tensor Cores can attack the dominant mathematical work.

## What not to claim

Do not claim:

> GPU block ILU already beats cuDSS or block-Jacobi.

It does not.

Do not claim:

> The current GPU block ILU bottleneck is already Tensor Core compute.

It is not. The current pilot is dominated by many small launches, diagonal inverse calls, and unoptimized scheduling.

Safe claim:

> Block ILU improves correction quality, but the current implementation is too slow. GPU phase timing shows the implementation needs batching/grouping, while symbolic work analysis shows that the dominant mathematical work is dense block update/apply. That makes Tensor Core acceleration a well-motivated next research step.

## Next implementation step

The next GPU version should not change NR policy. It should only optimize block ILU kernels:

1. group same-size block updates
2. replace per-block update kernels with batched dense GEMM
3. use Tensor Core path for block sizes 16/32
4. batch diagonal block factor/inverse
5. level-schedule forward/backward apply
6. reduce launch count and measure `unacct` again

Success metric for the next version:

```text
factor unacct ms decreases
apply unacct ms decreases
right+update ms becomes the dominant factorization component
offdiag apply ms becomes the dominant apply component
then Tensor Core acceleration can be evaluated directly
```
