# Report to GPT: cuPF NR Linear Systems, cuDSS, and GMRES Block-Jacobi

## Purpose

We are evaluating GPU iterative linear solvers for Newton-Raphson power-flow Jacobian systems. The target system is

```text
J dx = F
```

where `J` is a nonsymmetric sparse CSR Newton-Raphson Jacobian and `F` is the mismatch vector dumped from cuPF. The original experiment goal was to avoid ILU/SpSV bottlenecks by using right-preconditioned restarted GMRES with a METIS block-Jacobi dense-block preconditioner.

## Data

The data is not synthetic. It was generated from cuPF MATPOWER cases under:

```text
/workspace/gpu-powerflow/exp/20260607-2/raw/cupf_jf_dumps
```

cuPF iteration labels are zero-based. Therefore:

```text
J1/F1 = jacobian_iter1.txt / residual_iter1.txt
```

This corresponds to the second Newton-Raphson iteration.

For each case, aliases were created:

```text
J0.txt F0.txt
J1.txt F1.txt
J2.txt F2.txt
```

The full 1K-10K dump contains 23 cases. The current measurements use 5 representative cases by size:

| case | buses | linear n | nnz |
|---|---:|---:|---:|
| case1197 | 1197 | 2392 | 14344 |
| case2736sp | 2736 | 5237 | 33715 |
| case3375wp | 3374 | 6355 | 40717 |
| case6468rte | 6468 | 12643 | 87845 |
| case_ACTIVSg10k | 10000 | 18544 | 125174 |

## Implemented Tools

Main experiment directory:

```text
/workspace/gpu-powerflow/exp/20260607-2
```

Important files:

```text
src/tools/gmres_block_jacobi_bench.cpp
src/tools/cudss_jf_bench.cpp
scripts/dump_cupf_jf_linear_systems.py
scripts/run_cudss_representative_bench.py
scripts/run_gmres_representative_bench.py
scripts/summarize_cudss_vs_gmres_short.py
```

cuPF was patched to dump Jacobian CSR matrices:

```text
/workspace/gpu-powerflow-master/cuPF/cpp/src/newton_solver/ops/jacobian/fill_jacobian_gpu.cu
```

The existing cuPF mismatch dump already produced `residual_iterN.txt`.

## Residual Definitions

Final true residual:

```text
absolute residual = ||F - J dx||2
relative residual = ||F - J dx||2 / ||F||2
```

GMRES per-iteration residual history uses the internal Givens residual estimate. The final GMRES row reports the true residual computed by an actual SpMV at the end.

cuDSS residual is also computed as a true residual.

## cuDSS Results

cuDSS was measured with phase timing split into:

```text
ANALYSIS
FACTORIZATION
SOLVE
```

File parsing and H2D copies are excluded from these phase times. Measurements are FP64, 3-repeat average.

| case | n | nnz | analyze | factorize | solve | total | rel residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| case1197 | 2392 | 14344 | 1.347418e-02 | 4.228945e-03 | 1.074284e-03 | 1.877741e-02 | 1.441134e-13 |
| case2736sp | 5237 | 33715 | 1.572953e-02 | 4.464774e-03 | 1.200497e-03 | 2.139480e-02 | 1.568414e-13 |
| case3375wp | 6355 | 40717 | 1.653831e-02 | 4.413652e-03 | 1.169459e-03 | 2.212142e-02 | 7.791014e-13 |
| case6468rte | 12643 | 87845 | 2.202690e-02 | 4.587441e-03 | 1.219476e-03 | 2.783382e-02 | 9.823185e-14 |
| case_ACTIVSg10k | 18544 | 125174 | 2.761031e-02 | 4.838485e-03 | 1.324781e-03 | 3.377358e-02 | 4.126523e-13 |

For comparison against iterative solve-only time, cuDSS `factorize + solve` is:

| case | cuDSS factorize+solve |
|---|---:|
| case1197 | 5.303229e-03 |
| case2736sp | 5.665272e-03 |
| case3375wp | 5.583111e-03 |
| case6468rte | 5.806917e-03 |
| case_ACTIVSg10k | 6.163266e-03 |

## GMRES Configuration

Solver:

```text
right-preconditioned restarted GMRES
```

Preconditioner:

```text
METIS block-Jacobi dense diagonal blocks
FP32 default preconditioner
inverse_gemv apply mode
```

Default short sweep:

```text
block sizes: 32, 64
restart: 8, 16
max_iters: 16, 32
rtol: 1e-3
```

Best short result per case means:

1. If any run converged, pick fastest converged run.
2. Otherwise, pick lowest final relative residual.

No short run converged to `1e-3`.

| case | n | block | restart | max_iters | converged | iters | final rel residual | solve total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case1197 | 2392 | 64 | 16 | 32 | 0 | 32 | 7.877847e-01 | 1.186131e-02 |
| case2736sp | 5237 | 32 | 16 | 32 | 0 | 32 | 7.713123e-02 | 1.187265e-02 |
| case3375wp | 6355 | 64 | 16 | 32 | 0 | 32 | 2.741736e-01 | 1.194379e-02 |
| case6468rte | 12643 | 64 | 16 | 32 | 0 | 32 | 9.450246e-02 | 1.219000e-02 |
| case_ACTIVSg10k | 18544 | 64 | 16 | 32 | 0 | 32 | 7.022103e-02 | 1.204442e-02 |

## Absolute Residual Trend

The following table compares cuDSS `factorize + solve` time against short GMRES solve time and residual history.

| case | n | cuDSS factorize+solve | GMRES short solve | GMRES final abs residual | GMRES final rel residual | abs iter1 | abs iter8 | abs iter16 | abs iter32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case1197 | 2392 | 5.303229e-03 | 1.217361e-02 | 9.734751e-05 | 7.877847e-01 | 1.234805e-04 | 1.194554e-04 | 9.802927e-05 | 9.734751e-05 |
| case2736sp | 5237 | 5.665272e-03 | 1.187822e-02 | 4.110460e-01 | 7.713123e-02 | 5.660425e-01 | 4.245035e-01 | 4.183186e-01 | 4.110460e-01 |
| case3375wp | 6355 | 5.583111e-03 | 1.198329e-02 | 6.964919e-07 | 2.741736e-01 | 1.194293e-06 | 9.526637e-07 | 8.423043e-07 | 6.964919e-07 |
| case6468rte | 12643 | 5.806917e-03 | 1.209308e-02 | 1.025976e-04 | 9.450246e-02 | 5.383957e-04 | 1.299312e-04 | 1.116707e-04 | 1.025976e-04 |
| case_ACTIVSg10k | 18544 | 6.163266e-03 | 1.190892e-02 | 4.531151e-01 | 7.022103e-02 | 1.310058e+00 | 5.821123e-01 | 4.644756e-01 | 4.531151e-01 |

Observation: residual reduction flattens quickly. For example, `case_ACTIVSg10k` goes from `1.310` to `0.582` by iteration 8, but only reaches `0.453` by iteration 32. This suggests longer GMRES alone is unlikely to be useful with this preconditioner.

## GMRES Per-Iteration Timing

Best short settings show GMRES iteration speed is good:

| case | n | config | GMRES loop / iter | solve total / iter | dot / iter | SpMV / iter | preconditioner / iter |
|---|---:|---|---:|---:|---:|---:|---:|
| case1197 | 2392 | b64 r16 | 0.313 ms | 0.371 ms | 0.188 ms | 0.0057 ms | 0.0089 ms |
| case2736sp | 5237 | b32 r16 | 0.314 ms | 0.371 ms | 0.191 ms | 0.0067 ms | 0.0067 ms |
| case3375wp | 6355 | b64 r16 | 0.316 ms | 0.373 ms | 0.189 ms | 0.0070 ms | 0.0100 ms |
| case6468rte | 12643 | b64 r16 | 0.329 ms | 0.381 ms | 0.187 ms | 0.0082 ms | 0.0154 ms |
| case_ACTIVSg10k | 18544 | b64 r16 | 0.337 ms | 0.376 ms | 0.178 ms | 0.0090 ms | 0.0189 ms |

Average:

```text
GMRES loop / iter: 0.322 ms
solve total / iter: 0.374 ms
dot/reduction / iter: 0.187 ms
SpMV / iter: 0.007 ms
preconditioner apply / iter: 0.012 ms
orthogonalization / iter: 0.030 ms
```

The main per-iteration overhead is dot/reduction, not SpMV or block-Jacobi apply.

## Long GMRES Check

A stronger run was tested:

```text
block_size=64
restart=32
max_iters=128
apply=inverse_gemv
```

It still did not converge to `1e-3`.

| case | final rel residual | solve time |
|---|---:|---:|
| case1197 | 2.979132e-02 | 6.243341e-02 |
| case2736sp | 7.537503e-02 | 6.317502e-02 |
| case3375wp | 1.902535e-01 | 6.303739e-02 |
| case6468rte | 6.726378e-02 | 6.518066e-02 |
| case_ACTIVSg10k | 5.950960e-02 | 6.592602e-02 |

`lu_solve` apply mode produced essentially the same residuals but slower solve times, so it does not change the conclusion.

## Current Interpretation

1. GMRES iteration speed is acceptable.
2. The block-Jacobi preconditioner apply is very cheap.
3. The bottleneck inside GMRES is mostly dot/reduction from Arnoldi.
4. The main problem is not iteration cost; it is preconditioner quality.
5. Short GMRES is already slower than cuDSS `factorize + solve` and much less accurate.
6. Longer GMRES does not appear promising because residual curves flatten early.
7. Absolute residuals can look small for cases where `||F||2` is small, so relative residual should still be tracked for linear solve quality.

## Suggested Next Directions

The current METIS block-Jacobi dense-block preconditioner is likely too weak for these Jacobians. Useful next experiments would change the preconditioner structure rather than only increasing GMRES iterations:

1. Larger or physics-aware blocks, for example grouping voltage angle and magnitude variables by bus/area rather than only METIS partitions.
2. Block overlap or additive Schwarz style local solves.
3. Schur/compressed coupling between METIS blocks.
4. Hybrid approach: a few cheap GMRES iterations as correction after cuDSS or another stronger setup.
5. Reducing GMRES dot/reduction cost only after convergence quality improves, because current residuals are far from target.

## Key Result Files

```text
results/cudss_representative_j1.csv
results/cudss_representative_j1.md
results/gmres_representative_j1_sweep.csv
results/gmres_representative_j1_sweep_best.md
results/gmres_representative_j1_long.csv
results/gmres_representative_j1_long_lu.csv
results/cudss_vs_gmres_short_summary.csv
results/cudss_vs_gmres_short_summary.md
results/gmres_short_residual_history/
```
