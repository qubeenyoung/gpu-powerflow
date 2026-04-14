# Iterative Linear Solver Experiment Summary

Date: 2026-04-13

## What Was Built

- `dump_linear_systems`: dumps Newton linear systems from cuPF benchmark cases.
- `iterative_probe`: reads dumped systems and solves them with Eigen BiCGSTAB.

Build verified:

```bash
cmake -S /workspace/exp/20260413/iterative -B /workspace/exp/20260413/iterative/build -GNinja
cmake --build /workspace/exp/20260413/iterative/build --target dump_linear_systems iterative_probe -j
```

## Dumped Data

Command:

```bash
/workspace/exp/20260413/iterative/build/dump_linear_systems \
  --profile cuda_mixed_edge \
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --cases case118_ieee case2746wop_k case8387_pegase \
  --output-root /workspace/exp/20260413/iterative/dumps
```

Output root:

```text
/workspace/exp/20260413/iterative/dumps
```

Dumped snapshots:

| Case | Profile | Snapshots | Linear dim | NNZ | Precision |
| --- | --- | ---: | ---: | ---: | --- |
| `case118_ieee` | `cuda_mixed_edge` | 4 | 181 | 1,051 | FP32 `J`, FP32 `rhs=-F` |
| `case2746wop_k` | `cuda_mixed_edge` | 4 | 5,141 | 32,445 | FP32 `J`, FP32 `rhs=-F` |
| `case8387_pegase` | `cuda_mixed_edge` | 6 | 14,908 | 110,572 | FP32 `J`, FP32 `rhs=-F` |

Each snapshot has:

```text
J.csr
rhs.txt
x_direct.txt
meta.txt
```

## Probe Result

Command:

```bash
/workspace/exp/20260413/iterative/build/iterative_probe \
  --snapshot-root /workspace/exp/20260413/iterative/dumps \
  --solver bicgstab_ilut \
  --tolerance 1e-8 \
  --max-iter 1000 \
  --output-csv /workspace/exp/20260413/iterative/results/bicgstab_ilut.csv
```

Result CSV:

```text
/workspace/exp/20260413/iterative/results/bicgstab_ilut.csv
```

Aggregate probe results:

| Case | All Success | Iterations | Max relative residual | Solve time range |
| --- | --- | ---: | ---: | ---: |
| `case118_ieee` | yes | 2 | 1.870e-13 | 0.021-0.030 ms |
| `case2746wop_k` | yes | 5-6 | 9.454e-09 | 1.619-1.949 ms |
| `case8387_pegase` | yes | 7-8 | 5.136e-09 | 8.075-9.213 ms |

Notes:

- The default profile is `cuda_mixed_edge`, matching the benchmark CUDA mixed path: FP64 mismatch, FP32 Jacobian/solve, cuDSS32 direct solve.
- `x_direct.txt` is the direct cuDSS result from the same Newton iteration and is included for comparison, not as the iterative solver input.
- The current probe runs Eigen BiCGSTAB in FP64 over the dumped FP32-valued system, so direct residuals are not expected to match bit-for-bit.

## All PF Dataset Snapshot Dump

Command:

```bash
python3 /workspace/exp/20260413/iterative/dump_all_pf_datasets.py \
  --max-iter 10 \
  --converted-root /workspace/exp/20260413/iterative/pf_cupf_dumps \
  --output-root /workspace/exp/20260413/iterative/pf_dumps \
  --summary-csv /workspace/exp/20260413/iterative/results/pf_dataset_snapshot_summary.csv
```

Inputs:

```text
/workspace/datasets/pf_dataset/*.mat
```

Outputs:

```text
/workspace/exp/20260413/iterative/pf_cupf_dumps
/workspace/exp/20260413/iterative/pf_dumps
/workspace/exp/20260413/iterative/results/pf_dataset_snapshot_summary.csv
```

Result:

| Metric | Value |
| --- | ---: |
| PF `.mat` cases | 67 |
| Successful dump runs | 67 |
| Failed dump runs | 0 |
| Snapshots | 67 |
| Converged within `max_iter=10` | 32 |
| Not converged within `max_iter=10` | 35 |

Each case has one `cuda_mixed_edge/iter_000` snapshot. Convergence status,
iteration count, final mismatch, source `.mat`, converted cuPF dump path, and
snapshot path are recorded in the summary CSV.

## Max Iter 20 Retry For Max Iter 10 Non-Converged Cases

Command:

```bash
python3 /workspace/exp/20260413/iterative/dump_all_pf_datasets.py \
  --max-iter 20 \
  --case-list /workspace/exp/20260413/iterative/results/pf_not_converged_max10_cases.txt \
  --converted-root /workspace/exp/20260413/iterative/pf_cupf_dumps \
  --output-root /workspace/exp/20260413/iterative/pf_dumps_max20_not_converged_max10 \
  --summary-csv /workspace/exp/20260413/iterative/results/pf_not_converged_max10_max20_summary.csv
```

Result:

| Metric | Value |
| --- | ---: |
| Retried cases | 35 |
| Successful dump runs | 35 |
| Failed dump runs | 0 |
| Converged within `max_iter=20` | 0 |
| Still not converged within `max_iter=20` | 35 |
| Final mismatch improved vs `max_iter=10` | 11 |
| Final mismatch worsened or unchanged vs `max_iter=10` | 24 |

Notable improvements that still did not converge:

| Case | Final mismatch at 10 | Final mismatch at 20 | Ratio |
| --- | ---: | ---: | ---: |
| `case24464_goc` | 3.132e+13 | 1.155e+11 | 3.686e-03 |
| `case4917_goc` | 3.550e+10 | 2.813e+08 | 7.924e-03 |
| `case6495_rte` | 4.853e+10 | 3.056e+09 | 6.297e-02 |
| `case240_pserc` | 5.727e+06 | 4.197e+05 | 7.328e-02 |
| `case10000_goc` | 2.904e+09 | 5.323e+08 | 1.833e-01 |

## Iterative Solve Probe For All PF Snapshots

Command:

```bash
/workspace/exp/20260413/iterative/build/iterative_probe \
  --snapshot-root /workspace/exp/20260413/iterative/pf_dumps \
  --solver bicgstab_ilut \
  --tolerance 1e-8 \
  --max-iter 1000 \
  --output-csv /workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilut.csv
```

Merged result:

```text
/workspace/exp/20260413/iterative/results/pf_iterative_vs_newton_summary.csv
```

Classification criterion:

```text
iterative_solved = Eigen success flag AND measured relative_residual_inf <= 1e-8
```

Result:

| Iterative solved | Newton converged within `max_iter=10` | Count |
| --- | --- | ---: |
| yes | yes | 32 |
| yes | no | 32 |
| no | yes | 0 |
| no | no | 3 |

All 67 cases returned Eigen `success=true`. Under the stricter measured
residual criterion above, these 3 cases are counted as not solved:

- `case162_ieee_dtc`: relative residual 1.093e-08
- `case19402_goc`: relative residual 1.316e-08
- `case4020_goc`: relative residual 1.062e-08

## Planned Iterative Solver Comparison Result

Detailed report:

```text
/workspace/exp/20260413/iterative/ITERATIVE_SOLVER_EXPERIMENT_RESULTS.md
```

Aggregate comparison:

```text
/workspace/exp/20260413/iterative/results/pf_iterative_solver_comparison_summary.csv
```

Summary:

| Configuration | Solved / 67 | Mean iters | Mean total ms |
| --- | ---: | ---: | ---: |
| `BiCGSTAB + ILUT` | 64 | 5.43 | 24.16 |
| `BiCGSTAB + ILU(0)` | 54 | 321.97 | 480.40 |
| `BiCGSTAB + Block-Jacobi b32` | 27 | 621.54 | 357.69 |

## Iterative Solver Comparison With Tolerance 1e-3

Detailed report:

```text
/workspace/exp/20260413/iterative/ITERATIVE_SOLVER_TOL1E3_RESULTS.md
```

Summary:

| Configuration | Solved / 67 | Mean relative residual | Mean total ms |
| --- | ---: | ---: | ---: |
| `BiCGSTAB + ILUT` | 63 | 2.373e-04 | 19.73 |
| `BiCGSTAB + ILU(0)` | 57 | 4.752e-01 | 334.37 |
| `BiCGSTAB + ILU(1)` | 56 | 2.939e-01 | 376.40 |
| `BiCGSTAB + Block-Jacobi b32` | 41 | 1.351e+00 | 329.42 |
