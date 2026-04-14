# Iterative Solver Results With Tolerance 1e-3

Date: 2026-04-13

## Setup

Input snapshots:

```text
/workspace/exp/20260413/iterative/pf_dumps/<case>/cuda_mixed_edge/iter_000/
```

Cases:

```text
67 PF dataset snapshots
```

Precision:

```text
dump source: cuda_mixed_edge
J values: FP32-origin values
rhs=-F: FP32-origin values
offline iterative arithmetic: FP64
```

Convergence criterion used for this report:

```text
iterative_solved =
  solver success flag
  AND measured relative_residual_inf <= 1e-3

relative_residual_inf = ||Jx - rhs||_inf / ||rhs||_inf
```

Solver runtime settings:

```text
tolerance = 1e-3
max_iter = 1000
```

## Compared Configurations

| Name | Solver | Preconditioner | Parameters |
| --- | --- | --- | --- |
| `bicgstab_ilut_tol1e-3` | BiCGSTAB | ILUT | `drop_tol=1e-4`, `fill_factor=10` |
| `bicgstab_ilu0_tol1e-3` | BiCGSTAB | ILU(0) | original sparsity pattern |
| `bicgstab_ilu1_tol1e-3` | BiCGSTAB | ILU(1) | level-of-fill 1 |
| `bicgstab_block_jacobi_b32_tol1e-3` | BiCGSTAB | Block-Jacobi | contiguous block size 32 |

## Result Files

Raw CSVs:

```text
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilut_tol1e-3.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilu0_tol1e-3.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilu1_tol1e-3.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_block_jacobi_b32_tol1e-3.csv
```

Merged CSVs:

```text
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilut_tol1e-3_vs_newton.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilu0_tol1e-3_vs_newton.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilu1_tol1e-3_vs_newton.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_block_jacobi_b32_tol1e-3_vs_newton.csv
```

Aggregate comparison:

```text
/workspace/exp/20260413/iterative/results/pf_iterative_solver_comparison_tol1e-3_summary.csv
```

## Convergence

| Configuration | Solved / 67 | Not solved / 67 | Solver success flag / 67 |
| --- | ---: | ---: | ---: |
| `BiCGSTAB + ILUT` | 63 | 4 | 67 |
| `BiCGSTAB + ILU(0)` | 57 | 10 | 57 |
| `BiCGSTAB + ILU(1)` | 56 | 11 | 56 |
| `BiCGSTAB + Block-Jacobi b32` | 41 | 26 | 41 |

## Error

| Configuration | Mean relative residual | Median relative residual | Max relative residual |
| --- | ---: | ---: | ---: |
| `BiCGSTAB + ILUT` | 2.373e-04 | 8.362e-05 | 1.534e-03 |
| `BiCGSTAB + ILU(0)` | 4.752e-01 | 8.229e-04 | 1.031e+01 |
| `BiCGSTAB + ILU(1)` | 2.939e-01 | 7.833e-04 | 1.427e+01 |
| `BiCGSTAB + Block-Jacobi b32` | 1.351e+00 | 9.495e-04 | 3.892e+01 |

## Time

| Configuration | Mean setup ms | Mean solve ms | Mean total ms | Median total ms |
| --- | ---: | ---: | ---: | ---: |
| `BiCGSTAB + ILUT` | 12.87 | 6.86 | 19.73 | 4.27 |
| `BiCGSTAB + ILU(0)` | 5.87 | 328.50 | 334.37 | 35.20 |
| `BiCGSTAB + ILU(1)` | 13.30 | 363.10 | 376.40 | 37.88 |
| `BiCGSTAB + Block-Jacobi b32` | 8.94 | 320.47 | 329.42 | 100.72 |

## Iterations

| Configuration | Mean iterations | Median iterations | Max iterations |
| --- | ---: | ---: | ---: |
| `BiCGSTAB + ILUT` | 3.16 | 3 | 19 |
| `BiCGSTAB + ILU(0)` | 243.84 | 81 | 1000 |
| `BiCGSTAB + ILU(1)` | 221.99 | 63 | 1000 |
| `BiCGSTAB + Block-Jacobi b32` | 508.63 | 454 | 1000 |

## Not Solved Cases

`BiCGSTAB + ILUT`:

```text
case13659_pegase
case24464_goc
case2742_goc
case78484_epigrids
```

`BiCGSTAB + ILU(0)`:

```text
case13659_pegase
case1888_rte
case1951_rte
case2848_rte
case2868_rte
case6468_rte
case6470_rte
case6495_rte
case6515_rte
case9241_pegase
```

`BiCGSTAB + ILU(1)`:

```text
case13659_pegase
case1888_rte
case1951_rte
case240_pserc
case2848_rte
case2868_rte
case6468_rte
case6470_rte
case6495_rte
case6515_rte
case9241_pegase
```

`BiCGSTAB + Block-Jacobi b32`:

```text
case10192_epigrids
case10480_goc
case13659_pegase
case1888_rte
case1951_rte
case20758_epigrids
case2383wp_k
case24464_goc
case2736sp_k
case2848_rte
case2853_sdet
case2868_rte
case3012wp_k
case3022_goc
case3120sp_k
case3375wp_k
case4661_sdet
case5658_epigrids
case6468_rte
case6470_rte
case6495_rte
case6515_rte
case7336_epigrids
case78484_epigrids
case8387_pegase
case9241_pegase
```

## Interpretation

With the looser measured residual threshold `1e-3`, `BiCGSTAB + ILUT` remains
the best configuration. It solves 63/67 cases and is much faster than the two
local level-ILU/Block-Jacobi implementations.

`ILU(1)` did not improve over `ILU(0)` in this harness. It solved 56/67 versus
57/67 for `ILU(0)`, while also taking more mean total time. The median iteration
count was lower for `ILU(1)` than `ILU(0)`, but the solved-case count and total
time do not justify preferring it here.

Block-Jacobi remains too weak in the contiguous block-size-32 form.
