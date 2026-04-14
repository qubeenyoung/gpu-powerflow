# Iterative Solver Experiment Results

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

Newton convergence reference:

```text
/workspace/exp/20260413/iterative/results/pf_dataset_snapshot_summary.csv
```

Classification criterion:

```text
iterative_solved = solver success flag AND measured relative_residual_inf <= 1e-8
```

## Compared Configurations

| Name | Solver | Preconditioner | Parameters |
| --- | --- | --- | --- |
| `bicgstab_ilut` | BiCGSTAB | ILUT | `drop_tol=1e-4`, `fill_factor=10` |
| `bicgstab_ilu0` | BiCGSTAB | ILU(0) | original sparsity pattern |
| `bicgstab_block_jacobi_b32` | BiCGSTAB | Block-Jacobi | contiguous block size 32 |

Notes:

- `bicgstab_ilut` uses Eigen's built-in `BiCGSTAB + IncompleteLUT`.
- `bicgstab_ilu0` and `bicgstab_block_jacobi_b32` use the local experiment
  harness implementation added for this comparison.
- Timing should therefore be read as experiment-harness timing, not as a final
  production GPU timing estimate.

## Result Files

Raw CSVs:

```text
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilut.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilu0.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_block_jacobi_b32.csv
```

Merged with Newton convergence:

```text
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilut_vs_newton.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilu0_vs_newton.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_block_jacobi_b32_vs_newton.csv
```

Aggregate comparison:

```text
/workspace/exp/20260413/iterative/results/pf_iterative_solver_comparison_summary.csv
```

## Aggregate Results

| Configuration | Solved / 67 | Success flag / 67 | Mean iters | Median iters | Max iters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BiCGSTAB + ILUT` | 64 | 67 | 5.43 | 5 | 28 |
| `BiCGSTAB + ILU(0)` | 54 | 54 | 321.97 | 167 | 1000 |
| `BiCGSTAB + Block-Jacobi b32` | 27 | 27 | 621.54 | 947 | 1000 |

Timing:

| Configuration | Mean setup ms | Mean solve ms | Mean total ms | Median total ms |
| --- | ---: | ---: | ---: | ---: |
| `BiCGSTAB + ILUT` | 12.91 | 11.25 | 24.16 | 4.91 |
| `BiCGSTAB + ILU(0)` | 5.86 | 474.55 | 480.40 | 74.46 |
| `BiCGSTAB + Block-Jacobi b32` | 8.87 | 348.82 | 357.69 | 125.07 |

Newton cross-classification:

| Configuration | Iter solved + Newton converged | Iter solved + Newton not converged | Iter not solved + Newton converged | Iter not solved + Newton not converged |
| --- | ---: | ---: | ---: | ---: |
| `BiCGSTAB + ILUT` | 32 | 32 | 0 | 3 |
| `BiCGSTAB + ILU(0)` | 31 | 23 | 1 | 12 |
| `BiCGSTAB + Block-Jacobi b32` | 16 | 11 | 16 | 24 |

## Interpretation

`BiCGSTAB + ILUT` remains the clear baseline. It is both the most robust and the
fastest in this harness despite a higher setup cost than ILU(0).

`BiCGSTAB + ILU(0)` is cheaper to set up, but the lighter preconditioner causes
many more Krylov iterations. It solved 54/67 cases under the strict residual
criterion and failed one case that Newton did converge on (`case9241_pegase`).
As-is, ILU(0) is not a drop-in replacement for the ILUT baseline.

`BiCGSTAB + Block-Jacobi` with contiguous block size 32 is too weak as a direct
replacement candidate. It solved only 27/67 and often hit the 1000-iteration
limit. Its value is mostly as a lightweight/GPU-friendly reference point, not as
the next replacement candidate in this form.

## Recommendation

Keep `BiCGSTAB + ILUT` as the reference.

For the next iteration:

1. Try `ILU(1)` only as a targeted rescue for the `ILU(0)` path.
2. Do not spend time on broad Block-Jacobi sweeps unless the block definition
   becomes PF-structure-aware; contiguous block size 32 was too weak.
3. If a production replacement is the goal, focus next on making the ILUT/ILU
   family cheap enough or trying GMRES-style stability before investing more in
   plain Block-Jacobi.
