# CPU AMG FGMRES Experiment Results

Date: 2026-04-13

## Setup

Solver added to `iterative_probe`:

```text
fgmres_hypre_boomeramg
```

Configuration:

- Outer solver: HYPRE `ParCSRFlexGMRES`
- Krylov dimension: `30`
- Preconditioner: HYPRE `BoomerAMG`
- BoomerAMG use: one AMG cycle per preconditioner application
- Tolerance: `1e-3`
- Max iterations: `1000`
- Matrix/rhs source: existing `pf_dumps` Newton linear-system snapshots
- Runtime: CPU-side HYPRE, single-process MPI path

Command:

```bash
/workspace/exp/20260413/iterative/build/iterative_probe \
  --snapshot-root /workspace/exp/20260413/iterative/pf_dumps \
  --solver fgmres_hypre_boomeramg \
  --tolerance 1e-3 \
  --max-iter 1000 \
  --output-csv /workspace/exp/20260413/iterative/results/pf_iterative_fgmres_hypre_boomeramg_tol1e-3.csv
```

Success criterion:

```text
relative_residual_inf = ||Jx - rhs||_inf / ||rhs||_inf <= 1e-3
```

The final success flag is based on the measured residual above, not only on
HYPRE's internal estimated residual.

## Aggregate

| Solver | Solved / 67 | Mean iters | Median iters | Mean setup s | Median setup s | Mean solve s | Median solve s | Max solve s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BiCGSTAB + ILUT | 63 | 3.16 | 3 | 1.287e-02 | 3.238e-03 | 6.863e-03 | 1.096e-03 | 2.043e-01 |
| BiCGSTAB + ILU(0) | 57 | 243.84 | 81 | 5.872e-03 | 2.185e-03 | 3.285e-01 | 3.207e-02 | 8.223e+00 |
| BiCGSTAB + ILU(1) | 56 | 221.99 | 63 | 1.330e-02 | 5.063e-03 | 3.631e-01 | 3.257e-02 | 8.200e+00 |
| BiCGSTAB + Block-Jacobi b32 | 41 | 508.63 | 454 | 8.944e-03 | 4.185e-03 | 3.205e-01 | 9.656e-02 | 8.032e+00 |
| BiCGSTAB + HYPRE BoomerAMG | 40 | 309.10 | 42 | 1.830e-02 | 6.652e-03 | 1.166e+00 | 4.418e-02 | 3.007e+01 |
| FGMRES + HYPRE BoomerAMG | 35 | 352.37 | 78 | 1.836e-02 | 6.734e-03 | 7.759e-01 | 4.018e-02 | 1.766e+01 |

## Comparison To BiCGSTAB + HYPRE BoomerAMG

`FGMRES + HYPRE BoomerAMG` solved 35/67 cases. `BiCGSTAB + HYPRE BoomerAMG`
solved 40/67 cases.

FGMRES solved two cases that BiCGSTAB+AMG did not solve:

```text
case3012wp_k
case4661_sdet
```

BiCGSTAB+AMG solved seven cases that FGMRES+AMG did not solve:

```text
case162_ieee_dtc
case24464_goc
case2746wp_k
case2869_pegase
case4917_goc
case588_sdet
case89_pegase
```

On the 33 cases solved by both AMG variants:

- FGMRES was faster on 17 cases.
- BiCGSTAB was faster on 16 cases.
- FGMRES mean solve time: `1.358e-01 s`; median solve time: `8.932e-04 s`.
- BiCGSTAB mean solve time: `8.079e-02 s`; median solve time: `3.487e-03 s`.
- FGMRES mean iterations: `61.03`; median iterations: `21`.
- BiCGSTAB mean iterations: `28.70`; median iterations: `16`.

## Notes

FGMRES+BoomerAMG was not a stronger replacement candidate than the existing
ILUT baseline. It solved 35/67 versus ILUT's 63/67.

It also did not improve robustness over the previous CPU AMG attempt with
BiCGSTAB+BoomerAMG: 35/67 versus 40/67. Its median solve time was slightly
lower among all cases, but the lower solved count is the bigger signal here.

Several FGMRES failures stopped because HYPRE's estimated residual was below
`1e-3`, while the measured residual was still above `1e-3`. For this experiment,
those are counted as not solved.

## Artifacts

```text
/workspace/exp/20260413/iterative/results/pf_iterative_fgmres_hypre_boomeramg_tol1e-3.csv
/workspace/exp/20260413/iterative/results/pf_iterative_fgmres_hypre_boomeramg_tol1e-3.log
/workspace/exp/20260413/iterative/results/pf_iterative_solver_comparison_tol1e-3_with_hypre_fgmres_summary.csv
/workspace/exp/20260413/iterative/results/pf_iterative_fgmres_hypre_boomeramg_vs_bicgstab_hypre_boomeramg_tol1e-3.csv
```
