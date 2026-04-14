# CPU AMG Experiment Results

Date: 2026-04-13

## Setup

CPU AMG was tested with HYPRE BoomerAMG from `libhypre-dev`.

Solver added to `iterative_probe`:

```text
bicgstab_hypre_boomeramg
```

Configuration:

- Outer solver: HYPRE `ParCSRBiCGSTAB`
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
  --solver bicgstab_hypre_boomeramg \
  --tolerance 1e-3 \
  --max-iter 1000 \
  --output-csv /workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_hypre_boomeramg_tol1e-3.csv
```

Success criterion:

```text
relative_residual_inf = ||Jx - rhs||_inf / ||rhs||_inf <= 1e-3
```

## Aggregate

| Solver | Solved / 67 | Mean iters | Median iters | Mean setup s | Median setup s | Mean solve s | Median solve s | Max solve s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BiCGSTAB + ILUT | 63 | 3.16 | 3 | 1.287e-02 | 3.238e-03 | 6.863e-03 | 1.096e-03 | 2.043e-01 |
| BiCGSTAB + ILU(0) | 57 | 243.84 | 81 | 5.872e-03 | 2.185e-03 | 3.285e-01 | 3.207e-02 | 8.223e+00 |
| BiCGSTAB + ILU(1) | 56 | 221.99 | 63 | 1.330e-02 | 5.063e-03 | 3.631e-01 | 3.257e-02 | 8.200e+00 |
| BiCGSTAB + Block-Jacobi b32 | 41 | 508.63 | 454 | 8.944e-03 | 4.185e-03 | 3.205e-01 | 9.656e-02 | 8.032e+00 |
| BiCGSTAB + HYPRE BoomerAMG | 40 | 309.10 | 42 | 1.830e-02 | 6.652e-03 | 1.166e+00 | 4.418e-02 | 3.007e+01 |

## Notes

CPU HYPRE BoomerAMG did not beat the ILUT baseline on this PF Jacobian dataset.
It also did not beat ILU(0) on robustness: 40 solved versus 57 solved.

HYPRE BoomerAMG solved two cases that ILUT missed under the strict `1e-3`
measured residual criterion:

```text
case24464_goc
case2742_goc
```

But HYPRE BoomerAMG failed on 25 cases that ILUT solved. It also failed on 17
cases that ILU(0) solved.

The slowest HYPRE BoomerAMG cases were dominated by non-convergent runs hitting
or approaching `max_iter=1000`; the worst was:

```text
case78484_epigrids: 1000 iterations, 30.074 s solve time, relative_residual_inf 6.332e+07
```

Conclusion: default CPU `BiCGSTAB + HYPRE BoomerAMG` is not a good replacement
candidate for `BiCGSTAB + ILUT` on this dump set. If AMG remains interesting,
the next experiment should be a small parameter sweep on the two cases HYPRE
uniquely solved and a few ILUT-solved/HYPRE-failed cases, rather than running a
broad sweep blindly.

## Artifacts

```text
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_hypre_boomeramg_tol1e-3.csv
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_hypre_boomeramg_tol1e-3_vs_newton.csv
/workspace/exp/20260413/iterative/results/pf_iterative_solver_comparison_tol1e-3_with_hypre_summary.csv
/workspace/exp/20260413/iterative/results/pf_iterative_hypre_boomeramg_vs_ilut_ilu0_tol1e-3.csv
```
