# Clean Stale GMRES(1) Refinement

Implementation note: `stale_cudss` now owns a separate `J_stale_values` device buffer. `current_J` remains `ctx.d_J_values` for Jacobian fill, residual, and SpMV. Refresh copies `J_current_values -> J_stale_values` and then runs cuDSS factorize. Middle steps call stale cuDSS solve only; current J is never factorized in the middle path.

Pure cuDSS is pre-analyzed once before the NR loop. Timing comparison below uses warm NR-loop time unless explicitly marked total.

## Case Summary

| case | pure loop ms | stale loop ms | speedup | pure NR | stale NR | full cuDSS calls | accepted | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 10.092 | 20.204 | 0.500 | 5 | 11 | 2/5 | 9 | 0 |
| case6468rte | 3.678 | 3.910 | 0.940 | 3 | 3 | 2/3 | 1 | 0 |
| case6470rte | 3.645 | 3.829 | 0.952 | 3 | 3 | 2/3 | 1 | 0 |
| case6495rte | 2.426 | 2.461 | 0.985 | 2 | 2 | 2/2 | 0 | 0 |
| case6515rte | 3.871 | 4.034 | 0.960 | 3 | 3 | 2/3 | 1 | 0 |
| case8387pegase | 4.963 | 4.862 | 1.021 | 3 | 3 | 2/3 | 1 | 0 |
| case9241pegase | 9.940 | 24.452 | 0.407 | 6 | 16 | 2/6 | 14 | 0 |
| case_ACTIVSg10k | 6.033 | 7.690 | 0.784 | 4 | 5 | 2/4 | 3 | 0 |
| case_ACTIVSg25k | 12.609 | 11.326 | 1.113 | 4 | 4 | 2/4 | 2 | 0 |
| case_ACTIVSg70k | 41.648 | 32.334 | 1.288 | 6 | 6 | 2/6 | 4 | 0 |
| case_SyntheticUSA | 43.886 | 39.467 | 1.112 | 6 | 7 | 2/6 | 5 | 0 |

## Efficiency Summary

- converged: 11/11
- NR-loop wins vs pure cuDSS: 4/11
- total wins vs pure cuDSS including analyze/setup: 4/11
- NR iterations within pure + 1: 9/11
- stale solve count matches expectation on every middle row: True
- current-J SpMV count matches expectation on every middle row: True (2 algorithmic SpMVs + 1 final residual verification)
- average stale solve time per middle step, predictor + preconditioner: 1.287 ms
- average current-J SpMV time per middle step, residual + GMRES: 0.040 ms
- average GMRES dot/scalar sync time: 0.187 ms
- average GMRES orthogonalization time: 0.006 ms
- average GMRES update time: 0.014 ms
- cuBLAS pointer mode in this path: host, so dot/nrm2 scalar sync is present and measured as `gmres_scalar_sync_ms`.

## Answers

1. Stale GMRES(1) reduces NR-loop time on the largest cases, but not on all 4k+ cases. It wins mainly where full-J factorize is large enough to amortize stale solves.
2. NR iterations are within pure +1 on 9/11 cases. The cases that exceed this are trajectory problems, not linear algebra kernel speed problems.
3. Stale solve calls match the expected GMRES(1) pattern: one predictor stale solve plus one stale preconditioner solve per middle step. Current-J SpMV count is 3 because the benchmark also computes a final linear residual for logging.
4. The GMRES(1) bottleneck is not current-J SpMV. The dominant pieces are stale cuDSS solves and host-synchronized dot/nrm2 reductions; SpMV is small.
5. There is host sync: cuBLAS is in host pointer mode for the GMRES dot/nrm2 scalars. This is visible in `gmres_scalar_sync_ms`. Component-level GMRES unaccounted time is near zero by construction; NR-iteration unaccounted time mostly comes from surrounding NR plumbing.
6. Stale GMRES(1) is useful on large cases such as ACTIVSg25k/70k/SyntheticUSA. It is weak on cases where the stale trajectory grows longer, especially the difficult PEGASE/RTE cases with more Newton steps.
