# Stale-Preconditioned GMRES Refinement

Middle steps use the last full-J cuDSS factorization as `M`. Current `J_k` is never factorized in middle steps; it is used only for residual and SpMV. GMRES applies the right preconditioner by stale cuDSS solve `z = M^{-1} v`. Pure cuDSS is pre-analyzed, so NR-loop comparison is warm factorize+solve.

## Mode Summary

| mode | converged | loop wins | total wins | median loop speedup | median total speedup | cuDSS call reduction | stale solves | current-J SpMVs | accepted | fallback | mean NR iter delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stale_R2_Richardson | 11/11 | 1/11 | 2/11 | 0.917 | 0.990 | 21 | 129 | 105 | 33 | 2 | 1.09 |
| stale_R1_GMRES1_stale_prec | 11/11 | 5/11 | 6/11 | 0.980 | 1.005 | 23 | 104 | 123 | 41 | 0 | 1.64 |
| stale_R1_GMRES2_stale_prec | 11/11 | 2/11 | 4/11 | 0.867 | 0.983 | 24 | 114 | 124 | 31 | 0 | 0.64 |

## Best Mode Per Case By NR-Loop Time

| case | best mode | pure loop ms | best loop ms | speedup | pure calls | calls | NR iters | accepted | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | stale_R2_Richardson | 10.123 | 16.515 | 0.613 | 5 | 3 | 7 | 4 | 1 |
| case6468rte | stale_R1_GMRES1_stale_prec | 3.671 | 3.934 | 0.933 | 3 | 2 | 3 | 1 | 0 |
| case6470rte | stale_R1_GMRES1_stale_prec | 3.646 | 3.787 | 0.963 | 3 | 2 | 3 | 1 | 0 |
| case6495rte | stale_R1_GMRES1_stale_prec | 2.424 | 2.407 | 1.007 | 2 | 2 | 2 | 0 | 0 |
| case6515rte | stale_R1_GMRES1_stale_prec | 3.876 | 3.954 | 0.980 | 3 | 2 | 3 | 1 | 0 |
| case8387pegase | stale_R1_GMRES1_stale_prec | 4.964 | 4.783 | 1.038 | 3 | 2 | 3 | 1 | 0 |
| case9241pegase | stale_R1_GMRES2_stale_prec | 9.929 | 19.625 | 0.506 | 6 | 2 | 10 | 8 | 0 |
| case_ACTIVSg10k | stale_R1_GMRES1_stale_prec | 5.990 | 7.246 | 0.827 | 4 | 2 | 5 | 3 | 0 |
| case_ACTIVSg25k | stale_R1_GMRES1_stale_prec | 12.657 | 10.752 | 1.177 | 4 | 2 | 4 | 2 | 0 |
| case_ACTIVSg70k | stale_R1_GMRES1_stale_prec | 41.898 | 30.703 | 1.365 | 6 | 2 | 6 | 4 | 0 |
| case_SyntheticUSA | stale_R1_GMRES1_stale_prec | 44.423 | 39.348 | 1.129 | 6 | 2 | 7 | 5 | 0 |

## R2 Richardson Versus GMRES

| case | R2 loop ms | GMRES1 loop ms | GMRES2 loop ms | R2 NR | GMRES1 NR | GMRES2 NR |
|---|---:|---:|---:|---:|---:|---:|
| case13659pegase | 16.515 | 20.261 | 18.647 | 7 | 11 | 8 |
| case6468rte | 4.020 | 3.934 | 4.311 | 3 | 3 | 3 |
| case6470rte | 4.004 | 3.787 | 4.336 | 3 | 3 | 3 |
| case6495rte | 2.459 | 2.407 | 2.438 | 2 | 2 | 2 |
| case6515rte | 4.227 | 3.954 | 4.473 | 3 | 3 | 3 |
| case8387pegase | 5.118 | 4.783 | 5.368 | 3 | 3 | 3 |
| case9241pegase | 24.211 | 24.233 | 19.625 | 13 | 16 | 10 |
| case_ACTIVSg10k | 8.525 | 7.246 | 7.770 | 5 | 5 | 4 |
| case_ACTIVSg25k | 12.589 | 10.752 | 13.235 | 4 | 4 | 4 |
| case_ACTIVSg70k | 42.913 | 30.703 | 38.920 | 7 | 6 | 6 |
| case_SyntheticUSA | 45.859 | 39.348 | 40.540 | 7 | 7 | 6 |

## Factor Age

- stale_R2_Richardson: max factor_age=9, rows with age>=5=5, mean mismatch ratio at age>=5=0.304
- stale_R1_GMRES1_stale_prec: max factor_age=13, rows with age>=5=13, mean mismatch ratio at age>=5=0.394
- stale_R1_GMRES2_stale_prec: max factor_age=7, rows with age>=5=4, mean mismatch ratio at age>=5=0.171

## Answers

1. GMRES(1) and GMRES(2) both converged on all cases and reduced full cuDSS calls versus pure cuDSS, but neither is a universal NR-loop speed win.
2. GMRES(1) is the better cost/benefit point. GMRES(2) improves the trajectory on hard cases, but the extra stale solve and orthogonalization cost usually erase the benefit.
3. Compared with R2 Richardson, GMRES(1) is faster on several large cases and has no fallback, but R2 remains competitive and often cheaper on smaller 4k-9k cases.
4. Stale solve calls are still the main cost: GMRES(1) uses predictor + one stale preconditioner solve per middle step; GMRES(2) uses predictor + two stale preconditioner solves.
5. Factor age does not immediately break the method, but high age still correlates with longer trajectories on difficult cases.
