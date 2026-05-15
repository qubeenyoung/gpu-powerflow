# Stale Full-J Factorization Refinement

Full-J analyze is pre-loop. In stale modes, the last full cuDSS factorization is reused as `M`; middle steps do not factorize current J. Residuals are computed with current J. BiCGSTAB refinement uses no preconditioner.

## Mode Summary

| mode | converged | loop wins | total wins | median loop speedup | median total speedup | cuDSS call reduction | accepted | fallback | mean NR iter delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stale_R0 | 10/11 | 5/11 | 4/11 | 0.993 | 0.997 | 22 | 62 | 2 | 3.64 |
| stale_R1_Richardson | 11/11 | 4/11 | 5/11 | 0.922 | 0.998 | 18 | 52 | 5 | 3.09 |
| stale_R2_Richardson | 11/11 | 6/11 | 7/11 | 1.016 | 1.011 | 21 | 33 | 2 | 1.09 |
| stale_R1_BiCGSTAB1_no_prec | 11/11 | 1/11 | 0/11 | 0.815 | 0.969 | 15 | 36 | 9 | 1.91 |
| stale_R1_BiCGSTAB2_no_prec | 11/11 | 4/11 | 5/11 | 0.817 | 0.996 | 17 | 47 | 7 | 2.73 |

## Best Stale Mode Per Case

| case | best mode | pure loop ms | stale loop ms | loop speedup | pure calls | stale calls | NR iters | accepted | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | stale_R2_Richardson | 10.118 | 14.478 | 0.699 | 5 | 3 | 7 | 4 | 1 |
| case6468rte | stale_R0 | 3.664 | 3.236 | 1.132 | 3 | 2 | 3 | 1 | 0 |
| case6470rte | stale_R0 | 3.652 | 3.238 | 1.128 | 3 | 2 | 3 | 1 | 0 |
| case6495rte | stale_R2_Richardson | 2.434 | 2.137 | 1.139 | 2 | 2 | 2 | 0 | 0 |
| case6515rte | stale_R0 | 3.866 | 3.386 | 1.142 | 3 | 2 | 3 | 1 | 0 |
| case8387pegase | stale_R2_Richardson | 4.915 | 4.444 | 1.106 | 3 | 2 | 3 | 1 | 0 |
| case9241pegase | stale_R1_BiCGSTAB1_no_prec | 9.912 | 14.148 | 0.701 | 6 | 5 | 6 | 1 | 4 |
| case_ACTIVSg10k | stale_R1_Richardson | 5.995 | 6.502 | 0.922 | 4 | 2 | 5 | 3 | 0 |
| case_ACTIVSg25k | stale_R1_Richardson | 12.589 | 12.043 | 1.045 | 4 | 2 | 5 | 3 | 0 |
| case_ACTIVSg70k | stale_R0 | 41.777 | 32.496 | 1.286 | 6 | 2 | 9 | 7 | 0 |
| case_SyntheticUSA | stale_R0 | 44.810 | 40.555 | 1.105 | 6 | 2 | 11 | 9 | 0 |

## Factor Age

- stale_R0: max factor_age=17, fallback count=2, ages=[0, 0]
- stale_R1_Richardson: max factor_age=14, fallback count=5, ages=[0, 0, 0, 0, 0]
- stale_R2_Richardson: max factor_age=9, fallback count=2, ages=[0, 0]
- stale_R1_BiCGSTAB1_no_prec: max factor_age=9, fallback count=9, ages=[0, 1, 0, 0, 0, 0, 1, 0, 1]
- stale_R1_BiCGSTAB2_no_prec: max factor_age=10, fallback count=7, ages=[0, 8, 0, 0, 0, 0, 0]

## Answers

1. Best overall is `stale_R2_Richardson` by loop wins/median speed. `stale_R2_Richardson` is the strongest Richardson variant; R0 can be fast but fails one hard case and often lengthens the NR trajectory.
2. BiCGSTAB refinement does not clearly reduce fallback relative to Richardson. It adds SpMV/dot/update cost and is not better than R2 Richardson on this run.
3. Full cuDSS factorize calls are reduced substantially, but the saved factorize calls are partly spent on repeated stale solves and extra NR iterations.
4. NR trajectory is close on easier RTE cases. On hard PEGASE and the largest cases, factor age grows and the trajectory can become much longer.
5. Pure cuDSS NR-loop time is beaten on a subset of cases, but not consistently across all 4k+ cases.
