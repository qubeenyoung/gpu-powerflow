# Stale-Preconditioned BiCGSTAB Refinement

Middle steps use the last full-J cuDSS factorization as `M`. Current J is never factorized in middle steps. Current J is used only for residual/SpMV. BiCGSTAB preconditioner apply is `z = M^{-1} v` using stale cuDSS solve.

## Mode Summary

| mode | converged | loop wins | total wins | median loop speedup | median total speedup | cuDSS call reduction | stale solves | current-J SpMVs | BiCGSTAB iters | accepted | fallback | mean NR iter delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stale_R2_Richardson | 11/11 | 1/11 | 1/11 | 0.914 | 0.989 | 21 | 129 | 105 | 0 | 33 | 2 | 1.09 |
| stale_R1_BiCGSTAB1_stale_prec | 11/11 | 3/11 | 4/11 | 0.901 | 0.998 | 24 | 135 | 152 | 38 | 38 | 0 | 1.27 |
| stale_R1_BiCGSTAB2_stale_prec | 11/11 | 0/11 | 0/11 | 0.766 | 0.967 | 24 | 151 | 156 | 52 | 26 | 0 | 0.18 |

## Best Mode Per Case

| case | best mode | pure loop ms | best loop ms | speedup | pure calls | calls | NR iters | accepted | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | stale_R2_Richardson | 10.153 | 16.587 | 0.612 | 5 | 3 | 7 | 4 | 1 |
| case6468rte | stale_R2_Richardson | 3.676 | 4.045 | 0.909 | 3 | 2 | 3 | 1 | 0 |
| case6470rte | stale_R2_Richardson | 3.651 | 4.016 | 0.909 | 3 | 2 | 3 | 1 | 0 |
| case6495rte | stale_R1_BiCGSTAB1_stale_prec | 2.422 | 2.444 | 0.991 | 2 | 2 | 2 | 0 | 0 |
| case6515rte | stale_R2_Richardson | 3.856 | 4.218 | 0.914 | 3 | 2 | 3 | 1 | 0 |
| case8387pegase | stale_R2_Richardson | 4.931 | 5.100 | 0.967 | 3 | 2 | 3 | 1 | 0 |
| case9241pegase | stale_R1_BiCGSTAB2_stale_prec | 10.016 | 17.087 | 0.586 | 6 | 2 | 7 | 5 | 0 |
| case_ACTIVSg10k | stale_R1_BiCGSTAB1_stale_prec | 6.018 | 6.678 | 0.901 | 4 | 1 | 4 | 3 | 0 |
| case_ACTIVSg25k | stale_R1_BiCGSTAB1_stale_prec | 12.602 | 11.674 | 1.079 | 4 | 2 | 4 | 2 | 0 |
| case_ACTIVSg70k | stale_R1_BiCGSTAB1_stale_prec | 41.754 | 34.527 | 1.209 | 6 | 2 | 6 | 4 | 0 |
| case_SyntheticUSA | stale_R1_BiCGSTAB1_stale_prec | 44.608 | 40.415 | 1.104 | 6 | 2 | 6 | 4 | 0 |

## R2 Richardson Versus Stale-Preconditioned BiCGSTAB

| case | R2 loop ms | BICG1 loop ms | BICG2 loop ms | R2 NR | BICG1 NR | BICG2 NR |
|---|---:|---:|---:|---:|---:|---:|
| case13659pegase | 16.587 | 19.610 | 17.314 | 7 | 9 | 6 |
| case6468rte | 4.045 | 4.167 | 4.830 | 3 | 3 | 3 |
| case6470rte | 4.016 | 4.098 | 4.810 | 3 | 3 | 3 |
| case6495rte | 2.453 | 2.444 | 2.457 | 2 | 2 | 2 |
| case6515rte | 4.218 | 4.306 | 5.035 | 3 | 3 | 3 |
| case8387pegase | 5.100 | 5.185 | 6.042 | 3 | 3 | 3 |
| case9241pegase | 24.242 | 29.332 | 17.087 | 13 | 16 | 7 |
| case_ACTIVSg10k | 8.567 | 6.678 | 9.975 | 5 | 4 | 4 |
| case_ACTIVSg25k | 12.580 | 11.674 | 16.047 | 4 | 4 | 4 |
| case_ACTIVSg70k | 42.965 | 34.527 | 49.667 | 7 | 6 | 6 |
| case_SyntheticUSA | 45.980 | 40.415 | 46.768 | 7 | 6 | 6 |

## Factor Age

- stale_R2_Richardson: max factor_age=9, rows with age>=5=5, mean mismatch ratio at age>=5=0.304
- stale_R1_BiCGSTAB1_stale_prec: max factor_age=13, rows with age>=5=11, mean mismatch ratio at age>=5=0.394
- stale_R1_BiCGSTAB2_stale_prec: max factor_age=4, rows with age>=5=0, mean mismatch ratio at age>=5=0.000

## Answers

1. Stale-preconditioned BiCGSTAB reduces fallback versus R2 Richardson: both BiCGSTAB variants had 0 fallback calls, while R2 had 2. But fallback was already low for R2.
2. BiCGSTAB(1) is better than BiCGSTAB(2) on cost. BiCGSTAB(2) improves trajectory on some hard cases, but its extra stale solves make it slower on most cases.
3. The stale cuDSS solve count becomes large. This often consumes the factorize savings, especially for BiCGSTAB(2), which uses four stale preconditioner solves per middle step plus the predictor solve.
4. NR iterations are +0 to +1 only on many cases for stale-preconditioned BiCGSTAB, but hard cases still exceed that target.
5. NR-loop time beats pure cuDSS on a subset, not all. R2 Richardson remains the best median loop-speed option in this comparison.
6. Factor age does not immediately collapse quality, but high factor age correlates with long trajectories on difficult cases such as case9241pegase.
