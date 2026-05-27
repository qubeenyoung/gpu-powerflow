# Optimized cuDSS-Preconditioned GMRES(1)

Compared modes: pure cuDSS pre-analyzed, previous hybrid-style `stale_GMRES1`, and from-start `stale_GMRES1_refresh`. The refresh mode creates a full-J cuDSS factor as a preconditioner before the first step, applies GMRES(1) as the Newton correction, and refreshes the factor only after a rejected GMRES step. No block-Jacobi/A1/RAS/BpBpp is used.

## Case Summary

| case | mode | pure loop ms | loop ms | speedup | pure NR | NR | factor/solve calls | accepted | rejected | refresh/fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | hybrid_stale_GMRES1 | 10.055 | 18.817 | 0.534 | 5 | 11 | 2/5 | 9 | 0 | 0 |
| case6468rte | hybrid_stale_GMRES1 | 3.666 | 3.719 | 0.986 | 3 | 3 | 2/3 | 1 | 0 | 0 |
| case6470rte | hybrid_stale_GMRES1 | 3.653 | 3.699 | 0.988 | 3 | 3 | 2/3 | 1 | 0 | 0 |
| case6495rte | hybrid_stale_GMRES1 | 2.417 | 2.498 | 0.968 | 2 | 2 | 2/2 | 0 | 0 | 0 |
| case6515rte | hybrid_stale_GMRES1 | 3.893 | 3.941 | 0.988 | 3 | 3 | 2/3 | 1 | 0 | 0 |
| case8387pegase | hybrid_stale_GMRES1 | 5.108 | 4.741 | 1.077 | 3 | 3 | 2/3 | 1 | 0 | 0 |
| case9241pegase | hybrid_stale_GMRES1 | 9.981 | 22.398 | 0.446 | 6 | 16 | 2/6 | 14 | 0 | 0 |
| case_ACTIVSg10k | hybrid_stale_GMRES1 | 6.032 | 7.148 | 0.844 | 4 | 5 | 2/4 | 3 | 0 | 0 |
| case_ACTIVSg25k | hybrid_stale_GMRES1 | 12.662 | 10.964 | 1.155 | 4 | 4 | 2/4 | 2 | 0 | 0 |
| case_ACTIVSg70k | hybrid_stale_GMRES1 | 41.908 | 31.295 | 1.339 | 6 | 6 | 2/6 | 4 | 0 | 0 |
| case_SyntheticUSA | hybrid_stale_GMRES1 | 44.321 | 38.717 | 1.145 | 6 | 7 | 2/6 | 5 | 0 | 0 |
| case13659pegase | refresh_stale_GMRES1 | 10.055 | 29.136 | 0.345 | 5 | 17 | 1/5 | 17 | 0 | 0 |
| case6468rte | refresh_stale_GMRES1 | 3.666 | 4.288 | 0.855 | 3 | 3 | 1/3 | 3 | 0 | 0 |
| case6470rte | refresh_stale_GMRES1 | 3.653 | 4.181 | 0.874 | 3 | 3 | 1/3 | 3 | 0 | 0 |
| case6495rte | refresh_stale_GMRES1 | 2.417 | 2.926 | 0.826 | 2 | 2 | 1/2 | 2 | 0 | 0 |
| case6515rte | refresh_stale_GMRES1 | 3.893 | 4.323 | 0.900 | 3 | 3 | 1/3 | 3 | 0 | 0 |
| case8387pegase | refresh_stale_GMRES1 | 5.108 | 6.396 | 0.799 | 3 | 4 | 1/3 | 4 | 0 | 0 |
| case9241pegase | refresh_stale_GMRES1 | 9.981 | 31.322 | 0.319 | 6 | 20 | 3/6 | 19 | 1 | 2 |
| case_ACTIVSg10k | refresh_stale_GMRES1 | 6.032 | 8.180 | 0.737 | 4 | 6 | 1/4 | 6 | 0 | 0 |
| case_ACTIVSg25k | refresh_stale_GMRES1 | 12.662 | 12.015 | 1.054 | 4 | 5 | 1/4 | 5 | 0 | 0 |
| case_ACTIVSg70k | refresh_stale_GMRES1 | 41.908 | 31.097 | 1.348 | 6 | 7 | 1/6 | 7 | 0 | 0 |
| case_SyntheticUSA | refresh_stale_GMRES1 | 44.321 | 41.043 | 1.080 | 6 | 8 | 1/6 | 8 | 0 | 0 |

## Mode Summary

| mode | converged | loop wins | total wins | median loop speedup | median total speedup | NR <= pure+1 | call reduction | accepted | rejected | refresh/fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_stale_GMRES1 | 11/11 | 4/11 | 5/11 | 0.988 | 0.998 | 9/11 | 23 | 41 | 0 | 0 |
| refresh_stale_GMRES1 | 10/11 | 3/11 | 5/11 | 0.855 | 0.997 | 7/11 | 32 | 77 | 1 | 2 |

## hybrid_stale_GMRES1 middle-step average

- stale predictor solve: 0.639 ms
- stale preconditioner solve: 0.637 ms
- current-J residual SpMV: 0.020 ms
- GMRES SpMV: 0.020 ms
- GMRES fused dot/scalar sync: 0.054 ms
- GMRES total: 0.718 ms

## refresh_stale_GMRES1 middle-step average

- stale predictor solve: 0.579 ms
- stale preconditioner solve: 0.578 ms
- current-J residual SpMV: 0.019 ms
- GMRES SpMV: 0.018 ms
- GMRES fused dot/scalar sync: 0.049 ms
- GMRES total: 0.652 ms

## Decision

The from-start refresh mode did not improve median loop speed over the hybrid-style mode. Roll back to the hybrid-style stale_GMRES1 path as the safer baseline.
