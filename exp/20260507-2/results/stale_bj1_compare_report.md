# Stale Full-J Predictor + Block-Jacobi Correction

## What was tested

Warm-analyze NR-loop timing was used. Full-J cuDSS analysis is outside the NR loop.

Modes:

- `pure_cudss_preanalyzed`: current Jacobian factorize + solve every NR iteration.
- `stale_GMRES1`: one cached full-J solve for predictor, then one cached full-J solve as GMRES(1) preconditioner.
- `stale_BJ1`: one cached full-J solve for predictor, then one METIS block-Jacobi MR1 correction.

`stale_BJ1` was measured with `block_size=16` and `bj_setup=numeric_reuse_after_full_cudss`.

## Case-level result

| case | pure loop ms | stale_GMRES1 ms | stale_BJ1 ms | pure iters | GMRES1 iters | BJ1 iters | pure full cuDSS | GMRES1 full cuDSS | BJ1 full cuDSS | GMRES1 fallback | BJ1 fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case6468rte | 3.733 | 3.729 | 4.855 | 3 | 3 | 3 | 3 | 2 | 2 | 0 | 0 |
| case6470rte | 3.725 | 3.744 | 4.868 | 3 | 3 | 3 | 3 | 2 | 2 | 0 | 0 |
| case6495rte | 2.483 | 2.485 | 2.473 | 2 | 2 | 2 | 2 | 2 | 2 | 0 | 0 |
| case6515rte | 3.955 | 3.914 | 4.977 | 3 | 3 | 3 | 3 | 2 | 2 | 0 | 0 |
| case8387pegase | 4.973 | 4.741 | 7.450 | 3 | 3 | 4 | 3 | 2 | 2 | 0 | 0 |
| case9241pegase | 10.169 | 22.564 | 23.780 | 6 | 16 | 17 | 6 | 2 | 2 | 0 | 0 |
| case13659pegase | 10.223 | 17.943 | 30.236 | 5 | 11 | 16 | 5 | 2 | 3 | 0 | 1 |
| case_ACTIVSg10k | 6.082 | 7.203 | 8.509 | 4 | 5 | 5 | 4 | 2 | 2 | 0 | 0 |
| case_ACTIVSg25k | 12.706 | 10.683 | 18.383 | 4 | 4 | 6 | 4 | 2 | 2 | 0 | 0 |
| case_ACTIVSg70k | 41.108 | 30.262 | 81.852 | 6 | 6 | 8 | 6 | 2 | 4 | 0 | 2 |
| case_SyntheticUSA | 43.395 | 38.724 | 97.614 | 6 | 7 | 10 | 6 | 2 | 4 | 0 | 2 |

## Middle-step timing

Accepted middle steps only, excluding fallback steps:

| mode | accepted middle steps | median middle ms | median linear solve ms | median stale solve calls | predictor solve ms | correction solve/apply ms | median mismatch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| stale_GMRES1 | 41 | 0.843 | 0.814 | 2 | 0.377 | 0.374 stale solve | 0.241 |
| stale_BJ1 | 50 | 0.808 | 0.767 | 1 | 0.463 | 0.013 BJ apply | 0.276 |

## Interpretation

`stale_BJ1` does save the second cached cuDSS solve in the middle step. The middle-step median is slightly lower than `stale_GMRES1`.

However, the block-Jacobi correction is weaker than the stale-factor correction. It accepts more weak middle steps, grows the NR trajectory on hard cases, and introduces fallback on `case13659pegase`, `case_ACTIVSg70k`, and `case_SyntheticUSA`.

Bottom line: replacing the second stale solve with block-Jacobi is not worth keeping as the main path. The per-middle cost improves a little, but the nonlinear progress is worse enough that total NR-loop time loses on 10/11 cases.

## Output files

- `results/stale_bj1_compare_table.csv`
- `results/stale_bj1_warm_pure_summary.csv`
- `results/stale_bj1_warm_stale_gmres1_summary.csv`
- `results/stale_bj1_warm_bj1_reuse_summary.csv`
- `results/stale_bj1_warm_pure_iters.csv`
- `results/stale_bj1_warm_stale_gmres1_iters.csv`
- `results/stale_bj1_warm_bj1_reuse_iters.csv`
