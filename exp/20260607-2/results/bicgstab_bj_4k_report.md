# BiCGSTAB(2) + METIS block-Jacobi, 4k+ Warm cuDSS Recheck

All rows use preanalyzed full-J cuDSS: full-J analyze is setup-only, and NR-loop timing includes factorize+solve only for cuDSS steps. Hybrid uses only BiCGSTAB(2) + METIS block-Jacobi with FP32 inverse_gemv; A1, RAS, B′/B″, coarse, scaling, previous-dx, and block-ILU are excluded.

## Main Answer

- Block size 8 is better by median NR-loop speedup: 0.516x vs 0.441x for block size 16.
- Full cuDSS calls were reduced in aggregate by 4 calls for bs8 and 2 calls for bs16, but this did not translate into NR-loop speedup on any case.
- NR-loop wins vs pure cuDSS: bs8 1/11, bs16 1/11 numerically, but the only win is `case6495rte`, where no middle step runs and the difference is measurement noise. Real hybrid wins with accepted middle steps: 0/11.
- Accepted middle steps / fallback calls: bs8 15/21, bs16 17/23. The accepted steps are too weak; fallback and extra NR iterations erase the saved cuDSS calls.
- Conclusion: for 4k+ cases, this BiCGSTAB(2)+block-Jacobi direction should not be continued as a speed path unless the preconditioner quality changes substantially.

## Case Summary, Best of bs8/bs16 by NR-loop

| case | best bs | pure loop ms | hybrid loop ms | loop speedup | pure calls | hybrid calls | call reduction | NR iters | accepted | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 16 | 10.092 | 28.968 | 0.348 | 5 | 5 | 0 | 8 | 3 | 3 |
| case6468rte | 8 | 3.647 | 6.344 | 0.575 | 3 | 2 | 1 | 3 | 1 | 1 |
| case6470rte | 16 | 3.641 | 6.234 | 0.584 | 3 | 2 | 1 | 3 | 1 | 1 |
| case6495rte | 16 | 2.484 | 2.424 | 1.025 | 2 | 2 | 0 | 2 | 0 | 0 |
| case6515rte | 16 | 3.878 | 7.777 | 0.499 | 3 | 3 | 0 | 4 | 1 | 1 |
| case8387pegase | 16 | 4.931 | 9.439 | 0.522 | 3 | 3 | 0 | 4 | 1 | 1 |
| case9241pegase | 8 | 9.927 | 19.243 | 0.516 | 6 | 4 | 2 | 7 | 3 | 2 |
| case_ACTIVSg10k | 8 | 5.994 | 18.216 | 0.329 | 4 | 4 | 0 | 7 | 3 | 2 |
| case_ACTIVSg25k | 8 | 12.621 | 23.050 | 0.548 | 4 | 4 | 0 | 4 | 0 | 2 |
| case_ACTIVSg70k | 16 | 41.997 | 106.440 | 0.395 | 6 | 6 | 0 | 7 | 1 | 4 |
| case_SyntheticUSA | 8 | 42.424 | 120.514 | 0.352 | 6 | 6 | 0 | 7 | 1 | 4 |

## Block Size Comparison

| block size | median loop speedup | loop wins | total wins | cuDSS call reduction | accepted middle | fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.516 | 1/11 | 0/11 | 4 | 15 | 21 |
| 16 | 0.441 | 1/11 | 0/11 | 2 | 17 | 23 |

## Interpretation

1. bs8 is marginally better than bs16, mostly because it avoids some extra NR-loop cost, not because it becomes faster than cuDSS.
2. The hybrid often reduces full cuDSS calls, but the saved calls are replaced by BiCGSTAB trials plus fallback/polish calls and extra NR iterations.
3. No 4k+ case with actual accepted BiCGSTAB middle steps beats pure cuDSS on NR-loop-only time under this clean warm comparison.
4. Including setup makes hybrid much worse because METIS/block-Jacobi setup is large; this is reported separately and not used to hide the loop result.
