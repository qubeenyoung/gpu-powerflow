# BiCGSTAB(2) + METIS block-Jacobi, Reuse Path Recheck

This rerun uses `--bicgstab-fused-fixed2 true` and `--bj-setup numeric_reuse_after_full_cudss`. Pure cuDSS remains preanalyzed: full-J analyze is setup-only, and NR-loop time is warm factorize+solve only.

## Main Answer

- With the reuse path, block size 8 still looks better than 16 by median NR-loop speedup: 0.497x vs 0.495x.
- Real hybrid wins with accepted middle steps: bs8 0/11, bs16 0/11.
- Aggregate full cuDSS call reduction: bs8 2, bs16 3. Accepted middle / fallback: bs8 19/23, bs16 17/23.
- The reuse path removes much of the repeated block-Jacobi setup cost, but it still does not make BiCGSTAB(2)+block-Jacobi a winning speed path against warm cuDSS.

## Best Reuse Result Per Case

| case | best bs | pure loop ms | reuse hybrid loop ms | loop speedup | pure calls | hybrid calls | call reduction | accepted | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 16 | 10.092 | 24.497 | 0.412 | 5 | 5 | 0 | 3 | 3 |
| case6468rte | 8 | 3.647 | 5.147 | 0.709 | 3 | 2 | 1 | 1 | 1 |
| case6470rte | 16 | 3.641 | 5.167 | 0.705 | 3 | 2 | 1 | 1 | 1 |
| case6495rte | 16 | 2.484 | 4.728 | 0.525 | 2 | 2 | 0 | 0 | 0 |
| case6515rte | 8 | 3.878 | 7.797 | 0.497 | 3 | 3 | 0 | 1 | 1 |
| case8387pegase | 16 | 4.931 | 9.396 | 0.525 | 3 | 3 | 0 | 1 | 1 |
| case9241pegase | 16 | 9.927 | 16.626 | 0.597 | 6 | 4 | 2 | 3 | 3 |
| case_ACTIVSg10k | 8 | 5.994 | 17.055 | 0.351 | 4 | 4 | 0 | 3 | 2 |
| case_ACTIVSg25k | 16 | 12.621 | 28.064 | 0.450 | 4 | 4 | 0 | 1 | 2 |
| case_ACTIVSg70k | 16 | 41.997 | 106.281 | 0.395 | 6 | 6 | 0 | 1 | 4 |
| case_SyntheticUSA | 16 | 42.424 | 120.847 | 0.351 | 6 | 6 | 0 | 2 | 4 |

## Every-middle vs Reuse, Best Per Case

| case | old best bs | old loop ms | reuse best bs | reuse loop ms | change |
|---|---:|---:|---:|---:|---:|
| case13659pegase | 16 | 28.968 | 16 | 24.497 | 4.471 |
| case6468rte | 8 | 6.344 | 8 | 5.147 | 1.197 |
| case6470rte | 16 | 6.234 | 16 | 5.167 | 1.067 |
| case6495rte | 16 | 2.424 | 16 | 4.728 | -2.305 |
| case6515rte | 16 | 7.777 | 8 | 7.797 | -0.021 |
| case8387pegase | 16 | 9.439 | 16 | 9.396 | 0.043 |
| case9241pegase | 8 | 19.243 | 16 | 16.626 | 2.617 |
| case_ACTIVSg10k | 8 | 18.216 | 8 | 17.055 | 1.162 |
| case_ACTIVSg25k | 8 | 23.050 | 16 | 28.064 | -5.014 |
| case_ACTIVSg70k | 16 | 106.440 | 16 | 106.281 | 0.159 |
| case_SyntheticUSA | 8 | 120.514 | 16 | 120.847 | -0.333 |

## Interpretation

1. The previous 4k+ report did not use BJ setup reuse; this rerun does.
2. Reuse helps the loop time on several cases, but the middle correction is still too weak, causing fallback or extra NR iterations.
3. Under warm cuDSS comparison, no accepted-middle case clearly beats pure cuDSS.
