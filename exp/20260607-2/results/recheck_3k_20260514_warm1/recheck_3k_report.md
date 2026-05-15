# 3k-scale NR Solver Recheck

Settings: cases `case3012wp,case3120sp,case3375wp`; warmup=1; full cuDSS analyze before NR loop; cuITER/hybrid uses BiCGSTAB(2)+METIS block-Jacobi, block_size=8. Times below are NR-loop only and exclude one-time analyze/setup columns.

| case | mode | converged | NR iters | total NR-loop ms | ms / iter | cuDSS calls | iter calls | fallback | final mismatch inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case3012wp | pure_cudss | true | 3 | 2.843 | 0.948 | 3 | 0 | 0 | 7.542e-12 |
| case3120sp | pure_cudss | true | 6 | 5.738 | 0.956 | 6 | 0 | 0 | 7.555e-12 |
| case3375wp | pure_cudss | true | 2 | 1.851 | 0.926 | 2 | 0 | 0 | 1.235e-11 |
| case3012wp | cuiter_bicgstab2_bj_bs8_no_cudss | false | 2 | 2.577 | 1.289 | 0 | 2 | 0 | 4.694e-03 |
| case3120sp | cuiter_bicgstab2_bj_bs8_no_cudss | false | 2 | 2.588 | 1.294 | 0 | 2 | 0 | 4.736e+01 |
| case3375wp | cuiter_bicgstab2_bj_bs8_no_cudss | false | 1 | 1.372 | 1.372 | 0 | 1 | 0 | 0.000e+00 |
| case3012wp | hybrid_bicgstab2_bj_bs8 | true | 4 | 5.334 | 1.333 | 3 | 2 | 1 | 7.613e-12 |
| case3120sp | hybrid_bicgstab2_bj_bs8 | true | 9 | 14.278 | 1.586 | 6 | 7 | 4 | 9.800e-12 |
| case3375wp | hybrid_bicgstab2_bj_bs8 | true | 2 | 2.024 | 1.012 | 2 | 0 | 0 | 1.238e-11 |

Notes:
- `cuITER` mode disables cuDSS bootstrap/fallback/polish, so rejected iterative steps stop the run.
- `hybrid` mode uses bootstrap=1, polish threshold=1e-4, immediate fallback.
