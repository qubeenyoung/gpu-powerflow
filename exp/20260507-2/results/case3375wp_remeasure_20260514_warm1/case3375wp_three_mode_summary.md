# case3375wp three-mode remeasurement

Config: warmup=1, full-J cuDSS analyze outside NR loop. cuITER uses BiCGSTAB(2)+METIS block-Jacobi bs=8. cuDSS-preconditioned cuITER uses stale_GMRES1 with bootstrap=1 and polish disabled.

| mode | conv | NR | total ms | ms/iter | pure full cuDSS calls | actual full cuDSS calls | iter calls | final inf | avg linear setup+solve ms | avg middle ms | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pure_cuDSS | true | 2 | 2.041 | 1.021 | 2 | 2 | 0 | 1.238e-11 | 0.682 | 0.000 |  |
| cuITER_BiCGSTAB2_BJ_bs8 | false | 20 | 26.684 | 1.334 | 2 | 0 | 20 | 2.038e-03 | 1.024 | 0.231 | no accept/reject/fallback; max_iter reached |
| cuITER_GMRES1_cuDSS_prec | true | 2 | 2.158 | 1.079 | 2 | 1 | 1 | 1.989e-11 | 0.625 | 0.268 | 1 full factor+solve bootstrap, 1 stale GMRES step |

Detailed iteration CSVs are in the same directory.
