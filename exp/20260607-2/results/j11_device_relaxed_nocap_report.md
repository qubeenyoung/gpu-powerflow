# J11-only relaxed/no-cap device correction
- Policy: `accept=0.9`, no A1 accept cap, `fallback=immediate`, `bootstrap=1`, `polish=1e-4`, `block_size=16`, `BiCGSTAB(2)+BJ`, `bj_setup=numeric_reuse_after_full_cudss`, fused fixed2 on.
- New mode: `bicgstab_block_jacobi_j11_device`. It applies only `J11 dtheta = rP`; it skips J22 factor/solve and skips A1 round1 cross residual.
- Comparison A1 file: `results/a1_warm_nocap_preanalyze_summary.csv` / `_timing.csv`. J11 output: `results/j11_device_relaxed_nocap_summary.csv` / `_timing.csv`.

| case | method | NR | full cuDSS | middle | accepted | fallback | total ms | median accepted middle ms | median field wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | J11-only | 7 | 4 | 6 | 3 | 3 | 16.860 | 1.350 | 0.457 |
| case2383wp | A1 | 9 | 2 | 7 | 7 | 0 | 13.070 | 1.083 | 0.615 |
| case3120sp | J11-only | 6 | 4 | 5 | 2 | 3 | 11.900 | 1.015 | 0.539 |
| case3120sp | A1 | 6 | 3 | 4 | 3 | 1 | 10.418 | 1.205 | 0.731 |
| case9241pegase | J11-only | 6 | 4 | 4 | 2 | 2 | 18.473 | 1.330 | 0.774 |
| case9241pegase | A1 | 6 | 3 | 5 | 3 | 2 | 18.541 | 1.852 | 1.048 |
| case13659pegase | J11-only | 7 | 4 | 6 | 3 | 3 | 26.108 | 1.529 | 0.855 |
| case13659pegase | A1 | 6 | 3 | 4 | 3 | 1 | 20.609 | 1.923 | 1.351 |
| case6468rte | J11-only | 4 | 3 | 2 | 1 | 1 | 9.071 | 1.206 | 0.677 |
| case6468rte | A1 | 3 | 2 | 1 | 1 | 0 | 6.948 | 1.643 | 1.114 |

## Readout
- J11-only field wall is lower than A1 on `5/5` cases, as expected from skipping J22 and round1.
- Accepted middle time is lower than A1 on `4/5` cases, but the nonlinear step is weaker overall.
- J11-only has more fallback than A1 on `4/5` cases and more full-J cuDSS calls on `5/5` cases.
- The main outcome: J11-only saves field correction work, but it gives up too much correction quality; under relaxed/no-cap it falls back more often than A1.
