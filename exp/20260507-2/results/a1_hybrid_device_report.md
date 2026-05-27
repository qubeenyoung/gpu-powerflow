# A1 Device Hybrid NR Report

Settings: block_size=16, BiCGSTAB(2), METIS block-Jacobi, device-resident A1 field correction. Policy A uses accept=0.9. Policy B uses accept=0.5 and max_a1_middle_accepts=2.

## Pure cuDSS
| case | conv | NR | pure NR | full cuDSS | middle | acc | rej | fb | polish | total ms | pure total ms | total speedup | linear ms | pure linear ms | linear speedup | final inf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | true | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 47.75 | 47.75 | 1.00x | 40.34 | 40.34 | 1.00x | 3.97e-12 |
| case3120sp | true | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 25.32 | 25.32 | 1.00x | 23.42 | 23.42 | 1.00x | 7.59e-12 |
| case9241pegase | true | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 53.05 | 53.05 | 1.00x | 50.72 | 50.72 | 1.00x | 2.13e-09 |
| case13659pegase | true | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 63.62 | 63.62 | 1.00x | 61.36 | 61.36 | 1.00x | 2.29e-09 |
| case6468rte | true | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 34.32 | 34.32 | 1.00x | 33.17 | 33.17 | 1.00x | 7.77e-12 |

## BJ baseline
| case | conv | NR | pure NR | full cuDSS | middle | acc | rej | fb | polish | total ms | pure total ms | total speedup | linear ms | pure linear ms | linear speedup | final inf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | true | 8 | 6 | 5 | 7 | 3 | 4 | 4 | 0 | 56.31 | 47.75 | 0.85x | 48.00 | 40.34 | 0.84x | 6.26e-10 |
| case3120sp | true | 8 | 6 | 6 | 6 | 2 | 4 | 4 | 1 | 32.70 | 25.32 | 0.77x | 29.65 | 23.42 | 0.79x | 1.14e-11 |
| case9241pegase | true | 7 | 6 | 5 | 5 | 2 | 3 | 3 | 1 | 63.96 | 53.05 | 0.83x | 60.78 | 50.72 | 0.83x | 7.67e-12 |
| case13659pegase | true | 8 | 5 | 5 | 6 | 3 | 3 | 3 | 1 | 82.55 | 63.62 | 0.77x | 78.38 | 61.36 | 0.78x | 3.70e-12 |
| case6468rte | true | 4 | 3 | 2 | 3 | 2 | 1 | 1 | 0 | 39.05 | 34.32 | 0.88x | 37.45 | 33.17 | 0.89x | 8.59e-09 |

## A1 device (standard)
| case | conv | NR | pure NR | full cuDSS | A1 calls | acc | rej | fb | polish | total ms | pure total ms | total speedup | linear ms | pure linear ms | linear speedup | final inf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | true | 7 | 6 | 2 | 6 | 5 | 1 | 1 | 0 | 57.19 | 47.75 | 0.83x | 51.30 | 40.34 | 0.79x | 6.15e-09 |
| case3120sp | true | 9 | 6 | 3 | 7 | 6 | 1 | 1 | 1 | 37.79 | 25.32 | 0.67x | 37.35 | 23.42 | 0.63x | 7.58e-12 |
| case9241pegase | true | 7 | 6 | 3 | 5 | 4 | 1 | 1 | 1 | 69.23 | 53.05 | 0.77x | 69.52 | 50.72 | 0.73x | 1.59e-11 |
| case13659pegase | true | 6 | 5 | 3 | 4 | 3 | 1 | 1 | 1 | 82.82 | 63.62 | 0.77x | 83.44 | 61.36 | 0.74x | 3.04e-10 |
| case6468rte | true | 3 | 3 | 2 | 1 | 1 | 0 | 0 | 1 | 36.40 | 34.32 | 0.94x | 35.80 | 33.17 | 0.93x | 1.36e-10 |

## A1 device (strict)
| case | conv | NR | pure NR | full cuDSS | A1 calls | acc | rej | fb | polish | total ms | pure total ms | total speedup | linear ms | pure linear ms | linear speedup | final inf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | true | 6 | 6 | 4 | 2 | 2 | 0 | 0 | 1 | 51.53 | 47.75 | 0.93x | 44.87 | 40.34 | 0.90x | 7.13e-12 |
| case3120sp | true | 6 | 6 | 4 | 2 | 2 | 0 | 0 | 1 | 27.75 | 25.32 | 0.91x | 26.54 | 23.42 | 0.88x | 7.55e-12 |
| case9241pegase | true | 6 | 6 | 4 | 2 | 2 | 0 | 0 | 0 | 57.48 | 53.05 | 0.92x | 56.46 | 50.72 | 0.90x | 2.79e-09 |
| case13659pegase | true | 6 | 5 | 4 | 2 | 2 | 0 | 0 | 1 | 71.03 | 63.62 | 0.90x | 70.01 | 61.36 | 0.88x | 4.75e-12 |
| case6468rte | true | 3 | 3 | 2 | 1 | 1 | 0 | 0 | 1 | 36.42 | 34.32 | 0.94x | 35.80 | 33.17 | 0.93x | 1.36e-10 |

## Answers
1. A1 hybrid converged on all 5 cases for both standard and strict policies.
2. A1 standard reduced full-J cuDSS calls versus pure from 26 to 13; strict used 18.
3. Standard increased total NR iterations from pure 26 to 32; strict matched the pure total at 27.
4. A1 did not beat pure cuDSS in total time: mean total speedup was 0.80x standard and 0.92x strict.
5. A1 did not beat pure cuDSS in linear solve time on average: mean linear speedup was 0.76x standard and 0.90x strict.
6. Strict is better for NR trajectory: zero fallback, same total NR iteration count as pure, and fewer weak accepted A1 steps. Standard reduces cuDSS calls more but stretches trajectories.
7. This run only evaluates block_size=16, so it does not establish an 8 vs 16 winner.
8. A1 should not be kept as the default hybrid middle solver yet: it is robust under strict policy, but current total/linear timings are still not below pure cuDSS.

## A1 Middle Timing
A1 middle wall mean/max: 1.331/5.423 ms. cuDSS event-sum mean/max: 1.829/2.843 ms. non-cuDSS overhead mean/max: 0.488/4.698 ms.