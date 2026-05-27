# BiCGSTAB + METIS Block-Jacobi Report

Fixed policy: bootstrap cuDSS 1, polish threshold `1e-4`, accept `0.9`, reject `1.05`, fallback immediate, block size 64, FP32 inverse-GEMV block-Jacobi. BiCGSTAB uses fixed iterations, no ILU/ILUT/SpSV/coarse/scaling/AMGX.

## 1. Case-Level Hybrid Result

| case | solver | NR iters | cuDSS calls | middle calls | accepted | rejected | fallback | hybrid ms | pure cuDSS ms | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | MR1 | 12 | 5 | 10 | 7 | 3 | 3 | 32.11 | 46.62 | 1.452 |
| case3120sp | MR1 | 12 | 6 | 10 | 6 | 4 | 4 | 36.91 | 25.31 | 0.6856 |
| case9241pegase | MR1 | 12 | 4 | 10 | 8 | 2 | 2 | 73.88 | 52.71 | 0.7134 |
| case13659pegase | MR1 | 12 | 5 | 10 | 7 | 3 | 3 | 94.93 | 63.28 | 0.6666 |
| case6468rte | MR1 | 5 | 2 | 3 | 3 | 0 | 0 | 39.37 | 34.21 | 0.8691 |
| case2383wp | BiCGSTAB(1) | 9 | 5 | 7 | 4 | 3 | 3 | 29.67 | 46.57 | 1.57 |
| case3120sp | BiCGSTAB(1) | 8 | 6 | 6 | 2 | 4 | 4 | 32.28 | 25.21 | 0.7809 |
| case9241pegase | BiCGSTAB(1) | 9 | 4 | 7 | 5 | 2 | 2 | 66.79 | 52.97 | 0.7931 |
| case13659pegase | BiCGSTAB(1) | 10 | 6 | 8 | 4 | 4 | 4 | 91.2 | 63.56 | 0.697 |
| case6468rte | BiCGSTAB(1) | 4 | 2 | 2 | 2 | 0 | 0 | 37.12 | 34.22 | 0.9218 |
| case2383wp | BiCGSTAB(2) | 7 | 5 | 5 | 2 | 3 | 3 | 27.81 | 46.83 | 1.684 |
| case3120sp | BiCGSTAB(2) | 8 | 6 | 6 | 2 | 4 | 4 | 33.31 | 25.25 | 0.7578 |
| case9241pegase | BiCGSTAB(2) | 7 | 4 | 5 | 3 | 2 | 2 | 62.57 | 52.77 | 0.8433 |
| case13659pegase | BiCGSTAB(2) | 7 | 5 | 5 | 2 | 3 | 3 | 80.31 | 63.47 | 0.7903 |
| case6468rte | BiCGSTAB(2) | 3 | 2 | 1 | 1 | 0 | 0 | 35.35 | 34.29 | 0.9699 |
| case2383wp | BiCGSTAB(4) | 11 | 5 | 9 | 6 | 3 | 3 | 36.15 | 47.03 | 1.301 |
| case3120sp | BiCGSTAB(4) | 10 | 6 | 8 | 4 | 4 | 4 | 40.98 | 25.4 | 0.6197 |
| case9241pegase | BiCGSTAB(4) | 7 | 4 | 5 | 3 | 2 | 2 | 64.73 | 52.72 | 0.8145 |
| case13659pegase | BiCGSTAB(4) | 7 | 5 | 5 | 2 | 3 | 3 | 82.25 | 63.31 | 0.7698 |
| case6468rte | BiCGSTAB(4) | 3 | 2 | 2 | 1 | 1 | 1 | 38.07 | 34.47 | 0.9054 |

## 2. dx Quality: First Middle Trial

| solver | dx norm ratio | dx cosine | theta ratio | theta cosine | |V| ratio | middle trial ratio | linear rel res |
|---|---:|---:|---:|---:|---:|---:|---:|
| MR1 | 0.1193 | 0.3131 | 0.05043 | 0.2784 | 0.5897 | 0.366 | 0.2986 |
| BiCGSTAB(1) | 0.138 | 0.355 | 0.06543 | 0.3225 | 0.6626 | 0.2762 | 0.1963 |
| BiCGSTAB(2) | 0.1782 | 0.3954 | 0.1134 | 0.3587 | 0.7658 | 0.2018 | 0.1293 |
| BiCGSTAB(4) | 0.3032 | 0.4505 | 0.2482 | 0.4154 | 1.136 | 3.107 | 2.662 |

## 3. dx Quality: Accepted Non-Fallback Middle Trials

| solver | rows | dx norm ratio | dx cosine | theta ratio | theta cosine | |V| ratio | |V| cosine | middle trial ratio | linear rel res |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MR1 | 31 | 0.04791 | 0.2768 | 0.02237 | 0.2671 | 0.2434 | 0.3944 | 0.6669 | 0.6723 |
| BiCGSTAB(1) | 17 | 0.08351 | 0.2785 | 0.04095 | 0.2715 | 0.4797 | 0.5087 | 0.4803 | 0.4882 |
| BiCGSTAB(2) | 10 | 0.1411 | 0.2941 | 0.07263 | 0.286 | 0.6888 | 0.6618 | 0.2245 | 0.1843 |
| BiCGSTAB(4) | 16 | 0.2355 | 0.3211 | 0.1512 | 0.3127 | 1.147 | 0.7137 | 0.4339 | 0.4522 |

## 4. Timing Breakdown

| solver | middle total ms | block-Jacobi apply ms | BiCGSTAB SpMV ms | BiCGSTAB dot ms | BiCGSTAB update ms | MR1 SpMV ms | MR1 dot ms | MR1 update ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MR1 | 0.1282 | 0.01642 | 0 | 0 | 0 | 0.01152 | 0.02338 | 0.006721 |
| BiCGSTAB(1) | 0.205 | 0.03094 | 0.02262 | 0.115 | 0.02134 | 0 | 0 | 0 |
| BiCGSTAB(2) | 0.3285 | 0.05826 | 0.04199 | 0.1623 | 0.03596 | 0 | 0 | 0 |
| BiCGSTAB(4) | 0.5603 | 0.1093 | 0.07699 | 0.2535 | 0.06102 | 0 | 0 | 0 |

## 5. Judgment

- Fixed 1 and 2 BiCGSTAB improve the first middle correction versus MR1: larger dx norm ratio, better cosine, and lower first-step mismatch ratio.
- Accepted non-fallback trials show the same trend: BiCGSTAB(2) gives the best middle trial ratio (`0.224`) with cosine slightly above MR1.
- BiCGSTAB(4) is not attractive: it is slower and more unstable; it can make large bad trials and adds fallback in `case6468rte`.
- Fallback counts mostly do not decrease. BiCGSTAB reduces NR iterations, but the remaining fallback/cuDSS calls mean total hybrid time still beats pure cuDSS only for `case2383wp` in this run.
- Final decision: BiCGSTAB + block-Jacobi is a better middle-correction candidate than MR1, with fixed 2 as the best tested point, but it is not yet a reliable speedup strategy over cuDSS. The bottleneck is still correction quality/fallback, not BiCGSTAB kernel time.
