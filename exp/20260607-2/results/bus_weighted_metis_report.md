# Bus-weighted METIS block-Jacobi experiment

## 1. Case summary

| case | partition | converged | NR | cuDSS | MR1 | accepted | fallback | hybrid ms | pure cuDSS ms | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | unknown_metis | true | 12 | 5 | 10 | 7 | 3 | 94.77 | 63.34 | 0.668 |
| case13659pegase | bus_weighted_metis | true | 5 | 5 | 4 | 0 | 4 | 201.05 | 63.26 | 0.315 |
| case2383wp | unknown_metis | true | 12 | 5 | 10 | 7 | 3 | 32.05 | 46.95 | 1.465 |
| case2383wp | bus_weighted_metis | true | 7 | 6 | 5 | 1 | 4 | 44.25 | 47.32 | 1.069 |
| case3120sp | unknown_metis | true | 12 | 6 | 10 | 6 | 4 | 36.65 | 25.22 | 0.688 |
| case3120sp | bus_weighted_metis | true | 6 | 6 | 4 | 0 | 4 | 52.09 | 25.22 | 0.484 |
| case6468rte | unknown_metis | true | 5 | 2 | 3 | 3 | 0 | 38.91 | 34.33 | 0.882 |
| case6468rte | bus_weighted_metis | true | 3 | 2 | 2 | 1 | 1 | 94.14 | 36.53 | 0.388 |
| case9241pegase | unknown_metis | true | 12 | 4 | 10 | 8 | 2 | 75.33 | 52.76 | 0.700 |
| case9241pegase | bus_weighted_metis | true | 7 | 5 | 5 | 2 | 3 | 145.26 | 52.72 | 0.363 |

## 2. Partition stats

| case | partition | blocks | min | max | avg | std | diag nnz | diag weighted | J11 | J12 | J21 | J22 | theta/V split | P/Q split |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | unknown_metis | 517 | 32 | 64 | 44.92 | 15.10 | 0.711 | 0.955 | 0.944 | 0.690 | 0.688 | 0.981 | 3283 | 3283 |
| case13659pegase | bus_weighted_metis | 363 | 57 | 65 | 63.98 | 1.20 | 0.827 | 0.997 | 0.997 | 0.998 | 0.998 | 0.997 | 0 | 0 |
| case2383wp | unknown_metis | 91 | 32 | 64 | 48.77 | 15.08 | 0.785 | 0.935 | 0.919 | 0.819 | 0.819 | 0.959 | 585 | 585 |
| case2383wp | bus_weighted_metis | 70 | 61 | 65 | 63.40 | 1.26 | 0.902 | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 | 0 | 0 |
| case3120sp | unknown_metis | 120 | 32 | 64 | 49.92 | 15.25 | 0.794 | 0.937 | 0.927 | 0.683 | 0.688 | 0.966 | 765 | 765 |
| case3120sp | bus_weighted_metis | 94 | 60 | 65 | 63.73 | 1.21 | 0.895 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| case6468rte | unknown_metis | 261 | 32 | 64 | 48.44 | 15.42 | 0.762 | 0.986 | 0.994 | 0.896 | 0.899 | 0.992 | 1953 | 1953 |
| case6468rte | bus_weighted_metis | 198 | 60 | 65 | 63.85 | 1.04 | 0.874 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| case9241pegase | unknown_metis | 381 | 32 | 64 | 44.71 | 14.93 | 0.703 | 0.964 | 0.970 | 0.547 | 0.556 | 0.980 | 3157 | 3157 |
| case9241pegase | bus_weighted_metis | 267 | 59 | 65 | 63.81 | 1.18 | 0.839 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0 | 0 |

## 3. dx and middle quality

| case | dx unknown | dx bus | cos unknown | cos bus | theta unknown | theta bus | vmag unknown | vmag bus | trial unknown | trial bus | linear rel bus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 0.006383 | 0.005705 | 0.02662 | 0.01545 | 0.005751 | 0.00539 | 0.07109 | 0.1297 | 0.8572 | 0.9616 | 0.9852 |
| case2383wp | 0.02168 | 0.001722 | 0.516 | 0.08319 | 0.01139 | 0.001536 | 0.2253 | 0.06833 | 0.7542 | 0.9755 | 0.9857 |
| case3120sp | 0.03116 | 0.01153 | 0.4357 | 0.06787 | 0.01649 | 0.006919 | 0.1059 | 0.02004 | 0.8193 | 0.9542 | 0.9756 |
| case6468rte | 0.1438 | 0.1669 | 0.4564 | 0.2834 | 0.09502 | 0.1325 | 0.3464 | 0.2982 | 0.5034 | 0.7295 | 0.7443 |
| case9241pegase | 0.06108 | 0.1097 | 0.1145 | 0.07208 | 0.01778 | 0.05852 | 0.3173 | 0.2744 | 0.6858 | 0.7003 | 0.6837 |

## 4. Timing

| case | partition | partition ms | weighted graph ms | avg setup ms | avg middle solve ms |
|---|---|---:|---:|---:|---:|
| case13659pegase | unknown_metis | 125.99 | 0.00 | 2.880 | 0.154 |
| case13659pegase | bus_weighted_metis | 115.80 | 3.67 | 35.163 | 0.157 |
| case2383wp | unknown_metis | 18.50 | 0.00 | 0.707 | 0.106 |
| case2383wp | bus_weighted_metis | 15.63 | 0.43 | 4.281 | 0.106 |
| case3120sp | unknown_metis | 25.94 | 0.00 | 0.872 | 0.108 |
| case3120sp | bus_weighted_metis | 21.16 | 0.53 | 6.854 | 0.108 |
| case6468rte | unknown_metis | 59.19 | 0.00 | 1.358 | 0.127 |
| case6468rte | bus_weighted_metis | 52.62 | 1.44 | 30.256 | 0.132 |
| case9241pegase | unknown_metis | 85.58 | 0.00 | 2.025 | 0.143 |
| case9241pegase | bus_weighted_metis | 75.43 | 2.64 | 18.849 | 0.142 |

## 5. Judgment

- Bus-weighted METIS did what it was designed to do structurally: theta/Vm and P/Q split counts became zero, diagonal NNZ recovery increased, and weighted coupling recovery became almost 1.0 in every case.
- dx_norm_ratio improved only in case6468rte, case9241pegase; it dropped sharply on case2383wp and case3120sp, which were the most useful unknown-METIS cases.
- fallback decreased in none; bus-weighted either kept or increased fallback pressure.
- NR iterations decreased in case13659pegase, case2383wp, case3120sp, case6468rte, case9241pegase, but this mostly came from doing more direct cuDSS fallback/polish work rather than accepting better MR1 steps.
- Final call: bus-aware weighted partition improves block locality metrics, but it does not improve this MR1 block-Jacobi middle solver enough to be useful. The preconditioner-quality bottleneck is not fixed by preserving same-bus theta/Vm and high-weight bus couplings alone.
