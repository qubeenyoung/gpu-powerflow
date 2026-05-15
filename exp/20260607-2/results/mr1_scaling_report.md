# MR1 row/column scaling experiment

## 읽는 법

- **none**: 기존 MR1 + METIS block-Jacobi, coarse/gamma 없이 사용한다.
- **ruiz**: `A_s = Dr A_perm Dc`, `b_s = Dr b_perm`에서 MR1을 수행하고 `dx_perm = Dc y`로 되돌린다.
- **middle trial ratio**는 shadow 진단의 `gmres_nonlinear_ratio_inf` 평균이다. 작을수록 middle step이 mismatch를 더 줄인다.
- **scaled/unscaled linear residual**은 각각 scaled system과 원래 permuted system 기준이다.

## 1. NR 결과

| case | scaling | converged | NR iters | cuDSS calls | MR1 calls | accepted | fallback | hybrid time(ms) | pure cuDSS(ms) | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | none | true | 12 | 5 | 10 | 7 | 3 | 74.93 | 63.46 | 0.847 |
| case13659pegase | ruiz | true | 14 | 5 | 12 | 9 | 3 | 97.53 | 63.23 | 0.648 |
| case2383wp | none | true | 12 | 5 | 10 | 7 | 3 | 28.75 | 110.23 | 3.835 |
| case2383wp | ruiz | true | 10 | 5 | 8 | 5 | 3 | 31.63 | 46.79 | 1.480 |
| case3120sp | none | true | 12 | 6 | 10 | 6 | 4 | 32.40 | 25.20 | 0.778 |
| case3120sp | ruiz | true | 12 | 6 | 10 | 6 | 4 | 38.75 | 25.26 | 0.652 |
| case6468rte | none | true | 5 | 2 | 3 | 3 | 0 | 35.88 | 34.30 | 0.956 |
| case6468rte | ruiz | true | 5 | 2 | 3 | 3 | 0 | 35.66 | 31.89 | 0.894 |
| case9241pegase | none | true | 12 | 4 | 10 | 8 | 2 | 59.63 | 55.82 | 0.936 |
| case9241pegase | ruiz | true | 11 | 4 | 9 | 7 | 2 | 70.49 | 53.68 | 0.762 |

## 2. dx quality

| case | dx_norm none | dx_norm ruiz | dx_cos none | dx_cos ruiz | theta_norm none | theta_norm ruiz | vmag_norm none | vmag_norm ruiz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 0.006383 | 0.003482 | 0.02662 | 0.04337 | 0.005751 | 0.002743 | 0.07109 | 0.06443 |
| case2383wp | 0.02168 | 0.02448 | 0.516 | 0.4446 | 0.01139 | 0.0123 | 0.2253 | 0.278 |
| case3120sp | 0.03116 | 0.03595 | 0.4357 | 0.4068 | 0.01649 | 0.02138 | 0.1059 | 0.1188 |
| case6468rte | 0.1438 | 0.1431 | 0.4564 | 0.4639 | 0.09502 | 0.09471 | 0.3464 | 0.3503 |
| case9241pegase | 0.06108 | 0.06935 | 0.1145 | 0.1213 | 0.01778 | 0.02022 | 0.3173 | 0.3595 |

## 3. middle quality

| case | middle trial ratio none | middle trial ratio ruiz | scaled linear rel ruiz | unscaled linear rel ruiz |
|---|---:|---:|---:|---:|
| case13659pegase | 0.8572 | 0.8965 | 0.8283 | 0.8625 |
| case2383wp | 0.7542 | 0.7288 | 0.6273 | 0.6365 |
| case3120sp | 0.8193 | 0.8278 | 0.6329 | 0.7673 |
| case6468rte | 0.5034 | 0.5083 | 0.6764 | 0.5886 |
| case9241pegase | 0.6858 | 0.6796 | 0.5617 | 0.6365 |

## 4. timing

| case | scaling total ruiz(ms) | middle total none(ms) | middle total ruiz(ms) | setup none(ms) | setup ruiz(ms) |
|---|---:|---:|---:|---:|---:|
| case13659pegase | 0.9199 | 0.1538 | 0.1637 | 0.8945 | 2.3140 |
| case2383wp | 0.2107 | 0.1052 | 0.1095 | 0.3701 | 0.8199 |
| case3120sp | 0.2108 | 0.1078 | 0.1122 | 0.4395 | 0.8858 |
| case6468rte | 0.2297 | 0.1249 | 0.1272 | 0.3733 | 0.8974 |
| case9241pegase | 0.6885 | 0.1415 | 0.1503 | 0.6220 | 1.8029 |

## 5. 판단

- dx_norm_ratio가 2배 이상 오른 케이스: none.
- fallback이 줄어든 케이스: none.
- 성공 기준은 dx 크기/방향, middle trial ratio, fallback/NR 반복, 총 시간을 함께 본다.
- 이번 결과가 이 상태라면 raw scaling이 주된 원인이라는 가설은 약하다.
