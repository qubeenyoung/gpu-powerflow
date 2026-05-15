# Scaled MR1 + coarse experiment

## 읽는 법

- **MR1 + coarse**: 기존 방식이다. MR1이 만든 `dx`를 그대로 적용한다.
- **scaled MR1 + coarse**: 같은 `dx`에 대해 `gamma = 4, 2, 1`을 임시 적용해 보고, mismatch_inf가 가장 작은 gamma만 선택한다.
- **middle trial ratio**: middle solver 후보 step 후 `mismatch_inf_after / mismatch_inf_before`이다. 작을수록 Newton 진행이 좋다.
- **scaled dx 크기비**: shadow 진단의 `||dx_mr1|| / ||dx_cuDSS||`에 선택된 gamma를 곱한 값이다.

## 1. NR 결과

| 케이스 | 모드 | 수렴 | NR 반복 | cuDSS 호출 | MR1 호출 | Accepted | Rejected | Fallback | 시간(s) | Pure cuDSS(s) | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | MR1 + coarse | true | 8 | 5 | 6 | 3 | 3 | 3 | 0.0713 | 0.0630 | 0.883 |
| case13659pegase | scaled MR1 + coarse | true | 10 | 5 | 8 | 5 | 3 | 3 | 0.0777 | 0.0631 | 0.812 |
| case2383wp | MR1 + coarse | true | 10 | 5 | 8 | 5 | 3 | 3 | 0.0283 | 0.0480 | 1.697 |
| case2383wp | scaled MR1 + coarse | true | 10 | 5 | 9 | 5 | 4 | 4 | 0.0301 | 0.0459 | 1.525 |
| case3120sp | MR1 + coarse | true | 9 | 6 | 7 | 3 | 4 | 4 | 0.0309 | 0.0256 | 0.830 |
| case3120sp | scaled MR1 + coarse | true | 8 | 6 | 6 | 2 | 4 | 4 | 0.0315 | 0.0253 | 0.804 |
| case6468rte | MR1 + coarse | true | 4 | 3 | 2 | 1 | 1 | 1 | 0.0362 | 0.0337 | 0.930 |
| case6468rte | scaled MR1 + coarse | true | 4 | 3 | 2 | 1 | 1 | 1 | 0.0373 | 0.0342 | 0.917 |
| case9241pegase | MR1 + coarse | true | 12 | 4 | 10 | 8 | 2 | 2 | 0.0621 | 0.0531 | 0.854 |
| case9241pegase | scaled MR1 + coarse | true | 10 | 4 | 8 | 6 | 2 | 2 | 0.0617 | 0.0528 | 0.856 |

## 2. Middle step

| 케이스 | 모드 | total middle 평균(ms) | 추가 mismatch 평가(ms) | 선택 gamma 평균 | gamma count | scaled dx 크기비 | dx 방향유사도 | middle trial ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| case13659pegase | MR1 + coarse | 1.2626 | 0.0000 | 0 |  | 0.004491 | 0.3136 | 0.9503 |
| case13659pegase | scaled MR1 + coarse | 1.6930 | 0.4582 | 1.25 | 1:6, 2:2 | 0.004551 | 0.1843 | 0.8512 |
| case2383wp | MR1 + coarse | 0.6066 | 0.0000 | 0 |  | 0.02613 | 0.4437 | 0.7712 |
| case2383wp | scaled MR1 + coarse | 0.9003 | 0.3111 | 1.56 | 1:6, 2:2, 4:1 | 0.02086 | 0.2283 | 0.7611 |
| case3120sp | MR1 + coarse | 0.6345 | 0.0000 | 0 |  | 0.0389 | 0.4591 | 0.9087 |
| case3120sp | scaled MR1 + coarse | 0.9702 | 0.3361 | 1 | 1:6 | 0.04515 | 0.4241 | 0.8915 |
| case6468rte | MR1 + coarse | 1.0403 | 0.0000 | 0 |  | 0.1082 | 0.5375 | 0.9025 |
| case6468rte | scaled MR1 + coarse | 1.4278 | 0.3612 | 1 | 1:2 | 0.1082 | 0.5375 | 0.9025 |
| case9241pegase | MR1 + coarse | 0.9578 | 0.0000 | 0 |  | 0.0574 | 0.1706 | 0.7051 |
| case9241pegase | scaled MR1 + coarse | 1.3715 | 0.3876 | 1.12 | 1:7, 2:1 | 0.09609 | 0.1724 | 0.6623 |

## 3. 판단

- 평균 hybrid 시간: baseline `0.0458s`, scaled `0.0476s`.
- 평균 fallback 수: baseline `2.60`, scaled `2.80`.
- 성공 판단은 scaled가 fallback 또는 NR 반복을 줄이고, 추가 mismatch 평가 비용을 감수해도 총 시간이 개선되는지로 본다.
