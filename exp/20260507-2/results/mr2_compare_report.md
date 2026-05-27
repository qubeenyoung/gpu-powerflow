# 2D-MR coarse separated comparison

## 읽는 법

- **MR1 block-Jacobi**: local block-Jacobi correction만 사용한다.
- **MR1 + coarse merged**: local correction과 coarse correction을 먼저 더하고 scalar alpha 하나로 최소잔차 보정한다.
- **2D-MR + coarse separated**: local 방향과 coarse 방향에 서로 다른 계수 `a0`, `a1`을 둔 2D 최소잔차 보정이다.
- **중간 step mismatch ratio**: middle solver step 후 mismatch_inf / step 전 mismatch_inf. 작을수록 좋다.
- **dx 크기비**: `||dx_iterative||2 / ||dx_cuDSS||2`. 1에 가까울수록 cuDSS Newton step만큼 크다.
- **dx 방향유사도**: `cos(dx_iterative, dx_cuDSS)`. 1에 가까울수록 방향이 비슷하다.

## 1. NR 결과

| 케이스 | 모드 | 수렴 | Newton 반복 | Fallback | Hybrid 시간(s) |
|---|---|---:|---:|---:|---:|
| case13659pegase | MR1 block-Jacobi | true | 12 | 3 | 0.0745 |
| case13659pegase | MR1 block-Jacobi + coarse merged | true | 8 | 3 | 0.0724 |
| case13659pegase | 2D-MR block-Jacobi + coarse separated | true | 13 | 3 | 0.0760 |
| case2383wp | MR1 block-Jacobi | true | 12 | 3 | 0.0285 |
| case2383wp | MR1 block-Jacobi + coarse merged | true | 10 | 3 | 0.0250 |
| case2383wp | 2D-MR block-Jacobi + coarse separated | true | 12 | 3 | 0.0296 |
| case3120sp | MR1 block-Jacobi | true | 12 | 4 | 0.0336 |
| case3120sp | MR1 block-Jacobi + coarse merged | true | 9 | 4 | 0.0278 |
| case3120sp | 2D-MR block-Jacobi + coarse separated | true | 10 | 4 | 0.0317 |
| case6468rte | MR1 block-Jacobi | true | 5 | 0 | 0.0334 |
| case6468rte | MR1 block-Jacobi + coarse merged | true | 4 | 1 | 0.0368 |
| case6468rte | 2D-MR block-Jacobi + coarse separated | true | 5 | 0 | 0.0353 |
| case9241pegase | MR1 block-Jacobi | true | 12 | 2 | 0.0600 |
| case9241pegase | MR1 block-Jacobi + coarse merged | true | 12 | 2 | 0.0576 |
| case9241pegase | 2D-MR block-Jacobi + coarse separated | true | 13 | 2 | 0.0631 |

## 2. Middle step quality

| 케이스 | 모드 | middle solver 평균(ms) | dx 크기비 | dx 방향유사도 | 중간 step mismatch ratio |
|---|---|---:|---:|---:|---:|
| case13659pegase | MR1 block-Jacobi | 0.1530 | 0.006383 | 0.02662 | 0.8572 |
| case13659pegase | MR1 block-Jacobi + coarse merged | 0.3512 | 0.004491 | 0.3136 | 0.9503 |
| case13659pegase | 2D-MR block-Jacobi + coarse separated | 0.3167 | 0.0029 | 0.07808 | 0.8702 |
| case2383wp | MR1 block-Jacobi | 0.1053 | 0.02168 | 0.516 | 0.7542 |
| case2383wp | MR1 block-Jacobi + coarse merged | 0.1697 | 0.02613 | 0.4437 | 0.7711 |
| case2383wp | 2D-MR block-Jacobi + coarse separated | 0.1820 | 0.02167 | 0.5171 | 0.7529 |
| case3120sp | MR1 block-Jacobi | 0.1244 | 0.03116 | 0.4357 | 0.8193 |
| case3120sp | MR1 block-Jacobi + coarse merged | 0.1788 | 0.0389 | 0.4591 | 0.9087 |
| case3120sp | 2D-MR block-Jacobi + coarse separated | 0.1913 | 0.0395 | 0.4026 | 0.8021 |
| case6468rte | MR1 block-Jacobi | 0.1115 | 0.1438 | 0.4564 | 0.5034 |
| case6468rte | MR1 block-Jacobi + coarse merged | 0.2548 | 0.1082 | 0.5375 | 0.9025 |
| case6468rte | 2D-MR block-Jacobi + coarse separated | 0.2318 | 0.1438 | 0.4571 | 0.5034 |
| case9241pegase | MR1 block-Jacobi | 0.1419 | 0.06108 | 0.1145 | 0.6858 |
| case9241pegase | MR1 block-Jacobi + coarse merged | 0.2641 | 0.0574 | 0.1706 | 0.7051 |
| case9241pegase | 2D-MR block-Jacobi + coarse separated | 0.2903 | 0.05584 | 0.1246 | 0.7284 |

## 3. 판단

- 2D-MR가 성공하려면 merged coarse 대비 Fallback 또는 Newton 반복이 줄고, dx 크기비/방향유사도 또는 mismatch ratio가 개선되어야 한다.
- middle solver 평균 시간이 크게 증가하면, NR 반복 감소가 없을 때 총 시간에서 불리하다.
- 이번 결과에서는 **2D-MR separated가 성공하지 못했다.**
- Fallback 감소가 없고, 여러 케이스에서 Newton 반복이 merged coarse보다 늘었다.
- 평균 middle solver 시간은 MR1 block-Jacobi `0.127 ms`, merged coarse `0.244 ms`, 2D-MR separated `0.242 ms`이다.
- 평균 dx 크기비는 merged coarse `0.04702`, 2D-MR separated `0.05274`이다.
- 평균 dx 방향유사도는 merged coarse `0.3849`, 2D-MR separated `0.3159`이다.
