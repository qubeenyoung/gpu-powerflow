# MR1 + METIS block-Jacobi coarse correction 결과

## 읽는 법

- **기존**: coarse correction 없는 `MR1 + METIS block-Jacobi`
- **Coarse**: block 사이 coupling을 작은 dense coarse system으로 한 번 보정한 버전
- **Newton 반복**: 전체 NR iteration 수. 작을수록 좋다.
- **Fallback**: MR1 step이 충분히 좋지 않아서 같은 NR iteration에서 cuDSS로 다시 푼 횟수. 작을수록 좋다.
- **속도비**: `pure cuDSS 시간 / hybrid 시간`. 1보다 크면 hybrid가 더 빠르다.
- **dx 크기비**: `||dx_MR1||2 / ||dx_cuDSS||2`. 1에 가까울수록 cuDSS Newton step만큼 큰 correction이다.
- **dx 방향유사도**: `cos(dx_MR1, dx_cuDSS)`. 1이면 같은 방향, 0이면 거의 직교, 음수면 반대 방향이다.

## 1. 전체 비교

| 케이스 | 설정 | 수렴 | Newton 반복 | cuDSS 호출 | MR1 호출 | Fallback | 시간(s) | 속도비 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 기존 | true | 12 | 5 | 10 | 3 | 0.0285 | 1.62 |
| case3120sp | 기존 | true | 12 | 6 | 10 | 4 | 0.0322 | 0.78 |
| case9241pegase | 기존 | true | 12 | 4 | 10 | 2 | 0.0594 | 0.89 |
| case13659pegase | 기존 | true | 12 | 5 | 10 | 3 | 0.0748 | 0.85 |
| case6468rte | 기존 | true | 5 | 2 | 3 | 0 | 0.0333 | 0.96 |
| case2383wp | Coarse | true | 10 | 5 | 8 | 3 | 0.0251 | 1.76 |
| case3120sp | Coarse | true | 9 | 6 | 7 | 4 | 0.0279 | 0.82 |
| case9241pegase | Coarse | true | 12 | 4 | 10 | 2 | 0.0600 | 0.83 |
| case13659pegase | Coarse | true | 8 | 5 | 6 | 3 | 0.0722 | 0.88 |
| case6468rte | Coarse | true | 4 | 3 | 2 | 1 | 0.0368 | 0.93 |

요약하면, coarse correction은 Newton 반복 수를 줄이는 경우가 있지만 Fallback 감소는 아직 만들지 못했다.

## 2. pure cuDSS와 비교

이 표는 순수 cuDSS NR을 기준으로 hybrid가 실제로 빨라졌는지 보여준다.
`시간 차이`는 `hybrid 시간 - pure cuDSS 시간`이다. 음수면 hybrid가 빠르고, 양수면 hybrid가 느리다.

| 케이스 | 설정 | pure Newton 반복 | hybrid Newton 반복 | pure cuDSS 시간(s) | hybrid 시간(s) | 시간 차이(s) | 속도비 |
|---|---|---:|---:|---:|---:|---:|---:|
| case2383wp | 기존 | 6 | 12 | 0.0462 | 0.0285 | -0.0177 | 1.62 |
| case3120sp | 기존 | 6 | 12 | 0.0252 | 0.0322 | +0.0070 | 0.78 |
| case9241pegase | 기존 | 6 | 12 | 0.0527 | 0.0594 | +0.0067 | 0.89 |
| case13659pegase | 기존 | 5 | 12 | 0.0634 | 0.0748 | +0.0114 | 0.85 |
| case6468rte | 기존 | 3 | 5 | 0.0319 | 0.0333 | +0.0014 | 0.96 |
| case2383wp | Coarse | 6 | 10 | 0.0443 | 0.0251 | -0.0192 | 1.76 |
| case3120sp | Coarse | 6 | 9 | 0.0230 | 0.0279 | +0.0049 | 0.82 |
| case9241pegase | Coarse | 6 | 12 | 0.0501 | 0.0600 | +0.0099 | 0.83 |
| case13659pegase | Coarse | 5 | 8 | 0.0634 | 0.0722 | +0.0088 | 0.88 |
| case6468rte | Coarse | 3 | 4 | 0.0342 | 0.0368 | +0.0025 | 0.93 |

해석:
- `case2383wp`만 pure cuDSS보다 빠르다.
- 나머지 케이스는 hybrid가 Newton 반복을 일부 줄여도 pure cuDSS보다 느리다.
- 현재 병목은 middle solve 한 번의 속도보다, MR1 step이 충분히 강하지 않아 NR 반복/fallback을 늘리는 데 있다.

## 3. dx 품질 비교

이 표는 MR1이 만든 correction `dx`가 cuDSS Newton correction과 얼마나 비슷한지 본다.
`dx 크기비`는 클수록 좋고, `dx 방향유사도`는 1에 가까울수록 좋다.

| 케이스 | dx 크기비 기존 | dx 크기비 Coarse | dx 방향유사도 기존 | dx 방향유사도 Coarse |
|---|---:|---:|---:|---:|
| case13659pegase | 0.006383 | 0.004491 | 0.02662 | 0.3136 |
| case2383wp | 0.02168 | 0.02613 | 0.516 | 0.4437 |
| case3120sp | 0.03116 | 0.0389 | 0.4357 | 0.4591 |
| case6468rte | 0.1438 | 0.1082 | 0.4564 | 0.5375 |
| case9241pegase | 0.06108 | 0.0574 | 0.1145 | 0.1706 |

해석:
- coarse가 `dx` 방향은 여러 케이스에서 개선했다.
- 하지만 `dx` 크기는 2/5 케이스에서만 커졌고, 2배 이상 개선된 케이스는 없다.
- 따라서 coarse correction이 방향성은 일부 보정하지만 correction scale 문제는 아직 못 풀었다.

## 4. NR 진행 변화

| 케이스 | Newton 반복 기존 | Newton 반복 Coarse | Fallback 기존 | Fallback Coarse | 시간 기존(s) | 시간 Coarse(s) |
|---|---:|---:|---:|---:|---:|---:|
| case13659pegase | 12 | 8 | 3 | 3 | 0.0748 | 0.0722 |
| case2383wp | 12 | 10 | 3 | 3 | 0.0285 | 0.0251 |
| case3120sp | 12 | 9 | 4 | 4 | 0.0322 | 0.0279 |
| case6468rte | 5 | 4 | 0 | 1 | 0.0333 | 0.0368 |
| case9241pegase | 12 | 12 | 2 | 2 | 0.0594 | 0.0600 |

coarse가 NR 반복 수를 줄인 경우에도 Fallback 수가 줄지 않아서, 전체 시간 개선은 제한적이다.

## 5. Coarse overhead

- coarse correction 추가 비용 평균: **0.103 ms**
- coarse correction 추가 비용 최대: **0.179 ms**
- middle solver 평균 시간: 기존 **0.124 ms**, Coarse **0.242 ms**

목표였던 평균 0.3 ms 이하, 최대 0.5 ms 이하는 만족한다.

## 6. 판단

- **성공한 점**: coarse overhead는 충분히 작고, 일부 케이스에서 Newton 반복 수와 시간이 줄었다.
- **부족한 점**: Fallback 감소가 없고, `dx` 크기비가 기대만큼 커지지 않았다.
- **pure cuDSS 대비 결론**: 현재 설정에서 hybrid가 직접해법보다 확실히 빠른 케이스는 `case2383wp`뿐이다.
- **최종 결론**: 현재 1 coarse variable/block 방식은 방향 보정 효과는 있지만, hybrid NR을 안정적으로 빠르게 만들 만큼 correction scale을 키우지는 못했다.
