# cuDSS vs Custom 커널 읽기용 요약

Date: 2026-05-07

이 문서는 실험 1 결과에서 **왜 b256이 좋은지**를 보기 쉽게 보려고 만든 요약이다.

분류:

- `cuDSS`: `cudss` + `cudss_aux`
- `custom`: `ibus`, `jacobian_fill`, `mismatch`, `mismatch_norm`, `prepare_rhs`, `voltage_reconstruct`, `voltage_update_apply`

## 1. b256에서 시간이 어디로 가는가

먼저 이 표만 보면 된다. `cuDSS`가 전체 시간의 대부분인지, `custom`이 실제로 얼마나 남는지 보는 표다.

| case | total ms/scenario | cuDSS ms/scenario | custom ms/scenario | 한 줄 해석 |
| --- | ---: | ---: | ---: | --- |
| `case118` | 0.1026 | 0.1007 (98.2%) | 0.0019 (1.8%) | 거의 전부 cuDSS 고정비 |
| `case1197` | 0.2671 | 0.2524 (94.5%) | 0.0148 (5.5%) | cuDSS가 지배 |
| `case6468rte` | 0.8165 | 0.7364 (90.2%) | 0.0801 (9.8%) | cuDSS 9 : custom 1 |
| `case_ACTIVSg10k` | 1.9874 | 1.8036 (90.8%) | 0.1837 (9.2%) | cuDSS 9 : custom 1 |
| `case_ACTIVSg25k` | 3.9022 | 3.5138 (90.0%) | 0.3883 (10.0%) | cuDSS 9 : custom 1 |
| `case_ACTIVSg70k` | 16.1886 | 14.6415 (90.4%) | 1.5471 (9.6%) | cuDSS 9 : custom 1 |

결론: b256에서도 전체 시간은 거의 cuDSS가 결정한다. custom 커널은 큰 case에서도 약 10%다.

## 2. 그런데 GPU를 잘 쓰는 쪽은 어디인가

시간 비중과 GPU util이 서로 반대다.

| case | cuDSS SM util | custom SM util | 의미 |
| --- | ---: | ---: | --- |
| `case118` | 11.5% | 65.8% | 작은 case라 cuDSS 고정비가 큼 |
| `case1197` | 23.3% | 81.6% | custom은 이미 잘 참 |
| `case6468rte` | 31.5% | 81.3% | cuDSS가 낮은 util로 오래 돎 |
| `case_ACTIVSg10k` | 33.9% | 82.7% | custom 병목 아님 |
| `case_ACTIVSg25k` | 32.1% | 81.3% | cuDSS 병목 |
| `case_ACTIVSg70k` | 30.4% | 80.3% | cuDSS 병목 |

결론: custom 커널은 b256에서 GPU를 잘 채운다. 문제는 **시간 90%를 먹는 cuDSS가 30%대 util에 머무는 것**이다.

## 3. b256이 왜 b1보다 좋은가

b1에서 b256으로 가면 두 그룹 모두 scenario당 비용이 줄지만, 이유가 다르다.

| case | cuDSS speedup b1 -> b256 | custom speedup b1 -> b256 | 해석 |
| --- | ---: | ---: | --- |
| `case118` | 103.9x | 53.5x | 고정비 분산 효과가 압도적 |
| `case1197` | 52.1x | 9.5x | cuDSS 고정비/작은 grid가 크게 완화 |
| `case6468rte` | 23.2x | 3.6x | cuDSS batch 효과가 아직 큼 |
| `case_ACTIVSg10k` | 12.1x | 3.1x | cuDSS가 주된 개선원 |
| `case_ACTIVSg25k` | 8.6x | 2.6x | cuDSS 개선이 더 큼 |
| `case_ACTIVSg70k` | 3.9x | 2.4x | 큰 case라 b1도 이미 일이 큼 |

결론: b256이 좋은 가장 큰 이유는 custom kernel만 좋아져서가 아니다. **cuDSS 계열 비용이 256개 scenario로 분산되고, cuDSS uniform batch가 scenario당 factor/solve 비용을 낮추기 때문**이다.

## 4. b128에서 b256으로 더 키울 가치가 있는가

이 표는 micro-batch 결정을 위한 표다. `custom`은 b128에서 이미 거의 포화됐는지, `cuDSS`에 아직 이득이 남았는지 보면 된다.

| case | total gain b128 -> b256 | cuDSS gain | custom gain | 판단 |
| --- | ---: | ---: | ---: | --- |
| `case118` | 43.9% | 44.2% | 16.5% | b256 가치 큼 |
| `case1197` | 27.3% | 28.3% | 3.1% | b256 이득은 거의 cuDSS에서 옴 |
| `case6468rte` | 13.3% | 14.5% | 1.0% | b256 이득은 cuDSS 중심 |
| `case_ACTIVSg10k` | 6.9% | 7.5% | 0.8% | b256 이득 작지만 남아 있음 |
| `case_ACTIVSg25k` | 22.1% | 22.6% | 17.0% | b128 anomalous point, 재측정 필요 |
| `case_ACTIVSg70k` | 1.9% | 2.0% | 0.8% | b128과 b256 거의 동일 후보 |

결론:

- custom 커널 관점에서는 b64-b128 정도에서 대부분 포화된다.
- b128 이후 b256의 추가 이득은 거의 cuDSS에서 나온다.
- 70k급에서는 cuDSS도 plateau라 b128과 b256을 둘 다 micro-batch 후보로 둬야 한다.

## 최종 해석

```text
b256이 좋은 이유:
  작은/중간 case:
    cuDSS 고정비 + launch overhead가 256개 scenario로 분산됨

  custom kernel:
    b256에서 SM util 80% 전후까지 올라가지만,
    전체 시간 비중은 5-10%라 전체 성능을 지배하지 않음

  cuDSS:
    b256에서도 전체 시간의 약 90%
    b128 이후 추가 이득도 대부분 cuDSS에서 옴

b256이 큰 case에서 압도적이지 않은 이유:
  cuDSS factor/solve가 30%대 SM util에서 plateau
  custom kernel도 이미 b128에서 포화
```

실험 1 기준 micro-batch 해석:

- `case118`, `case1197`: b256 추천. 고정비 분산 효과가 큼.
- `case6468rte`, `case_ACTIVSg10k`: b256 우세. 다만 b128 이후 이득은 작아지기 시작.
- `case_ACTIVSg25k`: b256 우세로 보이지만 b128 anomaly 재측정 필요.
- `case_ACTIVSg70k`: b128/b256 둘 다 후보. throughput-only는 b256, PP/workspace 고려하면 b128도 유지.
