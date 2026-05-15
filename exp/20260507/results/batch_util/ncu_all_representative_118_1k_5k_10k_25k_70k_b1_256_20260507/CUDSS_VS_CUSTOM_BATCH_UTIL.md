# cuDSS vs Custom Batch별 GPU Util

Date: 2026-05-07

목적: 실험 1에서 batch size가 커질 때 **cuDSS 계열**과 **나머지 custom 커널**의 GPU util이 각각 어떻게 변하는지 본다.

분류:

- `cuDSS`: `cudss` + `cudss_aux`
- `custom`: `ibus`, `jacobian_fill`, `mismatch`, `mismatch_norm`, `prepare_rhs`, `voltage_reconstruct`, `voltage_update_apply`

Metric:

- NCU `Compute (SM) Throughput`
- duration-weighted average: `sum(duration_ns * compute_sm_pct) / sum(duration_ns)`

## 전체 대표 case 합산 SM util

이 표가 핵심이다.

| group | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 | b256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cuDSS` | 11.8 | 12.7 | 15.1 | 18.5 | 22.3 | 25.8 | 28.6 | 30.1 | 30.9 |
| `custom` | 31.3 | 48.0 | 61.4 | 69.9 | 75.1 | 78.1 | 79.6 | 80.3 | 80.7 |

읽는 법:

- custom 커널은 b1 -> b64 구간에서 빠르게 GPU util이 올라가고, b64-b128 근처에서 80% 부근으로 포화된다.
- cuDSS 계열은 batch를 키워도 훨씬 천천히 올라가며, b128-b256에서도 30%대 초반에 머문다.
- 따라서 batch를 키울 때 custom 커널은 먼저 포화되고, 전체 성능의 남은 병목은 cuDSS가 된다.

## case별 cuDSS SM util

| case | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 | b256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `case118` | 6.9 | 7.1 | 7.1 | 7.2 | 7.4 | 7.7 | 8.4 | 9.5 | 11.5 |
| `case1197` | 6.5 | 6.6 | 6.9 | 7.5 | 8.7 | 10.8 | 14.1 | 18.5 | 23.3 |
| `case6468rte` | 7.8 | 8.2 | 9.2 | 11.1 | 14.1 | 18.4 | 23.3 | 28.0 | 31.5 |
| `case_ACTIVSg10k` | 8.6 | 9.3 | 11.3 | 14.6 | 19.1 | 24.2 | 28.7 | 31.9 | 33.9 |
| `case_ACTIVSg25k` | 10.7 | 11.6 | 13.9 | 17.3 | 21.5 | 25.6 | 28.8 | 31.3 | 32.1 |
| `case_ACTIVSg70k` | 17.0 | 18.7 | 21.3 | 24.2 | 26.7 | 28.4 | 29.6 | 30.1 | 30.4 |

읽는 법:

- 작은 case에서는 cuDSS util이 매우 낮다. `case118`은 b256에서도 11.5%뿐이다.
- 1k-25k는 batch 증가에 따라 cuDSS util이 계속 올라가지만 b128-b256에서 30%대에 들어간 뒤 둔화된다.
- 70k는 b1부터 17.0%로 상대적으로 높지만 b64 이후 거의 포화다. b64 29.6%, b128 30.1%, b256 30.4%.

## case별 custom SM util

| case | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 | b256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `case118` | 1.2 | 2.4 | 4.6 | 8.6 | 16.0 | 27.1 | 41.2 | 55.0 | 65.8 |
| `case1197` | 8.7 | 16.0 | 27.8 | 42.4 | 56.5 | 67.0 | 74.7 | 79.1 | 81.6 |
| `case6468rte` | 22.8 | 37.8 | 53.0 | 64.2 | 72.4 | 77.1 | 79.5 | 80.7 | 81.3 |
| `case_ACTIVSg10k` | 26.6 | 43.1 | 57.8 | 68.2 | 75.1 | 79.1 | 81.1 | 82.2 | 82.7 |
| `case_ACTIVSg25k` | 31.7 | 48.4 | 62.2 | 70.6 | 75.8 | 78.8 | 80.3 | 81.1 | 81.3 |
| `case_ACTIVSg70k` | 34.3 | 51.3 | 63.8 | 71.2 | 75.7 | 78.2 | 79.4 | 79.9 | 80.3 |

읽는 법:

- custom 커널은 case가 커질수록 b1에서도 util이 높아진다.
- 1k 이상에서는 b64-b128부터 거의 80% 근처로 포화된다.
- b128 -> b256 custom util 증가는 매우 작다.

## 전체 대표 case 합산 memory throughput

SM util만 보면 custom이 훨씬 잘 차지만, memory throughput은 b256에서 두 그룹이 비슷해진다.

| group | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 | b256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cuDSS` | 9.8 | 10.5 | 13.2 | 16.8 | 21.1 | 24.9 | 28.0 | 29.5 | 30.4 |
| `custom` | 12.8 | 18.3 | 22.6 | 25.4 | 27.5 | 28.6 | 29.4 | 29.8 | 29.7 |

해석:

- custom은 SM util이 80%까지 올라가지만 memory throughput은 30% 근처다.
- cuDSS는 SM util과 memory throughput이 둘 다 30% 근처에서 포화된다.
- 즉 cuDSS는 단순히 DRAM bandwidth 하나만 꽉 차서 막힌 형태라기보다는 sparse factor/solve 특성상 parallelism과 dependency가 섞인 plateau로 보는 게 맞다.

## 결론

batch별 GPU util 관점에서 보면:

```text
custom:
  b1 31.3% -> b64 79.6% -> b256 80.7%
  b64-b128에서 거의 포화

cuDSS:
  b1 11.8% -> b64 28.6% -> b256 30.9%
  batch를 키워도 30%대 초반에서 포화
```

따라서 실험 1의 결론은:

- b256이 좋은 이유는 custom 커널 util이 계속 올라가서가 아니다. custom은 b64-b128에서 이미 거의 찬다.
- b128 이후 b256의 추가 이득은 주로 cuDSS의 scenario당 비용 감소에서 온다.
- 하지만 cuDSS util 자체는 30%대에서 plateau이므로, 큰 case에서는 b128과 b256 차이가 작다.
- micro-batch 관점에서는 custom kernel 기준 포화점은 b64-b128, cuDSS/throughput 기준 후보는 b128-b256이다.
