# Batch Utilization 진행 상태

Date: 2026-05-07

## 1차 smoke sweep

Run:

```text
exp/20260507/results/batch_util/smoke_representative_b1_256_w1_r3_20260507
```

조건:

```text
cases = case118, case1354pegase, case9241pegase, case_ACTIVSg25k, case_ACTIVSg70k
batch = 1, 2, 4, 8, 16, 32, 64, 128, 256
warmup = 1
repeats = 3
GPU sampling = nvidia-smi 100 ms coarse sample
```

결과 요약:

| case | buses | end-to-end best batch | solve-only best batch | max success batch |
| --- | ---: | ---: | ---: | ---: |
| `case118` | 118 | 256 | 256 | 256 |
| `case1354pegase` | 1,354 | 256 | 256 | 256 |
| `case9241pegase` | 9,241 | 256 | 128 | 256 |
| `case_ACTIVSg25k` | 25,000 | 256 | 128 | 256 |
| `case_ACTIVSg70k` | 70,000 | 256 | 128 | 256 |

임시 해석:

- 대표 case 모두 batch 256까지 수렴했고 OOM은 없었다.
- end-to-end `elapsed_ms_per_scenario`는 batch 256까지 계속 개선되었다.
- solve-only 기준으로는 9k 이상 case에서 batch 128이 batch 256보다 약간 좋거나 비슷한 구간이 나타났다.
- 큰 case에서는 batch 128 이후 GPU util coarse sample의 max가 100%에 도달하므로, 최종 micro-batch 후보는 `128`과 `256`을 함께 재측정해야 한다.
- `nvidia-smi` coarse sample은 작은 case에서 0% 또는 낮은 값이 자주 나오므로 작은 case utilization 판단에는 부적합하다. 작은 case는 Nsight Systems로 launch/sync gap을 확인해야 한다.

## 다음 단계

- 전체 78개 MATPOWER case에 대해 batch `1..256` timing sweep를 수행한다.
- 전체 sweep에서는 coarse GPU sampling을 끄고 timing/OOM/success 여부를 먼저 수집한다.
- 이후 대표 batch 후보 `64, 128, 256`을 골라 Nsight Systems/Compute로 utilization과 kernel breakdown을 확인한다.

## 전체 78개 case 1차 timing sweep

Run:

```text
exp/20260507/results/batch_util/all_matpower_b1_256_w1_r3_20260507
```

조건:

```text
cases = MATPOWER dump 전체 78개
batch = 1, 2, 4, 8, 16, 32, 64, 128, 256
warmup = 1
repeats = 3
GPU sampling = off
```

결과:

- 총 `78 * 9 = 702` case/batch aggregate row를 생성했다.
- 수렴 실패 aggregate row는 `0`개다.
- 모든 case가 batch 256까지 성공했다.
- end-to-end 기준 best batch는 `78/78` case에서 `256`이었다.
- solve-only 기준 best batch는 `75/78` case에서 `256`, `3/78` case에서 `128`이었다.

Size-bin별 1차 결론:

| size bin | cases | b1->b256 end-to-end speedup | b1->b256 solve speedup | b128->b256 end-to-end gain |
| --- | ---: | ---: | ---: | ---: |
| `<100` | 41 | 134.1x | 10.63x | 47.21% |
| `100-999` | 10 | 83.02x | 7.50x | 48.18% |
| `1k-9,999` | 22 | 24.17x | 2.66x | 14.87% |
| `10k-49,999` | 3 | 13.62x | 1.73x | 5.40% |
| `>=50k` | 2 | 7.88x | 1.30x | 6.58% |

임시 추천:

- DP throughput 우선: batch `256`.
- PP 또는 memory-sensitive schedule: batch `128`과 `256`을 둘 다 후보로 유지.
- 최종 판단 전 보강: `ncu` SpeedOfLight metric과 `warmup=3`, `repeats=10` 재측정.

상세 문서:

```text
exp/20260507/results/batch_util/all_matpower_b1_256_w1_r3_20260507/MICROBATCH_RECOMMENDATION.md
```

## Nsight Compute smoke

`nsys`는 현재 환경에 없고, `ncu`만 사용 가능하다.

경로 검증용으로 다음을 실행했다.

```text
case = case118
batch = 256
ncu set = basic
launch count = 20
```

산출물:

```text
exp/20260507/results/batch_util/ncu_smoke/case118_b256_basic_l20.csv
exp/20260507/results/batch_util/ncu_smoke/case118_b256_basic_l20_summary.csv
exp/20260507/results/batch_util/ncu_smoke/SUMMARY.md
```

해석:

- `ncu` 수집 경로는 정상 동작한다.
- launch-count 20은 초기 cuDSS kernel 위주라 최종 utilization 판단에는 쓰지 않는다.
- 다음에는 custom kernel 또는 steady solve 구간을 겨냥하기 위해 launch skip/count 또는 kernel-name filter를 정해서 대표 case/batch를 다시 측정한다.

## 대표 case ncu custom-kernel batch sweep

Run:

```text
exp/20260507/results/batch_util/ncu_custom_representative_100_1k_5k_10k_25k_70k_b1_256_20260507
```

조건:

```text
cases = case94pi, case1197, case6468rte, case_ACTIVSg10k, case_ACTIVSg25k, case_ACTIVSg70k
target sizes = 100, 1000, 5000, 10000, 25K, 70K
batch = 1, 2, 4, 8, 16, 32, 64, 128, 256
ncu set = basic
scope = cuPF custom kernels only
warmup = 0
repeats = 1
```

결과:

- 총 `6 * 9 = 54`개 ncu run을 완료했다.
- 모든 run이 수렴 성공했다.
- batch 증가에 따라 custom kernel의 평균 SM throughput, memory throughput, occupancy가 증가했다.
- b256 기준 큰 case에서 `ibus`는 SM throughput이 약 84-85%까지 올라간다.
- b256 기준 `jacobian_fill`/`mismatch`는 memory throughput이 높아 memory-side 최적화 후보로 보인다.
- 이 pass는 cuDSS 내부 kernel을 제외하므로 전체 solve utilization 결론은 아직 아니다.

상세 문서:

```text
exp/20260507/results/batch_util/ncu_custom_representative_100_1k_5k_10k_25k_70k_b1_256_20260507/NCU_CUSTOM_ANALYSIS.md
```

## 대표 case ncu all-kernel/cuDSS 포함 batch sweep

Run:

```text
exp/20260507/results/batch_util/ncu_all_representative_118_1k_5k_10k_25k_70k_b1_256_20260507
```

조건:

```text
cases = case118, case1197, case6468rte, case_ACTIVSg10k, case_ACTIVSg25k, case_ACTIVSg70k
target sizes = 118, 1000, 5000, 10000, 25K, 70K
batch = 1, 2, 4, 8, 16, 32, 64, 128, 256
ncu set = basic
scope = all CUDA kernels, including cuDSS internal kernels
warmup = 0
repeats = 1
```

결과:

- 총 `6 * 9 = 54`개 ncu run을 완료했다.
- `errors.json`은 생성되지 않았고 모든 benchmark run이 수렴 성공했다.
- 100급 대표 case는 사용자 지시에 따라 `case94pi`에서 `case118`로 바꾸었다.
- b256 기준 10k 이상 case에서 cuDSS 내부 kernel이 전체 profiled kernel duration의 약 85-90%를 차지한다.
- b256 기준 cuPF custom kernel은 큰 case에서 약 9-10% 수준이며, 이 중 `ibus`는 SM throughput이 높게 나온다.
- all-kernel 평균 SM throughput은 b256 큰 case에서도 약 14-17% 수준이다. 일부 kernel은 SM max가 90-97%까지 올라가지만, cuDSS sparse kernel들이 전체 평균 utilization을 낮추는 구조로 보인다.
- all-kernel ncu는 매우 무겁기 때문에 반복 통계용보다는 대표 breakdown용으로 사용하고, 최종 성능 판단은 timing sweep와 b128/b256 반복 측정으로 보강한다.

상세 문서:

```text
exp/20260507/results/batch_util/ncu_all_representative_118_1k_5k_10k_25k_70k_b1_256_20260507/NCU_ALL_CUDSS_ANALYSIS.md
```
