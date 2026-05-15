# NCU All-Kernel cuDSS 포함 Batch Sweep 분석

Date: 2026-05-07

## 목적

custom kernel만 본 이전 pass를 보강하기 위해 ncu kernel filter를 제거하고, cuDSS 내부 kernel까지 포함한 전체 CUDA kernel profile을 수행했다.

이번 pass의 100급 대표 case는 사용자 지시에 따라 `case94pi`가 아니라 `case118`로 변경했다.

## 실행 조건

```text
cases = case118, case1197, case6468rte, case_ACTIVSg10k, case_ACTIVSg25k, case_ACTIVSg70k
target sizes = 118, 1000, 5000, 10000, 25K, 70K
batch = 1, 2, 4, 8, 16, 32, 64, 128, 256
ncu set = basic
scope = all CUDA kernels, including cuDSS internal kernels
warmup = 0
repeats = 1
profile = cuda_mixed_edge
```

Run:

```text
exp/20260507/results/batch_util/ncu_all_representative_118_1k_5k_10k_25k_70k_b1_256_20260507
```

결과 파일:

```text
run_summary.csv
launch_metrics.csv
operator_summary.csv
case_batch_summary.csv
raw/*/b*/ncu_all_basic.csv
```

총 `6 * 9 = 54`개 ncu run을 완료했고, `errors.json`은 생성되지 않았다. 모든 benchmark run은 수렴 성공했다.

## b256 전체 kernel 요약

아래 duration은 ncu가 report한 profiled kernel duration 합계다. profiler가 붙은 상태의 wall time이 아니므로, 최종 성능 수치는 timing sweep 결과와 함께 봐야 한다.

| case | launches | duration ms sum | duration ms/scenario | SM mean % | SM max % | Mem mean % | Occ mean % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `case118` | 188 | 26.3 | 0.1026 | 8.8 | 79.9 | 7.1 | 24.1 |
| `case1197` | 224 | 68.4 | 0.2671 | 15.1 | 84.2 | 17.4 | 30.0 |
| `case6468rte` | 274 | 209.0 | 0.8165 | 14.2 | 87.5 | 19.4 | 29.6 |
| `case_ACTIVSg10k` | 338 | 508.8 | 1.9874 | 17.3 | 90.8 | 24.2 | 33.5 |
| `case_ACTIVSg25k` | 371 | 999.0 | 3.9022 | 14.1 | 94.2 | 20.0 | 31.6 |
| `case_ACTIVSg70k` | 465 | 4144.3 | 16.1886 | 15.5 | 97.3 | 23.5 | 37.1 |

## b256 operator group breakdown

| case | cuDSS | cuDSS aux | cuPF custom |
| --- | ---: | ---: | ---: |
| `case118` | 6.7 ms (25.6%, 152L) | 19.1 ms (72.6%, 12L) | 0.5 ms (1.8%, 24L) |
| `case1197` | 43.5 ms (63.6%, 188L) | 21.1 ms (30.8%, 12L) | 3.8 ms (5.5%, 24L) |
| `case6468rte` | 163.5 ms (78.2%, 238L) | 25.0 ms (12.0%, 12L) | 20.5 ms (9.8%, 24L) |
| `case_ACTIVSg10k` | 433.8 ms (85.3%, 288L) | 27.9 ms (5.5%, 12L) | 47.0 ms (9.2%, 38L) |
| `case_ACTIVSg25k` | 867.6 ms (86.8%, 328L) | 32.0 ms (3.2%, 12L) | 99.4 ms (10.0%, 31L) |
| `case_ACTIVSg70k` | 3714.0 ms (89.6%, 408L) | 34.2 ms (0.8%, 12L) | 396.1 ms (9.6%, 45L) |

해석:

- `case118`은 작은 문제라 고정성 cuDSS auxiliary kernel 비용이 지배적이다.
- 1k 이상에서는 batch 256 기준 cuDSS 내부 kernel이 전체 profiled duration의 대부분을 차지한다.
- 10k 이상에서는 cuDSS 내부 kernel 비중이 약 85-90%다.
- cuPF custom kernel은 큰 case에서 약 9-10% 수준이지만, `ibus`는 SM max/mean이 높아 batch가 커질수록 잘 채워지는 편이다.
- 전체 평균 SM throughput은 b256에서도 큰 case 기준 약 14-17% 수준이다. 반면 SM max는 90-97%까지 올라가므로, 일부 custom kernel은 GPU를 잘 쓰지만 cuDSS sparse kernel들이 평균 utilization을 낮추는 구조로 보인다.

## batch 64/128/256 비교

아래 값은 profiled kernel duration을 batch size로 나눈 ms/scenario다.

| case | b64 | b128 | b256 | b128 -> b256 개선 |
| --- | ---: | ---: | ---: | ---: |
| `case118` | 0.3434 | 0.1829 | 0.1026 | 43.9% |
| `case1197` | 0.5669 | 0.3674 | 0.2671 | 27.3% |
| `case6468rte` | 1.1885 | 0.9418 | 0.8165 | 13.3% |
| `case_ACTIVSg10k` | 2.4304 | 2.1350 | 1.9874 | 6.9% |
| `case_ACTIVSg25k` | 4.4884 | 5.0097 | 3.9022 | 22.1% |
| `case_ACTIVSg70k` | 19.7285 | 16.4993 | 16.1886 | 1.9% |

해석:

- all-kernel ncu 기준으로도 b256은 대부분의 case에서 ms/scenario가 가장 낮다.
- 70k에서는 b128과 b256 차이가 1.9%로 작다. timing sweep에서도 큰 case solve-only가 b128/b256 근처에서 비슷했으므로, 70k급 micro-batch 후보는 `128`과 `256`을 함께 유지한다.
- 25k의 b128은 launch 수가 392로 b64/b256의 371보다 많아 anomalous point로 보인다. 반복 측정 또는 cuDSS 내부 phase 분리가 필요하다.

## 과제 논리와 연결

- 올해 목표인 cuITER 반복 선형 솔버 관점에서는, 큰 case에서 cuDSS kernel duration 비중이 압도적이라는 점이 중요하다.
- cuDSS를 매 반복마다 완전 정확하게 호출하는 구조는 큰 case에서 비용 중심이 된다.
- 따라서 반복 중간 단계는 더 낮은 정밀도 또는 더 저렴한 approximate update로 처리하고, 필요한 시점에 cuDSS로 보정하는 hybrid 논리가 실험적으로 타당해 보인다.
- 이 주장은 아직 수치 오차 관점에서 완성된 결론이 아니다. 다음 실험 2의 batch 1 불일치 벡터 방향/크기 추적이 필요하다.

## 다음 액션

- cuDSS 내부 kernel을 factor/solve/setup 성격으로 더 나누기 위해 NVTX range 또는 phase별 ncu pass를 추가한다.
- b128/b256만 대상으로 `case_ACTIVSg25k`, `case_ACTIVSg70k`를 반복 측정해 anomalous point와 큰 case plateau를 확인한다.
- 실험 2에서 batch 1 모든 case의 반복별 불일치 벡터 크기와 방향을 추적해, 반복 중간 계산을 낮은 정밀도로 둬도 되는 논리를 검증한다.
- 실험 3에서 `mismatch`, `jacobian_fill`, `ibus`, cuDSS solve 관련 연산을 정밀도 민감 후보로 두고 오차 원인을 분석한다.
