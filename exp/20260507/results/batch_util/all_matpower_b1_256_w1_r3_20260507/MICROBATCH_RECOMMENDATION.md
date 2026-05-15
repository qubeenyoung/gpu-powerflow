# Micro-Batch Recommendation

- Created UTC: 2026-05-07T03:58:10.111145+00:00
- Source run: `exp/20260507/results/batch_util/all_matpower_b1_256_w1_r3_20260507`
- Scope: 1차 timing sweep (`warmup=1`, `repeats=3`), 최종 수치 확정 전 단계

## 전체 결론

- 모든 case가 batch 256까지 성공했다: `78 / 78`.
- end-to-end 기준 best batch 분포: `{'256': 78}`.
- solve-only 기준 best batch 분포: `{'128': 3, '256': 75}`.
- DP throughput 우선 micro-batch는 현재 데이터만 보면 전 bin에서 `256`이 1순위다.
- PP 또는 memory-sensitive schedule은 큰 case에서 `128`과 `256`을 모두 후보로 남긴다.
- 이 결론은 timing 기반 1차 판단이며, utilization 최종 판단은 Nsight Compute 보강 후 확정한다.

## Size-Bin 요약

| size bin | cases | b1 elapsed ms/scenario | b128 elapsed | b256 elapsed | b1->b256 elapsed speedup | b1 solve ms/scenario | b256 solve | b1->b256 solve speedup | 128->256 elapsed gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `<100` | 41 | 11.27 | 0.159 | 0.084 | 134.1x | 0.464 | 0.044 | 10.63x | 47.21% |
| `100-999` | 10 | 10.96 | 0.255 | 0.132 | 83.02x | 0.668 | 0.089 | 7.498x | 48.18% |
| `1k-9,999` | 22 | 21.60 | 1.050 | 0.894 | 24.17x | 2.108 | 0.792 | 2.661x | 14.87% |
| `10k-49,999` | 3 | 51.61 | 4.004 | 3.788 | 13.62x | 6.145 | 3.561 | 1.726x | 5.402% |
| `>=50k` | 2 | 170.7 | 23.18 | 21.66 | 7.880x | 27.18 | 20.97 | 1.296x | 6.578% |

## Solve-Only 예외

end-to-end best는 모든 case에서 `256`이지만, solve-only best는 아래 3개 case에서 `128`이었다.

| case | size bin | buses | best solve batch | best solve ms/scenario | best elapsed batch |
| --- | --- | ---: | ---: | ---: | ---: |
| `case1197` | `1k-9,999` | 1197 | 128 | 0.282 | 256 |
| `case59` | `<100` | 59 | 128 | 0.089 | 256 |
| `case_ACTIVSg25k` | `10k-49,999` | 25000 | 128 | 5.187 | 256 |

## 임시 추천

| 목적 | 추천 batch | 근거 | 보강 필요 |
| --- | ---: | --- | --- |
| DP throughput | 256 | 모든 case의 end-to-end ms/scenario가 256에서 최저 | final `warmup=3`, `repeats=10` 재측정 |
| PP stage balance | 128 또는 256 | 큰 case에서 128 이후 이득이 작아지고 solve-only 예외가 있음 | stage별 memory/workspace와 Nsight Compute |
| 작은 case launch amortization | 256 | `<100` bin에서 b1->b256 end-to-end speedup이 가장 큼 | Nsight Systems 대체 지표 필요 |
| 큰 case memory-sensitive | 128 우선, 256 확인 | `>=50k` bin은 b128->b256 end-to-end gain이 작고 memory 사용 증가 | b512 실패/성공 여부 및 memory cap 확인 |

## 다음 측정

- 대표 batch `64, 128, 256`에 대해 `ncu` SpeedOfLight metric을 수집한다.
- 대표 case는 `case118`, `case9241pegase`, `case_ACTIVSg25k`, `case_ACTIVSg70k`, `case_SyntheticUSA`로 둔다.
- 최종 보고용 timing은 전체 case가 아니라 후보 batch 중심으로 `warmup=3`, `repeats=10` 재측정한다.

## 실패 여부

- batch별 aggregate 기준 수렴 실패 row: `0`
