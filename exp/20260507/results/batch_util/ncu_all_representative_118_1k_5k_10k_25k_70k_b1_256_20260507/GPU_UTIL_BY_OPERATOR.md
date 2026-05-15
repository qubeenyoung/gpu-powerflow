# Operator별 GPU Util 분석

Date: 2026-05-07

## 데이터 위치

- 입력 분석 문서: `NCU_ALL_CUDSS_ANALYSIS.md`
- raw NCU CSV: `raw/*/b*/ncu_all_basic.csv` (54 files)
- 파싱된 launch 단위 데이터: `launch_metrics.csv` (16,107 launches)
- 기존 operator 집계: `operator_summary.csv`

## 기준

- 여기서 GPU util은 NCU `GPU Speed Of Light Throughput`의 `Compute (SM) Throughput`을 의미한다.
- `operator_summary.csv`의 `compute_sm_pct_mean`은 launch 단순 평균이다.
- 아래 핵심 표는 duration이 긴 kernel의 영향을 반영하기 위해 `sum(duration_ns * compute_sm_pct) / sum(duration_ns)`로 duration-weighted SM util을 다시 계산했다.
- `Mem util`은 같은 방식으로 weighted `Memory Throughput`을 계산한 값이다.

## b256 전체 operator별 요약

| operator | launches | duration ms | duration share | SM util weighted % | SM launch mean % | SM max % | Mem util weighted % | Occ weighted % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cudss` | 1602 | 5229.2 | 87.8% | 31.6 | 10.3 | 97.3 | 31.3 | 47.1 |
| `cudss_aux` | 72 | 159.2 | 2.7% | 5.6 | 5.8 | 7.8 | 1.5 | 8.3 |
| `ibus` | 30 | 477.5 | 8.0% | 84.8 | 84.0 | 84.8 | 20.5 | 92.4 |
| `jacobian_fill` | 24 | 41.1 | 0.7% | 55.2 | 62.9 | 81.4 | 93.8 | 76.2 |
| `mismatch` | 30 | 14.0 | 0.2% | 65.5 | 60.4 | 69.4 | 94.5 | 82.7 |
| `mismatch_norm` | 30 | 3.7 | 0.1% | 43.0 | 42.4 | 44.8 | 82.5 | 50.2 |
| `prepare_rhs` | 24 | 4.1 | 0.1% | 31.1 | 28.1 | 31.9 | 91.9 | 72.4 |
| `voltage_reconstruct` | 24 | 18.9 | 0.3% | 84.2 | 77.3 | 84.5 | 29.1 | 78.9 |
| `voltage_update_apply` | 24 | 7.9 | 0.1% | 32.5 | 30.3 | 36.1 | 83.5 | 80.4 |

## b256 case별 SM util weighted %

| case | `cudss` | `cudss_aux` | `ibus` | `jacobian_fill` | `mismatch` | `mismatch_norm` | `prepare_rhs` | `voltage_reconstruct` | `voltage_update_apply` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `case118` | 22.8 | 7.5 | 79.8 | 48.5 | 18.3 | 37.3 | 6.4 | 38.9 | 9.9 |
| `case1197` | 31.3 | 6.9 | 84.2 | 76.6 | 66.9 | 40.9 | 31.5 | 76.2 | 35.5 |
| `case6468rte` | 35.4 | 5.9 | 84.7 | 59.0 | 69.3 | 44.1 | 31.2 | 83.0 | 34.6 |
| `case_ACTIVSg10k` | 35.8 | 5.4 | 84.7 | 81.4 | 66.9 | 43.7 | 31.3 | 83.6 | 32.7 |
| `case_ACTIVSg25k` | 33.1 | 4.7 | 84.8 | 59.7 | 68.4 | 44.6 | 31.1 | 84.2 | 32.3 |
| `case_ACTIVSg70k` | 30.7 | 4.5 | 84.8 | 52.0 | 64.5 | 42.6 | 31.1 | 84.5 | 32.4 |

## b256 case별 duration share %

| case | `cudss` | `cudss_aux` | `ibus` | `jacobian_fill` | `mismatch` | `mismatch_norm` | `prepare_rhs` | `voltage_reconstruct` | `voltage_update_apply` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `case118` | 25.6 | 72.6 | 1.3 | 0.1 | 0.1 | 0.1 | 0.0 | 0.1 | 0.1 |
| `case1197` | 63.6 | 30.8 | 4.8 | 0.2 | 0.1 | 0.1 | 0.0 | 0.2 | 0.1 |
| `case6468rte` | 78.2 | 12.0 | 8.4 | 0.6 | 0.2 | 0.1 | 0.1 | 0.3 | 0.1 |
| `case_ACTIVSg10k` | 85.3 | 5.5 | 8.0 | 0.5 | 0.2 | 0.1 | 0.1 | 0.3 | 0.1 |
| `case_ACTIVSg25k` | 86.8 | 3.2 | 8.5 | 0.7 | 0.2 | 0.1 | 0.1 | 0.3 | 0.1 |
| `case_ACTIVSg70k` | 89.6 | 0.8 | 8.0 | 0.7 | 0.2 | 0.1 | 0.1 | 0.3 | 0.1 |

## batch scaling: 전체 case 합산 duration-weighted SM util %

| operator | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 | b256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cudss` | 18.9 | 22.2 | 24.6 | 27.0 | 29.0 | 30.3 | 31.0 | 31.5 | 31.6 |
| `cudss_aux` | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 |
| `ibus` | 59.8 | 77.6 | 83.3 | 84.1 | 84.4 | 84.6 | 84.7 | 84.7 | 84.8 |
| `jacobian_fill` | 30.1 | 39.5 | 45.6 | 49.9 | 52.4 | 54.1 | 54.4 | 55.1 | 55.2 |
| `mismatch` | 20.6 | 32.8 | 44.6 | 53.9 | 59.9 | 62.7 | 64.9 | 65.2 | 65.5 |
| `mismatch_norm` | 0.3 | 0.6 | 1.1 | 2.1 | 4.3 | 8.4 | 16.4 | 29.6 | 43.0 |
| `prepare_rhs` | 6.7 | 12.0 | 18.9 | 25.8 | 29.6 | 30.7 | 31.0 | 31.1 | 31.1 |
| `voltage_reconstruct` | 33.2 | 49.9 | 61.8 | 71.8 | 77.2 | 80.8 | 82.8 | 83.7 | 84.2 |
| `voltage_update_apply` | 9.4 | 16.7 | 24.0 | 28.6 | 30.9 | 32.0 | 32.3 | 32.5 | 32.5 |

## b256이 좋은 이유

`b256`이 좋은 이유는 전체 실행 시간이 작아서가 아니라, **1개 scenario당 비용이 가장 작아지기 때문**이다. batch가 256배 커져도 launch 수는 256배 늘지 않는다. 대표 case에서 전체 profiled launch 수는 다음 정도만 증가한다.

| case | launch count b1 | launch count b256 | batch 증가 | launch 증가 |
| --- | ---: | ---: | ---: | ---: |
| `case118` | 121 | 188 | 256x | 1.55x |
| `case1197` | 145 | 224 | 256x | 1.54x |
| `case6468rte` | 170 | 274 | 256x | 1.61x |
| `case_ACTIVSg10k` | 223 | 338 | 256x | 1.52x |
| `case_ACTIVSg25k` | 229 | 371 | 256x | 1.62x |
| `case_ACTIVSg70k` | 297 | 465 | 256x | 1.57x |

즉, batch는 kernel launch를 256번 반복하는 구조가 아니라, 같은 Newton stage launch 안에서 grid와 waves를 키우는 구조다. 이 때문에 launch overhead, cuDSS auxiliary/setup 성격의 고정비, 작은 reduction kernel 비용이 scenario 수로 분산된다.

전체 ms/scenario 개선은 다음과 같다.

| case | b1 ms/scenario | b128 ms/scenario | b256 ms/scenario | b1 -> b256 | b128 -> b256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `case118` | 10.570 | 0.183 | 0.103 | 103.0x | 43.9% |
| `case1197` | 13.297 | 0.367 | 0.267 | 49.8x | 27.3% |
| `case6468rte` | 17.356 | 0.942 | 0.816 | 21.3x | 13.3% |
| `case_ACTIVSg10k` | 22.374 | 2.135 | 1.987 | 11.3x | 6.9% |
| `case_ACTIVSg25k` | 31.313 | 5.010 | 3.902 | 8.0x | 22.1% |
| `case_ACTIVSg70k` | 60.598 | 16.499 | 16.189 | 3.7x | 1.9% |

개선 원인은 case 크기별로 다르다.

1. 작은 case에서는 고정비 분산이 지배적이다.
   - `case118` b256에서 `cudss_aux`는 19.1 ms로 여전히 총시간의 72.6%지만, scenario당으로는 0.0745 ms다.
   - b1에서는 `cudss_aux`가 9.536 ms/scenario였으므로, scenario당 128x 줄었다.
   - 작은 case는 실제 계산량보다 cuDSS auxiliary/setup과 launch overhead가 커서 batch 효과가 극단적으로 크다.

2. custom CUDA kernel은 batch가 커질수록 충분한 grid를 만들어 SM을 채운다.
   - 전체 case 합산 기준 `ibus` SM util은 b1 59.8%에서 b256 84.8%로 오른다.
   - `voltage_reconstruct`는 33.2%에서 84.2%로 오른다.
   - `mismatch`는 20.6%에서 65.5%로 오른다.
   - `jacobian_fill`은 30.1%에서 55.2%로 오른다.
   - `mismatch_norm`은 b1에서 거의 launch/reduction overhead만 보이다가 b256에서 43.0%까지 오른다.

3. 큰 case에서는 b1도 이미 어느 정도 큰 grid를 갖기 때문에 custom kernel 개선폭은 작아진다.
   - `case_ACTIVSg70k`의 `ibus`는 b1에서도 SM mean 61.9%이며 b256에서 84.8%다. scenario당 시간은 1.7795 ms에서 1.2938 ms로 1.4x 개선이다.
   - 반면 `mismatch_norm`은 reduction launch 구조상 b1 1.4952 ms/scenario에서 b256 0.0098 ms/scenario로 151.9x 개선된다.
   - 큰 case에서 b256의 이득은 custom compute를 256배 잘 압축한다기보다, reduction/auxiliary/fixed-cost와 cuDSS batched parallelism을 같이 분산하는 데서 온다.

4. cuDSS도 batch로 parallelism은 늘지만, b64 이후 포화가 빠르다.
   - 전체 case 합산 `cudss` SM util은 b1 18.9%, b64 31.0%, b128 31.5%, b256 31.6%다.
   - b256이 b1보다 좋은 이유에는 cuDSS grid/waves 증가가 포함되지만, b128 이후 추가 개선이 작은 이유도 바로 cuDSS plateau 때문이다.
   - `case_ACTIVSg70k`에서 `factorize_v3_ker`는 b1 평균 1.88 ms, b256 평균 251.1 ms다. 총 launch는 6회로 같고 grid는 약 52k에서 13.4M으로 커진다. scenario당으로 보면 약 0.98 ms라 b1 대비 1.9x 정도 좋아진다.
   - 같은 case에서 `factorize_ker`는 scenario당 약 0.80 ms로 b1 대비 1.4x, `bwd_ker`는 약 0.18 ms로 b1 대비 약 1.5x다. cuDSS 내부 solve/factor는 좋아지지만 custom kernel만큼 선형적으로 좋아지지는 않는다.

따라서 `b256`의 장점은 다음처럼 정리된다.

```text
batch 256
  = launch 수는 거의 유지
  + per-launch grid/waves 증가
  + custom kernel SM util 상승
  + reduction/auxiliary 고정비 분산
  + cuDSS uniform batch factor/solve의 per-scenario 비용 감소
```

반대로 `>=50k`에서 b256이 압도적이지 않은 이유도 명확하다. 큰 case는 b1부터 grid가 크고, cuDSS factor/solve가 전체 시간을 지배하며, cuDSS SM util이 31%대에서 포화된다. 그래서 `case_ACTIVSg70k`는 b128에서 b256으로 가도 1.9%만 개선된다. 이 경우 throughput-only는 b256이지만, PP stage balance와 workspace memory까지 고려하면 b128도 micro-batch 후보로 남겨야 한다.

## cuDSS vs 나머지 커널

여기서는 `cudss`와 `cudss_aux`를 합쳐 `cuDSS 계열`로 보고, 나머지 operator를 `custom 커널`로 묶었다.

### b256에서의 시간 비중

| case | cuDSS ms/scenario | cuDSS share | cuDSS SM util % | custom ms/scenario | custom share | custom SM util % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `case118` | 0.1007 | 98.2% | 11.5 | 0.0019 | 1.8% | 65.8 |
| `case1197` | 0.2524 | 94.5% | 23.3 | 0.0148 | 5.5% | 81.6 |
| `case6468rte` | 0.7364 | 90.2% | 31.5 | 0.0801 | 9.8% | 81.3 |
| `case_ACTIVSg10k` | 1.8036 | 90.8% | 33.9 | 0.1837 | 9.2% | 82.7 |
| `case_ACTIVSg25k` | 3.5138 | 90.0% | 32.1 | 0.3883 | 10.0% | 81.3 |
| `case_ACTIVSg70k` | 14.6415 | 90.4% | 30.4 | 1.5471 | 9.6% | 80.3 |

b256 기준으로 보면, `case118`을 제외한 1k 이상 case에서 custom 커널은 이미 약 80% SM util까지 올라가지만 전체 시간 비중은 5-10%다. 반대로 cuDSS 계열은 전체 시간의 약 90%를 차지하지만 SM util은 30%대에 머문다. 따라서 b256 이후의 전체 throughput은 custom 커널이 아니라 cuDSS 계열이 결정한다.

### b1 -> b256 개선 분해

| case | cuDSS b1 ms/scenario | cuDSS b256 ms/scenario | cuDSS speedup | custom b1 ms/scenario | custom b256 ms/scenario | custom speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `case118` | 10.4703 | 0.1007 | 103.9x | 0.0997 | 0.0019 | 53.5x |
| `case1197` | 13.1566 | 0.2524 | 52.1x | 0.1407 | 0.0148 | 9.5x |
| `case6468rte` | 17.0672 | 0.7364 | 23.2x | 0.2884 | 0.0801 | 3.6x |
| `case_ACTIVSg10k` | 21.7999 | 1.8036 | 12.1x | 0.5741 | 0.1837 | 3.1x |
| `case_ACTIVSg25k` | 30.3105 | 3.5138 | 8.6x | 1.0028 | 0.3883 | 2.6x |
| `case_ACTIVSg70k` | 56.9451 | 14.6415 | 3.9x | 3.6526 | 1.5471 | 2.4x |

작은 case에서 b256이 좋은 주된 이유는 cuDSS 계열 고정비가 scenario당 크게 줄기 때문이다. `case118`은 cuDSS 계열만 봐도 103.9x 개선된다. custom 커널도 53.5x 좋아지지만, b256에서 custom share가 1.8%밖에 안 되므로 전체 성능 설명력은 cuDSS 쪽이 훨씬 크다.

큰 case로 갈수록 두 그룹 모두 개선폭이 줄어든다. `case_ACTIVSg70k`는 cuDSS가 3.9x, custom이 2.4x만 개선된다. 큰 case에서는 b1부터 각 kernel의 grid가 이미 큰 편이라 launch amortization 이득이 작고, cuDSS factor/solve의 sparse 병렬성이 먼저 포화된다.

### b128 -> b256 추가 개선 분해

| case | cuDSS b128 ms/scenario | cuDSS b256 ms/scenario | cuDSS gain | custom b128 ms/scenario | custom b256 ms/scenario | custom gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `case118` | 0.1806 | 0.1007 | 44.2% | 0.0022 | 0.0019 | 16.5% |
| `case1197` | 0.3522 | 0.2524 | 28.3% | 0.0152 | 0.0148 | 3.1% |
| `case6468rte` | 0.8610 | 0.7364 | 14.5% | 0.0809 | 0.0801 | 1.0% |
| `case_ACTIVSg10k` | 1.9499 | 1.8036 | 7.5% | 0.1851 | 0.1837 | 0.8% |
| `case_ACTIVSg25k` | 4.5417 | 3.5138 | 22.6% | 0.4680 | 0.3883 | 17.0% |
| `case_ACTIVSg70k` | 14.9403 | 14.6415 | 2.0% | 1.5590 | 1.5471 | 0.8% |

1k 이상에서 custom 커널은 b128 시점에 거의 포화되어 있다. `case1197` 이상에서는 custom의 b128 -> b256 추가 개선이 대체로 0.8-3.1% 수준이다. 따라서 b128 이후 b256의 추가 이득은 대부분 cuDSS 계열에서 나온다. 단, `case_ACTIVSg25k`는 b128 launch count가 다른 anomalous point라 재측정 대상이다.

### 그룹별 해석

cuDSS 계열:

- 전체 시간의 대부분을 차지한다. b256에서도 1k 이상 case에서 약 90% share다.
- b1 -> b256 개선은 크지만, case가 커질수록 103.9x에서 3.9x까지 줄어든다.
- SM util은 b1에서 낮고 batch 증가로 올라가지만, 큰 case에서는 b128/b256에서 30% 안팎으로 포화된다.
- 따라서 b256의 전체 성능 이득과 plateau를 동시에 설명하는 주된 그룹이다.

custom 커널:

- b256에서 SM util은 대부분 80% 전후까지 올라간다.
- 하지만 전체 duration share는 큰 case에서도 약 10%라서, 전체 ms/scenario를 지배하지 않는다.
- b128 이후 추가 개선은 거의 없다. custom kernel 관점의 micro-batch 포화점은 대체로 b64-b128 근처다.
- custom 커널을 더 최적화하면 일부 개선은 가능하지만, 실험 1의 micro-batch 선택을 좌우하는 1차 병목은 cuDSS다.

실험 1 관점의 결론:

```text
b256이 좋은 이유
  작은/중간 case: cuDSS 고정비와 launch overhead를 scenario 수로 분산
  custom kernel: b64-b128 사이에 대부분 util 포화
  cuDSS kernel: b256까지 per-scenario 비용 감소가 남아 있음

b256이 압도적이지 않은 이유
  큰 case: cuDSS factor/solve가 전체 시간의 약 90%
  cuDSS SM util: b128/b256에서 30%대 plateau
  custom kernel: 이미 b128에서 거의 포화
```

## 관찰

- b256 전체 duration 기준으로 `cudss`가 87.8%를 차지한다. 그런데 duration-weighted SM util은 31.6%라서, 전체 평균 GPU util을 낮추는 가장 큰 원인이다.
- `ibus`는 b256에서 84.8% 수준으로 거의 모든 큰 case에서 안정적으로 높다. 하지만 duration share는 8.0%라 전체 시간을 지배하지는 않는다.
- `voltage_reconstruct`도 큰 case에서 83-85% 수준으로 잘 찬다. 다만 duration share는 0.3% 수준이다.
- `mismatch`는 b256 큰 case에서 64-69% 수준, `jacobian_fill`은 case에 따라 52-81%로 변동한다. 두 연산 모두 memory throughput이 높은 편이라 메모리 쪽 압력이 더 강하게 보인다.
- `prepare_rhs`와 `voltage_update_apply`는 b256에서도 각각 약 31%, 32% 수준에서 plateau가 생긴다.
- `cudss_aux`는 batch와 무관하게 4.5-7.5% 수준이며, 작은 case에서는 고정 overhead 때문에 duration share가 크게 보인다.
- batch scaling을 보면 custom operator들은 batch 증가에 따라 util이 크게 올라가지만, `cudss`는 b64 이후 31%대에서 거의 plateau다. cuDSS sparse 내부 kernel 특성이 전체 평균 util의 병목으로 보인다.
