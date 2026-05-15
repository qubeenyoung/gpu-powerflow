# NCU Custom Kernel Analysis

Date: 2026-05-07

## 측정 조건

대표 case:

| target size | selected case | buses | 비고 |
| ---: | --- | ---: | --- |
| 100 | `case94pi` | 94 | 100 bus 근사 |
| 1,000 | `case1197` | 1,197 | 1k 근사 |
| 5,000 | `case6468rte` | 6,468 | 정확한 5k case가 없어 3k-8k 중 선택 |
| 10,000 | `case_ACTIVSg10k` | 10,000 | 정확히 10k |
| 25,000 | `case_ACTIVSg25k` | 25,000 | 정확히 25k |
| 70,000 | `case_ACTIVSg70k` | 70,000 | 정확히 70k |

Batch:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
```

ncu scope:

```text
set = basic
profile = cuda_mixed_edge
warmup = 0
repeats = 1
target kernels = cuPF custom kernels only
```

대상 custom kernels:

```text
compute_ibus_kernel
compute_mismatch_from_ibus_kernel
reduce_mismatch_norm_kernel
fill_jacobian_gpu_kernel
prepare_rhs_kernel
apply_voltage_update_kernel
reconstruct_voltage_kernel
```

주의:

- 이 pass는 cuDSS 내부 kernel을 제외한다.
- ncu replay 때문에 `run_summary.csv`의 elapsed/solve time은 일반 benchmark timing으로 해석하지 않는다.
- batch별 kernel metric 비교에는 `case_batch_summary.csv`, `operator_summary.csv`, `launch_metrics.csv`를 사용한다.

## 전체 요약

- 총 `6 case * 9 batch = 54`개 ncu run이 모두 완료되었다.
- 모든 run이 수렴 성공했다.
- batch가 커질수록 custom kernel의 평균 SM throughput, memory throughput, occupancy가 증가했다.
- 큰 case에서는 batch 128-256 구간에서 custom kernel 평균 memory throughput이 65-72% 수준까지 올라간다.
- 개별 operator 기준으로는 `ibus`가 b256에서 SM throughput 약 80-85%까지 올라가고, `jacobian_fill`/`mismatch`는 memory throughput이 높은 쪽으로 간다.

## Case/Batch 핵심 지표

아래 표의 duration은 profiled custom kernel duration 합계이며, per-scenario는 `duration_sum / batch`이다.

| case | buses | batch | custom duration / scenario | SM mean | SM max | Memory mean | Occupancy mean | waves/SM max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `case94pi` | 94 | 1 | 129.3 us | 1.0% | 5.4% | 1.0% | 11.6% | 0.19 |
| `case94pi` | 94 | 128 | 2.45 us | 25.0% | 74.3% | 12.6% | 30.8% | 3.06 |
| `case94pi` | 94 | 256 | 1.98 us | 33.0% | 79.8% | 18.0% | 44.4% | 6.11 |
| `case1197` | 1,197 | 1 | 140.7 us | 6.0% | 29.9% | 3.1% | 16.0% | 2.43 |
| `case1197` | 1,197 | 128 | 15.25 us | 52.0% | 83.9% | 45.4% | 67.8% | 38.93 |
| `case1197` | 1,197 | 256 | 14.77 us | 59.2% | 84.2% | 54.4% | 73.8% | 77.85 |
| `case6468rte` | 6,468 | 1 | 287.7 us | 13.8% | 52.0% | 8.9% | 16.7% | 13.15 |
| `case6468rte` | 6,468 | 128 | 80.66 us | 56.2% | 84.6% | 64.5% | 72.6% | 210.34 |
| `case6468rte` | 6,468 | 256 | 79.95 us | 59.0% | 84.7% | 69.6% | 76.9% | 420.68 |
| `case_ACTIVSg10k` | 10,000 | 1 | 573.8 us | 16.6% | 56.0% | 11.6% | 17.8% | 20.33 |
| `case_ACTIVSg10k` | 10,000 | 128 | 184.8 us | 58.5% | 84.7% | 66.8% | 73.0% | 325.20 |
| `case_ACTIVSg10k` | 10,000 | 256 | 183.6 us | 61.0% | 84.7% | 72.0% | 77.0% | 650.41 |
| `case_ACTIVSg25k` | 25,000 | 1 | 999.9 us | 23.5% | 60.1% | 19.2% | 29.6% | 50.81 |
| `case_ACTIVSg25k` | 25,000 | 128 | 388.4 us | 56.3% | 84.7% | 64.8% | 73.2% | 813.01 |
| `case_ACTIVSg25k` | 25,000 | 256 | 468.1 us | 58.6% | 84.8% | 70.2% | 77.1% | 1626.02 |
| `case_ACTIVSg70k` | 70,000 | 1 | 3648.4 us | 32.6% | 62.1% | 31.0% | 51.3% | 142.28 |
| `case_ACTIVSg70k` | 70,000 | 128 | 1785.8 us | 54.3% | 84.8% | 67.3% | 72.5% | 2276.42 |
| `case_ACTIVSg70k` | 70,000 | 256 | 1547.7 us | 56.4% | 84.8% | 70.8% | 76.1% | 4552.85 |

## Operator 관측

b256 기준:

| case | dominant custom duration | high-SM operator | high-memory operator |
| --- | --- | --- | --- |
| `case94pi` | `ibus` 0.339 ms | `ibus` 79.7% | `jacobian_fill` 36.1% |
| `case1197` | `ibus` 3.258 ms | `ibus` 84.2% | `jacobian_fill` 89.7%, `mismatch` 77.0% |
| `case6468rte` | `ibus` 17.507 ms | `ibus` 84.7% | `jacobian_fill` 106.2%, `mismatch` 92.3% |
| `case_ACTIVSg10k` | `ibus` 40.581 ms | `ibus` 84.7% | `jacobian_fill` 112.3%, `mismatch` 93.5% |
| `case_ACTIVSg25k` | `ibus` 101.396 ms | `ibus` 84.8% | `mismatch` 94.3%, `prepare_rhs` 91.6% |
| `case_ACTIVSg70k` | `ibus` 331.205 ms | `ibus` 84.8% | `mismatch` 95.2%, `jacobian_fill` 93.4% |

해석:

- `ibus`는 custom kernel 중 duration 지배적이고, batch가 충분히 크면 SM throughput이 80% 이상으로 올라간다.
- `jacobian_fill`과 `mismatch`는 큰 batch에서 memory throughput이 높아져 memory-side 최적화 후보로 보인다.
- `mismatch_norm`, `prepare_rhs`, `voltage_update_apply`는 batch 증가로 개선되지만 전체 duration 지배력은 상대적으로 작다.
- b256에서 25K의 custom duration/scenario가 b128보다 나빠진 것은 ncu run에서 Newton iteration 수가 `5 -> 6`으로 늘어난 영향이 섞여 있다. 동일 반복 수 고정 profiling이 필요하다.

## 다음 작업

1. cuDSS 내부 kernel을 별도 ncu sweep로 측정한다.
2. `tolerance=-1`, `max_iter` 고정 방식으로 Newton iteration 수를 고정한 ncu pass를 추가한다.
3. custom kernel 결과와 cuDSS 결과를 합쳐 DP/PP micro-batch 추천을 확정한다.
