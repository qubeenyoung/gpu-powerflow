# Compute / Memory Throughput 및 Roofline 가능성

Date: 2026-05-07

## 가능 여부

가능하다. 단, 현재 `ncu --set basic` 결과로 가능한 것과 추가 측정이 필요한 것을 나눠야 한다.

현재 데이터로 가능:

- NCU `Compute (SM) Throughput`
- NCU `Memory Throughput`
- NCU `DRAM Throughput`
- batch별 cuDSS/custom의 compute-vs-memory proxy 분석

현재 데이터로 부족:

- FLOP/s
- byte/s의 실제 절대값
- arithmetic intensity = FLOPs / byte
- 진짜 roofline plot

정확한 roofline을 하려면 `SpeedOfLight_RooflineChart` 또는 hierarchical roofline section을 추가로 수집해야 한다.

```text
SpeedOfLight_RooflineChart
SpeedOfLight_HierarchicalSingleRooflineChart
SpeedOfLight_HierarchicalDoubleRooflineChart
SpeedOfLight_HierarchicalTensorRooflineChart
```

## 현재 데이터 기반 proxy roofline

아래 표는 대표 6개 case 전체를 duration-weighted로 합친 값이다.

| group | metric | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b128 | b256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cuDSS` | Compute % | 11.8 | 12.7 | 15.1 | 18.5 | 22.3 | 25.8 | 28.6 | 30.1 | 30.9 |
| `cuDSS` | Memory % | 9.8 | 10.5 | 13.2 | 16.8 | 21.1 | 24.9 | 28.0 | 29.5 | 30.4 |
| `cuDSS` | DRAM % | 0.9 | 1.0 | 1.4 | 2.0 | 2.7 | 3.4 | 4.0 | 4.3 | 4.3 |
| `custom` | Compute % | 31.3 | 48.0 | 61.4 | 69.9 | 75.1 | 78.1 | 79.6 | 80.3 | 80.7 |
| `custom` | Memory % | 12.8 | 18.3 | 22.6 | 25.4 | 27.5 | 28.6 | 29.4 | 29.8 | 29.7 |
| `custom` | DRAM % | 6.1 | 8.3 | 10.1 | 11.4 | 12.4 | 12.9 | 13.4 | 14.0 | 13.5 |

해석:

- `custom`은 batch 증가에 따라 compute throughput이 크게 오른다. b64-b128에서 약 80%로 포화된다.
- `custom`의 memory throughput은 b256에서도 약 30%다. 즉 custom 전체는 memory-bound라기보다 compute-side utilization이 높은 쪽이다.
- `cuDSS`는 compute와 memory가 같이 30% 근처까지 올라가고 b128-b256에서 포화된다.
- `cuDSS`의 DRAM throughput은 낮다. 따라서 cuDSS 전체를 단순 DRAM bandwidth-bound로 해석하면 안 된다. sparse factor/solve의 dependency, irregular access, synchronization, low effective parallelism이 섞인 plateau로 봐야 한다.

## b256 operator별 proxy 분류

| operator | duration ms | Compute % | Memory % | DRAM % | proxy 분류 |
| --- | ---: | ---: | ---: | ---: | --- |
| `cudss` | 5229.2 | 31.6 | 31.3 | 4.5 | 낮은 roofline plateau / sparse dependency |
| `cudss_aux` | 159.2 | 5.6 | 1.5 | 0.0 | underfilled / 고정비 |
| `ibus` | 477.5 | 84.8 | 20.5 | 1.7 | compute-leaning |
| `jacobian_fill` | 41.1 | 55.2 | 93.8 | 88.1 | memory-heavy |
| `mismatch` | 14.0 | 65.5 | 94.5 | 94.5 | high-throughput memory-heavy |
| `mismatch_norm` | 3.7 | 43.0 | 82.5 | 82.5 | memory/reduction-heavy |
| `prepare_rhs` | 4.1 | 31.1 | 91.9 | 91.9 | memory-bound |
| `voltage_reconstruct` | 18.9 | 84.2 | 29.1 | 29.1 | compute-leaning |
| `voltage_update_apply` | 7.9 | 32.5 | 83.5 | 83.5 | memory-bound |

주의:

- 이 표는 진짜 roofline이 아니다.
- `Compute %`와 `Memory %`의 상대 위치로 병목 성격을 보는 proxy다.
- FLOP count와 byte count가 없으므로 arithmetic intensity는 계산하지 않았다.

## 실험 1 관점 해석

batch 증가에 따른 util 변화는 두 가지 다른 모양을 보인다.

```text
custom:
  Compute %가 b1 31.3 -> b64 79.6 -> b256 80.7
  b64-b128부터 compute throughput 포화

cuDSS:
  Compute %가 b1 11.8 -> b64 28.6 -> b256 30.9
  Memory %도 b1 9.8 -> b64 28.0 -> b256 30.4
  둘 다 30% 근처에서 포화
```

따라서 b256이 좋은 이유는 custom kernel이 계속 좋아지기 때문이 아니다. custom은 b64-b128에서 이미 거의 포화된다. b128 이후 b256의 추가 이득은 주로 cuDSS의 scenario당 비용 감소에서 온다. 하지만 cuDSS의 compute/memory throughput이 둘 다 30% 근처에서 plateau라 큰 case에서는 b128과 b256 차이가 작다.

## 정확한 roofline 추가 측정 방법

정확한 roofline을 하려면 다음 중 하나를 추가로 수집한다.

가장 간단한 방법:

```bash
ncu --target-processes all \
  --section SpeedOfLight_RooflineChart \
  --kernel-name-base demangled \
  --csv \
  --log-file roofline_basic.csv \
  --force-overwrite \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case_ACTIVSg70k \
  --profile cuda_mixed_edge \
  --warmup 0 \
  --repeats 1 \
  --batch-size 256 \
  --tolerance 1e-8 \
  --max-iter 50 \
  --cudss-matching-alg DEFAULT \
  --cudss-pivot-epsilon AUTO
```

Mixed path라서 precision별로 더 보고 싶으면 아래 section을 나눠서 수집한다.

```bash
--section SpeedOfLight_HierarchicalSingleRooflineChart
--section SpeedOfLight_HierarchicalDoubleRooflineChart
--section SpeedOfLight_HierarchicalTensorRooflineChart
```

NVTX ON build를 쓰면 cuDSS factorize와 solve를 분리해서 roofline을 수집할 수 있다.

```bash
ncu --target-processes all \
  --section SpeedOfLight_RooflineChart \
  --nvtx \
  --nvtx-push-pop-scope process \
  --nvtx-include "NR.iteration.factorize" \
  --kernel-name-base demangled \
  --csv \
  --log-file roofline_factorize.csv \
  --force-overwrite \
  /workspace/gpu-powerflow-master/cuPF/build/bench-nvtx-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case_ACTIVSg70k \
  --profile cuda_mixed_edge \
  --warmup 0 \
  --repeats 1 \
  --batch-size 256 \
  --tolerance 1e-8 \
  --max-iter 50 \
  --cudss-matching-alg DEFAULT \
  --cudss-pivot-epsilon AUTO
```

`--nvtx-include "NR.iteration.solve"`로 바꾸면 triangular solve만 따로 볼 수 있다.

## 추가로 직접 계산할 metric 후보

NCU section 대신 metric을 직접 지정하려면 다음 계열이 필요하다.

FLOP proxy:

```text
sm__sass_thread_inst_executed_op_fadd_pred_on
sm__sass_thread_inst_executed_op_fmul_pred_on
sm__sass_thread_inst_executed_op_ffma_pred_on
sm__sass_thread_inst_executed_op_dadd_pred_on
sm__sass_thread_inst_executed_op_dmul_pred_on
sm__sass_thread_inst_executed_op_dfma_pred_on
```

Memory bytes:

```text
dram__bytes
dram__bytes_read
dram__bytes_write
l1tex__t_bytes
```

다만 직접 metric 방식은 FMA를 2 FLOPs로 계산하고 precision별 peak를 맞춰야 하므로, 우선은 NCU roofline section을 쓰는 편이 안전하다.
