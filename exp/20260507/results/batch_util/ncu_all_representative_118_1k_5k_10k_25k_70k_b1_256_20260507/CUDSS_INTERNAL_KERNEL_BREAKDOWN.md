# cuDSS 내부 Kernel GPU Util 분석

Date: 2026-05-07

## 기준

- 데이터는 `launch_metrics.csv`의 b256 launch rows를 사용했다.
- GPU util은 NCU `Compute (SM) Throughput`이며, duration-weighted 평균으로 계산했다.
- cuDSS phase는 NVTX range가 아니라 demangled kernel name으로 추정 분류했다.
- `factorize`: `factorize_v3_ker`, `factorize_ker`, `independent_ker`
- `triangular_solve`: `fwd_ker`, `bwd_ker`
- `copy_permute_map`: `copy_matrix_ker`, `perm_ker`, `map_ker`, `plain_map_ker`
- `symbolic_structure`: `nnz_per_col_ker`, `csc_rows_ker`, `define_superpanel_ker`, `dependency_map_ker`
- `sort`: `radix_sort_ker`

## b256 cuDSS phase breakdown

| phase | launches | duration ms | cuDSS duration share | SM util weighted % | Mem util weighted % | Occ weighted % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `factorize` | 69 | 3804.1 | 72.7% | 32.0 | 30.6 | 51.0 |
| `triangular_solve` | 133 | 1293.0 | 24.7% | 32.0 | 30.9 | 34.3 |
| `copy_permute_map` | 264 | 81.7 | 1.6% | 16.5 | 78.7 | 81.5 |
| `symbolic_structure` | 156 | 31.3 | 0.6% | 2.5 | 2.5 | 16.6 |
| `sort` | 48 | 10.0 | 0.2% | 81.3 | 81.3 | 29.2 |
| `other_cudss` | 932 | 9.1 | 0.2% | 15.5 | 29.8 | 32.5 |

## b256 case별 phase breakdown

| case | phase | launches | duration ms | cuDSS share | SM util weighted % | Mem util weighted % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `case118` | `factorize` | 6 | 3.7 | 54.6% | 28.7 | 21.5 |
| `case118` | `triangular_solve` | 8 | 1.8 | 27.5% | 23.3 | 21.3 |
| `case118` | `copy_permute_map` | 39 | 0.3 | 4.4% | 13.4 | 12.2 |
| `case118` | `symbolic_structure` | 26 | 0.5 | 6.8% | 0.6 | 0.5 |
| `case1197` | `factorize` | 9 | 23.0 | 52.9% | 38.4 | 31.2 |
| `case1197` | `triangular_solve` | 19 | 15.6 | 35.9% | 27.4 | 25.9 |
| `case6468rte` | `factorize` | 9 | 109.3 | 66.8% | 37.9 | 31.7 |
| `case6468rte` | `triangular_solve` | 19 | 47.0 | 28.7% | 32.3 | 29.8 |
| `case_ACTIVSg10k` | `factorize` | 15 | 305.3 | 70.4% | 37.7 | 32.2 |
| `case_ACTIVSg10k` | `triangular_solve` | 29 | 117.3 | 27.0% | 32.3 | 30.2 |
| `case_ACTIVSg25k` | `factorize` | 12 | 629.2 | 72.5% | 33.4 | 30.7 |
| `case_ACTIVSg25k` | `triangular_solve` | 24 | 213.5 | 24.6% | 33.7 | 32.0 |
| `case_ACTIVSg70k` | `factorize` | 18 | 2733.7 | 73.6% | 30.7 | 30.3 |
| `case_ACTIVSg70k` | `triangular_solve` | 34 | 897.8 | 24.2% | 31.7 | 30.8 |

For 10k 이상 case에서는 factorization이 cuDSS 내부 시간의 약 70-74%, triangular solve가 약 24-27%다.

## b256 cuDSS exact kernel top

| kernel | duration ms | cuDSS share | launches | SM util weighted % | Mem util weighted % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cudss::factorize_v3_ker<long, float, int, float, 128, 8, 0, 0, 0, 1, 1, 16, 32, 0>` | 2200.2 | 42.1% | 24 | 36.4 | 34.7 |
| `cudss::factorize_ker<long, float, int, float, 32, 0, 0, 0, 1, 0, 16, 32, 0>` | 1597.9 | 30.6% | 21 | 25.9 | 24.9 |
| `cudss::bwd_ker<long, float, int, 32, 32, 16, 2, 16, 1, 0, 1, 0>` | 796.4 | 15.2% | 45 | 30.5 | 28.4 |
| `cudss::fwd_ker<long, float, int, 32, 1, 0, 0, 32, 1, 16, 32>` | 422.1 | 8.1% | 21 | 36.4 | 36.4 |
| `cudss::fwd_ker<long, float, int, 128, 1, 0, 0, 128, 1, 16, 8>` | 72.7 | 1.4% | 24 | 24.2 | 24.0 |
| `cudss::copy_matrix_ker<long, float, int, int, float, 128>` | 53.4 | 1.0% | 48 | 11.3 | 93.5 |

상위 5개 kernel만으로 b256 cuDSS duration의 약 97.4%를 설명한다.

## case_ACTIVSg70k b256 반복 구조 관찰

`stdout.txt` 기준 `iterations=7`이며, launch sequence에서는 custom residual 관련 kernel이 7회, `jacobian_fill`, `prepare_rhs`, `voltage_update_apply`, `voltage_reconstruct` 및 cuDSS factor/solve kernel이 6회 반복된다. 이는 마지막 residual check에는 factor/solve/update가 붙지 않는 Newton 수렴 구조와 맞는다.

70k b256의 반복별 큰 kernel 패턴은 다음처럼 반복된다.

| step kernel | per-launch duration | SM util % | comment |
| --- | ---: | ---: | --- |
| `compute_ibus_kernel` | 47.3 ms | 84.8 | custom residual 계산, util 높음 |
| `fill_jacobian_gpu_kernel` | 5.1 ms | 52.0 | memory throughput 약 93-94% |
| `prepare_rhs_kernel` | 0.49 ms | 31.1 | memory throughput 약 93% |
| `copy_matrix_ker` | 4.29 ms + 2.57 ms | 9.7-10.8 | memory throughput 약 94-95% |
| `factorize_ker` | 204.3 ms | 25.6 | 낮은 SM util의 큰 duration kernel |
| `factorize_v3_ker` | 251.1 ms | 34.9 | 가장 큰 single kernel |
| `fwd_ker` | 52.2 ms + 6.0 ms | 23.3-35.5 | triangular solve |
| `bwd_ker` | 23.5 ms + 67.8 ms | 22.3-32.8 | triangular solve |
| `apply_voltage_update_kernel` | 0.93 ms | 32.5 | memory throughput 약 83-84% |
| `reconstruct_voltage_kernel` | 2.23 ms | 84.5 | util 높음 |

## 결론

- 전체 GPU util 병목은 cuDSS auxiliary/setup이 아니라 cuDSS numerical factorization과 triangular solve다.
- b256 전체 cuDSS 내부에서 factorization이 72.7%, triangular solve가 24.7%를 차지한다.
- 큰 case에서는 factorization 비중이 더 뚜렷하며, `case_ACTIVSg70k` b256에서는 factorization만 2733.7 ms다.
- `copy_matrix_ker`는 memory throughput이 93.5%로 높지만 cuDSS duration share는 1.0%라 최우선 병목은 아니다.
- custom kernel 중 `ibus`와 `voltage_reconstruct`는 이미 84%대 SM util까지 올라간다. 반면 전체 시간의 대부분을 차지하는 cuDSS factor/solve 계열은 30%대 SM util에 머문다.
- 다음 실험에서 NVTX range를 넣는다면 `cudss_analyze`, `cudss_factorize`, `cudss_solve`, `cudss_refine/update`를 분리해 현재 kernel-name 기반 추정을 확인하는 것이 좋다.

## NVTX follow-up

소스에는 이미 stage-level NVTX wrapper가 있다.

- `/workspace/gpu-powerflow-master/cuPF/cpp/src/newton_solver/core/newton_solver.cpp:266`: `NR.iteration.factorize`
- `/workspace/gpu-powerflow-master/cuPF/cpp/src/newton_solver/core/newton_solver.cpp:267`: `NR.iteration.solve`
- `/workspace/gpu-powerflow-master/cuPF/cpp/src/newton_solver/ops/linear_solve/cuda_cudss.cpp:180`: `CUDSS_PHASE_ANALYSIS`
- `/workspace/gpu-powerflow-master/cuPF/cpp/src/newton_solver/ops/linear_solve/cuda_cudss.cpp:190`: `CUDSS_PHASE_FACTORIZATION` 또는 `CUDSS_PHASE_REFACTORIZATION`
- `/workspace/gpu-powerflow-master/cuPF/cpp/src/newton_solver/ops/linear_solve/cuda_cudss.cpp:217`: `CUDSS_PHASE_SOLVE`

이번에 사용된 build는 둘 다 NVTX와 timing이 꺼져 있었다.

```text
/workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/CMakeCache.txt
ENABLE_NVTX:BOOL=OFF
ENABLE_TIMING:BOOL=OFF

/workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto-dump/CMakeCache.txt
ENABLE_NVTX:BOOL=OFF
ENABLE_TIMING:BOOL=OFF
```

다음 pass에서는 기존 source를 새 build directory로 NVTX ON 빌드한 뒤, NCU NVTX filter로 factorize와 solve를 분리해서 수집하면 된다. cuDSS MT layer가 host thread를 쓸 수 있으므로 `--nvtx-push-pop-scope process`를 붙인다.

```bash
cmake -S /workspace/gpu-powerflow-master/cuPF \
  -B /workspace/gpu-powerflow-master/cuPF/build/bench-nvtx-cudss-mt-auto \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_BENCHMARKS=ON \
  -DWITH_CUDA=ON \
  -DENABLE_LOG=OFF \
  -DENABLE_TIMING=ON \
  -DENABLE_NVTX=ON \
  -DCUPF_CUDSS_ENABLE_MT=ON \
  -DCUPF_CUDSS_REORDERING_ALG=DEFAULT \
  -DCUPF_CUDSS_HOST_NTHREADS=AUTO \
  -DCUPF_CUDSS_ND_NLEVELS=AUTO

cmake --build /workspace/gpu-powerflow-master/cuPF/build/bench-nvtx-cudss-mt-auto \
  --target cupf_case_benchmark -j
```

예시 NCU command:

```bash
ncu --target-processes all \
  --set basic \
  --kernel-name-base demangled \
  --csv \
  --nvtx \
  --nvtx-push-pop-scope process \
  --nvtx-include "NR.iteration.factorize" \
  --log-file factorize_basic.csv \
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

`--nvtx-include "NR.iteration.solve"`로 같은 command를 한 번 더 돌리면 triangular solve range만 따로 수집할 수 있다.
