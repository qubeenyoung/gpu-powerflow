# CUDSS_CONFIG_HOST_NTHREADS Sweep Plan

## Goal

Measure how `CUDSS_CONFIG_HOST_NTHREADS` affects the current `cuda_edge`
benchmark path.

The sweep keeps the previous reordering experiment's best broad baseline:

- profile: `cuda_edge` (`cuda_mixed_edge`, FP32 edge Jacobian + cuDSS32 solve)
- reordering: `CUDSS_ALG_DEFAULT`
- modes: `end2end`, `operators`
- warmup/repeats: `1 / 10`
- cases:
  - `case30_ieee`
  - `case118_ieee`
  - `case793_goc`
  - `case1354_pegase`
  - `case2746wop_k`
  - `case4601_goc`
  - `case8387_pegase`
  - `case9241_pegase`

## cuDSS Constraints

The local cuDSS header is version `0.7.1`, so use the CUDA 12.9.1 cuDSS docs
as the reference for this sweep.

- `CUDSS_CONFIG_HOST_NTHREADS` has associated parameter type `int`.
- The default is `-1`, which uses the value returned by
  `cudssGetMaxThreads()`.
- This setting only affects execution when cuDSS Multi-Threaded mode is
  enabled.
- MT mode is enabled by setting a threading layer via
  `cudssSetThreadingLayer()`. If that call receives `NULL`, cuDSS reads
  `CUDSS_THREADING_LIB` from the environment.
- The cuDSS package includes a GNU OpenMP threading layer:
  `libcudss_mtlayer_gomp.so`.
- In this workspace, setting only `CUDSS_THREADING_LIB` caused repeat-run
  SIGSEGVs during `cudssDestroy()` / threading-layer cleanup. The benchmark
  runner now also prepends the same library to `LD_PRELOAD` for MT runs, which
  keeps the MT layer loaded for the process lifetime and allowed all planned
  MT sweep values to complete.

Reference:

- <https://docs.nvidia.com/cuda/archive/12.9.1/cudss/types.html>
- <https://docs.nvidia.com/cuda/archive/12.9.1/cudss/advanced_features.html>

On this machine, `nproc` returned `32`, and the available threading layer is:

```text
/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0
```

## Implementation Plan

Add build-time controls:

- `CUPF_CUDSS_ENABLE_MT`
  - CMake cache boolean, default `OFF`.
  - When `ON`, compile `CUPF_CUDSS_ENABLE_MT` and call
    `cudssSetThreadingLayer(handle, nullptr)` immediately after
    `cudssCreate()`.
  - The benchmark runner will set `CUDSS_THREADING_LIB` in the process
    environment when MT mode is enabled.
- `CUPF_CUDSS_HOST_NTHREADS`
  - CMake cache string, default `AUTO`.
  - Accepted values: `AUTO` or integer `>= 1`.
  - `AUTO` means do not call `cudssConfigSet()` for this parameter, preserving
    cuDSS's internal default exactly.
  - Numeric values compile into `CUPF_CUDSS_HOST_NTHREADS=<value>`.
  - Apply with `cudssConfigSet(config, CUDSS_CONFIG_HOST_NTHREADS, ...)`
    before `CUDSS_PHASE_ANALYSIS`.

Update code paths:

- Extend `cpp/src/newton_solver/ops/linear_solve/cudss_config.hpp`.
- Apply the config in both cuDSS solvers:
  - `cuda_cudss32.cpp`
  - `cuda_cudss64.cpp`
- Keep reordering config separate and set all cuDSS config knobs before
  analysis.

Update harness:

- Add runner args:
  - `--cudss-enable-mt`
  - `--cudss-host-nthreads {AUTO,int}`
  - `--cudss-threading-lib PATH`
- Pass CMake definitions into the configure command.
- For MT runs, set both `CUDSS_THREADING_LIB` and `LD_PRELOAD` in the benchmark
  process environment.
- Add the selected values and the threading library path to `manifest.json`.
- Add the selected values to generated run README/SUMMARY files.

## Sweep Matrix

The first row keeps MT mode off and is the compatibility baseline. The remaining
rows enable the cuDSS threading layer and sweep the host thread count.

| label | `CUPF_CUDSS_ENABLE_MT` | `CUPF_CUDSS_HOST_NTHREADS` | `CUDSS_THREADING_LIB` |
|---|---:|---:|---|
| `no_mt` | `OFF` | `AUTO` | unset |
| `mt_auto` | `ON` | `AUTO` | `libcudss_mtlayer_gomp.so.0` |
| `mt_1` | `ON` | `1` | `libcudss_mtlayer_gomp.so.0` |
| `mt_2` | `ON` | `2` | `libcudss_mtlayer_gomp.so.0` |
| `mt_4` | `ON` | `4` | `libcudss_mtlayer_gomp.so.0` |
| `mt_8` | `ON` | `8` | `libcudss_mtlayer_gomp.so.0` |
| `mt_16` | `ON` | `16` | `libcudss_mtlayer_gomp.so.0` |
| `mt_32` | `ON` | `32` | `libcudss_mtlayer_gomp.so.0` |

## Output Layout

```text
/workspace/exp/20260412_2/mt/
  PLAN.md
  cases.txt
  build/
    no_mt/
    mt_auto/
    mt_1/
    mt_2/
    mt_4/
    mt_8/
    mt_16/
    mt_32/
  results/
    cuda_edge_mt_no_mt/
    cuda_edge_mt_auto/
    cuda_edge_mt_1/
    cuda_edge_mt_2/
    cuda_edge_mt_4/
    cuda_edge_mt_8/
    cuda_edge_mt_16/
    cuda_edge_mt_32/
  combined_aggregates.csv
  mt_comparison.csv
  operator_timer_comparison.csv
  SUMMARY.md
```

## Command Template

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --results-root /workspace/exp/20260412_2/mt/results \
  --run-name cuda_edge_mt_8 \
  --mode both \
  --case-list /workspace/exp/20260412_2/mt/cases.txt \
  --profiles cuda_edge \
  --warmup 1 \
  --repeats 10 \
  --cudss-reordering-alg DEFAULT \
  --cudss-enable-mt \
  --cudss-host-nthreads 8 \
  --cudss-threading-lib /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0 \
  --end2end-build-dir /workspace/exp/20260412_2/mt/build/mt_8/end2end \
  --operators-build-dir /workspace/exp/20260412_2/mt/build/mt_8/operators
```

## Metrics To Compare

Primary:

- `elapsed_sec_mean`
- `analyze_sec_mean`
- `solve_sec_mean`

Operator focus:

- `CUDA.analyze.cudss32.analysis.avg_sec`
- `CUDA.analyze.cudss32.setup.avg_sec`
- `CUDA.solve.factorization32.avg_sec`
- `CUDA.solve.refactorization32.avg_sec`
- `CUDA.solve.solve32.avg_sec`
- `NR.analyze.linear_solve.avg_sec`
- `NR.iteration.linear_solve.avg_sec`

Report:

- per-case time table
- sum of elapsed means across all cases
- geometric mean speedup vs `no_mt`
- best host thread count per case
- whether MT mode improves only analysis or also end-to-end time
