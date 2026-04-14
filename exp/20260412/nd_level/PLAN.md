# CUDSS_CONFIG_ND_NLEVELS Sweep Plan

## Goal

Measure how `CUDSS_CONFIG_ND_NLEVELS` affects the current `cuda_edge`
benchmark path when cuDSS uses the default reordering algorithm.

The sweep keeps everything except `ND_NLEVELS` fixed:

- profile: `cuda_edge` (`cuda_mixed_edge`, FP32 edge Jacobian + cuDSS32 solve)
- reordering: `CUDSS_ALG_DEFAULT`
- MT mode: disabled
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

The local cuDSS header is version `0.7.1`, so use the current cuDSS docs that
match the installed 0.7.x behavior as the reference for this sweep. A smoke run
with `CUDSS_CONFIG_ND_NLEVELS=512` failed with `CUDSS_STATUS_INVALID_VALUE` in
this environment, so this plan uses the documented small integer range instead
of the newer power-of-two values.

- `CUDSS_CONFIG_ND_NLEVELS` has associated parameter type `int`.
- It controls the minimum number of levels for nested-dissection reordering.
- It only works when `CUDSS_CONFIG_REORDERING_ALG` is `CUDSS_ALG_DEFAULT`.
- The value should be a positive integer.
- The documented default is `10`.
- NVIDIA describes it as an advanced performance knob and recommends trying
  values not too far from the default, such as `8` through `11`.

Reference:

- <https://docs.nvidia.com/cuda/cudss/doc_output/types.html>

## Implementation Plan

Add build-time control:

- `CUPF_CUDSS_ND_NLEVELS`
  - CMake cache string, default `AUTO`.
  - Accepted values: `AUTO` or integer `>= 0`.
  - `AUTO` means do not call `cudssConfigSet()` for this parameter, preserving
    cuDSS's internal default exactly.
  - Numeric values compile into `CUPF_CUDSS_ND_NLEVELS=<value>`.
  - Apply with `cudssConfigSet(config, CUDSS_CONFIG_ND_NLEVELS, ...)` before
    `CUDSS_PHASE_ANALYSIS`.

Validation:

- If `CUPF_CUDSS_REORDERING_ALG` is not `DEFAULT` and
  `CUPF_CUDSS_ND_NLEVELS` is not `AUTO`, fail at CMake configure time.
- Keep the main sweep close to the documented default. Do not include `512` in
  this environment because the installed cuDSS returns `CUDSS_STATUS_INVALID_VALUE`.

Update code paths:

- Extend `cpp/src/newton_solver/ops/linear_solve/cudss_config.hpp`.
- Apply the config in both cuDSS solvers:
  - `cuda_cudss32.cpp`
  - `cuda_cudss64.cpp`
- Keep the config call before analysis and after `cudssConfigCreate()`.

Update harness:

- Add runner arg:
  - `--cudss-nd-nlevels {AUTO,int}`
- Pass the CMake definition into the configure command.
- Add the selected value to `manifest.json`.
- Add the selected value to generated run README/SUMMARY files.

## Sweep Matrix

Main sweep:

| label | `CUPF_CUDSS_REORDERING_ALG` | `CUPF_CUDSS_ND_NLEVELS` |
|---|---|---:|
| `auto` | `DEFAULT` | `AUTO` |
| `nd_8` | `DEFAULT` | `8` |
| `nd_9` | `DEFAULT` | `9` |
| `nd_10` | `DEFAULT` | `10` |
| `nd_11` | `DEFAULT` | `11` |

Optional follow-up, only if the main sweep suggests a boundary effect:

| label | `CUPF_CUDSS_REORDERING_ALG` | `CUPF_CUDSS_ND_NLEVELS` |
|---|---|---:|
| `nd_7` | `DEFAULT` | `7` |
| `nd_12` | `DEFAULT` | `12` |

## Output Layout

```text
/workspace/exp/20260412_2/nd_level/
  PLAN.md
  cases.txt
  build/
    auto/
    nd_8/
    nd_9/
    nd_10/
    nd_11/
  results/
    cuda_edge_nd_auto/
    cuda_edge_nd_8/
    cuda_edge_nd_9/
    cuda_edge_nd_10/
    cuda_edge_nd_11/
  combined_aggregates.csv
  nd_level_comparison.csv
  operator_timer_comparison.csv
  SUMMARY.md
```

## Command Template

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --results-root /workspace/exp/20260412_2/nd_level/results \
  --run-name cuda_edge_nd_10 \
  --mode both \
  --case-list /workspace/exp/20260412_2/nd_level/cases.txt \
  --profiles cuda_edge \
  --warmup 1 \
  --repeats 10 \
  --cudss-reordering-alg DEFAULT \
  --cudss-nd-nlevels 10 \
  --end2end-build-dir /workspace/exp/20260412_2/nd_level/build/nd_10/end2end \
  --operators-build-dir /workspace/exp/20260412_2/nd_level/build/nd_10/operators
```

## Metrics To Compare

Primary:

- `elapsed_sec_mean`
- `analyze_sec_mean`
- `solve_sec_mean`

Operator focus:

- `CUDA.analyze.cudss32.analysis.avg_sec`
- `CUDA.solve.factorization32.avg_sec`
- `CUDA.solve.refactorization32.avg_sec`
- `CUDA.solve.solve32.avg_sec`
- `NR.analyze.linear_solve.avg_sec`
- `NR.iteration.linear_solve.avg_sec`

Report:

- per-case time table
- sum of elapsed means across all cases
- geometric mean speedup vs `auto`
- best `ND_NLEVELS` per case
- whether the knob improves analysis without hurting factorization/solve
