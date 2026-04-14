# cuPF Benchmark `cuda_edge_mt_no_mt`

## Setup

- Created UTC: 2026-04-12T09:14:57.889999+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: 

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010941 | 0.010259 | 0.000681 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011505 | 0.010717 | 0.000787 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015492 | 0.014082 | 0.001409 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.018864 | 0.016853 | 0.002011 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.024577 | 0.022340 | 0.002237 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.036360 | 0.032243 | 0.004117 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.053735 | 0.047746 | 0.005989 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.065002 | 0.054808 | 0.010194 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011272 | 0.010432 | 0.000839 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011796 | 0.010850 | 0.000945 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015940 | 0.014249 | 0.001690 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.019395 | 0.017038 | 0.002356 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.025301 | 0.022655 | 0.002645 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.036513 | 0.032263 | 0.004249 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.055787 | 0.048590 | 0.007196 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.059399 | 0.052153 | 0.007246 | n/a | 8.0 |

## Files

- `manifest.json`: run configuration and environment
- `summary.csv`: one row per measured run across all measurement modes
- `aggregates.csv`: grouped statistics by mode/case/profile
- `summary_<mode>.csv` and `aggregates_<mode>.csv`: mode-specific views
- `raw/<mode>/`: per-run timing payload

## Nsight Hints

Use the operators benchmark binary directly for profiling. Prefer `--warmup 1` to remove one-time CUDA setup from measured repeats.

```bash
nsys profile --trace=cuda,nvtx -o cupf_nsys \
  /workspace/exp/20260412_2/mt/build/no_mt/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/mt/build/no_mt/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
