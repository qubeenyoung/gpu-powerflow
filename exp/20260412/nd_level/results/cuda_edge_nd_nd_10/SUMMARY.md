# cuPF Benchmark `cuda_edge_nd_nd_10`

## Setup

- Created UTC: 2026-04-12T08:34:55.762491+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 10

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011151 | 0.010433 | 0.000718 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011701 | 0.010860 | 0.000840 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015709 | 0.014226 | 0.001482 | n/a | 5.6 |
| case1354_pegase | cuda_edge | True | 0.019216 | 0.017048 | 0.002168 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.025166 | 0.022653 | 0.002512 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.036388 | 0.032262 | 0.004126 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.053764 | 0.047797 | 0.005966 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.061173 | 0.052858 | 0.008315 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011293 | 0.010449 | 0.000844 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011899 | 0.010934 | 0.000964 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015954 | 0.014247 | 0.001706 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.019350 | 0.016995 | 0.002354 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.025383 | 0.022719 | 0.002664 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.037410 | 0.032623 | 0.004787 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.055710 | 0.048530 | 0.007179 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.061485 | 0.052901 | 0.008584 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/nd_level/build/nd_10/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/nd_level/build/nd_10/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
