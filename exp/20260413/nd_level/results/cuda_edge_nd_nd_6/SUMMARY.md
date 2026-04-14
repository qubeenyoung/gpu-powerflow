# cuPF Benchmark `cuda_edge_nd_nd_6`

## Setup

- Created UTC: 2026-04-13T00:09:29.155640+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 6
- cuDSS LD_PRELOAD: 

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011071 | 0.010355 | 0.000716 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011693 | 0.010772 | 0.000920 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014483 | 0.012938 | 0.001545 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.016580 | 0.014656 | 0.001922 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.021459 | 0.019152 | 0.002307 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.032958 | 0.027042 | 0.005915 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.048763 | 0.041077 | 0.007686 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.053914 | 0.045117 | 0.008797 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010938 | 0.010182 | 0.000756 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011616 | 0.010684 | 0.000931 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014189 | 0.012751 | 0.001437 | n/a | 5.6 |
| case1354_pegase | cuda_edge | True | 0.016602 | 0.014585 | 0.002016 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.021512 | 0.019112 | 0.002399 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.033091 | 0.027069 | 0.006021 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.049008 | 0.041176 | 0.007832 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.054212 | 0.045186 | 0.009025 | n/a | 8.0 |

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
  /workspace/exp/20260413/nd_level/build/nd_6/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260413/nd_level/build/nd_6/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
