# cuPF Benchmark `cuda_edge_mt_mt_16`

## Setup

- Created UTC: 2026-04-12T09:33:17.804149+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: 16
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011252 | 0.010510 | 0.000741 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011698 | 0.010855 | 0.000842 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014725 | 0.013191 | 0.001534 | n/a | 5.9 |
| case1354_pegase | cuda_edge | True | 0.017280 | 0.015115 | 0.002165 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.020474 | 0.017931 | 0.002543 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.027368 | 0.022760 | 0.004607 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.041582 | 0.033604 | 0.007977 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.042439 | 0.034003 | 0.008436 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011197 | 0.010426 | 0.000770 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011629 | 0.010776 | 0.000852 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014568 | 0.013109 | 0.001459 | n/a | 5.6 |
| case1354_pegase | cuda_edge | True | 0.016916 | 0.014789 | 0.002126 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.019970 | 0.017579 | 0.002391 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.026660 | 0.022405 | 0.004254 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.036804 | 0.030623 | 0.006181 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.040352 | 0.033064 | 0.007288 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/mt/build/mt_16/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/mt/build/mt_16/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
