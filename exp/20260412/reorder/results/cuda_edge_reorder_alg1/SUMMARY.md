# cuPF Benchmark `cuda_edge_reorder_alg1`

## Setup

- Created UTC: 2026-04-12T06:23:53.946144+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: ALG_1

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.004267 | 0.003133 | 0.001133 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.005701 | 0.003235 | 0.002465 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.017993 | 0.004127 | 0.013866 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.025754 | 0.004911 | 0.020843 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.048882 | 0.006779 | 0.042103 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.099161 | 0.010549 | 0.088611 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.171215 | 0.017809 | 0.153406 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.208207 | 0.019888 | 0.188318 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.004385 | 0.003145 | 0.001239 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.005805 | 0.003241 | 0.002563 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.018342 | 0.004099 | 0.014242 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.025912 | 0.004951 | 0.020961 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.049003 | 0.006775 | 0.042228 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.100491 | 0.011016 | 0.089474 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.169979 | 0.017486 | 0.152493 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.211194 | 0.020852 | 0.190341 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/build/alg1/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/build/alg1/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
