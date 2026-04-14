# cuPF Benchmark `cuda_edge_nd_nd_8`

## Setup

- Created UTC: 2026-04-12T08:28:17.609564+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 8

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010892 | 0.010209 | 0.000683 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011434 | 0.010653 | 0.000781 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014842 | 0.013262 | 0.001580 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.017445 | 0.015497 | 0.001947 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.024352 | 0.021421 | 0.002930 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.033080 | 0.028824 | 0.004256 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.046762 | 0.040456 | 0.006305 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.052698 | 0.044603 | 0.008094 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010952 | 0.010206 | 0.000747 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011518 | 0.010657 | 0.000861 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014871 | 0.013285 | 0.001585 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.017510 | 0.015465 | 0.002045 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.023042 | 0.020671 | 0.002371 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.033259 | 0.028867 | 0.004392 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.046991 | 0.040511 | 0.006480 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.052927 | 0.044624 | 0.008302 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/nd_level/build/nd_8/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/nd_level/build/nd_8/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
