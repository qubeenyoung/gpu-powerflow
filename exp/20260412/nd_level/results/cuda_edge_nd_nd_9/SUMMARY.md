# cuPF Benchmark `cuda_edge_nd_nd_9`

## Setup

- Created UTC: 2026-04-12T08:31:35.497984+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 9

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010878 | 0.010199 | 0.000679 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011500 | 0.010709 | 0.000790 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015400 | 0.013778 | 0.001622 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.017889 | 0.015980 | 0.001909 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.024419 | 0.021881 | 0.002538 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.034583 | 0.030414 | 0.004168 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.050022 | 0.044058 | 0.005963 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.057480 | 0.049130 | 0.008349 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010976 | 0.010224 | 0.000751 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011527 | 0.010669 | 0.000857 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015823 | 0.013924 | 0.001899 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.018089 | 0.016047 | 0.002042 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.023849 | 0.021483 | 0.002365 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.034733 | 0.030477 | 0.004256 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.050194 | 0.044087 | 0.006106 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.055762 | 0.048477 | 0.007284 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/nd_level/build/nd_9/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/nd_level/build/nd_9/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
