# cuPF Benchmark `cuda_edge_reorder_default`

## Setup

- Created UTC: 2026-04-12T06:20:20.800133+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010931 | 0.010245 | 0.000686 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011487 | 0.010696 | 0.000790 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015479 | 0.014043 | 0.001436 | n/a | 5.9 |
| case1354_pegase | cuda_edge | True | 0.019224 | 0.017054 | 0.002169 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.025181 | 0.022671 | 0.002509 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.037270 | 0.032647 | 0.004622 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.055869 | 0.048873 | 0.006996 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.061217 | 0.052891 | 0.008325 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011300 | 0.010461 | 0.000838 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011848 | 0.010905 | 0.000942 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015883 | 0.014217 | 0.001666 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.019367 | 0.017020 | 0.002346 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.025374 | 0.022712 | 0.002662 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.037495 | 0.032703 | 0.004792 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.053916 | 0.047772 | 0.006143 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.059480 | 0.052166 | 0.007313 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/build/default/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/build/default/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
