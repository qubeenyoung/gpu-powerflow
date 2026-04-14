# cuPF Benchmark `cuda_edge_reorder_alg2`

## Setup

- Created UTC: 2026-04-12T06:27:25.164946+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: ALG_2

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.004142 | 0.003028 | 0.001113 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.005711 | 0.003245 | 0.002465 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.018016 | 0.004038 | 0.013976 | n/a | 5.9 |
| case1354_pegase | cuda_edge | True | 0.025207 | 0.004598 | 0.020608 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.047942 | 0.006166 | 0.041775 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.098571 | 0.010034 | 0.088537 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.168247 | 0.015879 | 0.152367 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.207483 | 0.018750 | 0.188732 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.004346 | 0.003114 | 0.001231 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.005788 | 0.003225 | 0.002562 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.017823 | 0.003908 | 0.013914 | n/a | 5.9 |
| case1354_pegase | cuda_edge | True | 0.025302 | 0.004582 | 0.020720 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.048064 | 0.006207 | 0.041857 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.098540 | 0.009784 | 0.088756 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.170308 | 0.016800 | 0.153506 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.207239 | 0.018644 | 0.188594 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/build/alg2/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/build/alg2/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
