# cuPF Benchmark `cuda_edge_nd_nd_5`

## Setup

- Created UTC: 2026-04-13T00:06:11.805166+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 5
- cuDSS LD_PRELOAD: 

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010870 | 0.010181 | 0.000688 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011412 | 0.010578 | 0.000834 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014011 | 0.012592 | 0.001419 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.016299 | 0.014330 | 0.001969 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.022469 | 0.019580 | 0.002889 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.034669 | 0.027976 | 0.006692 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.050612 | 0.042732 | 0.007880 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.061776 | 0.049852 | 0.011924 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011000 | 0.010249 | 0.000751 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011480 | 0.010581 | 0.000898 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014095 | 0.012586 | 0.001508 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.016430 | 0.014360 | 0.002069 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.022614 | 0.019623 | 0.002990 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.034849 | 0.028061 | 0.006788 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.050752 | 0.042754 | 0.007997 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.056940 | 0.047647 | 0.009293 | n/a | 8.0 |

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
  /workspace/exp/20260413/nd_level/build/nd_5/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260413/nd_level/build/nd_5/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
