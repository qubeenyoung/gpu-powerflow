# cuPF Benchmark `cuda_edge_nd_nd_7`

## Setup

- Created UTC: 2026-04-13T00:12:46.054086+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 7
- cuDSS LD_PRELOAD: 

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010872 | 0.010187 | 0.000685 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011435 | 0.010647 | 0.000788 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014426 | 0.012982 | 0.001445 | n/a | 5.9 |
| case1354_pegase | cuda_edge | True | 0.016946 | 0.015003 | 0.001943 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.021779 | 0.019485 | 0.002293 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.031874 | 0.026914 | 0.004959 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.046804 | 0.040128 | 0.006676 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.052823 | 0.044156 | 0.008667 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010952 | 0.010201 | 0.000751 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011491 | 0.010629 | 0.000862 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014492 | 0.012992 | 0.001500 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.017091 | 0.015034 | 0.002057 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.021941 | 0.019524 | 0.002416 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.032010 | 0.026956 | 0.005053 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.046845 | 0.040004 | 0.006841 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.053043 | 0.044196 | 0.008847 | n/a | 8.0 |

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
  /workspace/exp/20260413/nd_level/build/nd_7/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260413/nd_level/build/nd_7/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
