# cuPF Benchmark `cuda_edge_nd_nd_11`

## Setup

- Created UTC: 2026-04-12T08:38:14.033062+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 11

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011232 | 0.010518 | 0.000713 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011561 | 0.010780 | 0.000780 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015450 | 0.014060 | 0.001390 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.019053 | 0.017169 | 0.001884 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.026543 | 0.024251 | 0.002292 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.038691 | 0.034466 | 0.004224 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.057211 | 0.051460 | 0.005750 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.062701 | 0.055867 | 0.006834 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011084 | 0.010332 | 0.000752 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011671 | 0.010814 | 0.000857 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015654 | 0.014132 | 0.001521 | n/a | 5.9 |
| case1354_pegase | cuda_edge | True | 0.019127 | 0.017167 | 0.001959 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.027208 | 0.024511 | 0.002697 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.038837 | 0.034490 | 0.004346 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.057290 | 0.051374 | 0.005916 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.062982 | 0.055954 | 0.007027 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/nd_level/build/nd_11/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/nd_level/build/nd_11/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
