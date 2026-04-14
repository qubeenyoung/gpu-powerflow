# cuPF Benchmark `cuda_edge_nd_auto`

## Setup

- Created UTC: 2026-04-12T08:25:00.666617+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010933 | 0.010246 | 0.000686 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011502 | 0.010710 | 0.000791 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015473 | 0.014037 | 0.001435 | n/a | 5.9 |
| case1354_pegase | cuda_edge | True | 0.018850 | 0.016852 | 0.001998 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.024611 | 0.022368 | 0.002242 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.036466 | 0.032349 | 0.004118 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.053810 | 0.047756 | 0.006053 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.059283 | 0.052100 | 0.007183 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011029 | 0.010270 | 0.000759 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011572 | 0.010718 | 0.000854 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015501 | 0.014017 | 0.001484 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.018956 | 0.016841 | 0.002113 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.024706 | 0.022375 | 0.002329 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.036540 | 0.032333 | 0.004207 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.053961 | 0.047850 | 0.006110 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.059502 | 0.052159 | 0.007342 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/nd_level/build/auto/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/nd_level/build/auto/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
