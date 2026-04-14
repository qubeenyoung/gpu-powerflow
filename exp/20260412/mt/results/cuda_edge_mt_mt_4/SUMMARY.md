# cuPF Benchmark `cuda_edge_mt_mt_4`

## Setup

- Created UTC: 2026-04-12T09:27:11.175322+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: 4
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010995 | 0.010303 | 0.000692 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011464 | 0.010675 | 0.000789 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015012 | 0.013557 | 0.001455 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.017498 | 0.015481 | 0.002017 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.020349 | 0.018101 | 0.002247 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.027422 | 0.023337 | 0.004084 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.042944 | 0.036018 | 0.006926 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.044694 | 0.037540 | 0.007153 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011039 | 0.010296 | 0.000743 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011561 | 0.010697 | 0.000863 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015466 | 0.013710 | 0.001755 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.017607 | 0.015485 | 0.002121 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.022941 | 0.019772 | 0.003168 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.029075 | 0.024840 | 0.004234 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.041508 | 0.035369 | 0.006137 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.044822 | 0.037497 | 0.007324 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/mt/build/mt_4/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/mt/build/mt_4/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
