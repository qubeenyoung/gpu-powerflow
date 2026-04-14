# cuPF Benchmark `cuda_edge_mt_mt_2`

## Setup

- Created UTC: 2026-04-12T09:24:08.087779+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: 2
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.010940 | 0.010257 | 0.000682 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011458 | 0.010666 | 0.000792 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015161 | 0.013773 | 0.001388 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.017708 | 0.015700 | 0.002007 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.021608 | 0.019383 | 0.002225 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.030123 | 0.026013 | 0.004110 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.045983 | 0.040004 | 0.005979 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.049366 | 0.042277 | 0.007089 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011048 | 0.010294 | 0.000754 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011588 | 0.010728 | 0.000860 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015904 | 0.014228 | 0.001676 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.017798 | 0.015677 | 0.002120 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.021692 | 0.019361 | 0.002330 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.030275 | 0.026044 | 0.004231 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.046135 | 0.040004 | 0.006131 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.049522 | 0.042258 | 0.007264 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/mt/build/mt_2/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/mt/build/mt_2/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
