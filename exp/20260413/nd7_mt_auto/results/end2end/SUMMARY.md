# cuPF Benchmark `end2end`

## Setup

- Created UTC: 2026-04-13T00:36:26.299840+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 7
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.012367 | 0.011683 | 0.000684 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011615 | 0.010817 | 0.000798 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015098 | 0.013616 | 0.001481 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.019753 | 0.017647 | 0.002105 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.019143 | 0.016777 | 0.002366 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.027148 | 0.022033 | 0.005114 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.042801 | 0.036122 | 0.006679 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.042344 | 0.033672 | 0.008672 | n/a | 8.0 |

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
  /workspace/exp/20260413/nd7_mt_auto/build/end2end/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260413/nd7_mt_auto/build/end2end/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
