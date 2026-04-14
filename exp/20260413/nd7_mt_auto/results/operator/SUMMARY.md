# cuPF Benchmark `operator`

## Setup

- Created UTC: 2026-04-13T00:38:24.008415+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 7
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011472 | 0.010627 | 0.000844 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011828 | 0.010954 | 0.000874 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.016207 | 0.014510 | 0.001697 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.016268 | 0.013982 | 0.002286 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.022307 | 0.019571 | 0.002735 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.028538 | 0.022882 | 0.005656 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.043304 | 0.035461 | 0.007842 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.044755 | 0.034596 | 0.010158 | n/a | 8.0 |

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
  /workspace/exp/20260413/nd7_mt_auto/build/operator/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260413/nd7_mt_auto/build/operator/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
