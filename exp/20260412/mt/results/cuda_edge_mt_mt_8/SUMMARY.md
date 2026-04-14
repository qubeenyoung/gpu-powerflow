# cuPF Benchmark `cuda_edge_mt_mt_8`

## Setup

- Created UTC: 2026-04-12T09:30:13.448220+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: 8
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011017 | 0.010334 | 0.000683 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011483 | 0.010689 | 0.000793 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014724 | 0.013315 | 0.001408 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.017045 | 0.015010 | 0.002034 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.020272 | 0.018028 | 0.002243 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.027584 | 0.023476 | 0.004108 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.038549 | 0.032578 | 0.005970 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.042080 | 0.035011 | 0.007069 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011045 | 0.010301 | 0.000744 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011517 | 0.010666 | 0.000851 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014848 | 0.013345 | 0.001503 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.017125 | 0.015020 | 0.002104 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.020241 | 0.017917 | 0.002323 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.027410 | 0.023201 | 0.004209 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.038802 | 0.032690 | 0.006111 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.042312 | 0.034977 | 0.007335 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/mt/build/mt_8/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/mt/build/mt_8/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
