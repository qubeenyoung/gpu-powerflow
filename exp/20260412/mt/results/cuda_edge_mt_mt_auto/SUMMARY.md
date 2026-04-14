# cuPF Benchmark `cuda_edge_mt_mt_auto`

## Setup

- Created UTC: 2026-04-12T09:18:01.271743+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011111 | 0.010419 | 0.000691 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.028650 | 0.027744 | 0.000905 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014705 | 0.013251 | 0.001453 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.016552 | 0.014548 | 0.002003 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.019735 | 0.017451 | 0.002284 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.026906 | 0.022309 | 0.004597 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.037113 | 0.031144 | 0.005968 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.040113 | 0.032963 | 0.007149 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011625 | 0.010872 | 0.000753 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011667 | 0.010805 | 0.000861 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.023451 | 0.021896 | 0.001555 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.016773 | 0.014652 | 0.002121 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.024749 | 0.022368 | 0.002380 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.026554 | 0.022350 | 0.004203 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.040252 | 0.034164 | 0.006087 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.040121 | 0.032857 | 0.007264 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/mt/build/mt_auto/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/mt/build/mt_auto/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
