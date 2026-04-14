# cuPF Benchmark `dump`

## Setup

- Created UTC: 2026-04-13T00:40:25.610005+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: operators
- Warmup: 1
- Repeats: 1
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: 7
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.049579 | 0.048376 | 0.001203 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.012558 | 0.011006 | 0.001552 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.018897 | 0.013054 | 0.005842 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.022696 | 0.013865 | 0.008831 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.029800 | 0.017154 | 0.012646 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.047489 | 0.021783 | 0.025706 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.077585 | 0.031899 | 0.045685 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.094318 | 0.034378 | 0.059940 | n/a | 8.0 |

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
  /workspace/exp/20260413/nd7_mt_auto/build/dump/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260413/nd7_mt_auto/build/dump/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
