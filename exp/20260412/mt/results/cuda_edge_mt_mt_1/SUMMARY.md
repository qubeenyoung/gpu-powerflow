# cuPF Benchmark `cuda_edge_mt_mt_1`

## Setup

- Created UTC: 2026-04-12T09:21:04.959793+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: 1
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011003 | 0.010314 | 0.000689 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011495 | 0.010711 | 0.000784 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015465 | 0.014050 | 0.001414 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.018856 | 0.016834 | 0.002021 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.024643 | 0.022411 | 0.002231 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.036414 | 0.032267 | 0.004146 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.053811 | 0.047834 | 0.005976 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.059346 | 0.052176 | 0.007169 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011028 | 0.010277 | 0.000751 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011551 | 0.010693 | 0.000857 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.015519 | 0.014040 | 0.001478 | n/a | 5.7 |
| case1354_pegase | cuda_edge | True | 0.018985 | 0.016860 | 0.002124 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.024754 | 0.022418 | 0.002335 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.036500 | 0.032244 | 0.004256 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.053942 | 0.047793 | 0.006149 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.059490 | 0.052144 | 0.007345 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/mt/build/mt_1/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/mt/build/mt_1/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
