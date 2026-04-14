# cuPF Benchmark `jacobian_edge_vs_vertex`

## Setup

- Created UTC: 2026-04-14T00:24:11.285613+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge, cuda_vertex
- Measurement modes: operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.025224 | 0.024460 | 0.000763 | n/a | 5.0 |
| case30_ieee | cuda_vertex | True | 0.011124 | 0.010374 | 0.000749 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.012523 | 0.011641 | 0.000881 | n/a | 5.0 |
| case118_ieee | cuda_vertex | True | 0.012832 | 0.011969 | 0.000863 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.025652 | 0.024087 | 0.001564 | n/a | 6.0 |
| case793_goc | cuda_vertex | True | 0.017038 | 0.015166 | 0.001870 | n/a | 5.9 |
| case1354_pegase | cuda_edge | True | 0.018856 | 0.016511 | 0.002344 | n/a | 6.0 |
| case1354_pegase | cuda_vertex | True | 0.022012 | 0.019348 | 0.002664 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.029277 | 0.026767 | 0.002509 | n/a | 5.0 |
| case2746wop_k | cuda_vertex | True | 0.019733 | 0.017332 | 0.002401 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.030172 | 0.025943 | 0.004228 | n/a | 6.0 |
| case4601_goc | cuda_vertex | True | 0.029096 | 0.024241 | 0.004855 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.041065 | 0.034916 | 0.006148 | n/a | 7.0 |
| case8387_pegase | cuda_vertex | True | 0.038944 | 0.032811 | 0.006133 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.041473 | 0.034213 | 0.007260 | n/a | 8.0 |
| case9241_pegase | cuda_vertex | True | 0.046992 | 0.039612 | 0.007380 | n/a | 8.0 |

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
  /workspace/cuPF/build/bench-operators-kcc-mt-auto-no-nd/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/cuPF/build/bench-operators-kcc-mt-auto-no-nd/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
