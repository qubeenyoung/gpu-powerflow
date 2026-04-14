# cuPF Benchmark `end2end_main_chain`

## Setup

- Created UTC: 2026-04-14T00:22:13.022597+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: pypower, cpp_naive, cpp, cuda_edge
- Measurement modes: end2end
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpp | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | pypower | True | 0.010009 | n/a | n/a | 0.01x | 4.0 |
| case30_ieee | cpp_naive | True | 0.000472 | 0.000000 | 0.000471 | 0.15x | 5.0 |
| case30_ieee | cpp | True | 0.000072 | 0.000023 | 0.000048 | 1.00x | 5.0 |
| case30_ieee | cuda_edge | True | 0.011365 | 0.010644 | 0.000721 | 0.01x | 5.0 |
| case118_ieee | pypower | True | 0.012020 | n/a | n/a | 0.02x | 4.0 |
| case118_ieee | cpp_naive | True | 0.001074 | 0.000001 | 0.001072 | 0.22x | 5.0 |
| case118_ieee | cpp | True | 0.000231 | 0.000080 | 0.000151 | 1.00x | 5.0 |
| case118_ieee | cuda_edge | True | 0.011752 | 0.010919 | 0.000832 | 0.02x | 5.0 |
| case793_goc | pypower | True | 0.021954 | n/a | n/a | 0.13x | 4.0 |
| case793_goc | cpp_naive | True | 0.009220 | 0.000006 | 0.009214 | 0.32x | 5.0 |
| case793_goc | cpp | True | 0.002947 | 0.000889 | 0.002058 | 1.00x | 5.0 |
| case793_goc | cuda_edge | True | 0.014885 | 0.013298 | 0.001587 | 0.20x | 6.0 |
| case1354_pegase | pypower | True | 0.038390 | n/a | n/a | 0.26x | 5.0 |
| case1354_pegase | cpp_naive | True | 0.019378 | 0.000025 | 0.019353 | 0.51x | 6.0 |
| case1354_pegase | cpp | True | 0.009811 | 0.003042 | 0.006768 | 1.00x | 6.0 |
| case1354_pegase | cuda_edge | True | 0.017621 | 0.015154 | 0.002465 | 0.56x | 6.0 |
| case2746wop_k | pypower | True | 0.054780 | n/a | n/a | 0.21x | 4.0 |
| case2746wop_k | cpp_naive | True | 0.033357 | 0.000019 | 0.033338 | 0.34x | 5.0 |
| case2746wop_k | cpp | True | 0.011473 | 0.003338 | 0.008134 | 1.00x | 5.0 |
| case2746wop_k | cuda_edge | True | 0.019929 | 0.017443 | 0.002485 | 0.58x | 5.0 |
| case4601_goc | pypower | True | 0.139758 | n/a | n/a | 0.30x | 5.0 |
| case4601_goc | cpp_naive | True | 0.102031 | 0.000036 | 0.101994 | 0.42x | 6.0 |
| case4601_goc | cpp | True | 0.042372 | 0.006423 | 0.035948 | 1.00x | 6.0 |
| case4601_goc | cuda_edge | True | 0.027054 | 0.022897 | 0.004158 | 1.57x | 6.0 |
| case8387_pegase | pypower | True | 0.240751 | n/a | n/a | 0.24x | 6.0 |
| case8387_pegase | cpp_naive | True | 0.177071 | 0.000091 | 0.176979 | 0.32x | 7.0 |
| case8387_pegase | cpp | True | 0.057038 | 0.012044 | 0.044993 | 1.00x | 7.0 |
| case8387_pegase | cuda_edge | True | 0.045148 | 0.038748 | 0.006399 | 1.26x | 7.0 |
| case9241_pegase | pypower | True | 0.316927 | n/a | n/a | 0.24x | 7.0 |
| case9241_pegase | cpp_naive | True | 0.241134 | 0.000423 | 0.240710 | 0.31x | 8.0 |
| case9241_pegase | cpp | True | 0.075085 | 0.013890 | 0.061195 | 1.00x | 8.0 |
| case9241_pegase | cuda_edge | True | 0.040821 | 0.033600 | 0.007220 | 1.84x | 8.0 |

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
  /workspace/cuPF/build/bench-end2end-kcc-mt-auto-no-nd/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/cuPF/build/bench-end2end-kcc-mt-auto-no-nd/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
