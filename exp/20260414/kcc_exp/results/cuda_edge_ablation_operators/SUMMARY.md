# cuPF Benchmark `cuda_edge_ablation_operators`

## Setup

- Created UTC: 2026-04-14T00:23:04.621760+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge, cuda_wo_cudss, cuda_wo_jacobian, cuda_fp64_edge
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
| case30_ieee | cuda_edge | True | 0.011297 | 0.010547 | 0.000749 | n/a | 5.0 |
| case30_ieee | cuda_wo_cudss | True | 0.000729 | 0.000218 | 0.000510 | n/a | 5.0 |
| case30_ieee | cuda_wo_jacobian | True | 0.011298 | 0.010430 | 0.000867 | n/a | 5.0 |
| case30_ieee | cuda_fp64_edge | True | 0.012365 | 0.011494 | 0.000870 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011672 | 0.010810 | 0.000861 | n/a | 5.0 |
| case118_ieee | cuda_wo_cudss | True | 0.001273 | 0.000255 | 0.001018 | n/a | 5.0 |
| case118_ieee | cuda_wo_jacobian | True | 0.011938 | 0.010760 | 0.001177 | n/a | 5.0 |
| case118_ieee | cuda_fp64_edge | True | 0.025970 | 0.024965 | 0.001005 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014622 | 0.013071 | 0.001550 | n/a | 6.0 |
| case793_goc | cuda_wo_cudss | True | 0.009144 | 0.000464 | 0.008679 | n/a | 5.2 |
| case793_goc | cuda_wo_jacobian | True | 0.016619 | 0.012909 | 0.003709 | n/a | 5.9 |
| case793_goc | cuda_fp64_edge | True | 0.014675 | 0.013097 | 0.001577 | n/a | 5.0 |
| case1354_pegase | cuda_edge | True | 0.044101 | 0.041946 | 0.002155 | n/a | 6.0 |
| case1354_pegase | cuda_wo_cudss | True | 0.018207 | 0.000681 | 0.017525 | n/a | 6.0 |
| case1354_pegase | cuda_wo_jacobian | True | 0.028191 | 0.021839 | 0.006351 | n/a | 6.0 |
| case1354_pegase | cuda_fp64_edge | True | 0.017437 | 0.014717 | 0.002720 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.019764 | 0.017345 | 0.002418 | n/a | 5.0 |
| case2746wop_k | cuda_wo_cudss | True | 0.032127 | 0.001045 | 0.031082 | n/a | 5.0 |
| case2746wop_k | cuda_wo_jacobian | True | 0.025716 | 0.017229 | 0.008486 | n/a | 5.0 |
| case2746wop_k | cuda_fp64_edge | True | 0.020601 | 0.017231 | 0.003370 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.031941 | 0.027702 | 0.004238 | n/a | 6.0 |
| case4601_goc | cuda_wo_cudss | True | 0.098806 | 0.001778 | 0.097028 | n/a | 6.0 |
| case4601_goc | cuda_wo_jacobian | True | 0.056321 | 0.032812 | 0.023509 | n/a | 6.0 |
| case4601_goc | cuda_fp64_edge | True | 0.030320 | 0.023801 | 0.006519 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.043224 | 0.037086 | 0.006137 | n/a | 7.0 |
| case8387_pegase | cuda_wo_cudss | True | 0.160659 | 0.004072 | 0.156586 | n/a | 7.0 |
| case8387_pegase | cuda_wo_jacobian | True | 0.078060 | 0.033615 | 0.044445 | n/a | 7.0 |
| case8387_pegase | cuda_fp64_edge | True | 0.051750 | 0.042664 | 0.009086 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.071922 | 0.064511 | 0.007410 | n/a | 8.0 |
| case9241_pegase | cuda_wo_cudss | True | 0.222786 | 0.005579 | 0.217206 | n/a | 8.0 |
| case9241_pegase | cuda_wo_jacobian | True | 0.098841 | 0.036504 | 0.062336 | n/a | 8.0 |
| case9241_pegase | cuda_fp64_edge | True | 0.047785 | 0.035443 | 0.012341 | n/a | 8.0 |

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
