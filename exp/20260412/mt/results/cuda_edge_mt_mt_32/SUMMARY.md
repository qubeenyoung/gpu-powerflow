# cuPF Benchmark `cuda_edge_mt_mt_32`

## Setup

- Created UTC: 2026-04-12T09:36:21.603960+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Profiles: cuda_edge
- Measurement modes: end2end, operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: 32
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011502 | 0.010779 | 0.000723 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.013846 | 0.012975 | 0.000870 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.014935 | 0.013521 | 0.001413 | n/a | 5.8 |
| case1354_pegase | cuda_edge | True | 0.016558 | 0.014560 | 0.001998 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.026708 | 0.024386 | 0.002322 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.036460 | 0.031832 | 0.004627 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.036538 | 0.030598 | 0.005939 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.045506 | 0.038393 | 0.007112 | n/a | 8.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case30_ieee | cuda_edge | True | 0.011784 | 0.011039 | 0.000745 | n/a | 5.0 |
| case118_ieee | cuda_edge | True | 0.011661 | 0.010797 | 0.000864 | n/a | 5.0 |
| case793_goc | cuda_edge | True | 0.019256 | 0.017677 | 0.001578 | n/a | 6.0 |
| case1354_pegase | cuda_edge | True | 0.017104 | 0.014993 | 0.002111 | n/a | 6.0 |
| case2746wop_k | cuda_edge | True | 0.020750 | 0.018312 | 0.002437 | n/a | 5.0 |
| case4601_goc | cuda_edge | True | 0.026519 | 0.022297 | 0.004222 | n/a | 6.0 |
| case8387_pegase | cuda_edge | True | 0.036829 | 0.030670 | 0.006158 | n/a | 7.0 |
| case9241_pegase | cuda_edge | True | 0.043115 | 0.035832 | 0.007282 | n/a | 8.0 |

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
  /workspace/exp/20260412_2/mt/build/mt_32/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260412_2/mt/build/mt_32/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/cuPF_benchmark_dumps/case118_ieee \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
