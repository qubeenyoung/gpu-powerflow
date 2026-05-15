# cuPF Benchmark `texas_gpu_mt_b4_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T02:19:26.479941+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/texas_univ_cases/cuPF_datasets`
- Cases: Base_Eastern_Interconnect_515GW, Base_Florida_42GW, Base_MIOHIN_76GW, Base_Texas_66GW, Base_West_Interconnect_121GW, MemphisCase2026_Mar7, Texas7k_20220923, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 4
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS matching: False
- cuDSS matching algorithm: DEFAULT
- cuDSS pivot epsilon: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | cuda_edge | True | 0.358380 | 0.141233 | 0.217147 | n/a | 7.0 |
| Base_Florida_42GW | cuda_edge | True | 0.048627 | 0.022633 | 0.025993 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_edge | True | 0.071077 | 0.031517 | 0.039559 | n/a | 5.0 |
| Base_Texas_66GW | cuda_edge | True | 0.055671 | 0.025007 | 0.030663 | n/a | 5.0 |
| Base_West_Interconnect_121GW | cuda_edge | True | 0.116952 | 0.048373 | 0.068578 | n/a | 5.7 |
| MemphisCase2026_Mar7 | cuda_edge | True | 0.025135 | 0.012605 | 0.012530 | n/a | 3.0 |
| Texas7k_20220923 | cuda_edge | True | 0.050257 | 0.023368 | 0.026888 | n/a | 4.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.019958 | 0.010333 | 0.009625 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.031413 | 0.015341 | 0.016071 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.128038 | 0.059035 | 0.069003 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.021975 | 0.011156 | 0.010819 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.310972 | 0.126382 | 0.184590 | n/a | 7.0 |

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
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/texas_univ_cases/cuPF_datasets/Base_Eastern_Interconnect_515GW \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/texas_univ_cases/cuPF_datasets/Base_Eastern_Interconnect_515GW \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
