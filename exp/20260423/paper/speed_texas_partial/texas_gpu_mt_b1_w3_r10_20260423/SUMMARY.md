# cuPF Benchmark `texas_gpu_mt_b1_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T02:18:57.712383+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/texas_univ_cases/cuPF_datasets`
- Cases: Base_Eastern_Interconnect_515GW, Base_Florida_42GW, Base_MIOHIN_76GW, Base_Texas_66GW, Base_West_Interconnect_121GW, MemphisCase2026_Mar7, Texas7k_20220923, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 1
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
| Base_Eastern_Interconnect_515GW | cuda_edge | True | 0.174192 | 0.144342 | 0.029850 | n/a | 6.9 |
| Base_Florida_42GW | cuda_edge | True | 0.024986 | 0.021601 | 0.003384 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_edge | True | 0.036747 | 0.031608 | 0.005138 | n/a | 5.0 |
| Base_Texas_66GW | cuda_edge | True | 0.029090 | 0.024914 | 0.004175 | n/a | 5.0 |
| Base_West_Interconnect_121GW | cuda_edge | True | 0.057127 | 0.048957 | 0.008170 | n/a | 5.5 |
| MemphisCase2026_Mar7 | cuda_edge | True | 0.014567 | 0.013663 | 0.000904 | n/a | 3.0 |
| Texas7k_20220923 | cuda_edge | True | 0.026822 | 0.023799 | 0.003022 | n/a | 4.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.011585 | 0.011125 | 0.000459 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.016655 | 0.014758 | 0.001896 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.065275 | 0.057137 | 0.008137 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.011926 | 0.011117 | 0.000808 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.148641 | 0.121755 | 0.026885 | n/a | 7.0 |

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
