# cuPF Benchmark `texas_gpu3_mt_auto_r10`

## Setup

- Created UTC: 2026-04-16T05:11:53.421854+00:00
- Dataset root: `/workspace/datasets/texas_univ_cases/cuPF_datasets`
- Cases: case_ACTIVSg200, case_ACTIVSg500, MemphisCase2026_Mar7, case_ACTIVSg2000, Base_Florida_42GW, Texas7k_20220923, Base_Texas_66GW, Base_MIOHIN_76GW, Base_West_Interconnect_121GW, case_ACTIVSg25k, case_ACTIVSg70k, Base_Eastern_Interconnect_515GW
- Profiles: cuda_edge, cuda_edge_modified
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
| case_ACTIVSg200 | cuda_edge | True | 0.013507 | 0.012917 | 0.000589 | n/a | 3.0 |
| case_ACTIVSg200 | cuda_edge_modified | True | 0.011671 | 0.011193 | 0.000478 | n/a | 3.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.013964 | 0.012900 | 0.001064 | n/a | 4.0 |
| case_ACTIVSg500 | cuda_edge_modified | True | 0.013303 | 0.012239 | 0.001063 | n/a | 5.0 |
| MemphisCase2026_Mar7 | cuda_edge | True | 0.019454 | 0.018353 | 0.001100 | n/a | 3.0 |
| MemphisCase2026_Mar7 | cuda_edge_modified | True | 0.014798 | 0.014007 | 0.000791 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.022484 | 0.020261 | 0.002222 | n/a | 4.0 |
| case_ACTIVSg2000 | cuda_edge_modified | True | 0.029311 | 0.027181 | 0.002129 | n/a | 5.0 |
| Base_Florida_42GW | cuda_edge | True | 0.038468 | 0.033854 | 0.004614 | n/a | 5.0 |
| Base_Florida_42GW | cuda_edge_modified | True | 0.031617 | 0.027647 | 0.003969 | n/a | 6.0 |
| Texas7k_20220923 | cuda_edge | True | 0.029982 | 0.026387 | 0.003594 | n/a | 4.0 |
| Texas7k_20220923 | cuda_edge_modified | True | 0.036895 | 0.034093 | 0.002802 | n/a | 4.7 |
| Base_Texas_66GW | cuda_edge | True | 0.031944 | 0.027569 | 0.004374 | n/a | 5.0 |
| Base_Texas_66GW | cuda_edge_modified | True | 0.034224 | 0.029286 | 0.004937 | n/a | 6.0 |
| Base_MIOHIN_76GW | cuda_edge | True | 0.043292 | 0.036957 | 0.006335 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_edge_modified | True | 0.051728 | 0.045571 | 0.006157 | n/a | 6.0 |
| Base_West_Interconnect_121GW | cuda_edge | True | 0.067207 | 0.057948 | 0.009258 | n/a | 5.8 |
| Base_West_Interconnect_121GW | cuda_edge_modified | True | 0.073826 | 0.065969 | 0.007856 | n/a | 5.1 |
| case_ACTIVSg25k | cuda_edge | True | 0.077873 | 0.067200 | 0.010673 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_edge_modified | True | 0.080817 | 0.070197 | 0.010619 | n/a | 6.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.231072 | 0.186326 | 0.044745 | n/a | 7.1 |
| case_ACTIVSg70k | cuda_edge_modified | True | 0.220098 | 0.171371 | 0.048726 | n/a | 12.0 |
| Base_Eastern_Interconnect_515GW | cuda_edge | True | 0.254041 | 0.212612 | 0.041429 | n/a | 6.7 |
| Base_Eastern_Interconnect_515GW | cuda_edge_modified | True | 0.227918 | 0.198313 | 0.029604 | n/a | 7.0 |

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | cuda_edge | True | 0.012402 | 0.011761 | 0.000641 | n/a | 3.0 |
| case_ACTIVSg200 | cuda_edge_modified | True | 0.018622 | 0.018103 | 0.000518 | n/a | 3.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.013144 | 0.012142 | 0.001002 | n/a | 4.0 |
| case_ACTIVSg500 | cuda_edge_modified | True | 0.013125 | 0.012123 | 0.001001 | n/a | 5.0 |
| MemphisCase2026_Mar7 | cuda_edge | True | 0.015326 | 0.014153 | 0.001172 | n/a | 3.0 |
| MemphisCase2026_Mar7 | cuda_edge_modified | True | 0.016337 | 0.015409 | 0.000927 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.018553 | 0.016463 | 0.002089 | n/a | 4.0 |
| case_ACTIVSg2000 | cuda_edge_modified | True | 0.019762 | 0.017575 | 0.002186 | n/a | 5.0 |
| Base_Florida_42GW | cuda_edge | True | 0.029021 | 0.024817 | 0.004203 | n/a | 5.0 |
| Base_Florida_42GW | cuda_edge_modified | True | 0.047284 | 0.042632 | 0.004651 | n/a | 6.0 |
| Texas7k_20220923 | cuda_edge | True | 0.031013 | 0.026997 | 0.004016 | n/a | 4.0 |
| Texas7k_20220923 | cuda_edge_modified | True | 0.030103 | 0.026636 | 0.003467 | n/a | 4.8 |
| Base_Texas_66GW | cuda_edge | True | 0.033846 | 0.028650 | 0.005196 | n/a | 5.0 |
| Base_Texas_66GW | cuda_edge_modified | True | 0.033508 | 0.028513 | 0.004995 | n/a | 6.0 |
| Base_MIOHIN_76GW | cuda_edge | True | 0.043023 | 0.036526 | 0.006496 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_edge_modified | True | 0.053632 | 0.046974 | 0.006657 | n/a | 6.0 |
| Base_West_Interconnect_121GW | cuda_edge | True | 0.066725 | 0.057281 | 0.009444 | n/a | 5.8 |
| Base_West_Interconnect_121GW | cuda_edge_modified | True | 0.063937 | 0.057406 | 0.006531 | n/a | 5.4 |
| case_ACTIVSg25k | cuda_edge | True | 0.082561 | 0.071138 | 0.011423 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_edge_modified | True | 0.095137 | 0.080791 | 0.014346 | n/a | 6.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.218554 | 0.176950 | 0.041604 | n/a | 7.1 |
| case_ACTIVSg70k | cuda_edge_modified | True | 0.220572 | 0.171013 | 0.049559 | n/a | 12.0 |
| Base_Eastern_Interconnect_515GW | cuda_edge | True | 0.241975 | 0.201349 | 0.040626 | n/a | 6.7 |
| Base_Eastern_Interconnect_515GW | cuda_edge_modified | True | 0.228857 | 0.203245 | 0.025611 | n/a | 7.1 |

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
  /workspace/exp/20260416/modified/build/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/texas_univ_cases/cuPF_datasets/case_ACTIVSg200 \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/exp/20260416/modified/build/operators/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/datasets/texas_univ_cases/cuPF_datasets/case_ACTIVSg200 \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
