# cuPF Benchmark `texas_cpu_ref_cpu_b1_w0_r10_20260423`

## Setup

- Created UTC: 2026-04-23T02:18:31.563615+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/texas_univ_cases/cuPF_datasets`
- Cases: Base_Eastern_Interconnect_515GW, Base_Florida_42GW, Base_MIOHIN_76GW, Base_Texas_66GW, Base_West_Interconnect_121GW, MemphisCase2026_Mar7, Texas7k_20220923, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k
- Profiles: cpp_naive, cpp
- Measurement modes: end2end
- Warmup: 0
- Repeats: 10
- Batch size: 1
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS matching: False
- cuDSS matching algorithm: DEFAULT
- cuDSS pivot epsilon: AUTO
- cuDSS LD_PRELOAD: 

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpp | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | cpp_naive | True | 2.493874 | 0.006402 | 2.487471 | 0.33x | 6.0 |
| Base_Eastern_Interconnect_515GW | cpp | True | 0.811523 | 0.094091 | 0.717431 | 1.00x | 6.0 |
| Base_Florida_42GW | cpp_naive | True | 0.074989 | 0.000090 | 0.074899 | 0.43x | 5.0 |
| Base_Florida_42GW | cpp | True | 0.032077 | 0.006743 | 0.025334 | 1.00x | 5.0 |
| Base_MIOHIN_76GW | cpp_naive | True | 0.178009 | 0.000171 | 0.177837 | 0.40x | 5.0 |
| Base_MIOHIN_76GW | cpp | True | 0.071937 | 0.009817 | 0.062119 | 1.00x | 5.0 |
| Base_Texas_66GW | cpp_naive | True | 0.102499 | 0.000108 | 0.102390 | 0.42x | 5.0 |
| Base_Texas_66GW | cpp | True | 0.042755 | 0.008715 | 0.034040 | 1.00x | 5.0 |
| Base_West_Interconnect_121GW | cpp_naive | True | 0.323019 | 0.000364 | 0.322655 | 0.41x | 5.0 |
| Base_West_Interconnect_121GW | cpp | True | 0.132311 | 0.025219 | 0.107091 | 1.00x | 5.0 |
| MemphisCase2026_Mar7 | cpp_naive | True | 0.007281 | 0.000017 | 0.007263 | 0.49x | 3.0 |
| MemphisCase2026_Mar7 | cpp | True | 0.003582 | 0.001389 | 0.002192 | 1.00x | 3.0 |
| Texas7k_20220923 | cpp_naive | True | 0.069410 | 0.000101 | 0.069308 | 0.45x | 4.0 |
| Texas7k_20220923 | cpp | True | 0.031343 | 0.007567 | 0.023776 | 1.00x | 4.0 |
| case_ACTIVSg200 | cpp_naive | True | 0.001411 | 0.000004 | 0.001406 | 0.37x | 3.0 |
| case_ACTIVSg200 | cpp | True | 0.000518 | 0.000226 | 0.000292 | 1.00x | 3.0 |
| case_ACTIVSg2000 | cpp_naive | True | 0.021312 | 0.000034 | 0.021278 | 0.52x | 4.0 |
| case_ACTIVSg2000 | cpp | True | 0.011038 | 0.002550 | 0.008487 | 1.00x | 4.0 |
| case_ACTIVSg25k | cpp_naive | True | 0.404314 | 0.000502 | 0.403812 | 0.29x | 5.0 |
| case_ACTIVSg25k | cpp | True | 0.118097 | 0.026430 | 0.091666 | 1.00x | 5.0 |
| case_ACTIVSg500 | cpp_naive | True | 0.004975 | 0.000010 | 0.004965 | 0.35x | 4.0 |
| case_ACTIVSg500 | cpp | True | 0.001737 | 0.000638 | 0.001098 | 1.00x | 4.0 |
| case_ACTIVSg70k | cpp_naive | True | 1.970876 | 0.001400 | 1.969475 | 0.33x | 7.0 |
| case_ACTIVSg70k | cpp | True | 0.642067 | 0.076142 | 0.565924 | 1.00x | 7.0 |

## Files

- `manifest.json`: run configuration and environment
- `summary.csv`: one row per measured run across all measurement modes
- `aggregates.csv`: grouped statistics by mode/case/profile
- `summary_<mode>.csv` and `aggregates_<mode>.csv`: mode-specific views
- `raw/<mode>/`: per-run timing payload

## Nsight Hints

This run did not include CUDA profiles. Build a CUDA benchmark run before using Nsight, for example:

```bash
python3 /workspace/gpu-powerflow-master/cuPF/benchmarks/run_benchmarks.py \
  --dataset-root /workspace/gpu-powerflow/datasets/texas_univ_cases/cuPF_datasets \
  --cases Base_Eastern_Interconnect_515GW \
  --profiles cuda_edge \
  --with-cuda --warmup 1 --repeats 1
```
