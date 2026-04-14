# cuPF Benchmark `end2end_main_chain`

## Setup

- Created UTC: 2026-04-14T01:52:31.518290+00:00
- Dataset root: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps`
- Cases: case_ACTIVSg200, case_ACTIVSg500, MemphisCase2026_Mar7, case_ACTIVSg2000, Base_Florida_42GW, Texas7k_20220923, Base_Texas_66GW, Base_MIOHIN_76GW, Base_West_Interconnect_121GW, case_ACTIVSg25k, case_ACTIVSg70k, Base_Eastern_Interconnect_515GW
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
| case_ACTIVSg200 | pypower | True | 0.007837 | n/a | n/a | 0.05x | 2.0 |
| case_ACTIVSg200 | cpp_naive | True | 0.001208 | 0.000002 | 0.001206 | 0.34x | 3.0 |
| case_ACTIVSg200 | cpp | True | 0.000413 | 0.000169 | 0.000243 | 1.00x | 3.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.012005 | 0.011437 | 0.000568 | 0.03x | 3.0 |
| case_ACTIVSg500 | pypower | True | 0.013935 | n/a | n/a | 0.10x | 3.0 |
| case_ACTIVSg500 | cpp_naive | True | 0.004358 | 0.000004 | 0.004354 | 0.34x | 4.0 |
| case_ACTIVSg500 | cpp | True | 0.001461 | 0.000486 | 0.000975 | 1.00x | 4.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.014972 | 0.013917 | 0.001054 | 0.10x | 4.0 |
| MemphisCase2026_Mar7 | pypower | True | 0.015833 | n/a | n/a | 0.20x | 2.0 |
| MemphisCase2026_Mar7 | cpp_naive | True | 0.006731 | 0.000008 | 0.006723 | 0.47x | 3.0 |
| MemphisCase2026_Mar7 | cpp | True | 0.003143 | 0.001193 | 0.001949 | 1.00x | 3.0 |
| MemphisCase2026_Mar7 | cuda_edge | True | 0.015391 | 0.014389 | 0.001002 | 0.20x | 3.0 |
| case_ACTIVSg2000 | pypower | True | 0.039359 | n/a | n/a | 0.28x | 3.0 |
| case_ACTIVSg2000 | cpp_naive | True | 0.023981 | 0.000015 | 0.023965 | 0.45x | 4.0 |
| case_ACTIVSg2000 | cpp | True | 0.010829 | 0.002397 | 0.008431 | 1.00x | 4.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.026648 | 0.024422 | 0.002226 | 0.41x | 4.0 |
| Base_Florida_42GW | pypower | True | 0.130211 | n/a | n/a | 0.29x | 4.0 |
| Base_Florida_42GW | cpp_naive | True | 0.092743 | 0.000047 | 0.092696 | 0.40x | 5.0 |
| Base_Florida_42GW | cpp | True | 0.037394 | 0.008089 | 0.029305 | 1.00x | 5.0 |
| Base_Florida_42GW | cuda_edge | True | 0.038677 | 0.035093 | 0.003583 | 0.97x | 5.0 |
| Texas7k_20220923 | pypower | True | 0.118120 | n/a | n/a | 0.31x | 3.0 |
| Texas7k_20220923 | cpp_naive | True | 0.083373 | 0.000110 | 0.083262 | 0.44x | 4.0 |
| Texas7k_20220923 | cpp | True | 0.036320 | 0.008724 | 0.027596 | 1.00x | 4.0 |
| Texas7k_20220923 | cuda_edge | True | 0.062879 | 0.058441 | 0.004438 | 0.58x | 4.0 |
| Base_Texas_66GW | pypower | True | 0.171963 | n/a | n/a | 0.30x | 4.0 |
| Base_Texas_66GW | cpp_naive | True | 0.127617 | 0.000070 | 0.127546 | 0.40x | 5.0 |
| Base_Texas_66GW | cpp | True | 0.050750 | 0.010586 | 0.040164 | 1.00x | 5.0 |
| Base_Texas_66GW | cuda_edge | True | 0.038766 | 0.032531 | 0.006234 | 1.31x | 5.0 |
| Base_MIOHIN_76GW | pypower | True | 0.276390 | n/a | n/a | 0.31x | 4.0 |
| Base_MIOHIN_76GW | cpp_naive | True | 0.216450 | 0.000876 | 0.215573 | 0.39x | 5.0 |
| Base_MIOHIN_76GW | cpp | True | 0.085000 | 0.011736 | 0.073264 | 1.00x | 5.0 |
| Base_MIOHIN_76GW | cuda_edge | True | 0.053974 | 0.048504 | 0.005469 | 1.57x | 5.0 |
| Base_West_Interconnect_121GW | pypower | True | 0.491610 | n/a | n/a | 0.32x | 4.0 |
| Base_West_Interconnect_121GW | cpp_naive | True | 0.405989 | 0.001484 | 0.404505 | 0.39x | 5.0 |
| Base_West_Interconnect_121GW | cpp | True | 0.158699 | 0.032776 | 0.125923 | 1.00x | 5.0 |
| Base_West_Interconnect_121GW | cuda_edge | True | 0.098847 | 0.082879 | 0.015967 | 1.61x | 5.7 |
| case_ACTIVSg25k | pypower | True | 0.484955 | n/a | n/a | 0.29x | 4.0 |
| case_ACTIVSg25k | cpp_naive | True | 0.425859 | 0.001734 | 0.424125 | 0.34x | 5.0 |
| case_ACTIVSg25k | cpp | True | 0.142993 | 0.033558 | 0.109435 | 1.00x | 5.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.092060 | 0.079751 | 0.012308 | 1.55x | 5.0 |
| case_ACTIVSg70k | pypower | True | 2.348869 | n/a | n/a | 0.33x | 6.0 |
| case_ACTIVSg70k | cpp_naive | True | 2.472949 | 0.005136 | 2.467813 | 0.31x | 7.0 |
| case_ACTIVSg70k | cpp | True | 0.778530 | 0.098583 | 0.679946 | 1.00x | 7.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.214567 | 0.182512 | 0.032056 | 3.63x | 7.0 |
| Base_Eastern_Interconnect_515GW | pypower | True | 2.652132 | n/a | n/a | 0.37x | 5.0 |
| Base_Eastern_Interconnect_515GW | cpp_naive | True | 2.874062 | 0.005872 | 2.868189 | 0.34x | 6.0 |
| Base_Eastern_Interconnect_515GW | cpp | True | 0.979670 | 0.119497 | 0.860173 | 1.00x | 6.0 |
| Base_Eastern_Interconnect_515GW | cuda_edge | True | 0.272745 | 0.221955 | 0.050790 | 3.59x | 6.8 |

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
  --case-dir /workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps/case_ACTIVSg200 \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/cuPF/build/bench-end2end-kcc-mt-auto-no-nd/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps/case_ACTIVSg200 \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
