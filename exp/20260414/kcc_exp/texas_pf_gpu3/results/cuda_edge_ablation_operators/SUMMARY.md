# cuPF Benchmark `cuda_edge_ablation_operators`

## Setup

- Created UTC: 2026-04-14T01:57:25.008645+00:00
- Dataset root: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps`
- Cases: case_ACTIVSg200, case_ACTIVSg500, MemphisCase2026_Mar7, case_ACTIVSg2000, Base_Florida_42GW, Texas7k_20220923, Base_Texas_66GW, Base_MIOHIN_76GW, Base_West_Interconnect_121GW, case_ACTIVSg25k, case_ACTIVSg70k, Base_Eastern_Interconnect_515GW
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
| case_ACTIVSg200 | cuda_edge | True | 0.012050 | 0.011414 | 0.000635 | n/a | 3.0 |
| case_ACTIVSg200 | cuda_wo_cudss | True | 0.001473 | 0.000329 | 0.001143 | n/a | 3.0 |
| case_ACTIVSg200 | cuda_wo_jacobian | True | 0.012277 | 0.011321 | 0.000955 | n/a | 3.0 |
| case_ACTIVSg200 | cuda_fp64_edge | True | 0.013511 | 0.012754 | 0.000756 | n/a | 3.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.014433 | 0.013281 | 0.001151 | n/a | 4.0 |
| case_ACTIVSg500 | cuda_wo_cudss | True | 0.004680 | 0.000425 | 0.004255 | n/a | 4.0 |
| case_ACTIVSg500 | cuda_wo_jacobian | True | 0.014399 | 0.012244 | 0.002155 | n/a | 4.0 |
| case_ACTIVSg500 | cuda_fp64_edge | True | 0.013723 | 0.012337 | 0.001385 | n/a | 4.0 |
| MemphisCase2026_Mar7 | cuda_edge | True | 0.015311 | 0.014019 | 0.001292 | n/a | 3.0 |
| MemphisCase2026_Mar7 | cuda_wo_cudss | True | 0.006797 | 0.000583 | 0.006214 | n/a | 3.0 |
| MemphisCase2026_Mar7 | cuda_wo_jacobian | True | 0.021227 | 0.018617 | 0.002609 | n/a | 3.0 |
| MemphisCase2026_Mar7 | cuda_fp64_edge | True | 0.015832 | 0.014293 | 0.001539 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.021861 | 0.019516 | 0.002344 | n/a | 4.0 |
| case_ACTIVSg2000 | cuda_wo_cudss | True | 0.024263 | 0.001035 | 0.023227 | n/a | 4.0 |
| case_ACTIVSg2000 | cuda_wo_jacobian | True | 0.026630 | 0.020602 | 0.006028 | n/a | 4.0 |
| case_ACTIVSg2000 | cuda_fp64_edge | True | 0.020429 | 0.017334 | 0.003095 | n/a | 4.0 |
| Base_Florida_42GW | cuda_edge | True | 0.028253 | 0.024581 | 0.003672 | n/a | 5.0 |
| Base_Florida_42GW | cuda_wo_cudss | True | 0.088667 | 0.002303 | 0.086364 | n/a | 5.0 |
| Base_Florida_42GW | cuda_wo_jacobian | True | 0.052300 | 0.032124 | 0.020175 | n/a | 5.0 |
| Base_Florida_42GW | cuda_fp64_edge | True | 0.030046 | 0.024504 | 0.005542 | n/a | 5.0 |
| Texas7k_20220923 | cuda_edge | True | 0.029194 | 0.025999 | 0.003195 | n/a | 4.0 |
| Texas7k_20220923 | cuda_wo_cudss | True | 0.081435 | 0.002509 | 0.078925 | n/a | 4.0 |
| Texas7k_20220923 | cuda_wo_jacobian | True | 0.044188 | 0.026686 | 0.017501 | n/a | 4.0 |
| Texas7k_20220923 | cuda_fp64_edge | True | 0.033970 | 0.029115 | 0.004854 | n/a | 4.0 |
| Base_Texas_66GW | cuda_edge | True | 0.032509 | 0.027954 | 0.004554 | n/a | 5.0 |
| Base_Texas_66GW | cuda_wo_cudss | True | 0.122374 | 0.003726 | 0.118647 | n/a | 5.0 |
| Base_Texas_66GW | cuda_wo_jacobian | True | 0.054026 | 0.028965 | 0.025061 | n/a | 5.0 |
| Base_Texas_66GW | cuda_fp64_edge | True | 0.035568 | 0.028480 | 0.007088 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_edge | True | 0.042297 | 0.036788 | 0.005508 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_wo_cudss | True | 0.205162 | 0.005843 | 0.199318 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_wo_jacobian | True | 0.074597 | 0.038477 | 0.036120 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_fp64_edge | True | 0.045319 | 0.036810 | 0.008509 | n/a | 5.0 |
| Base_West_Interconnect_121GW | cuda_edge | True | 0.073352 | 0.061088 | 0.012263 | n/a | 5.7 |
| Base_West_Interconnect_121GW | cuda_wo_cudss | True | 0.525865 | 0.058541 | 0.467323 | n/a | 5.9 |
| Base_West_Interconnect_121GW | cuda_wo_jacobian | True | 0.148322 | 0.075118 | 0.073203 | n/a | 5.1 |
| Base_West_Interconnect_121GW | cuda_fp64_edge | True | 0.075791 | 0.063588 | 0.012203 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.076749 | 0.068137 | 0.008611 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_wo_cudss | True | 0.382646 | 0.011768 | 0.370878 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_wo_jacobian | True | 0.156780 | 0.079930 | 0.076849 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_fp64_edge | True | 0.082397 | 0.066670 | 0.015727 | n/a | 5.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.202184 | 0.170776 | 0.031408 | n/a | 7.0 |
| case_ACTIVSg70k | cuda_wo_cudss | True | 2.537047 | 0.240330 | 2.296717 | n/a | 7.0 |
| case_ACTIVSg70k | cuda_wo_jacobian | True | 0.591499 | 0.184070 | 0.407428 | n/a | 7.0 |
| case_ACTIVSg70k | cuda_fp64_edge | True | 0.221613 | 0.171784 | 0.049829 | n/a | 7.0 |
| Base_Eastern_Interconnect_515GW | cuda_edge | True | 0.228100 | 0.194116 | 0.033983 | n/a | 6.7 |
| Base_Eastern_Interconnect_515GW | cuda_wo_cudss | True | 2.736822 | 0.047932 | 2.688889 | n/a | 6.3 |
| Base_Eastern_Interconnect_515GW | cuda_wo_jacobian | True | 0.602132 | 0.204335 | 0.397796 | n/a | 6.1 |
| Base_Eastern_Interconnect_515GW | cuda_fp64_edge | True | 0.248563 | 0.198201 | 0.050362 | n/a | 6.0 |

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
  --case-dir /workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps/case_ACTIVSg200 \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/cuPF/build/bench-operators-kcc-mt-auto-no-nd/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps/case_ACTIVSg200 \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
