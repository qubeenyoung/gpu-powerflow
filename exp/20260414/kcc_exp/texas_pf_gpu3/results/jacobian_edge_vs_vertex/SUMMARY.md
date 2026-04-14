# cuPF Benchmark `jacobian_edge_vs_vertex`

## Setup

- Created UTC: 2026-04-14T02:02:22.905012+00:00
- Dataset root: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps`
- Cases: case_ACTIVSg200, case_ACTIVSg500, MemphisCase2026_Mar7, case_ACTIVSg2000, Base_Florida_42GW, Texas7k_20220923, Base_Texas_66GW, Base_MIOHIN_76GW, Base_West_Interconnect_121GW, case_ACTIVSg25k, case_ACTIVSg70k, Base_Eastern_Interconnect_515GW
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
| case_ACTIVSg200 | cuda_edge | True | 0.011837 | 0.011201 | 0.000634 | n/a | 3.0 |
| case_ACTIVSg200 | cuda_vertex | True | 0.011715 | 0.011139 | 0.000576 | n/a | 3.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.013178 | 0.012182 | 0.000995 | n/a | 4.0 |
| case_ACTIVSg500 | cuda_vertex | True | 0.525304 | 0.270812 | 0.254492 | n/a | 4.0 |
| MemphisCase2026_Mar7 | cuda_edge | True | 0.038399 | 0.037362 | 0.001036 | n/a | 3.0 |
| MemphisCase2026_Mar7 | cuda_vertex | True | 0.033029 | 0.031789 | 0.001239 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.018738 | 0.016611 | 0.002127 | n/a | 4.0 |
| case_ACTIVSg2000 | cuda_vertex | True | 0.018885 | 0.016773 | 0.002112 | n/a | 4.0 |
| Base_Florida_42GW | cuda_edge | True | 0.036327 | 0.032656 | 0.003671 | n/a | 5.0 |
| Base_Florida_42GW | cuda_vertex | True | 0.033506 | 0.029842 | 0.003664 | n/a | 5.0 |
| Texas7k_20220923 | cuda_edge | True | 0.033306 | 0.030106 | 0.003199 | n/a | 4.0 |
| Texas7k_20220923 | cuda_vertex | True | 0.029139 | 0.025919 | 0.003220 | n/a | 4.0 |
| Base_Texas_66GW | cuda_edge | True | 0.032179 | 0.027672 | 0.004507 | n/a | 5.0 |
| Base_Texas_66GW | cuda_vertex | True | 0.038215 | 0.031935 | 0.006280 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_edge | True | 0.043057 | 0.036540 | 0.006517 | n/a | 5.0 |
| Base_MIOHIN_76GW | cuda_vertex | True | 0.041381 | 0.035875 | 0.005506 | n/a | 5.0 |
| Base_West_Interconnect_121GW | cuda_edge | True | 0.067236 | 0.058167 | 0.009069 | n/a | 5.6 |
| Base_West_Interconnect_121GW | cuda_vertex | True | 0.065103 | 0.056656 | 0.008446 | n/a | 5.2 |
| case_ACTIVSg25k | cuda_edge | True | 0.083793 | 0.071501 | 0.012292 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_vertex | True | 0.078724 | 0.069935 | 0.008789 | n/a | 5.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.215249 | 0.184632 | 0.030617 | n/a | 7.0 |
| case_ACTIVSg70k | cuda_vertex | True | 0.205569 | 0.170156 | 0.035413 | n/a | 7.0 |
| Base_Eastern_Interconnect_515GW | cuda_edge | True | 0.233006 | 0.197571 | 0.035434 | n/a | 6.8 |
| Base_Eastern_Interconnect_515GW | cuda_vertex | True | 0.254927 | 0.206345 | 0.048581 | n/a | 6.3 |

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
