# cuPF Benchmark `pypower_operator_profile`

## Setup

- Created UTC: 2026-04-14T01:48:38.956272+00:00
- Dataset root: `/workspace/datasets/texas_univ_cases/pf_dataset`
- Cases: case_ACTIVSg200, case_ACTIVSg500, MemphisCase2026_Mar7, case_ACTIVSg2000, Base_Florida_42GW, Texas7k_20220923, Base_Texas_66GW, Base_MIOHIN_76GW, Base_West_Interconnect_121GW, case_ACTIVSg25k, case_ACTIVSg70k, Base_Eastern_Interconnect_515GW
- Profiles: pypower
- Measurement modes: operators
- Warmup: 1
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: 

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | pypower | True | 0.007976 | n/a | 0.005698 | n/a | 2.0 |
| case_ACTIVSg500 | pypower | True | 0.014064 | n/a | 0.011494 | n/a | 3.0 |
| MemphisCase2026_Mar7 | pypower | True | 0.015353 | n/a | 0.012135 | n/a | 2.0 |
| case_ACTIVSg2000 | pypower | True | 0.039190 | n/a | 0.034457 | n/a | 3.0 |
| Base_Florida_42GW | pypower | True | 0.129837 | n/a | 0.120214 | n/a | 4.0 |
| Texas7k_20220923 | pypower | True | 0.116866 | n/a | 0.107087 | n/a | 3.0 |
| Base_Texas_66GW | pypower | True | 0.172279 | n/a | 0.160350 | n/a | 4.0 |
| Base_MIOHIN_76GW | pypower | True | 0.274336 | n/a | 0.257837 | n/a | 4.0 |
| Base_West_Interconnect_121GW | pypower | True | 0.491434 | n/a | 0.461053 | n/a | 4.0 |
| case_ACTIVSg25k | pypower | True | 0.483208 | n/a | 0.451761 | n/a | 4.0 |
| case_ACTIVSg70k | pypower | True | 2.329371 | n/a | 2.248268 | n/a | 6.0 |
| Base_Eastern_Interconnect_515GW | pypower | True | 2.636497 | n/a | 2.522186 | n/a | 5.0 |

## Files

- `manifest.json`: run configuration and environment
- `summary.csv`: one row per measured run across all measurement modes
- `aggregates.csv`: grouped statistics by mode/case/profile
- `summary_<mode>.csv` and `aggregates_<mode>.csv`: mode-specific views
- `raw/<mode>/`: per-run timing payload

## Nsight Hints

This run did not include CUDA profiles. Build a CUDA benchmark run before using Nsight, for example:

```bash
python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --dataset-root /workspace/datasets/texas_univ_cases/pf_dataset \
  --cases case_ACTIVSg200 \
  --profiles cuda_mixed_vertex \
  --with-cuda --warmup 1 --repeats 1
```
