# cuPF Benchmark `cuda_edge_operator_25k_plus`

## Setup

- Created UTC: 2026-04-14T02:09:40.858344+00:00
- Dataset root: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps`
- Cases: case_ACTIVSg25k, case_ACTIVSg70k, Base_Eastern_Interconnect_515GW
- Profiles: cuda_edge
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
| case_ACTIVSg25k | cuda_edge | True | 0.071206 | 0.062526 | 0.008679 | n/a | 5.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.206322 | 0.174902 | 0.031419 | n/a | 7.0 |
| Base_Eastern_Interconnect_515GW | cuda_edge | True | 0.253404 | 0.210584 | 0.042819 | n/a | 6.6 |

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
  --case-dir /workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps/case_ACTIVSg25k \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/cuPF/build/bench-operators-kcc-mt-auto-no-nd/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps/case_ACTIVSg25k \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
