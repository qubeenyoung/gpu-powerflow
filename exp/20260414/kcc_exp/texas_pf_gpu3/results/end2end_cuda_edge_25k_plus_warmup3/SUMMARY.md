# cuPF Benchmark `end2end_cuda_edge_25k_plus_warmup3`

## Setup

- Created UTC: 2026-04-14T02:19:11.202943+00:00
- Dataset root: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps`
- Cases: case_ACTIVSg25k, case_ACTIVSg70k, Base_Eastern_Interconnect_515GW
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg25k | cuda_edge | True | 0.071433 | 0.062962 | 0.008471 | n/a | 5.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.210143 | 0.180149 | 0.029994 | n/a | 7.0 |
| Base_Eastern_Interconnect_515GW | cuda_edge | True | 0.223569 | 0.191917 | 0.031651 | n/a | 6.7 |

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
  --case-dir /workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps/case_ACTIVSg25k \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/cuPF/build/bench-end2end-kcc-mt-auto-no-nd/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps/case_ACTIVSg25k \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
