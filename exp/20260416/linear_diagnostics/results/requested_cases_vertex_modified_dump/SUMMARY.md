# cuPF Benchmark `requested_cases_vertex_modified_dump`

## Setup

- Created UTC: 2026-04-16T10:09:39.668734+00:00
- Dataset root: `/workspace/gpu-powerflow/exp/20260414/amgx/cupf_dumps`
- Cases: MemphisCase2026_Mar7, Texas7k_20220923, Base_West_Interconnect_121GW, case_ACTIVSg25k, Base_Eastern_Interconnect_515GW
- Profiles: cuda_vertex_modified
- Measurement modes: end2end
- Warmup: 0
- Repeats: 1
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS matching: False
- cuDSS matching algorithm: DEFAULT
- cuDSS pivot epsilon: AUTO
- cuDSS LD_PRELOAD:

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| MemphisCase2026_Mar7 | cuda_vertex_modified | True | 0.291175 | 0.258048 | 0.033127 | n/a | 3.0 |
| Texas7k_20220923 | cuda_vertex_modified | True | 0.355704 | 0.198308 | 0.157396 | n/a | 4.0 |
| Base_West_Interconnect_121GW | cuda_vertex_modified | True | 0.853019 | 0.247887 | 0.605131 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_vertex_modified | True | 1.082183 | 0.269333 | 0.812850 | n/a | 6.0 |
| Base_Eastern_Interconnect_515GW | cuda_vertex_modified | True | 3.779199 | 0.493024 | 3.286175 | n/a | 7.0 |

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
  /tmp/cupf-cmake-cuda-dump-diagnostics/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/exp/20260414/amgx/cupf_dumps/MemphisCase2026_Mar7 \
  --profile cuda_mixed_vertex_modified --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /tmp/cupf-cmake-cuda-dump-diagnostics/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/exp/20260414/amgx/cupf_dumps/MemphisCase2026_Mar7 \
  --profile cuda_mixed_vertex_modified --warmup 1 --repeats 1
```
