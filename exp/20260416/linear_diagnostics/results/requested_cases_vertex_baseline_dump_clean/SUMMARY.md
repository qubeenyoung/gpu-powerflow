# cuPF Benchmark `requested_cases_vertex_baseline_dump_clean`

## Setup

- Created UTC: 2026-04-16T09:57:13.432203+00:00
- Dataset root: `/workspace/gpu-powerflow/exp/20260414/amgx/cupf_dumps`
- Cases: MemphisCase2026_Mar7, Texas7k_20220923, Base_West_Interconnect_121GW, case_ACTIVSg25k, Base_Eastern_Interconnect_515GW
- Profiles: cuda_vertex
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
| MemphisCase2026_Mar7 | cuda_vertex | True | 0.293919 | 0.260997 | 0.032921 | n/a | 3.0 |
| Texas7k_20220923 | cuda_vertex | True | 0.353902 | 0.196742 | 0.157159 | n/a | 4.0 |
| Base_West_Interconnect_121GW | cuda_vertex | True | 0.852334 | 0.249023 | 0.603311 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_vertex | True | 0.923964 | 0.265034 | 0.658930 | n/a | 5.0 |
| Base_Eastern_Interconnect_515GW | cuda_vertex | True | 3.222037 | 0.492822 | 2.729215 | n/a | 6.0 |

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
  --profile cuda_mixed_vertex --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /tmp/cupf-cmake-cuda-dump-diagnostics/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/exp/20260414/amgx/cupf_dumps/MemphisCase2026_Mar7 \
  --profile cuda_mixed_vertex --warmup 1 --repeats 1
```
