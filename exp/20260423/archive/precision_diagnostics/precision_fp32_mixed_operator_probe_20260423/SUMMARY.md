# cuPF Benchmark `precision_fp32_mixed_operator_probe_20260423`

## Setup

- Created UTC: 2026-04-23T03:23:20.343502+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case118, case_ACTIVSg2000, case_ACTIVSg70k
- Profiles: cuda_fp32_edge, cuda_edge
- Measurement modes: operators
- Warmup: 3
- Repeats: 5
- Batch size: 1
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS matching: False
- cuDSS matching algorithm: DEFAULT
- cuDSS pivot epsilon: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case118 | cuda_fp32_edge | False | 0.011769 | 0.010040 | 0.001729 | n/a | 10.0 |
| case118 | cuda_edge | True | 0.010652 | 0.010027 | 0.000624 | n/a | 4.0 |
| case_ACTIVSg2000 | cuda_fp32_edge | False | 0.020683 | 0.014894 | 0.005788 | n/a | 10.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.016726 | 0.014729 | 0.001996 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_fp32_edge | False | 0.165473 | 0.126160 | 0.039312 | n/a | 10.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.156367 | 0.129239 | 0.027128 | n/a | 7.0 |

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
  /workspace/gpu-powerflow-master/cuPF/build/bench-operators-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case118 \
  --profile cuda_fp32_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow-master/cuPF/build/bench-operators-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case118 \
  --profile cuda_fp32_edge --warmup 1 --repeats 1
```
