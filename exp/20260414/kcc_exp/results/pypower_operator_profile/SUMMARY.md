# cuPF Benchmark `pypower_operator_profile`

## Setup

- Created UTC: 2026-04-13T14:58:33.639369+00:00
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
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
| case30_ieee | pypower | True | 0.009885 | n/a | 0.007908 | n/a | 4.0 |
| case118_ieee | pypower | True | 0.011062 | n/a | 0.008927 | n/a | 4.0 |
| case793_goc | pypower | True | 0.021947 | n/a | 0.019087 | n/a | 4.0 |
| case1354_pegase | pypower | True | 0.039306 | n/a | 0.035373 | n/a | 5.0 |
| case2746wop_k | pypower | True | 0.054710 | n/a | 0.049644 | n/a | 4.0 |
| case4601_goc | pypower | True | 0.138914 | n/a | 0.130875 | n/a | 5.0 |
| case8387_pegase | pypower | True | 0.242652 | n/a | 0.227525 | n/a | 6.0 |
| case9241_pegase | pypower | True | 0.316770 | n/a | 0.300652 | n/a | 7.0 |

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
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --cases case118_ieee \
  --profiles cuda_mixed_vertex \
  --with-cuda --warmup 1 --repeats 1
```
