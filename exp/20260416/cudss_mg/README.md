# cuDSS MG Newton Solver Benchmark

This experiment compares the cuPF Newton solver with the normal cuDSS handle
against cuDSS multi-GPU handle creation (`cudssCreateMg`).

The cuPF code changes are intentionally small:

- `CUPF_CUDSS_ENABLE_MG=OFF`: baseline path uses `cudssCreate`.
- `CUPF_CUDSS_ENABLE_MG=ON`: MG path uses `cudssCreateMg` and applies
  `CUDSS_CONFIG_DEVICE_COUNT` / `CUDSS_CONFIG_DEVICE_INDICES`.
- Storage layout and Jacobian kernels stay unchanged. The benchmark can now
  select either the standard Newton schedule or the modified factorize-first
  schedule via profile names.

## Run

Smoke test:

```bash
python3 /workspace/exp/20260416/cudss_mg/run_cudss_mg_benchmark.py \
  --cases case30_ieee \
  --profiles cuda_edge \
  --warmup 1 \
  --repeats 1 \
  --mg-device-indices 0,1
```

Default Texas University dataset run:

```bash
python3 /workspace/exp/20260416/cudss_mg/run_cudss_mg_benchmark.py \
  --case-list /workspace/exp/20260416/cudss_mg/cases.txt \
  --profiles cuda_edge \
  --warmup 1 \
  --repeats 10 \
  --mg-device-indices 0,1
```

The default dataset root is:

```text
/workspace/datasets/texas_univ_cases/cuPF_datasets
```

For an isolated two-GPU process, combine CUDA visibility with matching local
indices:

```bash
python3 /workspace/exp/20260416/cudss_mg/run_cudss_mg_benchmark.py \
  --cuda-visible-devices 0,1 \
  --mg-device-indices 0,1
```

Modified Newton schedule profiles append `_modified`:

```bash
python3 /workspace/exp/20260416/cudss_mg/run_cudss_mg_benchmark.py \
  --case-list /workspace/exp/20260416/cudss_mg/cases.txt \
  --profiles cuda_edge cuda_edge_modified \
  --cuda-visible-devices 2,3 \
  --mg-device-indices 0,1
```

## Outputs

Results are written to:

```text
/workspace/exp/20260416/cudss_mg/results/<run-name>/
```

Important files:

- `summary.csv`: one row per measured repeat.
- `aggregates.csv`: grouped means/stdevs by mode, variant, case, profile.
- `mg_comparison.csv`: baseline vs MG top-level timing.
- `operator_comparison.csv`: baseline vs MG timing for every collected timer.
- `SUMMARY.md`: readable tables for end-to-end and operator timings.
- `manifest.json`: build/run commands and environment snapshot.

The script builds two variants for each selected measurement mode:

```text
build/baseline/<mode>  # CUPF_CUDSS_ENABLE_MG=OFF
build/mg/<mode>        # CUPF_CUDSS_ENABLE_MG=ON
```
