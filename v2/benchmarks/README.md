# Benchmarks

`run_benchmarks.py` stores benchmark runs under `results/<run_name>/`.

Each run directory contains:

- `manifest.json`: benchmark configuration, build info, and experiment environment
- `README.md`: human-readable summary and reproduction hints
- `summary.csv`: one row per implementation/case/repeat
- `aggregates.csv`: grouped statistics per implementation/case
- `cases/*.json`: case metadata
- `raw/<implementation>/<case>/run_XX.json`: per-run raw payloads

The current benchmark set compares:

- `pypower`
- `cpp_pypowerlike`
- `cpp_optimized`

Typical usage:

```bash
PYTHONPATH=/workspace python3 /workspace/cuPF/benchmarks/run_benchmarks.py \
  --cases 118_ieee 1354_pegase \
  --warmup 1 \
  --repeats 3
```
