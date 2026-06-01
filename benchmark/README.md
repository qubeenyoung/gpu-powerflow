# Benchmark

All benchmark execution and aggregation entry points live in `python/tests/`.
`benchmark/scripts/run_benchmark.bash` is a thin build-and-forward wrapper.

## Layout

```
benchmark/
  scripts/
    build_eval.bash
    run_benchmark.bash
  results/
    <run-name>/
      <variant>/runs.csv
      <variant>/run.json
      <variant>/timing.csv
      summary.csv
      summary.md
  docs/
    usage.md
    variants.md
```

## Quick Start

```bash
bash benchmark/scripts/run_benchmark.bash --build cpu \
  --cases case9 --warmup 0 --repeats 1 \
  --variants pypower-pandapower cupf-cpu-klu-pybind cupf-cpu-klu-native
```

Full representative run:

```bash
bash benchmark/scripts/run_benchmark.bash --build all --run-name full --warmup 1 --repeats 5
```

MATLAB/MATPOWER variants are skipped with `SKIPPED.txt` if MATLAB or a license
is unavailable. GPU/custom cuPF variants are skipped the same way when the
runtime device or build artifact is unavailable.
