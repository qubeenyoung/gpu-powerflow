# Benchmark Usage

## Build Evaluation Artifacts

```bash
bash benchmark/scripts/build_eval.bash cpu
bash benchmark/scripts/build_eval.bash gpu
bash benchmark/scripts/build_eval.bash gpu-custom
```

Build outputs are written under `cuPF/build/eval-<kind>/`. The pybind runner
uses `_cupf`; the native runner uses `cupf_cpp_evaluate`.

## Run

```bash
python3 -m python.tests.run_benchmark --cases case9 --warmup 0 --repeats 1
```

Equivalent wrapper with build:

```bash
bash benchmark/scripts/run_benchmark.bash --build cpu \
  --cases case9 --warmup 0 --repeats 1 \
  --variants pypower-pandapower cupf-cpu-klu-pybind cupf-cpu-klu-native
```

Full representative run:

```bash
bash benchmark/scripts/run_benchmark.bash --build all --run-name full --warmup 1 --repeats 5
```

## Common Options

| Option | Meaning |
|---|---|
| `--dataset-root PATH` | MATPOWER case root. Defaults to `/datasets/matpower`. |
| `--cases case9 case118` | Restrict to selected case names or `.m` paths. |
| `--limit N` | Limit the resolved case list. |
| `--variants ...` | Restrict to selected full variant IDs. |
| `--warmup N` | Warmup initialize+solve cycles per case. Default `1`. |
| `--repeats N` | Measured repeats per case. Default `5`; use `10` for longer runs. |
| `--skip-matlab` | Skip MATLAB/MATPOWER variants. |
| `--skip-cupf` | Skip all cuPF variants. |
| `--no-aggregate` | Skip `summary.csv` and `summary.md` generation. |
| `--output-root PATH` | Override the default `benchmark/results` root. |

## Individual Runners

```bash
python3 -m python.tests.run_pypower --cases case9 --warmup 0 --repeats 1
python3 -m python.tests.run_matpower --cases case9 --warmup 0 --repeats 1
python3 -m python.tests.run_cupf_pybind --cases case9 --variants cupf-cpu-klu-pybind
python3 -m python.tests.run_cupf_native --cases case9 --variants cupf-cpu-klu-native
```

## Aggregate Existing Results

```bash
python3 -m python.tests.aggregate_results benchmark/results/full
```

Each variant directory with a `runs.csv` contributes to `summary.csv` and
`summary.md`; directories with `SKIPPED.txt` are listed as skipped.
