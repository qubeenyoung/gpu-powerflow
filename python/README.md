# Python Utilities

Python-side tooling is split by responsibility:

- `python/tests/` is the single benchmark package. It contains MATPOWER case
  loading, the pypower baseline, MATLAB/MATPOWER runner, cuPF pybind/native
  runners, orchestration, and aggregation.
- `python/prepare/` contains dataset and dump preparation tools only.
- `cuPF/python/` stays user-facing package code; benchmark evaluators do not
  live there.

## `python/tests/`

- `matpower_data.py` - MATPOWER case load, Ybus/Sbus/V0/ref/pv/pq creation,
  reference solve, residual helpers, and cuPF dump writers.
- `run_pypower.py` - pandapower.pypower/SciPy NR baseline.
- `run_matpower.py` - MATLAB MATPOWER `runpf` baseline variants.
- `run_cupf_pybind.py` - cuPF `_cupf` pybind `NewtonSolver` variants.
- `run_cupf_native.py` - native `cupf_cpp_evaluate` variants.
- `run_benchmark.py` - representative matrix orchestration.
- `aggregate_results.py` - `runs.csv` to `summary.csv` / `summary.md`.

```bash
python3 -m python.tests.run_benchmark --cases case9 --warmup 0 --repeats 1
python3 -m python.tests.run_pypower --cases case9 case14
python3 -m python.tests.aggregate_results benchmark/results/<run-name>
```

## `python/prepare/`

- `prepare.py` - parse MATPOWER cases, solve the reference PF, and write cuPF
  dump directories.
- `convert_linear_system.py` - build Newton Jacobian linear systems for the
  custom linear solver.
- `convert_m_to_mat.py` - convert MATPOWER `.m` files into `.mat` files.

```bash
python3 -m python.prepare.prepare --dataset-root /datasets/matpower --cases case9 case14
python3 -m python.prepare.convert_linear_system --dataset-root /datasets/matpower --cases case9
python3 -m python.prepare.convert_m_to_mat \
  --input-root /datasets/matpower --output-root /datasets/matpower_mat
```

## Default Benchmark Matrix

The default full run uses the 78 cases under `/datasets/matpower`, warmup 1,
and repeats 5. Repeats 10 is supported with `--repeats 10`.

| Variant | Entry | Backend | Compute | Linear solver |
|---|---|---|---|---|
| `pypower-pandapower` | pypower | cpu | fp64 | SciPy `spsolve` |
| `matpower-default` | MATLAB | cpu | fp64 | MATPOWER default |
| `matpower-lu5` | MATLAB | cpu | fp64 | `LU5` |
| `cupf-cpu-klu-pybind` | pybind | cpu | fp64 | KLU |
| `cupf-cpu-klu-native` | native C++ | cpu | fp64 | KLU |
| `cupf-fp64-cudss-pybind` | pybind | cuda | fp64 | cuDSS |
| `cupf-fp64-cudss-native` | native C++ | cuda | fp64 | cuDSS |
| `cupf-mixed-cudss-pybind` | pybind | cuda | mixed | cuDSS |
| `cupf-mixed-cudss-native` | native C++ | cuda | mixed | cuDSS |
| `cupf-fp64-custom-pybind` | pybind | cuda | fp64 | custom |
| `cupf-fp64-custom-native` | native C++ | cuda | fp64 | custom |

```bash
bash benchmark/scripts/run_benchmark.bash --build all --run-name full --warmup 1 --repeats 5
```

Each variant writes `runs.csv` and `run.json` under
`benchmark/results/<run-name>/<variant>/`. Native C++ variants also copy
`timing.csv` when the evaluator provides it.
