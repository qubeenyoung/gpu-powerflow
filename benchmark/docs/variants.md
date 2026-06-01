# Benchmark Variants

Every variant writes the same `runs.csv` schema:

`mode, variant, case_name, case_path, backend, compute, linear_solver,
entrypoint, repeat_idx, warmup, success, converged, iterations, error_message,
n_bus, ybus_nnz, n_ref, n_pv, n_pq, initialize_ms, solve_ms, total_ms,
output_mismatch`.

## Default Matrix

| Variant | Mode | Entrypoint | Backend | Compute | Linear solver | Build |
|---|---|---|---|---|---|---|
| `pypower-pandapower` | pypower | pandapower.pypower | cpu | fp64 | SciPy `spsolve` | Python only |
| `matpower-default` | matpower | MATLAB `runpf` | cpu | fp64 | default | MATLAB + MATPOWER |
| `matpower-lu5` | matpower | MATLAB `runpf` | cpu | fp64 | `LU5` | MATLAB + MATPOWER |
| `cupf-cpu-klu-pybind` | cupf | pybind | cpu | fp64 | KLU | `cpu` |
| `cupf-cpu-klu-native` | cupf | native C++ | cpu | fp64 | KLU | `cpu` |
| `cupf-fp64-cudss-pybind` | cupf | pybind | cuda | fp64 | cuDSS | `gpu` |
| `cupf-fp64-cudss-native` | cupf | native C++ | cuda | fp64 | cuDSS | `gpu` |
| `cupf-mixed-cudss-pybind` | cupf | pybind | cuda | mixed | cuDSS | `gpu` |
| `cupf-mixed-cudss-native` | cupf | native C++ | cuda | mixed | cuDSS | `gpu` |
| `cupf-fp64-custom-pybind` | cupf | pybind | cuda | fp64 | custom | `gpu-custom` |
| `cupf-fp64-custom-native` | cupf | native C++ | cuda | fp64 | custom | `gpu-custom` |

`cupf-fp32-cudss-*` is diagnostic-only and enabled with
`--include-diagnostic-fp32`.

Vertex/edge choices are not default variants because the current public runtime
API does not expose a stable switch for them.
