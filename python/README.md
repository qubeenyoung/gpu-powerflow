# Python Utilities

This directory mirrors and reorganizes the useful Python-side workflows from
`/workspace/v1/python`.

## Converters

- `converters/convert_m_to_mat.py`
  Converts MATPOWER `.m` files from `/datasets/pglib-opf` into `.mat` files in
  `/workspace/datasets/pf_dataset`.
- `converters/convert_mat_to_nr_data.py`
  Converts workspace `.mat` cases into the legacy `nr_dataset` layout:
  `Ybus.npz`, `Sbus.npy`, `V0.npy`, `pv.npy`, `pq.npy`.
- `converters/convert_mat_to_cupf_input.py`
  Converts workspace `.mat` cases into the dump format consumed directly by
  `cuPF/tests/cpp/dump_case_loader.cpp`.

## PYPOWER

- `pypower/runpf.py`
  Workspace-local `runpf` wrapper that uses the local Newton implementation.
- `pypower/newtonpf.py`
  Local Newton-Raphson implementation with structured timing hooks.
- `pypower/benchmark.py`
  Benchmarks the target cases and stores results under `/workspace/exp/pypower_benchmark`.

## Path Policy

These scripts assume a closed workspace layout.

- MATPOWER `.mat` input: `/workspace/datasets/pf_dataset`
- Legacy `nr_dataset` output: `/workspace/datasets/nr_dataset`
- cuPF dump output: `/workspace/datasets/cuPF_datasets`
- PYPOWER benchmark output: `/workspace/exp/pypower_benchmark`

Input is handled as case names such as `118_ieee` or `pglib_opf_case118_ieee`,
not arbitrary filesystem paths.

## Execution

Modules in `converters/` and `pypower/` use relative imports only.
Run them as package modules, for example:

- `python -m python.converters.convert_mat_to_nr_data --cases 118_ieee`
- `python -m python.converters.convert_mat_to_cupf_input --cases 118_ieee`
- `python -m python.pypower.runpf 118_ieee --timing`
- `python -m python.pypower.benchmark --cases 118_ieee`

## Default Benchmark Cases

- `118_ieee`
- `793_goc`
- `1354_pegase`
- `2746wop_k`
- `4601_goc`
- `8387_pegase`
- `9241_pegase`
