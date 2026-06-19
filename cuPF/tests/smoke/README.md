# cuPF Python Tests

This directory is for cuPF unit and smoke tests only.

- `test_bindings.py` checks the pybind `NewtonSolver` API on a tiny CPU case.
- `test_torch_autograd.py` checks the CUDA torch/autograd path when a compatible
  build and CUDA device are available.

Benchmark runners and evaluators live in `python/tests/`:

```bash
python3 -m python.tests.run_benchmark --cases case9 --warmup 0 --repeats 1
python3 -m python.tests.run_cupf_pybind --cases case9 --variants cupf-cpu-klu-pybind
python3 -m python.tests.run_cupf_native --cases case9 --variants cupf-cpu-klu-native
```
