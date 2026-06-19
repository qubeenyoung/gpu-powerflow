# cuPF Python Tests

This directory is for cuPF unit and smoke tests only.

- `test_bindings.py` checks the pybind `NewtonSolver` API on a tiny CPU case.
- `test_torch_autograd.py` checks the CUDA torch/autograd path when a compatible
  build and CUDA device are available.

cuPF itself is run via `cuPF/tests/run_cupf.py`; the cross-tool baseline matrix and
shared scaffolding live in `benchmark/`:

```bash
python3 cuPF/tests/run_cupf.py --path python --backend cpu-klu --precision fp64
python3 -m benchmark.common.run_benchmark --cases case9 --warmup 0 --repeats 1
```
