# cuPF Evaluator

This evaluator uses MATPOWER case files from `/datasets/matpower` (normally
`/datasets/matpower/raw/case*.m`) and writes CSV results.

## Modes

- `prepare`: parse MATPOWER `.m` cases, build Ybus/Sbus/V0/PV/PQ, solve a
  SciPy reference Newton PF, and write cuPF dump directories.
- `python`: evaluate the pybind `NewtonSolver.initialize()` + `solve()` path.
- `torch`: evaluate the CUDA torch zero-copy forward path.
- `cpp`: prepare dumps, then run the C++ evaluator executable.

All timed modes separate `initialize_ms` and `solve_ms`. `--warmup` runs full
initialize+solve cycles before measured repeats.

## C++ Detailed Timing

The C++ evaluator can export cuPF internal `ScopedTimer` samples to
`timing.csv` when cuPF is rebuilt with timing enabled:

```bash
cmake -S cuPF -B cuPF/build/eval-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DBUILD_EVALUATORS=ON \
  -DENABLE_TIMING=ON \
  -DBUILD_TESTING=OFF
cmake --build cuPF/build/eval-cuda --target cupf_cpp_evaluate -j
```

Then run:

```bash
python3 -m evaluator.cupf_evaluate cpp \
  --dataset-root /datasets/matpower \
  --executable cuPF/build/eval-cuda/cupf_cpp_evaluate \
  --backend cuda --compute mixed \
  --warmup 1 --repeats 5
```

## Python / Torch

Python binding evaluation requires `_cupf` to be importable. Either install the
cuPF wheel or pass a build directory with `--cupf-python-path`.

```bash
python3 -m evaluator.cupf_evaluate python \
  --dataset-root /datasets/matpower \
  --cupf-python-path cuPF/build/eval-cuda \
  --backend cuda --compute mixed

python3 -m evaluator.cupf_evaluate torch \
  --dataset-root /datasets/matpower \
  --cupf-python-path cuPF/build/eval-cuda \
  --backend cuda --compute mixed
```

Outputs are written under `evaluator/results/<timestamp>/<mode>/` unless
`--output-dir` is provided.
