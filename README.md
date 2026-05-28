# sparse_direct_solver

Docker environment **and** a CUDA multifrontal sparse direct solver
(`mysolver-gpu`), built to be broadly competitive with cuDSS on power-grid
Newton-Raphson Jacobians and circuit-simulation matrices. The image installs
cuDSS, CPU/GPU sparse direct solvers, profiling tools, Python power-system
tooling, and build-time datasets so the comparison is fully reproducible.

## Project status (cy344, 2026-05-28)

vs cuDSS, warm per-NR-iteration kernel (the regime NR / repeated factorization
uses), 8-matrix benchmark, berr <= cuDSS, `gpu_test` 4/4:

| metric | result |
|---|---|
| **F (factor)** | **WIN 8/8** |
| **S (solve)** | **WIN 6/8**, 2 close (ACTIVSg25k 1.03x, onetone2 1.13x) |
| **F+S (per iter)** | **WIN 8/8** |

Same matrices, **cold single one-shot solve** (same tool, end-to-end
A+F+S): **WIN 5/8**; the three cold losses are all dominated by the analysis
(METIS nested-dissection) phase on the largest/deepest matrices.

Full per-metric tables, methodology, and the key techniques (ncu-guided
micro-wins, partitioned-inverse pivot block, size-adaptive amalgamation cap,
production plan reuse, onetone2 GPU shift-retry) are in
[docs/results-and-methodology.md](docs/results-and-methodology.md).

## Quick Start

Build for RTX 3090:

```bash
docker build --progress=plain \
  --build-arg CUDA_ARCHITECTURES=86 \
  -t sparse-direct-solver:3090 .
```

Run with GPU access, shared memory, a stable container name, and the current
project mounted into the container:

```bash
docker run --rm -it \
  --name sparse-direct-solver \
  --gpus all \
  --shm-size=16g \
  -v "$PWD":/workspace/host_sparse_direct_solver \
  sparse-direct-solver:3090
```

The image workspace is:

```text
/workspace/sparse_direct_solver
```

Build-time datasets are stored in:

```text
/datasets
```

## Installed Stack

The image includes:

- cuDSS
- oneMKL PARDISO
- METIS, ParMETIS
- SuiteSparse direct-solver components, including CHOLMOD, KLU, UMFPACK, SPQR
- SuperLU, SuperLU_MT
- GLU3.0
- MUMPS
- PaStiX CPU/CUDA
- PanguLU
- STRUMPACK CPU/CUDA
- Nsight Systems, Nsight Compute
- Claude Code, Codex CLI
- `pypower`, `pandapower`, `matpower`, `matpowercaseframes`
- MATPOWER case files
- MATPOWER-derived Newton-Raphson Jacobian linear systems
- SuiteSparse Matrix Collection benchmark matrices
- Standard `rhs.mtx` and `x_true.mtx` companion files for solver benchmarks
- C++ `compute_error` API for componentwise backward error and absolute solution error

MAGMA is intentionally excluded because this image is focused on sparse direct
solver comparison rather than sparse iterative workflows.

## Smoke Checks

List downloaded datasets:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'find /datasets -maxdepth 3 -type f | sort | head -80'
```

Check generated power-system Jacobians:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'find /datasets/power_system/nr_linear_systems -name J.mtx | sort'
```

Check standardized benchmark companion files:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'find /datasets \( -name rhs.mtx -o -name x_true.mtx \) -type f | sort | head -40'
```

Check Python power-system packages:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'python3 -c "import pypower, pandapower, matpower, matpowercaseframes; print(\"power packages ok\")"'
```

Check representative solver libraries and solver commands:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'find /opt/third_party/install /opt/nvidia/cudss -type f \( -name "libklu.so*" -o -name "libsuperlu*.so*" -o -name "libstrumpack*.so*" -o -name "libpastix*.so*" -o -name "libpangulu.so*" -o -name "libcudss.so*" \) | sort; command -v glu3-lu_cmd; command -v pangulu_example'
```

## Documentation

Start with the docs index:

- [Documentation index](docs/README.md)
- [**Results and methodology**](docs/results-and-methodology.md) — what we built, how it compares to cuDSS, methodology, key techniques
- [Build and run guide](docs/build-and-run.md)
- [Dataset overview](docs/data/README.md)
- [Benchmark matrices](docs/data/benchmark-matrices.md)
- [Power-system datasets](docs/data/power-system.md)
- [Benchmarking and error metrics](docs/benchmarking.md)
- [Solver stack reference](docs/reference/solver-stack.md)
- [Toolchain and dependency reference](docs/reference/dependencies.md)
