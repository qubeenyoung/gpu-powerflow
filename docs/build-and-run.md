# Build and Run

This guide contains the operational commands for building and running the
Docker image.

## Prerequisites

- Docker
- NVIDIA driver and NVIDIA Container Toolkit for GPU runtime tests
- RTX 3090 uses CUDA architecture `86`

GPU access is not required to build the image. It is required to run GPU solver
experiments and CUDA profiler tools.

## Build

Default build for RTX 3090. This downloads the selected 5-matrix SuiteSparse
benchmark set into `/datasets/benchmark_matrices` and generates the selected
MATPOWER Newton-Raphson linear systems under `/datasets/power_system`.

```bash
docker build --progress=plain \
  --build-arg CUDA_ARCHITECTURES=86 \
  -t sparse-direct-solver:3090 .
```

Build with a custom SuiteSparse Matrix Collection set if you intentionally want
to replace the default benchmark matrix set:

```bash
docker build --progress=plain \
  --build-arg CUDA_ARCHITECTURES=86 \
  --build-arg SUITESPARSE_MATRIX_URLS='https://sparse.tamu.edu/MM/Wang/wang3.tar.gz' \
  -t sparse-direct-solver:3090 .
```

Build with a pinned MATPOWER ref:

```bash
docker build --progress=plain \
  --build-arg CUDA_ARCHITECTURES=86 \
  --build-arg MATPOWER_REF='8.0-release' \
  -t sparse-direct-solver:3090 .
```

Build with a custom power-system NR case set:

```bash
docker build --progress=plain \
  --build-arg CUDA_ARCHITECTURES=86 \
  --build-arg POWER_NR_CASES='case30,case118,case_ACTIVSg2000,case_ACTIVSg25k,case_SyntheticUSA' \
  -t sparse-direct-solver:3090 .
```

## Run

Minimal interactive shell:

```bash
docker run --rm -it sparse-direct-solver:3090
```

GPU shell:

```bash
docker run --rm -it --gpus all sparse-direct-solver:3090
```

Recommended experiment shell with GPU access, shared memory, a stable
container name, and the current directory mounted:

```bash
docker run --rm -it \
  --name sparse-direct-solver \
  --gpus all \
  --shm-size=16g \
  -v "$PWD":/workspace/host_sparse_direct_solver \
  sparse-direct-solver:3090
```

Container workspace:

```text
/workspace/sparse_direct_solver
```

Mounted host workspace:

```text
/workspace/host_sparse_direct_solver
```

## Smoke Checks

GPU visibility:

```bash
docker run --rm --gpus all sparse-direct-solver:3090 \
  bash -lc 'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader'
```

Dataset visibility:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'find /datasets -maxdepth 3 -type f | sort | head -80'
```

MATPOWER case files:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'test -f /datasets/power_system/matpower/case118.m && echo ok'
```

Generated Newton-Raphson Jacobians:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'find /datasets/power_system/nr_linear_systems -name J.mtx | sort'
```

Standardized companion vectors:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'find /datasets \( -name rhs.mtx -o -name x_true.mtx \) -type f | sort | head -40'
```

See [Benchmarking](benchmarking.md) for the standardized `A`, `rhs.mtx`,
`x_true.mtx` contract and the `compute_error` API.

Benchmark matrices:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'find /datasets/benchmark_matrices/matrices -type f | sort'
```

Python power-system packages:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'python3 -c "import pypower, pandapower, matpower, matpowercaseframes; print(\"power packages ok\")"'
```

Representative solver libraries:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'for p in \
    /opt/nvidia/cudss/lib/libcudss.so \
    /opt/third_party/install/common/lib/libklu.so \
    /opt/third_party/install/common/lib/libcholmod.so \
    /opt/third_party/install/common/lib/libsuperlu.so \
    /opt/third_party/install/common/lib/libsuperlu_mt_OPENMP.so \
    /opt/third_party/install/pangulu/lib/libpangulu.so \
    /opt/third_party/install/pastix-cuda/lib/libpastix_kernels_cuda.so \
    /opt/third_party/install/strumpack-cuda/lib/libstrumpack.so \
    /opt/third_party/install/mumps/lib/libdmumps.so; do test -e "$p" && echo "$p"; done'
```

GLU3.0 installs a command-line binary instead of a shared solver library:

```bash
docker run --rm sparse-direct-solver:3090 \
  bash -lc 'command -v glu3-lu_cmd && command -v pangulu_example'
```

## Profiling Build

For local profiling inside the container, build the benchmark target with timer
and NVTX ranges enabled:

```bash
cmake -S /workspace/sparse_direct_solver -B /tmp/profile-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TIMER=ON \
  -DENABLE_CUDA_TIMER=ON \
  -DENABLE_NVTX=ON

cmake --build /tmp/profile-build --target benchmark -j
```

Run with Nsight Systems:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output report/nsys/benchmark \
  /tmp/profile-build/benchmark --matrix-set smoke --solver cudss-gpu
```

Run with Nsight Compute focused on cuDSS solver ranges:

```bash
ncu \
  --nvtx \
  --nvtx-include 'solve.*.cudss-gpu' \
  --target-processes all \
  /tmp/profile-build/benchmark --matrix-set smoke --solver cudss-gpu
```
