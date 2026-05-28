# Toolchain and Dependency Reference

This document lists non-solver dependencies, tools, and datasets used by the
Docker image.

Related documents:

- [Build and run](../build-and-run.md)
- [Solver stack reference](solver-stack.md)
- [Dataset overview](../data/README.md)

## Base System

| Dependency | Version or source | Notes |
|---|---|---|
| Base image | `nvidia/cuda:12.8.1-devel-ubuntu22.04` | CUDA development image on Ubuntu 22.04 |
| CUDA toolkit | `12.8.1` from base image | Used by CUDA solver builds |
| CUDA solver architectures | `CUDA_ARCHITECTURES=86` | RTX 3090 target; passed to CUDA-enabled CMake projects |
| cuDSS pip dependencies | `CUDSS_PIP_INSTALL_DEPS=0` | Avoids pulling Python-packaged CUDA 12.x dependencies into the CUDA base image |
| Ubuntu | `22.04` from base image | apt packages resolve against this release |
| OpenMPI | Ubuntu apt package | Used by MPI-enabled solvers |
| OpenBLAS / LAPACK / ScaLAPACK | Ubuntu apt packages | Used by STRUMPACK and general numerical tooling |
| GNU compilers | Ubuntu apt packages | `gcc`, `g++`, `gfortran` through `build-essential` and `gfortran` |
| CMake | `CMAKE_PIP_SPEC=cmake==3.31.6` | Installed over the Ubuntu `3.22.1` package because current MUMPS superbuild requires CMake `>=3.25` |
| Ninja | Ubuntu apt package | CMake generator |
| oneMKL PARDISO | `intel-oneapi-mkl-classic-devel` | Installs the classic oneMKL headers and libraries without the broader SYCL devel meta-package |
| StarPU bootstrap tools | `gettext`, `libtool`, `libtool-bin` | Required by StarPU `autogen.sh`; `libtool-bin` provides the `libtool` executable |
| PaStiX ordering backend | METIS | PaStiX is configured with source-built METIS paths; SCOTCH ordering is disabled |
| PaStiX CUDA include path | `/usr/local/cuda/include` | Added explicitly for generated C targets that include `cuda_runtime.h` |
| MUMPS GPU backend | `MUMPS_gpu=ON` | The build script now requests CUDA GPU support and fails if CMake leaves `MUMPS_gpu` disabled |
| MAGMA | Not installed | Excluded because the current focus is sparse direct solvers, not MAGMA sparse iterative routines |
| OpenGL GLU | Not installed intentionally | `libglu1-mesa-dev`, `libgl1-mesa-dev`, and `libopengl0` are not explicit dependencies; `glu` in this project means sparse LU solver work, not OpenGL GLU |

oneMKL libraries are exported from `/opt/intel/oneapi/mkl/latest/lib`.
`/opt/intel/oneapi/mkl/latest/lib/intel64` is kept in runtime search paths only
as a compatibility fallback for older oneAPI layouts.

`libtool` and `libtool-bin` are intentionally both present. On Ubuntu,
`libtool` provides the Autotools support files, while `libtool-bin` provides
the `libtool` command that StarPU's bootstrap script invokes.

## Profiling and CUDA Tools

| Tool | Version or package | Command |
|---|---|---|
| Nsight Compute | `nsight-compute-2026.1.1_2026.1.1.2-1_amd64.deb` | `ncu` |
| Nsight Systems | `nsight-systems-2026.2.1_2026.2.1.210-1_amd64.deb` | `nsys` |

The `.deb` packages are downloaded from NVIDIA's Ubuntu 22.04 devtools
repository during the Docker build.

## cuDSS Wheel Extraction

| Argument | Default |
|---|---|
| `CUDSS_PIP_SPEC` | `nvidia-cudss-cu12==0.7.1.6` |
| `CUDSS_PIP_INSTALL_DEPS` | `0` |

The cuDSS wheel does not expose an importable `nvidia.cudss` Python module.
Its headers and libraries are packaged under `nvidia/cu12`, then copied to:

```text
/opt/nvidia/cudss
```

Set `CUDSS_PIP_INSTALL_DEPS=1` only if you intentionally want pip to install
NVIDIA's Python CUDA dependency wheels such as `nvidia-cublas-cu12`.

## AI Coding CLIs

| Tool | Version or spec | Command |
|---|---|---|
| Node.js | `NODE_MAJOR=22` through NodeSource | `node`, `npm` |
| Claude Code | `@anthropic-ai/claude-code` | `claude` |
| OpenAI Codex CLI | `@openai/codex` | `codex` |

The npm package specs can be pinned at build time:

```bash
docker build \
  --build-arg CLAUDE_CODE_NPM_SPEC='@anthropic-ai/claude-code@latest' \
  --build-arg CODEX_NPM_SPEC='@openai/codex@latest' \
  -t sparse-direct-solver:3090 .
```

Do not bake API keys into the image. Use runtime environment variables or each
CLI's login flow.

## Power System Packages

| Tool | Version or spec | Installed by | Dataset policy |
|---|---|---|---|
| PYPOWER | `pypower` | pip | package only |
| pandapower | `pandapower` | pip | package only |
| MATPOWER Python package | `matpower` | pip | package only |
| MATPOWER Case Frames | `matpowercaseframes` | pip | package only |
| MATPOWER case files | `MATPOWER_REF=master` | temporary git clone | copied to `/datasets/power_system/matpower`; source/docs removed |
| MATPOWER `.mat` conversions | `POWER_NR_CASES` | build script | generated under `/datasets/power_system/matpower_mat` |
| Newton-Raphson linear systems | `POWER_NR_DUMP_ITERATION=2` | build script | generated under `/datasets/power_system/nr_linear_systems` |
| Linear-system companion files | `LINEAR_SYSTEM_RANDOM_SEED=20260521` | `prepare_dataset_vectors` | generates `rhs.mtx` and `x_true.mtx` |

Default Python package build argument:

```text
POWER_PYTHON_PACKAGES=pypower,pandapower,matpower,matpowercaseframes
```

MATPOWER source is cloned only to extract power-system case files from its
`data` directory. The clone is deleted before the layer finishes.

The `matpower[octave]` extra and the system Octave package are intentionally
not installed by default. The current image policy keeps MATPOWER as a Python
package, stores MATPOWER case files under `/datasets/power_system/matpower`,
and stores generated NR datasets under `/datasets/power_system`.

The Docker build then runs `scripts/docker/generate_power_nr_datasets.sh` to
generate `.mat` conversions and Newton-Raphson Jacobian systems for
`POWER_NR_CASES`.

After SuiteSparse and MATPOWER datasets exist, the build runs
`scripts/docker/generate_linear_system_companions.sh`. SuiteSparse matrices get
fixed-seed random `x_true.mtx` and computed `rhs.mtx`; MATPOWER NR systems use
`rhs.mtx = F.mtx` and solve `J * x_true = rhs` with SuiteSparse KLU.

## Dataset Roots

| Dataset type | Image path |
|---|---|
| All project datasets | `/datasets` |
| Power-system datasets | `/datasets/power_system` |
| MATPOWER case files | `/datasets/power_system/matpower` |
| MATPOWER `.mat` conversions | `/datasets/power_system/matpower_mat` |
| Newton-Raphson linear systems | `/datasets/power_system/nr_linear_systems` |
| Benchmark matrices | `/datasets/benchmark_matrices` |
| Benchmark matrix downloads | `/datasets/benchmark_matrices/downloads` |
| Benchmark matrix extracts | `/datasets/benchmark_matrices/matrices` |
| Linear-system companion manifest | `/datasets/LINEAR_SYSTEM_COMPANIONS.txt` |

Default SuiteSparse Matrix Collection download settings are defined by
`SUITESPARSE_MATRIX_URLS` in the Dockerfile. The default is the selected
5-matrix benchmark set:

| Matrix | Purpose |
|---|---|
| `Hamm/memplus` | Memory-circuit case with explicit-zero entries |
| `Rajat/rajat27` | Circuit case with many dmperm blocks and rank deficiency |
| `Wang/wang3` | 3D semiconductor-device matrix |
| `ATandT/onetone2` | Harmonic-balance circuit matrix |
| `Rajat/rajat15` | Larger Rajat circuit matrix |

See [Benchmark matrices](../data/benchmark-matrices.md) for the exact URLs and
the extended stress candidates that are not downloaded by default.

Override `SUITESPARSE_MATRIX_URLS` with comma-separated URLs to change the
matrix set:

```bash
docker build \
  --build-arg SUITESPARSE_MATRIX_URLS='https://sparse.tamu.edu/MM/Wang/wang3.tar.gz' \
  -t sparse-direct-solver:3090 .
```

When a single matrix URL is used, the script uses the URL basename as the
downloaded archive name. When multiple URLs are used, the script names archives
with numeric prefixes to avoid collisions.

## Build Script Policy

`scripts/docker` is limited to scripts required by the Docker build:

```text
build_sparse_stack.sh
download_suitesparse_matrix.sh
install_cudss.sh
install_power_tools.sh
sparse-env.sh
```

Solver experiments and validation should live outside `scripts/docker` so the
Docker build context stays focused on image construction.
