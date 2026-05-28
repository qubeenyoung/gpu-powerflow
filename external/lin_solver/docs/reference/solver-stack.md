# Solver Stack Reference

This document tracks the sparse direct solver stack inside the Docker image.
It focuses on solver versions, source locations, build directories, and install
prefixes.

Related documents:

- [Build and run](../build-and-run.md)
- [Toolchain and dependencies](dependencies.md)
- [Dataset overview](../data/README.md)

Version values are Docker build defaults. For git-based dependencies, the
final resolved commit is recorded during image build in:

```text
/opt/third_party/VERSIONS.txt
```

## Root Layout

| Purpose | Path |
|---|---|
| Third-party root | `/opt/third_party` |
| Source root | `/opt/third_party/src` |
| Build root | `/opt/third_party/build` |
| Install root | `/opt/third_party/install` |
| Common solver prefix | `/opt/third_party/install/common` |
| Dataset root | `/datasets` |

Most install prefixes contain `bin`, `include`, `lib` or `lib64`, and package
metadata under `share` when the upstream project provides it.

## Solver Matrix

| Solver / package | Version or ref | Install prefix | Source | Build tree |
|---|---|---|---|---|
| cuDSS | `nvidia-cudss-cu12==0.7.1.6` | `/opt/nvidia/cudss` | Extracted from pip wheel path `nvidia/cu12` | Not used |
| oneMKL PARDISO | apt package `intel-oneapi-mkl-classic-devel`, resolved at build time | `/opt/intel/oneapi/mkl/latest` | apt package | Not used |
| GKlib | `master` | `/opt/third_party/install/common` | `/opt/third_party/src/gklib` | `/opt/third_party/build/gklib` |
| METIS | `master` | `/opt/third_party/install/common` | `/opt/third_party/src/metis` | Built in source tree |
| ParMETIS | `main` | `/opt/third_party/install/common` | `/opt/third_party/src/parmetis` | Built in source tree |
| SuiteSparse | `v7.12.1` | `/opt/third_party/install/common` | `/opt/third_party/src/suitesparse` | `/opt/third_party/build/suitesparse` |
| SuiteSparse KLU | included in SuiteSparse `v7.12.1` | `/opt/third_party/install/common` | `/opt/third_party/src/suitesparse` | `/opt/third_party/build/suitesparse` |
| SuiteSparse CHOLMOD | included in SuiteSparse `v7.12.1`, CUDA disabled | `/opt/third_party/install/common` | `/opt/third_party/src/suitesparse` | `/opt/third_party/build/suitesparse` |
| SuperLU | `v7.0.0` | `/opt/third_party/install/common` | `/opt/third_party/src/superlu` | `/opt/third_party/build/superlu` |
| SuperLU_MT | `v4.0.2` | `/opt/third_party/install/common` | `/opt/third_party/src/superlu_mt` | `/opt/third_party/build/superlu_mt` |
| GLU3.0 | `master` | `/opt/third_party/install/glu3` | `/opt/third_party/src/glu3` | Built in source tree |
| StarPU | `master` | `/opt/third_party/install/common` | `/opt/third_party/src/starpu` | `/opt/third_party/build/starpu` |
| PaStiX CPU | `master`, METIS ordering enabled | `/opt/third_party/install/pastix-cpu` | `/opt/third_party/src/pastix` | `/opt/third_party/build/pastix-cpu` |
| PaStiX CUDA | `master`, StarPU + CUDA + METIS ordering enabled | `/opt/third_party/install/pastix-cuda` | `/opt/third_party/src/pastix` | `/opt/third_party/build/pastix-cuda` |
| STRUMPACK CPU | `master` | `/opt/third_party/install/strumpack-cpu` | `/opt/third_party/src/strumpack` | `/opt/third_party/build/strumpack-cpu` |
| STRUMPACK CUDA | `master`, CUDA enabled, MAGMA disabled | `/opt/third_party/install/strumpack-cuda` | `/opt/third_party/src/strumpack` | `/opt/third_party/build/strumpack-cuda` |
| PanguLU | `v5.0.0`, CUDA + METIS enabled | `/opt/third_party/install/pangulu` | `/opt/third_party/src/pangulu` | Built in source tree |
| MUMPS GPU | `mumps-superbuild` `main`, `MUMPS_gpu=ON` | `/opt/third_party/install/mumps` | `/opt/third_party/src/mumps-superbuild` | `/opt/third_party/build/mumps` |

## Build Configuration

| Setting | Default |
|---|---|
| CUDA architectures | `86` |
| C flags | `-O3 -DNDEBUG` |
| C++ flags | `-O3 -DNDEBUG` |
| Fortran flags | `-O3 -DNDEBUG` |
| CMake frontend | `cmake==3.31.6` from `CMAKE_PIP_SPEC` |
| CMake build type | `Release` |
| CMake generator | `Ninja` |

## Runtime Search Paths

The Docker image installs `/etc/profile.d/sparse-solvers.sh`. Interactive
shells get these solver paths through `PATH`, `LD_LIBRARY_PATH`,
`CMAKE_PREFIX_PATH`, `PKG_CONFIG_PATH`, and `CPATH`.

The profile script preserves variables already supplied at runtime, so
`docker run -e BENCHMARK_MATRIX_ROOT=...` or custom `THIRD_PARTY_*_PREFIX`
values are not overwritten by the default image paths.

Important exported variables:

| Variable | Value |
|---|---|
| `THIRD_PARTY_ROOT` | `/opt/third_party` |
| `THIRD_PARTY_COMMON_PREFIX` | `/opt/third_party/install/common` |
| `THIRD_PARTY_STRUMPACK_CPU_PREFIX` | `/opt/third_party/install/strumpack-cpu` |
| `THIRD_PARTY_STRUMPACK_CUDA_PREFIX` | `/opt/third_party/install/strumpack-cuda` |
| `THIRD_PARTY_GLU3_PREFIX` | `/opt/third_party/install/glu3` |
| `THIRD_PARTY_PASTIX_CPU_PREFIX` | `/opt/third_party/install/pastix-cpu` |
| `THIRD_PARTY_PASTIX_CUDA_PREFIX` | `/opt/third_party/install/pastix-cuda` |
| `THIRD_PARTY_PANGULU_PREFIX` | `/opt/third_party/install/pangulu` |
| `THIRD_PARTY_MUMPS_PREFIX` | `/opt/third_party/install/mumps` |
| `CUDSS_DIR` | `/opt/nvidia/cudss` |
| `MKLROOT` | `/opt/intel/oneapi/mkl/latest` |
| oneMKL library path | `/opt/intel/oneapi/mkl/latest/lib` with `/opt/intel/oneapi/mkl/latest/lib/intel64` kept as a compatibility fallback |

## Docker Build Scripts

The image keeps only build/install/download helpers under `/opt/docker-scripts`.
Ad-hoc solver test scripts are intentionally not copied into the Docker image.

## Notes

- METIS and ParMETIS follow their upstream `make config` build flow, so their
  temporary object files are created inside the source trees.
- cuDSS is installed from the NVIDIA Python wheel and copied into a normal C/C++
  prefix under `/opt/nvidia/cudss`.
- cuDSS pip dependencies are skipped by default because the base image already
  provides CUDA 12.8. Set `CUDSS_PIP_INSTALL_DEPS=1` if you want pip to install
  NVIDIA's Python-packaged CUDA dependencies as well.
- SuiteSparse receives explicit oneMKL BLAS/LAPACK include and library paths
  during CMake configure. This avoids relying on CMake's MKL header discovery,
  which can miss `mkl.h` in oneAPI layouts.
- SuiteSparse CUDA support is disabled intentionally. CHOLMOD CUDA is excluded
  because this project is not benchmarking CHOLMOD-style symmetric positive
  definite factorization.
- SuiteSparse binaries are intentionally limited to sparse-direct-solver
  relevant projects: AMD, BTF, CAMD, CCOLAMD, COLAMD, CHOLMOD, CXSparse, LDL,
  KLU, UMFPACK, and SPQR. The full upstream source tree is still kept under
  `/opt/third_party/src/suitesparse`.
- GLU3.0 is cloned from `https://github.com/sheldonucr/GLU_public.git` and
  built with its native Makefile. The installed command is
  `/opt/third_party/install/glu3/bin/glu3-lu_cmd`; the full upstream source
  remains under `/opt/third_party/src/glu3`.
- STRUMPACK CUDA is built with `TPL_ENABLE_MAGMA=OFF` to avoid pulling MAGMA into
  the image.
- StarPU is bootstrapped from source with `autogen.sh`; the image includes both
  `libtool` and `libtool-bin` because Ubuntu splits Autotools support files and
  the `libtool` executable into separate packages.
- PaStiX is configured with `PASTIX_ORDERING_METIS=ON` and SCOTCH ordering
  disabled. The script passes the source-built METIS include and library paths
  explicitly because PaStiX's upstream METIS discovery can be brittle with
  non-system installs.
- PaStiX CUDA receives `/usr/local/cuda/include` explicitly through C/C++ flags
  because generated C targets include `cuda_runtime.h` directly.
- PanguLU is cloned from
  `https://github.com/SuperScientificSoftwareLaboratory/PanguLU.git` and built
  with CUDA and METIS enabled through its `make.inc`. The installed command is
  `/opt/third_party/install/pangulu/bin/pangulu_example`; the full upstream
  source remains under `/opt/third_party/src/pangulu`.
- MUMPS superbuild currently requires CMake `>=3.25`, so the image installs
  `cmake==3.31.6` from Python wheels to shadow Ubuntu 22.04's `3.22.1` package.
- MUMPS is configured with `MUMPS_gpu=ON`, `CUDAToolkit_ROOT=/usr/local/cuda`,
  and `CMAKE_CUDA_ARCHITECTURES=86`. The build script fails if CMake does not
  keep `MUMPS_gpu:BOOL=ON` in `/opt/third_party/build/mumps/CMakeCache.txt`.
- The MUMPS superbuild has previously ignored some ordering-related variables;
  METIS, ParMETIS, SCOTCH, and OpenMP should be rechecked after the next image
  rebuild.
- MATPOWER is not stored here as solver source. The Docker image installs the
  Python package and keeps only MATPOWER power-system case files under
  `/datasets/power_system/matpower`.
