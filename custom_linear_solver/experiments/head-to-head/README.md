# Head-to-head harness: custom solver vs STRUMPACK(+MAGMA)

Purpose: test whether our **approach** differs from MAGMA/STRUMPACK (novelty), not speed
(cuDSS competitive speed is established separately). See
`../../docs/05-reports/06-head-to-head-2026-06-16.md` for results.

## Build the baselines (RTX 3090 / CUDA 12.8 / Ubuntu 22.04)
```
apt-get install -y gfortran libopenmpi-dev libmetis-dev libopenblas-dev liblapack-dev

# MAGMA 2.8.0 (patch band-kernel min/max vs <cooperative_groups> for CUDA 12.8):
#   in magmablas/{s,d,c,z}gbtf2_kernels.cu add `#undef min` / `#undef max`
#   before `#include <cooperative_groups.h>`
cmake -S magma-2.8.0 -B magma-2.8.0/build -DGPU_TARGET=Ampere -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBLA_VENDOR=OpenBLAS -DUSE_FORTRAN=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build magma-2.8.0/build --target magma -j   # install libmagma.so+headers to /opt/magma

# STRUMPACK with CUDA+MAGMA (non-MPI). Patch src/sparse/fronts/FrontMAGMA.cpp:
#   replace `if (!mpi_rank())` with `if (true)` (mpi_rank undefined in non-MPI build)
cmake -S STRUMPACK -B STRUMPACK/build -DCMAKE_BUILD_TYPE=Release \
  -DSTRUMPACK_USE_MPI=OFF -DSTRUMPACK_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DTPL_ENABLE_SLATE=OFF -DTPL_ENABLE_SCOTCH=OFF -DTPL_ENABLE_PARMETIS=OFF \
  -DTPL_ENABLE_BPACK=OFF -DTPL_ENABLE_ZFP=OFF -DTPL_ENABLE_MAGMA=ON \
  -DMAGMA_INCLUDE_DIR=/opt/magma/include -DMAGMA_LIBRARIES=/opt/magma/lib/libmagma.so \
  -DTPL_METIS_INCLUDE_DIRS=/usr/include -DTPL_METIS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libmetis.so \
  -DBLA_VENDOR=OpenBLAS
cmake --build STRUMPACK/build --target strumpack -j
```

## Build & run the harness
```
g++ -O2 -std=c++17 strumpack_bench.cpp -o strumpack_bench \
  -I STRUMPACK/src -I STRUMPACK/build -I/usr/local/cuda/include \
  STRUMPACK/build/libstrumpack.a -L/opt/magma/lib -lmagma \
  -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -lcusparse \
  -lopenblas -lmetis -lgfortran -lgomp -lpthread -lm
LD_LIBRARY_PATH=/opt/magma/lib ./strumpack_bench J.mtx <B> <repeat> [--sp_verbose]
```
Reports reorder/factor/solve ms and per-system batch timing (reorder once + B*(refactor+solve)).
