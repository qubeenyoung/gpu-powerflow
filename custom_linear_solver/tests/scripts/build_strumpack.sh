#!/usr/bin/env bash
# Build the STRUMPACK head-to-head bench (GPU multifrontal direct LU) in BOTH
# GPU variants:
#   strumpack_bench_magma    STRUMPACK + MAGMA vbatched path   (TPL_ENABLE_MAGMA=ON build)
#   strumpack_bench_nomagma  STRUMPACK native CUDA path        (MAGMA off, still GPU)
# Output: tests/build/strumpack_bench_{magma,nomagma}
#
# External dependency (outside the repo; not vendored): a CUDA-enabled STRUMPACK
# built twice — once with MAGMA, once without. STRUMPACK is a static lib here, so
# the link pulls its TPLs (OpenBLAS, METIS, CUDA, gfortran, [MAGMA]). The exact
# link line below mirrors STRUMPACK's own example link.txt on this machine.
#
# Env overrides (defaults match this machine):
#   STRUMPACK_SRC            headers (StrumpackSparseSolver.hpp)  [/root/baselines/STRUMPACK/src]
#   STRUMPACK_BUILD_MAGMA    libstrumpack.a + generated headers   [/root/baselines/STRUMPACK/build]
#   STRUMPACK_BUILD_NOMAGMA  libstrumpack.a + generated headers   [/root/baselines/STRUMPACK/build_nomagma]
#   MAGMA_PREFIX             MAGMA install                        [/opt/magma]
#   CUDA_HOME, CXX
#
# Run (MAGMA must be on the loader path for the magma variant):
#   LD_LIBRARY_PATH=$MAGMA_PREFIX/lib ./strumpack_bench_magma <J.mtx> [B] [repeat] [--sp_* opts]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../runners/strumpack_bench.cpp"

STRUMPACK_SRC="${STRUMPACK_SRC:-/root/baselines/STRUMPACK/src}"
STRUMPACK_BUILD_MAGMA="${STRUMPACK_BUILD_MAGMA:-/root/baselines/STRUMPACK/build}"
STRUMPACK_BUILD_NOMAGMA="${STRUMPACK_BUILD_NOMAGMA:-/root/baselines/STRUMPACK/build_nomagma}"
MAGMA_PREFIX="${MAGMA_PREFIX:-/opt/magma}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CXX="${CXX:-g++}"

# TPLs shared by both variants (from STRUMPACK's example link.txt).
common_libs=(
  /usr/lib/x86_64-linux-gnu/libpthread.a
  /usr/lib/x86_64-linux-gnu/libopenblas.so
  /usr/lib/x86_64-linux-gnu/libmetis.so
  "$CUDA_HOME/lib64/libcudart.so"
  "$CUDA_HOME/lib64/libcusolver.so"
  "$CUDA_HOME/lib64/libcusparse.so"
  "$CUDA_HOME/lib64/libcublas.so"
)
tail_libs=( -lgfortran -lquadmath -lcudadevrt -lcudart_static -lrt -lpthread -ldl )

# build_variant <build_dir> <out_name> <use_magma:0|1>
build_variant() {
  local bdir="$1" out="$HERE/$2" magma="$3"
  [ -f "$bdir/libstrumpack.a" ] || { echo "skip $2: $bdir/libstrumpack.a not found (set the build path)"; return 0; }

  local magma_inc=() magma_lib=()
  if [ "$magma" = 1 ]; then
    magma_inc=( -I"$MAGMA_PREFIX/include" )
    magma_lib=( "$MAGMA_PREFIX/lib/libmagma.so" )
  fi

  "$CXX" -O3 -std=c++17 -fopenmp "$SRC" \
    -I"$STRUMPACK_SRC" -I"$bdir" -I"$CUDA_HOME/include" "${magma_inc[@]}" \
    -L/usr/lib/gcc/x86_64-linux-gnu/11 -L"$CUDA_HOME/targets/x86_64-linux/lib" \
    -Wl,-rpath,"$CUDA_HOME/lib64:$MAGMA_PREFIX/lib" \
    "$bdir/libstrumpack.a" "${common_libs[@]}" "${magma_lib[@]}" "${tail_libs[@]}" \
    -o "$out"
  echo "built: $out"
}

build_variant "$STRUMPACK_BUILD_MAGMA"   strumpack_bench_magma   1
build_variant "$STRUMPACK_BUILD_NOMAGMA" strumpack_bench_nomagma 0

echo "run:   LD_LIBRARY_PATH=$MAGMA_PREFIX/lib $HERE/strumpack_bench_magma ../datasets/power/case_ACTIVSg25k/J.mtx 1 10 --sp_reordering_method metis"
