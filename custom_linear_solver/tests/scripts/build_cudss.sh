#!/usr/bin/env bash
# Build the cuDSS head-to-head bench (NVIDIA cuDSS GPU sparse direct LU).
# Output: tests/build/cudss_bench
#
# External dependency (outside the repo; not vendored): cuDSS.
# Env overrides (defaults match this machine):
#   CUDSS_INC  cudss.h dir            (default /usr/include/libcudss/12)
#   CUDSS_LIB  libcudss.so dir        (default /usr/lib/x86_64-linux-gnu/libcudss/12)
#   CUDA_HOME  CUDA toolkit           (default /usr/local/cuda)
#   CXX        host compiler          (default g++)
#
# Run (cuDSS is not on the default loader path):
#   LD_LIBRARY_PATH=$CUDSS_LIB ./cudss_bench <A.mtx> [repeat]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../runners/cudss_bench.cpp"
OUT="$HERE/cudss_bench"

CUDSS_INC="${CUDSS_INC:-/usr/include/libcudss/12}"
CUDSS_LIB="${CUDSS_LIB:-/usr/lib/x86_64-linux-gnu/libcudss/12}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CXX="${CXX:-g++}"

[ -f "$CUDSS_INC/cudss.h" ]   || { echo "cudss.h not found in $CUDSS_INC (set CUDSS_INC)"; exit 1; }
[ -d "$CUDSS_LIB" ]          || { echo "cuDSS lib dir not found: $CUDSS_LIB (set CUDSS_LIB)"; exit 1; }

"$CXX" -O3 -std=c++17 "$SRC" \
  -I"$CUDSS_INC" -I"$CUDA_HOME/include" \
  -L"$CUDSS_LIB" -lcudss \
  -L"$CUDA_HOME/lib64" -lcudart \
  -Wl,-rpath,"$CUDSS_LIB:$CUDA_HOME/lib64" \
  -o "$OUT"

echo "built: $OUT"
echo "run:   LD_LIBRARY_PATH=$CUDSS_LIB $OUT ../datasets/power/case_ACTIVSg25k/J.mtx 10"
