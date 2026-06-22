#!/usr/bin/env bash
# Build the GLU head-to-head bench (UC Riverside GPU sparse LU for circuit simulation).
# Output: tests/scripts/glu_bench
#
# External dependency (outside the repo; not vendored): GLU v3.0 source tree, which
# bundles NICSLU. We compile our runner against GLU's headers and link the GLU object
# files (numeric/symbolic/Timer/preprocess) plus the NICSLU static libs. GLU is built
# first via its own Makefile (make MAIN) if its objects are missing.
#
# Env overrides (defaults match this machine):
#   GLU_SRC    GLU src/ dir (numeric.cu, symbolic.cc, preprocess/, nicslu/)
#              (default /root/baselines/GLU_public/src)
#   CUDA_HOME  CUDA toolkit          (default /usr/local/cuda)
#   CXX        host compiler         (default g++)
#
# Run:
#   ./glu_bench <A.mtx> [repeat] [warmup] [-p]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../runners/glu_bench.cpp"
OUT="$HERE/glu_bench"

GLU_SRC="${GLU_SRC:-/root/baselines/GLU_public/src}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CXX="${CXX:-g++}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"

[ -d "$GLU_SRC" ] || { echo "GLU_SRC not found: $GLU_SRC (set GLU_SRC)"; exit 1; }

# Build GLU (and NICSLU) if the object files we link against are missing.
if [ ! -f "$GLU_SRC/numeric.o" ] || [ ! -f "$GLU_SRC/nicslu/lib/nicslu.a" ]; then
  echo "building GLU (make MAIN) ..."
  make -C "$GLU_SRC" MAIN
fi

GLU_INC=( -I"$GLU_SRC" -I"$GLU_SRC/../include" -I"$GLU_SRC/preprocess" -I"$GLU_SRC/nicslu/include" )
GLU_OBJS=(
  "$GLU_SRC/numeric.o"
  "$GLU_SRC/symbolic.o"
  "$GLU_SRC/Timer.o"
  "$GLU_SRC/preprocess/preprocess.o"
  "$GLU_SRC/nicslu/lib/nicslu.a"
  "$GLU_SRC/nicslu/util/nicslu_util.a"
)

# Compile our runner against GLU headers ...
"$CXX" -O3 -std=c++11 "${GLU_INC[@]}" -DGLU_DEBUG=0 -c "$SRC" -o "$HERE/glu_bench.o"

# ... then link with nvcc to pull in CUDA runtime alongside GLU's numeric.o.
"$NVCC" -O3 "$HERE/glu_bench.o" "${GLU_OBJS[@]}" \
  -lrt -lpthread -lm \
  -o "$OUT"

rm -f "$HERE/glu_bench.o"
echo "built: $OUT"
echo "run:   $OUT ../datasets/power/case_ACTIVSg25k/J.mtx 10"
