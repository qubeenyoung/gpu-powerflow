#!/usr/bin/env bash
# Build the custom solver runner (custom_linear_solver_run) via the project CMake.
# Output: <repo>/custom_linear_solver/build/custom_linear_solver_run
#
# Env overrides:
#   CLS_ARCH   CUDA architecture (default 86; must be >= 80 for TF32 mma)
#   BUILD_DIR  CMake build dir (default custom_linear_solver/build)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS_ROOT="$(cd "$HERE/../.." && pwd)"          # custom_linear_solver/
ARCH="${CLS_ARCH:-86}"
BUILD_DIR="${BUILD_DIR:-$CLS_ROOT/build}"

cmake -S "$CLS_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DCLS_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$BUILD_DIR" -j --target custom_linear_solver_run

echo "built: $BUILD_DIR/custom_linear_solver_run"
echo "run:   $BUILD_DIR/custom_linear_solver_run --matrix J.mtx --rhs F.mtx --precision tf32 --batch 64"
