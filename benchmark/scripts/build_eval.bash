#!/usr/bin/env bash
#
# Build the cuPF evaluation artifacts used by the variant benchmark matrix:
#   - _cupf            pybind11 module (standard measurement path, gives the
#                      independent ‖F(V)‖ residual via python.tests)
#   - cupf_cpp_evaluate  C++ evaluator executable (optional, ScopedTimer
#                      breakdown when run with --with-cpp)
#
# Usage:
#   benchmark/scripts/build_eval.bash cpu        # CPU/KLU build  -> cuPF/build/eval-cpu
#   benchmark/scripts/build_eval.bash gpu        # CUDA/cuDSS build -> cuPF/build/eval-gpu
#   benchmark/scripts/build_eval.bash gpu-custom # CUDA + custom solver -> cuPF/build/eval-gpu-custom
#
# Options:
#   -j, --jobs N     Parallel build jobs (default: nproc)
#   --configure-only Configure only, skip the build step
#   -h, --help       Show this help
#
# Notes:
#   - The GPU builds require nvcc + cuDSS at build time and a GPU device at run
#     time. This script can BUILD them on a CPU-only host but they will not RUN
#     there; execute the matrix on a GPU host.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KIND=""
JOBS="$(nproc 2>/dev/null || echo 4)"
CONFIGURE_ONLY=0

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    cpu|gpu|gpu-custom) KIND="$1"; shift ;;
    -j|--jobs) JOBS="${2:?missing value for $1}"; shift 2 ;;
    --configure-only) CONFIGURE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${KIND}" ]]; then
  echo "error: build kind required (cpu | gpu | gpu-custom)" >&2
  usage >&2
  exit 1
fi

BUILD_DIR="${REPO_ROOT}/cuPF/build/eval-${KIND}"
CMAKE_ARGS=(
  -S "${REPO_ROOT}/cuPF"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_PYTHON_BINDINGS=ON
  -DBUILD_EVALUATORS=ON
  -DBUILD_TESTING=OFF
  -DENABLE_TIMING=ON
)
case "${KIND}" in
  cpu)        CMAKE_ARGS+=(-DWITH_CUDA=OFF) ;;
  gpu)        CMAKE_ARGS+=(-DWITH_CUDA=ON) ;;
  gpu-custom) CMAKE_ARGS+=(-DWITH_CUDA=ON -DCUPF_ENABLE_CUSTOM_SOLVER=ON) ;;
esac

echo "[build_eval] configure (${KIND}) -> ${BUILD_DIR}"
cmake "${CMAKE_ARGS[@]}"

if [[ "${CONFIGURE_ONLY}" -eq 1 ]]; then
  echo "[build_eval] configure-only complete"
  exit 0
fi

echo "[build_eval] build _cupf + cupf_cpp_evaluate (-j ${JOBS})"
cmake --build "${BUILD_DIR}" --target _cupf cupf_cpp_evaluate -j "${JOBS}"

echo "[build_eval] done. artifacts under ${BUILD_DIR}"
find "${BUILD_DIR}" -name '_cupf*.so' -o -name 'cupf_cpp_evaluate' 2>/dev/null | sed 's/^/[build_eval]   /'
