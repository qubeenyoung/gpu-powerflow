#!/usr/bin/env bash
# Build cuPF: the single library + cupf_bench + python/torch bindings + tests.
# One cmake build covers all three run_cupf.py paths (cpp / python / torch);
# cuPF selects the backend (cudss/custom/cpu) at runtime, so there is no
# per-backend build split (unlike custom_linear_solver's cudss/strumpack scripts).
#
#   cuPF/tests/scripts/build_cupf.sh            # build into cuPF/build
#   BUILD_DIR=/tmp/cupf cuPF/tests/scripts/build_cupf.sh
#
# run_cupf.py --build wraps this same configuration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root
BUILD="${BUILD_DIR:-${ROOT}/cuPF/build}"

# torch is needed for the bindings + torch path; locate its cmake package.
TORCH_CMAKE="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null || true)"

cmake -S "${ROOT}/cuPF" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DENABLE_TIMING=ON \
  -DCUPF_ENABLE_CUSTOM_SOLVER=ON \
  -DBUILD_EVALUATORS=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DCUPF_WITH_TORCH=ON \
  ${TORCH_CMAKE:+-DCMAKE_PREFIX_PATH="${TORCH_CMAKE}"}

cmake --build "${BUILD}" -j
echo "cuPF built into ${BUILD}"
