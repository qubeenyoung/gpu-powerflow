#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/workspace/cuPF"
DEFAULT_PRESET="cpu-release"
PRESET="${DEFAULT_PRESET}"
RUN_TESTS=0
CONFIGURE_ONLY=0
JOBS=""

usage() {
  cat <<'EOF'
Usage: build_cupf.bash [options]

Options:
  --preset NAME        CMake preset to use.
                       Default: cpu-release
                       Common values:
                         cpu-release
                         cuda-release
                         cuda-timing
                         py-release
                         bench-release
  --cpu                Shortcut for --preset cpu-release
  --cuda               Shortcut for --preset cuda-release
  --cuda-timing        Shortcut for --preset cuda-timing
  --py                 Shortcut for --preset py-release
  --bench              Shortcut for --preset bench-release
  --test               Run ctest after build when the preset has tests
  --configure-only     Run cmake --preset only
  -j, --jobs N         Parallel build jobs
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="${2:?missing value for --preset}"
      shift 2
      ;;
    --cpu)
      PRESET="cpu-release"
      shift
      ;;
    --cuda)
      PRESET="cuda-release"
      shift
      ;;
    --cuda-timing)
      PRESET="cuda-timing"
      shift
      ;;
    --py)
      PRESET="py-release"
      shift
      ;;
    --bench)
      PRESET="bench-release"
      shift
      ;;
    --test)
      RUN_TESTS=1
      shift
      ;;
    --configure-only)
      CONFIGURE_ONLY=1
      shift
      ;;
    -j|--jobs)
      JOBS="${2:?missing value for $1}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "cuPF source directory not found: ${ROOT_DIR}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

echo "[build_cupf] configure preset=${PRESET}"
cmake --preset "${PRESET}"

if [[ "${CONFIGURE_ONLY}" -eq 1 ]]; then
  echo "[build_cupf] configure-only complete"
  exit 0
fi

BUILD_CMD=(cmake --build --preset "${PRESET}")
if [[ -n "${JOBS}" ]]; then
  BUILD_CMD+=(--parallel "${JOBS}")
fi

echo "[build_cupf] build preset=${PRESET}"
"${BUILD_CMD[@]}"

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  case "${PRESET}" in
    cpu-release|cuda-release|cuda-timing)
      echo "[build_cupf] test preset=${PRESET}"
      ctest --preset "${PRESET}"
      ;;
    *)
      echo "[build_cupf] skip tests: preset ${PRESET} has no ctest preset"
      ;;
  esac
fi
