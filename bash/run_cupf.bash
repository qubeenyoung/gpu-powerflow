#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/workspace/v2"
DEFAULT_CASE_DIR="/workspace/v1/core/dumps/case30_ieee"

PRESET="cpu-release"
BACKEND=""
JACOBIAN="edge_based"
ALGORITHM="optimized"
CASE_DIR="${DEFAULT_CASE_DIR}"
TOLERANCE="1e-8"
MAX_ITER="15"
COMPARE=0
COMPARE_TOL="1e-10"

usage() {
  cat <<'EOF'
Usage: run_cupf.bash [options] [-- extra_args_for_cupf_case_smoke]

Options:
  --preset NAME        Build preset / build dir to use.
                       Default: cpu-release
  --cpu                Shortcut for --preset cpu-release and --backend cpu
  --cuda               Shortcut for --preset cuda-release and --backend cuda
  --cuda-timing        Shortcut for --preset cuda-timing and --backend cuda
  --backend NAME       cpu | cuda
  --jacobian NAME      edge_based | vertex_based
  --algorithm NAME     optimized | pypower_like
  --case-dir PATH      Dump case directory
  --tolerance VALUE    NR tolerance
  --max-iter N         Maximum NR iterations
  --compare            Run optimized vs pypower_like compare mode
  --compare-tol VALUE  Threshold used with --compare
  -h, --help           Show this help

Examples:
  run_cupf.bash --cpu
  run_cupf.bash --cuda --jacobian vertex_based
  run_cupf.bash --cpu --algorithm pypower_like --case-dir /workspace/v1/core/dumps/case118_ieee
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
      BACKEND="cpu"
      shift
      ;;
    --cuda)
      PRESET="cuda-release"
      BACKEND="cuda"
      shift
      ;;
    --cuda-timing)
      PRESET="cuda-timing"
      BACKEND="cuda"
      shift
      ;;
    --backend)
      BACKEND="${2:?missing value for --backend}"
      shift 2
      ;;
    --jacobian)
      JACOBIAN="${2:?missing value for --jacobian}"
      shift 2
      ;;
    --algorithm)
      ALGORITHM="${2:?missing value for --algorithm}"
      shift 2
      ;;
    --case-dir)
      CASE_DIR="${2:?missing value for --case-dir}"
      shift 2
      ;;
    --tolerance)
      TOLERANCE="${2:?missing value for --tolerance}"
      shift 2
      ;;
    --max-iter)
      MAX_ITER="${2:?missing value for --max-iter}"
      shift 2
      ;;
    --compare)
      COMPARE=1
      shift
      ;;
    --compare-tol)
      COMPARE_TOL="${2:?missing value for --compare-tol}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

EXTRA_ARGS=("$@")

if [[ -z "${BACKEND}" ]]; then
  case "${PRESET}" in
    cpu-release|bench-release|py-release)
      BACKEND="cpu"
      ;;
    cuda-release|cuda-timing)
      BACKEND="cuda"
      ;;
    *)
      BACKEND="cpu"
      ;;
  esac
fi

EXE="${ROOT_DIR}/build/${PRESET}/cupf_case_smoke"

if [[ ! -x "${EXE}" ]]; then
  echo "Executable not found: ${EXE}" >&2
  echo "Build it first, for example:" >&2
  echo "  /workspace/bash/build_cupf.bash --preset ${PRESET}" >&2
  exit 1
fi

CMD=(
  "${EXE}"
  --case-dir "${CASE_DIR}"
  --backend "${BACKEND}"
  --jacobian "${JACOBIAN}"
  --algorithm "${ALGORITHM}"
  --tolerance "${TOLERANCE}"
  --max-iter "${MAX_ITER}"
)

if [[ "${COMPARE}" -eq 1 ]]; then
  CMD+=(--compare --compare-tol "${COMPARE_TOL}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[run_cupf] preset=${PRESET} backend=${BACKEND} jacobian=${JACOBIAN} case_dir=${CASE_DIR}"
exec "${CMD[@]}"
