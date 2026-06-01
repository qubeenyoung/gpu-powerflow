#!/usr/bin/env bash
#
# Thin wrapper around python.tests.run_benchmark. It optionally builds cuPF
# evaluator artifacts first, then forwards benchmark arguments unchanged.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="none"
JOBS="$(nproc 2>/dev/null || echo 4)"
PYTHON_BIN="${PYTHON:-python3}"
ARGS=()

usage() {
  cat <<'EOF'
Usage:
  benchmark/scripts/run_benchmark.bash [--build cpu|gpu|gpu-custom|all|none] [--jobs N] [benchmark args...]

Examples:
  benchmark/scripts/run_benchmark.bash --build cpu --cases case9 --warmup 0 --repeats 1
  benchmark/scripts/run_benchmark.bash --build all --run-name full --warmup 1 --repeats 5

Benchmark args are passed to:
  python3 -m python.tests.run_benchmark
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      BUILD="${2:?missing value for --build}"
      shift 2
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
      ARGS+=("$1")
      shift
      ;;
  esac
done

case "${BUILD}" in
  cpu|gpu|gpu-custom)
    bash "${REPO_ROOT}/benchmark/scripts/build_eval.bash" "${BUILD}" --jobs "${JOBS}"
    ;;
  all)
    bash "${REPO_ROOT}/benchmark/scripts/build_eval.bash" cpu --jobs "${JOBS}"
    bash "${REPO_ROOT}/benchmark/scripts/build_eval.bash" gpu --jobs "${JOBS}"
    bash "${REPO_ROOT}/benchmark/scripts/build_eval.bash" gpu-custom --jobs "${JOBS}" || true
    ;;
  none)
    ;;
  *)
    echo "Unknown --build value: ${BUILD}" >&2
    usage >&2
    exit 1
    ;;
esac

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m python.tests.run_benchmark "${ARGS[@]}"
