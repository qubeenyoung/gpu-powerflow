#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/exp/20290529/cudss_exp"
BIN="${CUDSS_RUN:-/tmp/custom_linear_solver_lin_solver_bench/cudss_run}"
DATA="${DATA_ROOT:-/datasets/matpower_linear_systems}"
REPEAT="${REPEAT:-3}"

mkdir -p "$EXP/results/ncu"

tail -n +2 "$EXP/cases.csv" | while IFS=, read -r case n nnz tier reason; do
  echo "[ncu-basic] $case repeat=$REPEAT"
  ncu \
    --target-processes all \
    --kernel-name-base demangled \
    --print-summary per-kernel \
    --set basic \
    --csv \
    --log-file "$EXP/results/ncu/${case}_basic.csv" \
    "$BIN" "$DATA/$case" --repeat "$REPEAT"
done
