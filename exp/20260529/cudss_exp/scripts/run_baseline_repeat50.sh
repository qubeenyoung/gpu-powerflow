#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/exp/20290529/cudss_exp"
BIN="${CUDSS_RUN:-/tmp/custom_linear_solver_lin_solver_bench/cudss_run}"
DATA="${DATA_ROOT:-/datasets/matpower_linear_systems}"
OUT="$EXP/results/baseline/cudss_repeat50.log"

mkdir -p "$EXP/results/baseline"
: > "$OUT"

tail -n +2 "$EXP/cases.csv" | while IFS=, read -r case n nnz tier reason; do
  echo "[case] $case n=$n nnz=$nnz tier=$tier" | tee -a "$OUT"
  "$BIN" "$DATA/$case" --repeat 50 | tee -a "$OUT"
  echo | tee -a "$OUT"
done
