#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/exp/20290529/cudss_exp"
BIN="${CUDSS_RUN:-/tmp/custom_linear_solver_cudss_profile/cudss_run}"
DATA="${DATA_ROOT:-/datasets/matpower_linear_systems}"

mkdir -p "$EXP/results/ncu"

tail -n +2 "$EXP/cases.csv" | while IFS=, read -r case n nnz tier reason; do
  for phase in factorize solve; do
    range="cudss_${phase}_0/"
    out="$EXP/results/ncu/${case}_${phase}0_basic.csv"
    echo "[ncu] $case $range"
    ncu \
      --target-processes all \
      --kernel-name-base demangled \
      --nvtx \
      --nvtx-include "$range" \
      --print-summary per-kernel \
      --set basic \
      --csv \
      --log-file "$out" \
      "$BIN" "$DATA/$case" --repeat 1
  done
done
