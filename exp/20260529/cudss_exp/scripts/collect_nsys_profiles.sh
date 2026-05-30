#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP="$ROOT/exp/20290529/cudss_exp"
BIN="${CUDSS_RUN:-/tmp/custom_linear_solver_cudss_profile/cudss_run}"
DATA="${DATA_ROOT:-/datasets/matpower_linear_systems}"
REPEAT="${REPEAT:-5}"

mkdir -p "$EXP/results/nsys"

tail -n +2 "$EXP/cases.csv" | while IFS=, read -r case n nnz tier reason; do
  out="$EXP/results/nsys/${case}_repeat${REPEAT}"
  echo "[nsys] $case repeat=$REPEAT"
  nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --stats=false \
    --output "$out" \
    "$BIN" "$DATA/$case" --repeat "$REPEAT"
  nsys export \
    --type sqlite \
    --force-overwrite=true \
    --output "${out}.sqlite" \
    "${out}.nsys-rep"
done
