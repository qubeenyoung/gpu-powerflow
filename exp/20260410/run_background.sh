#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/exp/20260410"
SCRIPT="$ROOT/scripts/run_five_way_benchmark.py"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

RUN_NAME="${1:-selected_cases_20260410_gpu3_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
PID_FILE="$LOG_DIR/${RUN_NAME}.pid"

nohup env PYTHONPATH=/workspace python3 "$SCRIPT" \
  --run-name "$RUN_NAME" \
  --benchmark-binary /workspace/cuPF/build/bench-cuda-timing/cupf_case_benchmark \
  --warmup 1 \
  --cpu-repeats 10 \
  --gpu-repeats 10 \
  --gpu-device 3 \
  >"$LOG_FILE" 2>&1 &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"

printf 'RUN_NAME=%s\n' "$RUN_NAME"
printf 'PID=%s\n' "$PID"
printf 'LOG=%s\n' "$LOG_FILE"
printf 'PID_FILE=%s\n' "$PID_FILE"
