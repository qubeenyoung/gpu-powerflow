#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(cd -- "$EXP_DIR/../../.." && pwd)

BUILD_DIR=${BUILD_DIR:-"$EXP_DIR/build"}
RESULTS_DIR=${RESULTS_DIR:-"$EXP_DIR/results"}
DATA_ROOT=${DATA_ROOT:-"datasets/matpower8.1/cupf_all_dumps"}
MODE=${MODE:-both}
ITERS=${ITERS:-100}
WARMUP=${WARMUP:-10}
NCU_ITERS=${NCU_ITERS:-1}
NCU_WARMUP=${NCU_WARMUP:-0}
NCU_BIN=${NCU_BIN:-ncu}
NCU_METRICS=${NCU_METRICS:-"gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed,smsp__thread_inst_executed.sum,smsp__inst_executed.sum,launch__thread_count,launch__registers_per_thread,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_red,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum,l1tex__t_requests_pipe_lsu_mem_global_op_red.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld,smsp__sass_average_data_bytes_per_sector_mem_global_op_st,smsp__sass_average_data_bytes_per_sector_mem_global_op_atom,smsp__sass_average_data_bytes_per_sector_mem_global_op_red"}

if [[ "$DATA_ROOT" = /* ]]; then
  DATA_ARG=$DATA_ROOT
else
  DATA_ARG="$WORKSPACE_ROOT/$DATA_ROOT"
fi

mkdir -p "$BUILD_DIR" "$RESULTS_DIR"

cmake -S "$EXP_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"

TIMING_CSV="$RESULTS_DIR/timing_matpower_all.csv"
NCU_BASENAME="$RESULTS_DIR/ncu_matpower_all_lane_occ"
NCU_CSV="$NCU_BASENAME.csv"
NCU_TXT="$NCU_BASENAME.txt"
NCU_LOG="$NCU_BASENAME.stderr"

"$BUILD_DIR/jac_asm_bench" \
  --data "$DATA_ARG" \
  --mode "$MODE" \
  --iters "$ITERS" \
  --warmup "$WARMUP" \
  > "$TIMING_CSV"

"$NCU_BIN" \
  --target-processes all \
  --metrics "$NCU_METRICS" \
  --csv \
  --page details \
  --force-overwrite \
  --export "$NCU_BASENAME" \
  "$BUILD_DIR/jac_asm_bench" \
    --data "$DATA_ARG" \
    --mode "$MODE" \
    --iters "$NCU_ITERS" \
    --warmup "$NCU_WARMUP" \
  > "$NCU_CSV" \
  2> "$NCU_LOG"

"$NCU_BIN" --import "$NCU_BASENAME.ncu-rep" --page details > "$NCU_TXT"

python3 "$SCRIPT_DIR/summarize_ncu_lane.py" \
  --timing "$TIMING_CSV" \
  --ncu "$NCU_CSV" \
  --out-csv "$RESULTS_DIR/ncu_matpower_all_lane_occ_summary.csv" \
  --out-md "$RESULTS_DIR/jac_asm_matpower_summary.md" \
  --data-root "$DATA_ROOT" \
  --mode "$MODE" \
  --timing-iters "$ITERS" \
  --timing-warmup "$WARMUP" \
  --ncu-iters "$NCU_ITERS" \
  --ncu-warmup "$NCU_WARMUP"

echo "Wrote $RESULTS_DIR/jac_asm_matpower_summary.md"
