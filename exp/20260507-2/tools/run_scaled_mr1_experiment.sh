#!/usr/bin/env bash
set -euo pipefail

CASES="case2383wp,case3120sp,case9241pegase,case13659pegase,case6468rte"
COMMON_ARGS=(
    --case "${CASES}"
    --solver hybrid
    --cudss-bootstrap-iters 1
    --force-gmres-min-steps 0
    --cudss-polish-threshold 1e-4
    --accept-mismatch-ratio 0.9
    --reject-mismatch-ratio 1.05
    --fallback-policy immediate
    --gmres-fixed-iter-mode true
    --gmres-restart 1
    --gmres-max-iters 1
    --middle-solver mr1_block_jacobi_coarse
    --preconditioner metis_block_jacobi_coarse
    --block-size 64
    --block-jacobi-precision fp32
    --block-jacobi-apply inverse_gemv
    --coarse-vars-per-block 1
    --coarse-refresh bootstrap_only
    --coarse-precision fp32
    --coarse-diag-shift-scale 1e-6
    --shadow-dx-diagnostic true
)

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --enable-scaled-mr1-step false \
    --output results/scaled_mr1_summary_baseline.csv \
    --iter-output results/scaled_mr1_iters_baseline.csv \
    --shadow-output results/scaled_mr1_shadow_baseline.csv \
    --timing-output results/scaled_mr1_timing_baseline.csv

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --enable-scaled-mr1-step true \
    --scaled-mr1-gammas 4.0,2.0,1.0 \
    --output results/scaled_mr1_summary_scaled.csv \
    --iter-output results/scaled_mr1_iters_scaled.csv \
    --shadow-output results/scaled_mr1_shadow_scaled.csv \
    --timing-output results/scaled_mr1_timing_scaled.csv

python3 tools/postprocess_scaled_mr1.py
