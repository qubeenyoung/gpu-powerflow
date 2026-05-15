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
    --block-size 64
    --block-jacobi-precision fp32
    --block-jacobi-apply inverse_gemv
    --shadow-dx-diagnostic true
)

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --middle-solver mr1_block_jacobi \
    --preconditioner metis_block_jacobi \
    --output results/mr2_compare_summary_mr1_bj.csv \
    --iter-output results/mr2_compare_iters_mr1_bj.csv \
    --shadow-output results/mr2_compare_shadow_mr1_bj.csv \
    --timing-output results/mr2_compare_timing_mr1_bj.csv

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --middle-solver mr1_block_jacobi_coarse \
    --preconditioner metis_block_jacobi_coarse \
    --coarse-vars-per-block 1 \
    --coarse-refresh bootstrap_only \
    --coarse-precision fp32 \
    --coarse-diag-shift-scale 1e-6 \
    --output results/mr2_compare_summary_mr1_coarse.csv \
    --iter-output results/mr2_compare_iters_mr1_coarse.csv \
    --shadow-output results/mr2_compare_shadow_mr1_coarse.csv \
    --timing-output results/mr2_compare_timing_mr1_coarse.csv

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --middle-solver mr2_block_jacobi_coarse \
    --preconditioner metis_block_jacobi_coarse \
    --coarse-vars-per-block 1 \
    --coarse-refresh bootstrap_only \
    --coarse-precision fp32 \
    --coarse-diag-shift-scale 1e-6 \
    --output results/mr2_compare_summary_mr2_coarse.csv \
    --iter-output results/mr2_compare_iters_mr2_coarse.csv \
    --shadow-output results/mr2_compare_shadow_mr2_coarse.csv \
    --timing-output results/mr2_compare_timing_mr2_coarse.csv

python3 tools/postprocess_mr2_comparison.py
