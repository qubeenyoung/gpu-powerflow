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
    --output results/mr1_coarse_hybrid_summary_off.csv \
    --iter-output results/mr1_coarse_hybrid_iters_off.csv \
    --shadow-output results/mr1_coarse_shadow_dx_off.csv \
    --timing-output results/mr1_coarse_timing_breakdown_off.csv

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --middle-solver mr1_block_jacobi_coarse \
    --preconditioner metis_block_jacobi_coarse \
    --coarse-vars-per-block 1 \
    --coarse-refresh bootstrap_only \
    --coarse-precision fp32 \
    --coarse-diag-shift-scale 1e-6 \
    --output results/mr1_coarse_hybrid_summary_on.csv \
    --iter-output results/mr1_coarse_hybrid_iters_on.csv \
    --shadow-output results/mr1_coarse_shadow_dx_on.csv \
    --timing-output results/mr1_coarse_timing_breakdown_on.csv

python3 tools/postprocess_mr1_coarse_results.py
