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
    --middle-solver mr1_block_jacobi
    --preconditioner metis_block_jacobi
    --block-size 64
    --block-jacobi-precision fp32
    --block-jacobi-apply inverse_gemv
    --shadow-dx-diagnostic true
    --enable-scaled-mr1-step false
)

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --linear-scaling none \
    --output results/mr1_scaling_summary_none.csv \
    --iter-output results/mr1_scaling_iters_none.csv \
    --shadow-output results/mr1_scaling_shadow_none.csv \
    --timing-output results/mr1_scaling_timing_none.csv

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --linear-scaling ruiz \
    --scaling-iters 3 \
    --scaling-norm l2 \
    --scaling-clamp 1e6 \
    --scaling-eps 1e-30 \
    --log-scaling-stats true \
    --output results/mr1_scaling_summary_ruiz.csv \
    --iter-output results/mr1_scaling_iters_ruiz.csv \
    --shadow-output results/mr1_scaling_shadow_ruiz.csv \
    --timing-output results/mr1_scaling_timing_ruiz.csv

python3 tools/postprocess_mr1_scaling.py
