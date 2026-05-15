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
    --linear-scaling none
    --shadow-dx-diagnostic true
    --enable-scaled-mr1-step false
)

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --partition-mode unknown_metis \
    --target-block-unknowns 64 \
    --output results/bus_weighted_metis_summary_unknown.csv \
    --iter-output results/bus_weighted_metis_iters_unknown.csv \
    --shadow-output results/bus_weighted_metis_shadow_unknown.csv \
    --timing-output results/bus_weighted_metis_timing_unknown.csv \
    --partition-stats-output results/bus_weighted_metis_partition_stats_unknown.csv

./build/hybrid_nr_bench \
    "${COMMON_ARGS[@]}" \
    --partition-mode bus_weighted_metis \
    --bus-edge-weight jacobian_frobenius \
    --bus-edge-weight-scale 1000 \
    --bus-edge-weight-clamp 1000000 \
    --target-block-unknowns 64 \
    --output results/bus_weighted_metis_summary_bus.csv \
    --iter-output results/bus_weighted_metis_iters_bus.csv \
    --shadow-output results/bus_weighted_metis_shadow_bus.csv \
    --timing-output results/bus_weighted_metis_timing_bus.csv \
    --partition-stats-output results/bus_weighted_metis_partition_stats_bus.csv

python3 tools/postprocess_bus_weighted_metis.py
