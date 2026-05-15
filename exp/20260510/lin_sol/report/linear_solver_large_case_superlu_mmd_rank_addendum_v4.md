# SuperLU_DIST MMD Rank-Sweep Addendum v4

This addendum continues the large-case diagnosis after the v4 report found that SuperLU_DIST's catastrophic slow path was tied to `NATURAL` ordering. It reruns `MMD_AT_PLUS_A` with both `LargeDiag_MC64` and `NOROWPERM` for `np=1`, `np=2`, and `np=4`.

All runs used the local MPICH launcher paired with the SuperLU_DIST build and `OMP_NUM_THREADS=1`.

## Best Large-Case Results

| case | np | rowperm | conv | analysis ms | factor ms | solve ms | solver ms | scaled resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case2869pegase | 1.000 | LargeDiag_MC64 | yes | 11.296 | 143.061 | 5.731e-01 | 155.019 | 4.534e-15 |
| case9241pegase | 1.000 | NOROWPERM | yes | 36.502 | 148.649 | 1.887 | 187.298 | 1.394e-14 |

## Full MMD Rank Sweep

| case | np | rowperm | conv | analysis ms | factor ms | solve ms | solver ms | scaled resid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_validation | 1.000 | LargeDiag_MC64 | yes | 4.644e-02 | 234.036 | 4.003e-02 | 234.136 | 6.478e-17 |
| synthetic_validation | 2.000 | LargeDiag_MC64 | yes | 1.817e-01 | 221.643 | 6.135e-02 | 221.891 | 6.478e-17 |
| synthetic_validation | 4.000 | LargeDiag_MC64 | yes | 2.045e-01 | 378.076 | 5.621e-02 | 378.337 | 6.478e-17 |
| synthetic_validation | 1.000 | NOROWPERM | yes | 3.036e-02 | 139.020 | 3.978e-02 | 139.100 | 1.122e-16 |
| synthetic_validation | 2.000 | NOROWPERM | yes | 1.315e-01 | 203.971 | 6.938e-02 | 204.179 | 1.122e-16 |
| synthetic_validation | 4.000 | NOROWPERM | yes | 1.900e-01 | 344.223 | 7.124e-02 | 344.483 | 1.122e-16 |
| case2869pegase | 1.000 | LargeDiag_MC64 | yes | 11.296 | 143.061 | 5.731e-01 | 155.019 | 4.534e-15 |
| case9241pegase | 1.000 | LargeDiag_MC64 | yes | 39.059 | 149.727 | 1.965 | 191.042 | 1.341e-14 |
| case2869pegase | 2.000 | LargeDiag_MC64 | yes | 12.336 | 207.774 | 1.002 | 221.059 | 5.152e-15 |
| case9241pegase | 2.000 | LargeDiag_MC64 | yes | 40.383 | 213.575 | 2.419 | 256.101 | 1.383e-14 |
| case2869pegase | 4.000 | LargeDiag_MC64 | yes | 14.643 | 345.130 | 9.206e-01 | 360.562 | 5.307e-15 |
| case9241pegase | 4.000 | LargeDiag_MC64 | yes | 44.862 | 353.655 | 2.227 | 400.526 | 1.393e-14 |
| case2869pegase | 1.000 | NOROWPERM | yes | 10.566 | 145.133 | 5.770e-01 | 156.359 | 5.403e-15 |
| case9241pegase | 1.000 | NOROWPERM | yes | 36.502 | 148.649 | 1.887 | 187.298 | 1.394e-14 |
| case2869pegase | 2.000 | NOROWPERM | yes | 11.061 | 205.826 | 1.017 | 217.994 | 4.649e-15 |
| case9241pegase | 2.000 | NOROWPERM | yes | 36.882 | 215.108 | 2.445 | 254.705 | 1.461e-14 |
| case2869pegase | 4.000 | NOROWPERM | yes | 12.894 | 344.606 | 9.434e-01 | 358.500 | 5.314e-15 |
| case9241pegase | 4.000 | NOROWPERM | yes | 41.467 | 347.163 | 2.233 | 391.081 | 1.401e-14 |

## Interpretation

The MMD rank sweep confirms that the earlier slow SuperLU_DIST timing was caused by the NATURAL ordering path, not by process launch, MatrixMarket loading, or a general inability of SuperLU_DIST to solve the large Jacobians. For these single-node runs, increasing MPI ranks did not improve the best MMD configuration; `np=1` remained the fastest or competitive option. This reinforces the annual-report interpretation: SuperLU_DIST is a valid external MPI/hybrid direct-solver baseline when configured with a supported fill-reducing ordering, but it is still less natural as a cuPF default than cuDSS because it relies on MPI/host-distributed integration and lacks a validated reusable repeated-Newton timing path in this wrapper.

CSV: `measurement_audit/results/superlu_dist_mmd_rank_sweep.csv`
