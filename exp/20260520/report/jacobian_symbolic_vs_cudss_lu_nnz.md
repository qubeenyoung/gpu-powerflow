# Jacobian Symbolic LU Pattern vs cuDSS Analyze LU_NNZ

Matrix basis: Newton power-flow Jacobian from `pf_case_loader`, not Ybus.

Local path: Jacobian CSR -> METIS NodeND -> apply `iperm` as old-to-new symmetric permutation -> reordered CSR -> symbolic LU reach -> CSC L/U patterns. Diagonal is counted in both L and U, so `our_LU_unique_nnz = L_nnz + U_nnz - n`.

cuDSS path: `CUDSS_DATA_LU_NNZ` read immediately after `CUDSS_PHASE_ANALYSIS` from `exp/20260519/report/cudss_metis_user_perm_fillin.csv`. The `metis` cuDSS column is with `CUDSS_DATA_USER_PERM` set before analysis.

| case | dim | J nnz | local L+U-diag | local/J | cuDSS default LU | default/J | cuDSS METIS LU | METIS/J | local - cuDSS METIS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case118 | 181 | 1,051 | 1,405 | 1.337 | 1,487 | 1.415 | 1,405 | 1.337 | 0 |
| case1197 | 2,392 | 14,344 | 17,344 | 1.209 | 18,856 | 1.315 | 17,344 | 1.209 | 0 |
| case3012wp | 5,725 | 36,263 | 69,127 | 1.906 | 69,789 | 1.925 | 69,127 | 1.906 | 0 |
| case6468rte | 12,643 | 87,845 | 156,357 | 1.780 | 160,245 | 1.824 | 156,357 | 1.780 | 0 |
| case8387pegase | 14,908 | 110,572 | 202,868 | 1.835 | 208,734 | 1.888 | 202,868 | 1.835 | 0 |

## Notes

- Local symbolic counts match cuDSS METIS-user-permutation `CUDSS_DATA_LU_NNZ` exactly on all tested Jacobians.
- Direction detail: METIS `perm` is the ordering vector cuDSS accepts as `CUDSS_DATA_USER_PERM`, while local `apply_symmetric_permutation()` expects old-index to new-index, so it must apply METIS `iperm`.
- cuDSS default ordering produces slightly larger LU_NNZ than METIS user permutation on these five cases.

CSV: `exp/20260520/report/jacobian_symbolic_vs_cudss_lu_nnz.csv`
