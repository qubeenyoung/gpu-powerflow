# cuDSS Fill-In and METIS User Permutation Experiment

- precision: `fp64`
- rhs_mode: `synthetic`
- warmup/repeats: `3/10`
- cuDSS threading layer: `enabled`

Fill-in is measured as cuDSS `CUDSS_DATA_LU_NNZ` immediately after `CUDSS_PHASE_ANALYSIS`; `metis` means METIS NodeND was supplied through `CUDSS_DATA_USER_PERM` before analysis.

## Fill Metrics

| case | n | J nnz | cuDSS LU default | default/J nnz | cuDSS LU METIS user | METIS/J nnz | LU METIS/default |
|---|---:|---:|---:|---:|---:|---:|---:|
| case118 | 181 | 1,051 | 1,487 | 1.415 | 1,405 | 1.337 | 0.945 |
| case1197 | 2,392 | 14,344 | 18,856 | 1.315 | 17,344 | 1.209 | 0.920 |
| case3012wp | 5,725 | 36,263 | 69,789 | 1.925 | 69,127 | 1.906 | 0.991 |
| case6468rte | 12,643 | 87,845 | 160,245 | 1.824 | 156,357 | 1.780 | 0.976 |
| case8387pegase | 14,908 | 110,572 | 208,734 | 1.888 | 202,868 | 1.835 | 0.972 |

## Timing Impact

| case | analysis default ms | analysis METIS ms | factor default ms | factor METIS ms | factor delta | solve default ms | solve METIS ms | solve delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case118 | 8.707 | 7.586 | 0.088 | 0.088 | -0.3% | 0.070 | 0.071 | 1.2% |
| case1197 | 9.696 | 11.457 | 0.194 | 0.370 | 90.4% | 0.096 | 0.084 | -12.1% |
| case3012wp | 12.225 | 18.592 | 0.402 | 1.281 | 218.8% | 0.190 | 0.216 | 13.6% |
| case6468rte | 17.696 | 33.052 | 0.589 | 3.055 | 419.0% | 0.249 | 0.351 | 40.8% |
| case8387pegase | 21.591 | 40.541 | 0.812 | 3.995 | 391.9% | 0.310 | 0.415 | 34.1% |
