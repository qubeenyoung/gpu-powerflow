# Symbolic Fill Counts for Reordered Ybus

Input matrices are `dump_Ybus.mtx` from `datasets/matpower8.1/cupf_all_dumps`.
Each matrix is loaded as CSR, reordered with METIS NodeND, applied with `iperm` as old-to-new symmetric permutation, and symbolically factorized into CSC `L` and `U` patterns.

The diagonal is stored in both `L` and `U`, so:

`LU_unique_nnz = L_nnz + U_nnz - n`

`fill_in_nnz = LU_unique_nnz - reordered_nnz`

| case | n | A nnz | L nnz | U nnz | L+U-diag nnz | fill-in nnz | fill ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case14 | 14 | 54 | 38 | 38 | 62 | 8 | 1.148 |
| case118 | 118 | 476 | 419 | 419 | 720 | 244 | 1.513 |

CSV: `exp/20260520/report/symbolic_fill_ybus_counts.csv`
