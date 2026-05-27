# cuDSS Analysis Reorder vs METIS NodeND

- matrix: `newton_jacobian_at_v0`
- precision used for cuDSS analysis: `fp64`
- cuDSS permutation interpretation in main metrics: raw `PERM_REORDER_ROW/COL` as original-index to reordered-position

| case | n | nnz | row exact | row rho | row mean delta | row p95 delta | col exact | col rho | col mean delta | cuDSS row=col | METIS BW | cuDSS BW | METIS profile | cuDSS profile | figure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case118 | 181 | 1051 | 0.00% | 0.170 | 0.301 | 0.694 | 0.00% | 0.170 | 0.301 | 100.00% | 168 | 178 | 11121 | 10902 | figs/cudss_vs_metis/case118_jacobian_cudss_vs_metis.png |
| case1197 | 2392 | 14344 | 0.08% | -0.015 | 0.342 | 0.757 | 0.08% | -0.015 | 0.342 | 100.00% | 2361 | 2283 | 1995810 | 1858888 | figs/cudss_vs_metis/case1197_jacobian_cudss_vs_metis.png |
| case3012wp | 5725 | 36259 | 0.03% | -0.004 | 0.341 | 0.759 | 0.03% | -0.004 | 0.341 | 100.00% | 5684 | 5707 | 11530427 | 11509546 | figs/cudss_vs_metis/case3012wp_jacobian_cudss_vs_metis.png |
| case6468rte | 12643 | 87743 | 0.02% | 0.022 | 0.328 | 0.775 | 0.02% | 0.022 | 0.328 | 100.00% | 12613 | 12614 | 56231336 | 57006082 | figs/cudss_vs_metis/case6468rte_jacobian_cudss_vs_metis.png |
| case8387pegase | 14908 | 110109 | 0.01% | 0.005 | 0.332 | 0.777 | 0.01% | 0.005 | 0.332 | 100.00% | 14830 | 14875 | 78295100 | 78920620 | figs/cudss_vs_metis/case8387pegase_jacobian_cudss_vs_metis.png |

## Reordered Matrix Overlap

| case | common nnz | METIS only | cuDSS only | Jaccard | common fraction | relative Frobenius diff | common-value relative diff | matrix figure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| case118 | 225 | 826 | 826 | 0.120 | 0.214 | 1.164e+00 | 1.057e+00 | figs/cudss_vs_metis/case118_reordered_matrix_compare.png |
| case1197 | 3178 | 11166 | 11166 | 0.125 | 0.222 | 1.411e+00 | 1.409e+00 | figs/cudss_vs_metis/case1197_reordered_matrix_compare.png |
| case3012wp | 6075 | 30184 | 30184 | 0.091 | 0.168 | 1.327e+00 | 1.269e+00 | figs/cudss_vs_metis/case3012wp_reordered_matrix_compare.png |
| case6468rte | 12809 | 74934 | 74934 | 0.079 | 0.146 | 1.376e+00 | 1.347e+00 | figs/cudss_vs_metis/case6468rte_reordered_matrix_compare.png |
| case8387pegase | 14930 | 95179 | 95179 | 0.073 | 0.136 | 1.340e+00 | 1.278e+00 | figs/cudss_vs_metis/case8387pegase_reordered_matrix_compare.png |

## Orientation Check

These values compare raw cuDSS arrays and their inverses against METIS. Lower mean delta is closer.

- `case118`: raw row mean=0.300982, inverse row mean=0.309392; raw col mean=0.300982, inverse col mean=0.309392
- `case1197`: raw row mean=0.342127, inverse row mean=0.334141; raw col mean=0.342127, inverse col mean=0.334141
- `case3012wp`: raw row mean=0.340609, inverse row mean=0.350737; raw col mean=0.340609, inverse col mean=0.350737
- `case6468rte`: raw row mean=0.327962, inverse row mean=0.334807; raw col mean=0.327962, inverse col mean=0.334807
- `case8387pegase`: raw row mean=0.332445, inverse row mean=0.327578; raw col mean=0.332445, inverse col mean=0.327578
