# Ginkgo BiCGSTAB Preconditioner Comparison

- Input: `J1/F1` from `raw/cupf_jf_dumps`.
- Fixed outer solver: Ginkgo `BiCGSTAB`, max iterations `2`.
- `ginkgo_block_jacobi` is Ginkgo's block-Jacobi preconditioner with max block size `16`.
- `ginkgo_parilu0_ilu` is Ginkgo ParILU(0) plus triangular ILU preconditioner. This is scalar-pattern ILU(0), not our dense block graph ILU(0).

| case | preconditioner | setup ms | solve ms | total ms | true rel residual |
|---|---:|---:|---:|---:|---:|
| case2383wp | ginkgo_block_jacobi | 4.090 | 1.657 | 5.751 | 1.062e-01 |
| case2383wp | ginkgo_parilu0_ilu | 2.375 | 0.704 | 3.082 | 1.333e-02 |
| case3120sp | ginkgo_block_jacobi | 2.056 | 0.203 | 2.262 | 7.363e-01 |
| case3120sp | ginkgo_parilu0_ilu | 2.569 | 0.767 | 3.340 | 1.247e-01 |
| case6468rte | ginkgo_block_jacobi | 2.944 | 0.272 | 3.220 | 4.170e+01 |
| case6468rte | ginkgo_parilu0_ilu | 3.060 | 0.590 | 3.653 | 1.321e+01 |
| case9241pegase | ginkgo_block_jacobi | 3.573 | 0.282 | 3.858 | 6.986e-02 |
| case9241pegase | ginkgo_parilu0_ilu | 4.631 | 2.165 | 6.799 | 1.762e+01 |
| case13659pegase | ginkgo_block_jacobi | 4.372 | 0.396 | 4.771 | 2.768e-01 |
| case13659pegase | ginkgo_parilu0_ilu | 5.243 | 2.406 | 7.652 | nan |

## Short Read

- ParILU/ILU residual better than block-Jacobi on `3/5` cases.
- ParILU/ILU solve phase faster than block-Jacobi on `1/5` cases. Setup is usually not comparable to block-Jacobi because it includes ParILU factor sweeps.
- This benchmark answers whether Ginkgo's built-in preconditioners change BiCGSTAB quality/cost. It does not replace the custom dense block ILU pilot comparison.
