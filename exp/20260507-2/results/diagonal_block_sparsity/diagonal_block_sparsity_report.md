# Diagonal Block Sparsity

- `input_density` = nonzeros in original diagonal block `A[Bi,Bi]` divided by `dim(Bi)^2`.
- `inverse_density` = nonzeros in the stored dense inverse block after setup, using the numeric tolerance.
- For block-ILU, the block partition is the same METIS block family but block order is coloring; block order does not change input diagonal-block membership.

| case | method | block | input avg | input p50 | input p90 | factor Uii avg | inverse avg | dense-storage waste |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | block_jacobi | 8 | 49.6% | 50.0% | 62.5% | nan | 80.8% | 50.4% |
| case2383wp | block_ilu0 | 8 | 49.6% | 50.0% | 62.5% | 58.9% | 89.7% | 50.4% |
| case2383wp | block_jacobi | 16 | 29.4% | 29.7% | 34.4% | nan | 78.9% | 70.6% |
| case2383wp | block_ilu0 | 16 | 29.4% | 29.7% | 34.4% | 35.5% | 87.9% | 70.6% |
| case2383wp | block_jacobi | 32 | 16.4% | 16.6% | 18.4% | nan | 83.9% | 83.6% |
| case2383wp | block_ilu0 | 32 | 16.4% | 16.6% | 18.4% | 20.6% | 91.4% | 83.6% |
| case3120sp | block_jacobi | 8 | 51.9% | 53.1% | 62.5% | nan | 83.5% | 48.0% |
| case3120sp | block_ilu0 | 8 | 51.9% | 53.1% | 62.5% | 59.7% | 91.7% | 48.0% |
| case3120sp | block_jacobi | 16 | 30.1% | 30.5% | 34.4% | nan | 80.7% | 69.9% |
| case3120sp | block_ilu0 | 16 | 30.1% | 30.5% | 34.4% | 36.6% | 91.0% | 69.9% |
| case3120sp | block_jacobi | 32 | 17.2% | 17.2% | 18.4% | nan | 86.9% | 82.8% |
| case3120sp | block_ilu0 | 32 | 17.2% | 17.2% | 18.4% | 20.0% | 92.3% | 82.8% |
| case6468rte | block_jacobi | 8 | 48.4% | 50.0% | 62.5% | nan | 77.0% | 51.6% |
| case6468rte | block_ilu0 | 8 | 48.4% | 50.0% | 62.5% | 62.5% | 88.5% | 51.6% |
| case6468rte | block_jacobi | 16 | 29.6% | 30.5% | 36.0% | nan | 76.3% | 70.4% |
| case6468rte | block_ilu0 | 16 | 29.6% | 30.5% | 36.0% | 39.8% | 88.5% | 70.4% |
| case6468rte | block_jacobi | 32 | 17.3% | 17.5% | 19.5% | nan | 81.5% | 82.7% |
| case6468rte | block_ilu0 | 32 | 17.3% | 17.5% | 19.5% | 22.9% | 91.2% | 82.7% |
| case9241pegase | block_jacobi | 8 | 49.6% | 50.0% | 62.5% | nan | 78.9% | 50.4% |
| case9241pegase | block_ilu0 | 8 | 49.6% | 50.0% | 62.5% | 63.4% | 89.8% | 50.4% |
| case9241pegase | block_jacobi | 16 | 30.5% | 30.5% | 37.5% | nan | 76.1% | 69.5% |
| case9241pegase | block_ilu0 | 16 | 30.5% | 30.5% | 37.5% | 43.3% | 89.1% | 69.5% |
| case9241pegase | block_jacobi | 32 | 17.8% | 17.2% | 21.2% | nan | 77.8% | 82.2% |
| case9241pegase | block_ilu0 | 32 | 17.8% | 17.2% | 21.2% | 26.5% | 89.9% | 82.2% |
| case13659pegase | block_jacobi | 8 | 44.4% | 43.8% | 62.5% | nan | 71.8% | 55.4% |
| case13659pegase | block_ilu0 | 8 | 44.4% | 43.8% | 62.5% | 62.3% | 86.6% | 55.4% |
| case13659pegase | block_jacobi | 16 | 28.6% | 28.1% | 36.7% | nan | 73.9% | 71.3% |
| case13659pegase | block_ilu0 | 16 | 28.6% | 28.1% | 36.7% | 42.4% | 87.7% | 71.3% |
| case13659pegase | block_jacobi | 32 | 17.0% | 16.4% | 20.3% | nan | 76.8% | 83.0% |
| case13659pegase | block_ilu0 | 32 | 17.0% | 16.4% | 20.3% | 26.8% | 89.3% | 83.0% |

## Interpretation

- The original diagonal blocks are structurally sparse, especially as the target block size grows.
- The stored inverse blocks are nearly dense. So apply uses dense math, while setup/scatter starts from sparse input.
- This supports the Tensor Core argument for block-ILU apply/factor kernels: the algorithm stores and updates dense small blocks even when the source Jacobian blocks are sparse.
