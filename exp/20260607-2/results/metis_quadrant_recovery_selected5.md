# METIS Quadrant Recovery Rate

- Recovery rate = diagonal-block nnz / total nnz for the same Jacobian quadrant.
- Total nnz for each quadrant is `diagonal + offdiagonal`; lower recovery means more entries leave the METIS block diagonal and are ignored by block-Jacobi.
- Main setting below is block_size=64, matching the recent hybrid/shadow experiments.

## block_size=64

| case | J11 P-theta | J12 P-Vm | J21 Q-theta | J22 Q-Vm | most lost |
|---|---:|---:|---:|---:|---|
| case13659pegase | 82.8% | 56.4% | 56.4% | 88.2% | J12_P_Vm (56.4%) |
| case2383wp | 88.6% | 65.4% | 65.4% | 94.5% | J12_P_Vm (65.4%) |
| case3120sp | 89.5% | 67.3% | 67.3% | 93.6% | J12_P_Vm (67.3%) |
| case6468rte | 90.1% | 62.0% | 62.0% | 90.4% | J12_P_Vm (62.0%) |
| case9241pegase | 86.5% | 51.0% | 51.0% | 91.4% | J12_P_Vm (51.0%) |

## Weighted Average, block_size=32

| quadrant | recovery | lost/off-block |
|---|---:|---:|
| J11_P_theta | 78.5% | 21.5% |
| J12_P_Vm | 74.0% | 26.0% |
| J21_Q_theta | 74.0% | 26.0% |
| J22_Q_Vm | 77.7% | 22.3% |

## Weighted Average, block_size=64

| quadrant | recovery | lost/off-block |
|---|---:|---:|
| J11_P_theta | 86.1% | 13.9% |
| J12_P_Vm | 57.4% | 42.6% |
| J21_Q_theta | 57.4% | 42.6% |
| J22_Q_Vm | 90.4% | 9.6% |

## Interpretation

- For block_size=64, J11 and J22 are mostly preserved inside diagonal blocks, usually around 83-95%.
- J12 and J21 have the lowest recovery in every case, only about 51-67% for block_size=64.
- This means the entries most often pushed outside the block diagonal are the cross couplings P-Vm and Q-theta.
- The worst case here is case9241pegase, where only about 51.0% of J12/J21 remains inside diagonal blocks.
