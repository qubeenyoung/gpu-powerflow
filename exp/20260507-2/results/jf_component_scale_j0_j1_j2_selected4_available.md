# J/F Component Scale Summary

- Scale metric shown below is RMS of values in each component.
- Jacobian components: `J11=P-theta`, `J12=P-Vm`, `J21=Q-theta`, `J22=Q-Vm`.
- RHS components: `F_P` and `F_Q` split by the same NR ordering.

## Jacobian RMS by Quadrant

| case | iter | J11 P-theta | J12 P-Vm | J21 Q-theta | J22 Q-Vm | largest |
|---|---:|---:|---:|---:|---:|---|
| case2383wp | J0 | 3.084e+03 | 1.077e+02 | 1.159e+02 | 2.898e+03 | J11_P_theta |
| case2383wp | J1 | 2.754e+03 | 9.911e+01 | 9.966e+01 | 2.681e+03 | J11_P_theta |
| case2383wp | J2 | 2.739e+03 | 9.854e+01 | 9.854e+01 | 2.670e+03 | J11_P_theta |
| case3120sp | J0 | 2.135e+03 | 3.496e+02 | 3.508e+02 | 2.061e+03 | J11_P_theta |
| case3120sp | J1 | 2.320e+03 | 3.691e+02 | 3.900e+02 | 2.169e+03 | J11_P_theta |
| case3120sp | J2 | 2.429e+03 | 3.773e+02 | 4.078e+02 | 2.229e+03 | J11_P_theta |
| case6468rte | J0 | 1.618e+03 | 4.257e+02 | 4.480e+02 | 1.540e+03 | J11_P_theta |
| case6468rte | J1 | 1.618e+03 | 4.257e+02 | 4.480e+02 | 1.540e+03 | J11_P_theta |
| case6468rte | J2 | 1.618e+03 | 4.257e+02 | 4.480e+02 | 1.540e+03 | J11_P_theta |
| case9241pegase | J0 | 1.383e+03 | 2.194e+02 | 2.255e+02 | 1.418e+03 | J22_Q_Vm |
| case9241pegase | J1 | 1.383e+03 | 2.195e+02 | 2.256e+02 | 1.419e+03 | J22_Q_Vm |
| case9241pegase | J2 | 1.383e+03 | 2.194e+02 | 2.255e+02 | 1.418e+03 | J22_Q_Vm |

## RHS RMS by P/Q

| case | iter | F_P | F_Q | F_Q / F_P |
|---|---:|---:|---:|---:|
| case2383wp | F0 | 6.200e+00 | 1.387e+02 | 22.374 |
| case2383wp | F1 | 3.243e-01 | 1.072e+01 | 33.059 |
| case2383wp | F2 | 6.243e-02 | 3.004e-01 | 4.812 |
| case3120sp | F0 | 8.342e+00 | 3.102e+01 | 3.719 |
| case3120sp | F1 | 4.649e-01 | 2.355e+00 | 5.066 |
| case3120sp | F2 | 2.098e-01 | 4.516e-01 | 2.152 |
| case6468rte | F0 | 1.475e-03 | 9.786e-05 | 0.066 |
| case6468rte | F1 | 3.875e-06 | 1.323e-05 | 3.415 |
| case6468rte | F2 | 9.896e-10 | 6.657e-10 | 0.673 |
| case9241pegase | F0 | 1.122e+00 | 1.482e-01 | 0.132 |
| case9241pegase | F1 | 8.858e-01 | 2.694e-01 | 0.304 |
| case9241pegase | F2 | 1.823e-01 | 7.741e-02 | 0.425 |

## Aggregate RMS Across Cases

| iter | J11 | J12 | J21 | J22 | F_P | F_Q |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.055e+03 | 2.756e+02 | 2.851e+02 | 1.979e+03 | 3.916e+00 | 4.247e+01 |
| 1 | 2.019e+03 | 2.783e+02 | 2.908e+02 | 1.952e+03 | 4.188e-01 | 3.337e+00 |
| 2 | 2.042e+03 | 2.802e+02 | 2.950e+02 | 1.964e+03 | 1.136e-01 | 2.074e-01 |
