# J/F Component Scale Summary

- Scale metric shown below is RMS of values in each component.
- Jacobian components: `J11=P-theta`, `J12=P-Vm`, `J21=Q-theta`, `J22=Q-Vm`.
- RHS components: `F_P` and `F_Q` split by the same NR ordering.

## Jacobian RMS by Quadrant

| case | iter | J11 P-theta | J12 P-Vm | J21 Q-theta | J22 Q-Vm | largest |
|---|---:|---:|---:|---:|---:|---|
| case1197 | J0 | 2.544e+00 | 1.764e+00 | 1.764e+00 | 2.544e+00 | J11_P_theta |
| case1197 | J1 | 2.539e+00 | 1.745e+00 | 1.727e+00 | 2.541e+00 | J22_Q_Vm |
| case1197 | J2 | 2.539e+00 | 1.744e+00 | 1.725e+00 | 2.541e+00 | J22_Q_Vm |
| case2736sp | J0 | 2.123e+03 | 2.706e+02 | 2.911e+02 | 1.964e+03 | J11_P_theta |
| case2736sp | J1 | 2.086e+03 | 2.673e+02 | 2.840e+02 | 1.943e+03 | J11_P_theta |
| case2736sp | J2 | 2.084e+03 | 2.672e+02 | 2.839e+02 | 1.942e+03 | J11_P_theta |
| case3375wp | J0 | 2.745e+03 | 3.830e+02 | 4.068e+02 | 2.490e+03 | J11_P_theta |
| case3375wp | J1 | 2.745e+03 | 3.830e+02 | 4.068e+02 | 2.490e+03 | J11_P_theta |
| case3375wp | J2 | 2.745e+03 | 3.830e+02 | 4.068e+02 | 2.490e+03 | J11_P_theta |
| case6468rte | J0 | 1.618e+03 | 4.257e+02 | 4.480e+02 | 1.540e+03 | J11_P_theta |
| case6468rte | J1 | 1.618e+03 | 4.257e+02 | 4.480e+02 | 1.540e+03 | J11_P_theta |
| case6468rte | J2 | 1.618e+03 | 4.257e+02 | 4.480e+02 | 1.540e+03 | J11_P_theta |
| case_ACTIVSg10k | J0 | 1.287e+03 | 1.815e+02 | 1.853e+02 | 1.365e+03 | J22_Q_Vm |
| case_ACTIVSg10k | J1 | 1.280e+03 | 1.816e+02 | 1.856e+02 | 1.361e+03 | J22_Q_Vm |
| case_ACTIVSg10k | J2 | 1.279e+03 | 1.816e+02 | 1.856e+02 | 1.361e+03 | J22_Q_Vm |

## RHS RMS by P/Q

| case | iter | F_P | F_Q | F_Q / F_P |
|---|---:|---:|---:|---:|
| case1197 | F0 | 1.481e-05 | 4.868e-06 | 0.329 |
| case1197 | F1 | 2.626e-06 | 2.424e-06 | 0.923 |
| case1197 | F2 | 1.641e-08 | 1.218e-08 | 0.742 |
| case2736sp | F0 | 1.088e+00 | 7.118e+00 | 6.542 |
| case2736sp | F1 | 1.160e-02 | 1.058e-01 | 9.122 |
| case2736sp | F2 | 8.791e-05 | 4.882e-04 | 5.553 |
| case3375wp | F0 | 1.876e-03 | 1.115e-02 | 5.943 |
| case3375wp | F1 | 1.088e-08 | 4.506e-08 | 4.139 |
| case3375wp | F2 | 3.115e-11 | 6.947e-12 | 0.223 |
| case6468rte | F0 | 1.475e-03 | 9.786e-05 | 0.066 |
| case6468rte | F1 | 3.875e-06 | 1.323e-05 | 3.415 |
| case6468rte | F2 | 9.896e-10 | 6.657e-10 | 0.673 |
| case_ACTIVSg10k | F0 | 8.926e-01 | 2.972e+00 | 3.330 |
| case_ACTIVSg10k | F1 | 1.170e-02 | 6.865e-02 | 5.866 |
| case_ACTIVSg10k | F2 | 2.622e-04 | 3.055e-03 | 11.650 |

## Aggregate RMS Across Cases

| iter | J11 | J12 | J21 | J22 | F_P | F_Q |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.555e+03 | 2.525e+02 | 2.666e+02 | 1.472e+03 | 3.968e-01 | 2.020e+00 |
| 1 | 1.546e+03 | 2.519e+02 | 2.652e+02 | 1.467e+03 | 4.663e-03 | 3.490e-02 |
| 2 | 1.546e+03 | 2.518e+02 | 2.652e+02 | 1.467e+03 | 7.003e-05 | 7.086e-04 |
