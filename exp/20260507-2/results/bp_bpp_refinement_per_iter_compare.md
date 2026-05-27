## case2383wp
| mode | iter | solver | iter compute ms | linear ms | mismatch inf |
|---|---:|---|---:|---:|---:|
| pure cuDSS | 0 | cudss_pure | 0.698 | 0.583 | 1.34e+03->1.16e+02 |
| pure cuDSS | 1 | cudss_pure | 0.617 | 0.505 | 1.16e+02->6.10e+00 |
| pure cuDSS | 2 | cudss_pure | 0.614 | 0.501 | 6.10e+00->5.57e-01 |
| pure cuDSS | 3 | cudss_pure | 0.618 | 0.506 | 5.57e-01->7.80e-03 |
| pure cuDSS | 4 | cudss_pure | 0.632 | 0.503 | 7.80e-03->1.50e-06 |
| pure cuDSS | 5 | cudss_pure | 0.638 | 0.503 | 1.50e-06->4.55e-12 |
| BJ+BpBpp | 0 | cudss_bootstrap | 0.723 | 0.606 | 1.34e+03->1.16e+02 |
| BJ+BpBpp | 1 | bicgstab_bpbpp_refine_middle | 1.498 | 1.387 | 1.16e+02->1.52e+01 |
| BJ+BpBpp | 2 | cudss_fallback | 2.058 | 1.845 | 1.52e+01->1.47e+01 |
| BJ+BpBpp | 3 | cudss_fallback | 2.052 | 1.837 | 1.47e+01->3.51e+00 |
| BJ+BpBpp | 4 | bicgstab_bpbpp_refine_middle | 1.446 | 1.334 | 3.51e+00->1.60e+00 |
| BJ+BpBpp | 5 | cudss_direct | 0.618 | 0.506 | 1.60e+00->5.21e-02 |
| BJ+BpBpp | 6 | cudss_direct | 0.616 | 0.506 | 5.21e-02->6.07e-05 |
| BJ+BpBpp | 7 | cudss_polish | 0.628 | 0.501 | 6.07e-05->1.02e-10 |

## case3120sp
| mode | iter | solver | iter compute ms | linear ms | mismatch inf |
|---|---:|---|---:|---:|---:|
| pure cuDSS | 0 | cudss_pure | 0.765 | 0.655 | 6.11e+02->4.72e+01 |
| pure cuDSS | 1 | cudss_pure | 0.644 | 0.542 | 4.72e+01->1.22e+01 |
| pure cuDSS | 2 | cudss_pure | 0.646 | 0.544 | 1.22e+01->9.08e-01 |
| pure cuDSS | 3 | cudss_pure | 0.646 | 0.544 | 9.08e-01->7.53e-03 |
| pure cuDSS | 4 | cudss_pure | 0.650 | 0.540 | 7.53e-03->7.16e-07 |
| pure cuDSS | 5 | cudss_pure | 0.673 | 0.547 | 7.16e-07->7.63e-12 |
| BJ+BpBpp | 0 | cudss_bootstrap | 0.834 | 0.718 | 6.11e+02->4.72e+01 |
| BJ+BpBpp | 1 | cudss_fallback | 2.371 | 2.156 | 4.72e+01->1.22e+01 |
| BJ+BpBpp | 2 | cudss_fallback | 2.364 | 2.150 | 1.22e+01->9.08e-01 |
| BJ+BpBpp | 3 | cudss_fallback | 2.354 | 2.141 | 9.08e-01->7.53e-03 |
| BJ+BpBpp | 4 | cudss_fallback | 2.361 | 2.141 | 7.53e-03->7.16e-07 |
| BJ+BpBpp | 5 | cudss_polish | 0.760 | 0.622 | 7.16e-07->4.22e-12 |

## case9241pegase
| mode | iter | solver | iter compute ms | linear ms | mismatch inf |
|---|---:|---|---:|---:|---:|
| pure cuDSS | 0 | cudss_pure | 1.301 | 1.173 | 4.15e+01->6.04e+01 |
| pure cuDSS | 1 | cudss_pure | 1.222 | 1.090 | 6.04e+01->1.25e+01 |
| pure cuDSS | 2 | cudss_pure | 1.220 | 1.089 | 1.25e+01->2.37e+00 |
| pure cuDSS | 3 | cudss_pure | 1.223 | 1.092 | 2.37e+00->1.23e-01 |
| pure cuDSS | 4 | cudss_pure | 1.231 | 1.092 | 1.23e-01->3.23e-04 |
| pure cuDSS | 5 | cudss_pure | 1.236 | 1.097 | 3.23e-04->2.13e-09 |
| BJ+BpBpp | 0 | cudss_bootstrap | 1.493 | 1.345 | 4.15e+01->6.04e+01 |
| BJ+BpBpp | 1 | bicgstab_bpbpp_refine_middle | 3.018 | 2.874 | 6.04e+01->1.00e+01 |
| BJ+BpBpp | 2 | cudss_fallback | 4.326 | 4.057 | 1.00e+01->2.13e+00 |
| BJ+BpBpp | 3 | bicgstab_bpbpp_refine_middle | 2.944 | 2.799 | 2.13e+00->6.38e-01 |
| BJ+BpBpp | 4 | cudss_direct | 1.408 | 1.265 | 6.38e-01->5.58e-03 |
| BJ+BpBpp | 5 | cudss_direct | 1.414 | 1.254 | 5.58e-03->4.34e-07 |
| BJ+BpBpp | 6 | cudss_polish | 1.423 | 1.253 | 4.34e-07->8.22e-12 |

## case13659pegase
| mode | iter | solver | iter compute ms | linear ms | mismatch inf |
|---|---:|---|---:|---:|---:|
| pure cuDSS | 0 | cudss_pure | 1.603 | 1.444 | 6.30e+01->2.92e+01 |
| pure cuDSS | 1 | cudss_pure | 1.507 | 1.353 | 2.92e+01->4.59e+00 |
| pure cuDSS | 2 | cudss_pure | 1.503 | 1.350 | 4.59e+00->2.00e-01 |
| pure cuDSS | 3 | cudss_pure | 1.509 | 1.347 | 2.00e-01->3.95e-04 |
| pure cuDSS | 4 | cudss_pure | 1.515 | 1.352 | 3.95e-04->2.29e-09 |
| BJ+BpBpp | 0 | cudss_bootstrap | 1.819 | 1.640 | 6.30e+01->2.92e+01 |
| BJ+BpBpp | 1 | cudss_fallback | 5.570 | 5.244 | 2.92e+01->4.59e+00 |
| BJ+BpBpp | 2 | cudss_fallback | 5.521 | 5.196 | 4.59e+00->2.00e-01 |
| BJ+BpBpp | 3 | cudss_fallback | 5.515 | 5.183 | 2.00e-01->3.95e-04 |
| BJ+BpBpp | 4 | cudss_fallback | 5.526 | 5.186 | 3.95e-04->2.29e-09 |

## case6468rte
| mode | iter | solver | iter compute ms | linear ms | mismatch inf |
|---|---:|---|---:|---:|---:|
| pure cuDSS | 0 | cudss_pure | 0.933 | 0.807 | 1.19e-01->7.41e-04 |
| pure cuDSS | 1 | cudss_pure | 0.848 | 0.716 | 7.41e-04->4.17e-08 |
| pure cuDSS | 2 | cudss_pure | 0.851 | 0.712 | 4.17e-08->1.07e-11 |
| BJ+BpBpp | 0 | cudss_bootstrap | 1.035 | 0.897 | 1.19e-01->7.41e-04 |
| BJ+BpBpp | 1 | cudss_fallback | 3.396 | 3.127 | 7.41e-04->4.17e-08 |
| BJ+BpBpp | 2 | cudss_polish | 0.979 | 0.819 | 4.17e-08->1.11e-11 |

## Case totals
| case | pure iter-compute total ms | pure linear total ms | hybrid iter-compute total ms | hybrid linear total ms | pure linear median ms | hybrid linear median ms |
|---|---:|---:|---:|---:|---:|---:|
| case2383wp | 3.817 | 3.101 | 9.641 | 8.521 | 0.504 | 0.970 |
| case3120sp | 4.024 | 3.372 | 11.044 | 9.929 | 0.544 | 2.141 |
| case9241pegase | 7.433 | 6.633 | 16.026 | 14.848 | 1.092 | 1.345 |
| case13659pegase | 7.636 | 6.846 | 23.950 | 22.448 | 1.352 | 5.186 |
| case6468rte | 2.632 | 2.234 | 5.411 | 4.844 | 0.716 | 0.897 |
