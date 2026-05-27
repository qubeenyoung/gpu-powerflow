# GLU CUDA Numeric Factorization vs cuDSS

- cuDSS source: `exp/20260519/report/cudss_pf_case_timings.csv`
- mode: `fp64`, `synthetic` RHS, cuDSS threading enabled
- GLU runs: warmup/repeats `1/5`
- GLU factor time uses `numeric_gpu_event_ms`, parsed from GLU `Total GPU time` inside `LUonDevice`.
- GLU uses single-precision numeric kernels because upstream GLU defines `REAL` as `float`.

## Summary

- GLU CUDA factorization is slower than cuDSS on all 5 cases.
- Arithmetic mean GLU/cuDSS factor ratio: `7.28x`.
- Geometric mean GLU/cuDSS factor ratio: `6.85x`.
- Best observed ratio: `5.40x`; worst observed ratio: `13.07x`.

| case | n_bus | dim | nnz | cuDSS factor ms | GLU CUDA factor ms | GLU/cuDSS | cuDSS/GLU speedup | GLU rel residual | GLU rel error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case118 | 118 | 181 | 1051 | 0.089 | 1.162 | 13.1 | 0.076 | 1.587e-07 | 1.923e-06 |
| case1197 | 1197 | 2392 | 14344 | 0.211 | 1.312 | 6.2 | 0.161 | 3.752e-08 | 5.367e-05 |
| case3012wp | 3012 | 5725 | 36263 | 0.458 | 2.669 | 5.8 | 0.171 | 5.848e-07 | 5.177e-05 |
| case6468rte | 6468 | 12643 | 87845 | 0.692 | 3.735 | 5.4 | 0.185 | 5.122e-07 | 1.789e-05 |
| case8387pegase | 8387 | 14908 | 110572 | 0.936 | 5.527 | 5.9 | 0.169 | 4.345e-07 | 1.589e-04 |
