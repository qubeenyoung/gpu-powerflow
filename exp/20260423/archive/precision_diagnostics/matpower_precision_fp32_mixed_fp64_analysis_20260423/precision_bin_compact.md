# MATPOWER GPU Precision Compact Bin Summary

Conditions: batch 1, end2end timing, tolerance `1e-8`, max iter 10. `final mismatch` is bin median from timing repeats.

| bus size | precision | solve ms/iter | final mismatch |
|---|---|---:|---:|
| <100 | FP32 | 0.1192 | 1.769e-05 |
| <100 | Mixed | 0.1092 | 5.862e-11 |
| <100 | FP64 | 0.1212 | 3.451e-12 |
| 100-999 | FP32 | 0.1703 | 1.613e-04 |
| 100-999 | Mixed | 0.1562 | 2.329e-11 |
| 100-999 | FP64 | 0.1826 | 1.444e-12 |
| 1k-9,999 | FP32 | 0.4941 | 4.666e-03 |
| 1k-9,999 | Mixed | 0.4591 | 2.171e-11 |
| 1k-9,999 | FP64 | 0.6112 | 9.089e-12 |
| 10k-49,999 | FP32 | 1.1967 | 7.943e-03 |
| 10k-49,999 | Mixed | 1.1499 | 1.351e-09 |
| 10k-49,999 | FP64 | 1.644 | 2.289e-09 |
| >=50k | FP32 | 4.1069 | 1.289e-02 |
| >=50k | Mixed | 4.0521 | 1.633e-09 |
| >=50k | FP64 | 5.9775 | 7.064e-11 |

Source timing runs:
- `matpower_precision_fp32_fp64_end2end_b1_tol1e-8_maxit10_w3_r10_20260423`
- `matpower_precision_mixed_end2end_b1_tol1e-8_maxit10_w3_r10_20260423`