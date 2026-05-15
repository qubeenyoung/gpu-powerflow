# Fair GPU Precision Bin Summary

Performance timing uses a fixed 10 full Newton updates (`tolerance=-1`, `max_iter=10`) so all precision profiles do the same amount of update work. Mismatch uses a separate convergence run (`tolerance=1e-8`, `max_iter=10`).

| bus size | precision | solve/update ms | final mismatch | success |
|---|---|---:|---:|---:|
| <100 | FP32 | 0.1194 | 1.531e-05 | 0/41 |
| <100 | Mixed | 0.1241 | 5.204e-11 | 41/41 |
| <100 | FP64 | 0.142 | 3.451e-12 | 41/41 |
| 100-999 | FP32 | 0.1702 | 1.995e-04 | 0/10 |
| 100-999 | Mixed | 0.1776 | 2.299e-11 | 10/10 |
| 100-999 | FP64 | 0.2168 | 1.444e-12 | 10/10 |
| 1k-9,999 | FP32 | 0.4949 | 4.116e-03 | 0/22 |
| 1k-9,999 | Mixed | 0.5151 | 2.640e-11 | 22/22 |
| 1k-9,999 | FP64 | 0.7303 | 9.196e-12 | 22/22 |
| 10k-49,999 | FP32 | 1.1968 | 6.447e-03 | 0/3 |
| 10k-49,999 | Mixed | 1.2624 | 1.381e-09 | 3/3 |
| 10k-49,999 | FP64 | 1.9184 | 2.289e-09 | 3/3 |
| >=50k | FP32 | 4.1121 | 1.453e-02 | 0/2 |
| >=50k | Mixed | 4.3959 | 5.306e-10 | 2/2 |
| >=50k | FP64 | 6.735 | 7.066e-11 | 2/2 |

Source runs:
- `matpower_precision_fp32_mixed_fp64_fixed10updates_b1_w3_r10_20260423`
- `matpower_precision_fp32_mixed_fp64_convergence_b1_tol1e-8_maxit10_w3_r10_20260423`