# MATPOWER GPU Precision Experiment

Generated: 2026-04-23

## Setup

- Dataset: MATPOWER dump cases, 78 cases.
- Profiles: `cuda_fp32_edge`, `cuda_fp64_edge`.
- Batch size: 1.
- Tolerance: `1e-8`.
- Max Newton iterations: 10.
- Timing run: end2end, warmup 3, repeats 10, residual dump OFF.
- Residual run: end2end, warmup 0, repeats 1, residual dump ON. Timing from this run is not used.
- cuDSS MT auto was enabled.
- Per-iteration end2end time is computed as `solve_sec / iterations` for each repeat, then averaged.

## Dataset Summary

| profile | success cases | failed cases | mean iter | mean solve ms | mean solve ms/iter | median final mismatch | max final mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| cuda_fp32_edge | 0/78 | 78 | 10 | 3.7517 | 0.3752 | 1.613e-04 | 8.878e-02 |
| cuda_fp64_edge | 78/78 | 0 | 4.59 | 2.6196 | 0.476 | 8.995e-12 | 9.558e-09 |

## Bus-Bin Summary

| bus bin | profile | success cases | mean iter | mean solve ms | mean solve ms/iter | median final mismatch | max final mismatch |
|---|---|---:|---:|---:|---:|---:|---:|
| <100 | cuda_fp32_edge | 0/41 | 10 | 1.1916 | 0.1192 | 1.769e-05 | 2.034e-03 |
| <100 | cuda_fp64_edge | 41/41 | 4.41 | 0.5432 | 0.1212 | 3.451e-12 | 9.085e-09 |
| 100-999 | cuda_fp32_edge | 0/10 | 10 | 1.7031 | 0.1703 | 1.613e-04 | 8.878e-02 |
| 100-999 | cuda_fp64_edge | 10/10 | 4.3 | 0.7894 | 0.1826 | 1.444e-12 | 2.116e-09 |
| 1k-9,999 | cuda_fp32_edge | 0/22 | 10 | 4.9413 | 0.4941 | 4.666e-03 | 1.360e-02 |
| 1k-9,999 | cuda_fp64_edge | 22/22 | 4.73 | 2.923 | 0.6112 | 9.089e-12 | 9.558e-09 |
| 10k-49,999 | cuda_fp32_edge | 0/3 | 10 | 11.9671 | 1.1967 | 7.943e-03 | 2.803e-02 |
| 10k-49,999 | cuda_fp64_edge | 3/3 | 5.33 | 8.7231 | 1.644 | 2.289e-09 | 4.509e-09 |
| >=50k | cuda_fp32_edge | 0/2 | 10 | 41.0693 | 4.1069 | 1.289e-02 | 2.615e-02 |
| >=50k | cuda_fp64_edge | 2/2 | 7 | 41.8426 | 5.9775 | 7.064e-11 | 1.212e-10 |

## Convergence Notes

- FP32 reached `1e-8` within 10 iterations on 0/78 cases.
- FP64 reached `1e-8` within 10 iterations on 78/78 cases.
- Cases where FP32 failed but FP64 succeeded: 78.

| case | buses | fp32 final mismatch | fp64 final mismatch | fp32 min residual dump | fp64 min residual dump |
|---|---:|---:|---:|---:|---:|
| case_SyntheticUSA | 82000 | 2.615e-02 | 1.212e-10 | 7.860e-03 | 1.209e-10 |
| case_ACTIVSg70k | 70000 | 1.818e-02 | 3.766e-11 | 3.526e-03 | 1.772e-11 |
| case_ACTIVSg25k | 25000 | 1.563e-02 | 1.210e-10 | 4.071e-03 | 1.210e-10 |
| case13659pegase | 13659 | 4.802e-03 | 2.289e-09 | 2.101e-03 | 2.289e-09 |
| case_ACTIVSg10k | 10000 | 2.803e-02 | 4.509e-09 | 6.021e-03 | 4.509e-09 |
| case9241pegase | 9241 | 5.272e-03 | 2.129e-09 | 2.819e-03 | 2.129e-09 |
| case8387pegase | 8387 | 8.239e-03 | 1.938e-11 | 5.250e-03 | 9.001e-12 |
| case6515rte | 6515 | 7.660e-03 | 1.189e-11 | 4.090e-03 | 8.087e-12 |
| case6495rte | 6495 | 6.990e-03 | 7.424e-11 | 4.119e-03 | 7.427e-11 |
| case6470rte | 6470 | 7.560e-03 | 1.146e-11 | 3.572e-03 | 1.069e-11 |
| case6468rte | 6468 | 8.356e-03 | 1.766e-11 | 4.175e-03 | 1.708e-11 |
| case3375wp | 3374 | 9.320e-03 | 1.983e-11 | 3.928e-03 | 1.235e-11 |
| case3120sp | 3120 | 6.086e-03 | 1.136e-11 | 4.062e-03 | 7.580e-12 |
| case3012wp | 3012 | 6.088e-03 | 1.139e-11 | 3.954e-03 | 1.132e-11 |
| case2869pegase | 2869 | 4.085e-03 | 2.154e-09 | 2.019e-03 | 2.154e-09 |

## Largest FP32 Final Mismatches

| case | buses | fp32 success | fp32 final mismatch | fp64 final mismatch |
|---|---:|---:|---:|---:|
| case141 | 141 | False | 8.878e-02 | 2.116e-09 |
| case_ACTIVSg10k | 10000 | False | 2.803e-02 | 4.509e-09 |
| case_SyntheticUSA | 82000 | False | 2.615e-02 | 1.212e-10 |
| case_ACTIVSg70k | 70000 | False | 1.818e-02 | 3.766e-11 |
| case_ACTIVSg25k | 25000 | False | 1.563e-02 | 1.210e-10 |
| case2746wp | 2746 | False | 1.360e-02 | 7.810e-10 |
| case2746wop | 2746 | False | 1.239e-02 | 2.971e-09 |
| case3375wp | 3374 | False | 9.320e-03 | 1.983e-11 |
| case2848rte | 2848 | False | 9.034e-03 | 9.558e-09 |
| case2868rte | 2868 | False | 8.490e-03 | 1.568e-11 |

## Best FP32 Per-Iteration Speedups vs FP64

| case | buses | fp32 ms/iter | fp64 ms/iter | speedup | fp32 success | fp64 success |
|---|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg70k | 70000 | 3.9101 | 5.771 | 1.48x | False | True |
| case_SyntheticUSA | 82000 | 4.3038 | 6.184 | 1.44x | False | True |
| case13659pegase | 13659 | 1.0613 | 1.5093 | 1.42x | False | True |
| case_ACTIVSg25k | 25000 | 1.7182 | 2.3723 | 1.38x | False | True |
| case9241pegase | 9241 | 0.8965 | 1.2364 | 1.38x | False | True |
| case_ACTIVSg10k | 10000 | 0.8106 | 1.0505 | 1.3x | False | True |
| case6468rte | 6468 | 0.6066 | 0.7791 | 1.28x | False | True |
| case3120sp | 3120 | 0.4677 | 0.6002 | 1.28x | False | True |
| case6515rte | 6515 | 0.649 | 0.8282 | 1.28x | False | True |
| case_ACTIVSg2000 | 2000 | 0.553 | 0.6984 | 1.26x | False | True |

## Output Files

- `precision_dataset_summary.csv`
- `precision_bin_summary.csv`
- `precision_timing_per_case.csv`
- `precision_residual_per_case.csv`
- `precision_case_comparison.csv`
- `precision_timing_residual_merged.csv`

## Source Runs

- Timing: `matpower_precision_fp32_fp64_end2end_b1_tol1e-8_maxit10_w3_r10_20260423`
- Residuals: `matpower_precision_fp32_fp64_residuals_b1_tol1e-8_maxit10_r1_20260423`
