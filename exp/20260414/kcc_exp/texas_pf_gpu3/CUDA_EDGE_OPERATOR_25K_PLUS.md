# CUDA Edge Operator Timing: 25K+ Cases

- Source run: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/results/cuda_edge_operator_25k_plus`
- Dataset: `/workspace/datasets/texas_univ_cases/pf_dataset`
- cuPF dumps: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps`
- Profile: `cuda_edge` / `cuda_mixed_edge`
- Measurement: `CUDA_VISIBLE_DEVICES=3`, warmup 1, repeats 10, cuDSS MT host threads `AUTO`
- ND level: CLI option was not passed (`--cudss-nd-nlevels` not present in manifest commands)

## Overall
| case | bus | Ybus nnz | iter mean | elapsed ms | elapsed stdev | analyze ms | analyze % | solve ms | solve % | success | max mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_ACTIVSg25k | 25000 | 85220 | 5.0 | 71.21 | 1.639 | 62.53 | 87.8% | 8.679 | 12.2% | True | 3.206e-09 |
| case_ACTIVSg70k | 70000 | 236636 | 7.0 | 206.3 | 27.42 | 174.9 | 84.8% | 31.42 | 15.2% | True | 9.894e-09 |
| Base_Eastern_Interconnect_515GW | 78478 | 294392 | 6.6 | 253.4 | 36.40 | 210.6 | 83.1% | 42.82 | 16.9% | True | 6.945e-09 |

## Analyze Breakdown
| case | storage_prepare ms | storage_prepare % analyze | jacobian_builder ms | jacobian_builder % analyze | linear_solve_analyze ms | linear_solve_analyze % analyze | cudss_setup ms | cudss_setup % analyze | cudss_analysis ms | cudss_analysis % analyze |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_ACTIVSg25k | 0.933 | 1.5% | 9.340 | 14.9% | 52.25 | 83.6% | 1.586 | 2.5% | 50.66 | 81.0% |
| case_ACTIVSg70k | 2.471 | 1.4% | 32.04 | 18.3% | 140.4 | 80.3% | 1.675 | 1.0% | 138.7 | 79.3% |
| Base_Eastern_Interconnect_515GW | 5.098 | 2.4% | 41.42 | 19.7% | 164.1 | 77.9% | 1.808 | 0.9% | 162.2 | 77.0% |

## NR Solve Operators
| case | upload total ms | upload % solve | upload per call ms | upload count | mismatch total ms | mismatch % solve | mismatch per call ms | mismatch count | jacobian total ms | jacobian % solve | jacobian per call ms | jacobian count | linear_solve total ms | linear_solve % solve | linear_solve per call ms | linear_solve count | voltage_update total ms | voltage_update % solve | voltage_update per call ms | voltage_update count | download total ms | download % solve | download per call ms | download count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_ACTIVSg25k | 0.837 | 9.6% | 0.837 | 1.0 | 0.555 | 6.4% | 0.111 | 5.0 | 0.096 | 1.1% | 0.024 | 4.0 | 6.978 | 80.4% | 1.744 | 4.0 | 0.117 | 1.3% | 0.029 | 4.0 | 0.086 | 1.0% | 0.086 | 1.0 |
| case_ACTIVSg70k | 3.641 | 11.6% | 3.641 | 1.0 | 2.250 | 7.2% | 0.321 | 7.0 | 0.383 | 1.2% | 0.064 | 6.0 | 24.56 | 78.2% | 4.094 | 6.0 | 0.285 | 0.9% | 0.047 | 6.0 | 0.281 | 0.9% | 0.281 | 1.0 |
| Base_Eastern_Interconnect_515GW | 7.357 | 17.2% | 7.357 | 1.0 | 4.985 | 11.6% | 0.755 | 6.6 | 0.451 | 1.1% | 0.081 | 5.6 | 28.83 | 67.3% | 5.149 | 5.6 | 0.318 | 0.7% | 0.057 | 5.6 | 0.854 | 2.0% | 0.854 | 1.0 |

## cuDSS Solve Sub-Operators
| case | rhs_prepare ms | rhs_prepare % linear | rhs_prepare % solve | rhs_prepare per call ms | rhs_prepare count | factorization ms | factorization % linear | factorization % solve | factorization per call ms | factorization count | refactorization ms | refactorization % linear | refactorization % solve | refactorization per call ms | refactorization count | cudss_solve ms | cudss_solve % linear | cudss_solve % solve | cudss_solve per call ms | cudss_solve count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_ACTIVSg25k | 0.303 | 4.3% | 3.5% | 0.076 | 4.0 | 1.254 | 18.0% | 14.5% | 1.254 | 1.0 | 3.428 | 49.1% | 39.5% | 1.143 | 3.0 | 1.933 | 27.7% | 22.3% | 0.483 | 4.0 |
| case_ACTIVSg70k | 1.426 | 5.8% | 4.5% | 0.238 | 6.0 | 2.905 | 11.8% | 9.2% | 2.905 | 1.0 | 14.14 | 57.6% | 45.0% | 2.828 | 5.0 | 5.933 | 24.2% | 18.9% | 0.989 | 6.0 |
| Base_Eastern_Interconnect_515GW | 3.648 | 12.7% | 8.5% | 0.651 | 5.6 | 3.482 | 12.1% | 8.1% | 3.482 | 1.0 | 15.69 | 54.4% | 36.6% | 3.411 | 4.6 | 5.733 | 19.9% | 13.4% | 1.024 | 5.6 |

## Notes
- Analyze dominates elapsed: about 83-88% on the 25K+ cases, mostly cuDSS analysis/setup.
- Inside the NR solve phase, `NR.iteration.linear_solve` dominates: about 80% of solve time.
- CUDA edge Jacobian update itself is small in this run: 0.095 ms, 0.386 ms, 0.466 ms total for 25K, 70K, and Eastern Interconnect respectively.
- For the two largest cases, refactorization is the largest cuDSS sub-operator within the linear solve.
