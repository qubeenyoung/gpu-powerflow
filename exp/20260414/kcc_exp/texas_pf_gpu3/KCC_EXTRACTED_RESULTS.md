# Texas PF Extracted Results

- Result root: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/results`
- MAT dataset: `/workspace/datasets/texas_univ_cases/pf_dataset`; cuPF dumps: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/cupf_dumps`
- Cases: `case_ACTIVSg200`, `case_ACTIVSg500`, `MemphisCase2026_Mar7`, `case_ACTIVSg2000`, `Base_Florida_42GW`, `Texas7k_20220923`, `Base_Texas_66GW`, `Base_MIOHIN_76GW`, `Base_West_Interconnect_121GW`, `case_ACTIVSg25k`, `case_ACTIVSg70k`, `Base_Eastern_Interconnect_515GW`
- Measurement: warmup 1, repeats 10 unless noted; End-To-End Chain `cuda_edge` for 25K+ cases uses warmup 3, repeats 10
- CUDA runs: `CUDA_VISIBLE_DEVICES=3`, cuDSS MT enabled, host threads `AUTO`
- ND level: CLI option was not passed for the regenerated GPU runs

## Case Sizes
| case | buses | Ybus nnz | PV buses | PQ buses |
| --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 200 | 690 | 37 | 162 |
| case_ACTIVSg500 | 500 | 1668 | 55 | 444 |
| MemphisCase2026_Mar7 | 993 | 3669 | 187 | 805 |
| case_ACTIVSg2000 | 2000 | 7334 | 391 | 1608 |
| Base_Florida_42GW | 5658 | 21384 | 104 | 5553 |
| Texas7k_20220923 | 6717 | 24009 | 589 | 6127 |
| Base_Texas_66GW | 7336 | 27174 | 239 | 7096 |
| Base_MIOHIN_76GW | 10189 | 39589 | 190 | 9998 |
| Base_West_Interconnect_121GW | 20758 | 78550 | 766 | 19991 |
| case_ACTIVSg25k | 25000 | 85220 | 2752 | 22247 |
| case_ACTIVSg70k | 70000 | 236636 | 5894 | 64105 |
| Base_Eastern_Interconnect_515GW | 78478 | 294392 | 2038 | 76439 |

## Selected Five-Case Outputs
- Tables and figures for the requested five cases are in `KCC_SELECTED_5CASE_RESULTS.md`.
- Base Florida PYPOWER operator pie: `figures/base_florida_pypower_newtonpf_pie.png`
- CUDA edge analyze/solve stack by bus count: `figures/cuda_edge_analyze_solve_stack_by_bus.png`

## PYPOWER Newtonpf Operator Profile
| case | mismatch ms | mismatch % | jacobian ms | jacobian % | solve ms | solve % | update ms | update % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 0.117 | 2.1% | 4.232 | 74.9% | 1.238 | 21.9% | 0.054 | 1.0% |
| case_ACTIVSg500 | 0.174 | 1.5% | 7.335 | 64.2% | 3.794 | 33.2% | 0.115 | 1.0% |
| MemphisCase2026_Mar7 | 0.150 | 1.2% | 6.112 | 50.6% | 5.676 | 47.0% | 0.124 | 1.0% |
| case_ACTIVSg2000 | 0.267 | 0.8% | 13.02 | 37.9% | 20.76 | 60.4% | 0.312 | 0.9% |
| Base_Florida_42GW | 0.636 | 0.5% | 36.13 | 30.1% | 82.20 | 68.5% | 1.054 | 0.9% |
| Texas7k_20220923 | 0.551 | 0.5% | 30.31 | 28.4% | 75.09 | 70.2% | 0.925 | 0.9% |
| Base_Texas_66GW | 0.812 | 0.5% | 44.65 | 27.9% | 113.3 | 70.7% | 1.360 | 0.8% |
| Base_MIOHIN_76GW | 1.137 | 0.4% | 57.94 | 22.5% | 196.5 | 76.3% | 1.957 | 0.8% |
| Base_West_Interconnect_121GW | 2.139 | 0.5% | 112.1 | 24.3% | 342.5 | 74.4% | 3.845 | 0.8% |
| case_ACTIVSg25k | 2.334 | 0.5% | 127.9 | 28.4% | 316.3 | 70.1% | 4.586 | 1.0% |
| case_ACTIVSg70k | 8.685 | 0.4% | 545.8 | 24.3% | 1668.5 | 74.4% | 18.69 | 0.8% |
| Base_Eastern_Interconnect_515GW | 8.540 | 0.3% | 512.9 | 20.4% | 1974.1 | 78.6% | 17.16 | 0.7% |

Note: this table excludes `init_index`; the full runpf/newtonpf breakdown is in `tables/pypower_operator_pie.csv`.

## End-To-End Chain
| case | pypower ms | cpp naive ms | cpp optimized ms | cuda edge ms | cuda vs pypower | cuda vs cpp opt |
| --- | --- | --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 7.837 | 1.208 | 0.413 | 12.01 | 0.65x | 0.03x |
| case_ACTIVSg500 | 13.93 | 4.358 | 1.461 | 14.97 | 0.93x | 0.10x |
| MemphisCase2026_Mar7 | 15.83 | 6.731 | 3.143 | 15.39 | 1.03x | 0.20x |
| case_ACTIVSg2000 | 39.36 | 23.98 | 10.83 | 26.65 | 1.48x | 0.41x |
| Base_Florida_42GW | 130.2 | 92.74 | 37.39 | 38.68 | 3.37x | 0.97x |
| Texas7k_20220923 | 118.1 | 83.37 | 36.32 | 62.88 | 1.88x | 0.58x |
| Base_Texas_66GW | 172.0 | 127.6 | 50.75 | 38.77 | 4.44x | 1.31x |
| Base_MIOHIN_76GW | 276.4 | 216.5 | 85.00 | 53.97 | 5.12x | 1.57x |
| Base_West_Interconnect_121GW | 491.6 | 406.0 | 158.7 | 98.85 | 4.97x | 1.61x |
| case_ACTIVSg25k | 485.0 | 425.9 | 143.0 | 71.43 | 6.79x | 2.00x |
| case_ACTIVSg70k | 2348.9 | 2472.9 | 778.5 | 210.1 | 11.18x | 3.70x |
| Base_Eastern_Interconnect_515GW | 2652.1 | 2874.1 | 979.7 | 223.6 | 11.86x | 4.38x |

Note: the 25K+ `cuda_edge` end-to-end values above are updated from `results/end2end_cuda_edge_25k_plus_warmup3`; the other end-to-end values remain from `results/end2end_main_chain`.

## CUDA Edge Ablation
| case | profile | elapsed mean ms | solve mean ms | elapsed / full | solve / full |
| --- | --- | --- | --- | --- | --- |
| case_ACTIVSg200 | full | 12.05 | 0.635 | 1.00x | 1.00x |
| case_ACTIVSg200 | w/o cuDSS | 1.473 | 1.143 | 0.12x | 1.80x |
| case_ACTIVSg200 | w/o Jacobian | 12.28 | 0.955 | 1.02x | 1.50x |
| case_ACTIVSg200 | w/o mixed precision | 13.51 | 0.756 | 1.12x | 1.19x |
| case_ACTIVSg500 | full | 14.43 | 1.151 | 1.00x | 1.00x |
| case_ACTIVSg500 | w/o cuDSS | 4.680 | 4.255 | 0.32x | 3.70x |
| case_ACTIVSg500 | w/o Jacobian | 14.40 | 2.155 | 1.00x | 1.87x |
| case_ACTIVSg500 | w/o mixed precision | 13.72 | 1.385 | 0.95x | 1.20x |
| MemphisCase2026_Mar7 | full | 15.31 | 1.292 | 1.00x | 1.00x |
| MemphisCase2026_Mar7 | w/o cuDSS | 6.797 | 6.214 | 0.44x | 4.81x |
| MemphisCase2026_Mar7 | w/o Jacobian | 21.23 | 2.609 | 1.39x | 2.02x |
| MemphisCase2026_Mar7 | w/o mixed precision | 15.83 | 1.539 | 1.03x | 1.19x |
| case_ACTIVSg2000 | full | 21.86 | 2.344 | 1.00x | 1.00x |
| case_ACTIVSg2000 | w/o cuDSS | 24.26 | 23.23 | 1.11x | 9.91x |
| case_ACTIVSg2000 | w/o Jacobian | 26.63 | 6.028 | 1.22x | 2.57x |
| case_ACTIVSg2000 | w/o mixed precision | 20.43 | 3.095 | 0.93x | 1.32x |
| Base_Florida_42GW | full | 28.25 | 3.672 | 1.00x | 1.00x |
| Base_Florida_42GW | w/o cuDSS | 88.67 | 86.36 | 3.14x | 23.52x |
| Base_Florida_42GW | w/o Jacobian | 52.30 | 20.18 | 1.85x | 5.49x |
| Base_Florida_42GW | w/o mixed precision | 30.05 | 5.542 | 1.06x | 1.51x |
| Texas7k_20220923 | full | 29.19 | 3.195 | 1.00x | 1.00x |
| Texas7k_20220923 | w/o cuDSS | 81.43 | 78.93 | 2.79x | 24.70x |
| Texas7k_20220923 | w/o Jacobian | 44.19 | 17.50 | 1.51x | 5.48x |
| Texas7k_20220923 | w/o mixed precision | 33.97 | 4.854 | 1.16x | 1.52x |
| Base_Texas_66GW | full | 32.51 | 4.554 | 1.00x | 1.00x |
| Base_Texas_66GW | w/o cuDSS | 122.4 | 118.6 | 3.76x | 26.05x |
| Base_Texas_66GW | w/o Jacobian | 54.03 | 25.06 | 1.66x | 5.50x |
| Base_Texas_66GW | w/o mixed precision | 35.57 | 7.088 | 1.09x | 1.56x |
| Base_MIOHIN_76GW | full | 42.30 | 5.508 | 1.00x | 1.00x |
| Base_MIOHIN_76GW | w/o cuDSS | 205.2 | 199.3 | 4.85x | 36.18x |
| Base_MIOHIN_76GW | w/o Jacobian | 74.60 | 36.12 | 1.76x | 6.56x |
| Base_MIOHIN_76GW | w/o mixed precision | 45.32 | 8.509 | 1.07x | 1.54x |
| Base_West_Interconnect_121GW | full | 73.35 | 12.26 | 1.00x | 1.00x |
| Base_West_Interconnect_121GW | w/o cuDSS | 525.9 | 467.3 | 7.17x | 38.11x |
| Base_West_Interconnect_121GW | w/o Jacobian | 148.3 | 73.20 | 2.02x | 5.97x |
| Base_West_Interconnect_121GW | w/o mixed precision | 75.79 | 12.20 | 1.03x | 1.00x |
| case_ACTIVSg25k | full | 76.75 | 8.611 | 1.00x | 1.00x |
| case_ACTIVSg25k | w/o cuDSS | 382.6 | 370.9 | 4.99x | 43.07x |
| case_ACTIVSg25k | w/o Jacobian | 156.8 | 76.85 | 2.04x | 8.92x |
| case_ACTIVSg25k | w/o mixed precision | 82.40 | 15.73 | 1.07x | 1.83x |
| case_ACTIVSg70k | full | 202.2 | 31.41 | 1.00x | 1.00x |
| case_ACTIVSg70k | w/o cuDSS | 2537.0 | 2296.7 | 12.55x | 73.13x |
| case_ACTIVSg70k | w/o Jacobian | 591.5 | 407.4 | 2.93x | 12.97x |
| case_ACTIVSg70k | w/o mixed precision | 221.6 | 49.83 | 1.10x | 1.59x |
| Base_Eastern_Interconnect_515GW | full | 228.1 | 33.98 | 1.00x | 1.00x |
| Base_Eastern_Interconnect_515GW | w/o cuDSS | 2736.8 | 2688.9 | 12.00x | 79.12x |
| Base_Eastern_Interconnect_515GW | w/o Jacobian | 602.1 | 397.8 | 2.64x | 11.71x |
| Base_Eastern_Interconnect_515GW | w/o mixed precision | 248.6 | 50.36 | 1.09x | 1.48x |

## CUDA Ablation Operator Snapshot
| case | profile | jacobian ms | linear solve ms | factor ms | refactor ms | cudss solve ms | cpu naive J ms | cpu SuperLU ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_ACTIVSg25k | full | 0.095 | 6.925 | 1.257 | 3.423 | 1.910 | - | - |
| case_ACTIVSg25k | w/o cuDSS | 0.126 | 368.9 | - | - | - | - | 345.4 |
| case_ACTIVSg25k | w/o Jacobian | 65.56 | 7.221 | 1.361 | 3.436 | 1.946 | 48.34 | - |
| case_ACTIVSg25k | w/o mixed precision | 0.299 | 12.24 | 2.113 | 6.089 | 3.150 | - | - |
| case_ACTIVSg70k | full | 0.381 | 24.55 | 2.901 | 14.12 | 5.945 | - | - |
| case_ACTIVSg70k | w/o cuDSS | 0.463 | 2289.6 | - | - | - | - | 2186.2 |
| case_ACTIVSg70k | w/o Jacobian | 354.7 | 29.84 | 3.197 | 14.19 | 6.134 | 269.0 | - |
| case_ACTIVSg70k | w/o mixed precision | 0.962 | 40.93 | 4.945 | 24.29 | 9.107 | - | - |
| Base_Eastern_Interconnect_515GW | full | 0.451 | 26.52 | 3.418 | 15.56 | 5.805 | - | - |
| Base_Eastern_Interconnect_515GW | w/o cuDSS | 0.459 | 2682.6 | - | - | - | - | 2573.9 |
| Base_Eastern_Interconnect_515GW | w/o Jacobian | 350.4 | 26.36 | 3.645 | 13.59 | 5.340 | 267.8 | - |
| Base_Eastern_Interconnect_515GW | w/o mixed precision | 0.972 | 38.03 | 5.634 | 21.99 | 7.591 | - | - |

Full operator trace for all cases/profiles is in `tables/cuda_edge_ablation_operator_breakdown.csv`.

## Edge Vs Vertex Jacobian Update
| case | edge J ms | vertex J ms | edge speedup | edge per call ms | vertex per call ms |
| --- | --- | --- | --- | --- | --- |
| case_ACTIVSg200 | 0.026 | 0.023 | 0.88x | 0.013 | 0.011 |
| case_ACTIVSg500 | 0.033 | 0.036 | 1.08x | 0.011 | 0.012 |
| MemphisCase2026_Mar7 | 0.023 | 0.028 | 1.21x | 0.012 | 0.014 |
| case_ACTIVSg2000 | 0.034 | 0.040 | 1.19x | 0.011 | 0.013 |
| Base_Florida_42GW | 0.057 | 0.080 | 1.41x | 0.014 | 0.020 |
| Texas7k_20220923 | 0.043 | 0.064 | 1.51x | 0.014 | 0.021 |
| Base_Texas_66GW | 0.057 | 0.104 | 1.82x | 0.014 | 0.026 |
| Base_MIOHIN_76GW | 0.081 | 0.111 | 1.36x | 0.020 | 0.028 |
| Base_West_Interconnect_121GW | 0.106 | 0.185 | 1.75x | 0.023 | 0.044 |
| case_ACTIVSg25k | 0.103 | 0.200 | 1.93x | 0.026 | 0.050 |
| case_ACTIVSg70k | 0.380 | 0.781 | 2.05x | 0.063 | 0.130 |
| Base_Eastern_Interconnect_515GW | 0.462 | 0.802 | 1.74x | 0.080 | 0.151 |

## Validation
| run | profile | success | rows | final mismatch max |
| --- | --- | --- | --- | --- |
| pypower_operator_profile | pypower | True | 120 | 1.0249396060157314e-09 |
| end2end_main_chain | cpp | True | 120 | 1.025027793975e-09 |
| end2end_main_chain | cpp_naive | True | 120 | 1.024859125266e-09 |
| end2end_main_chain | cuda_edge | True | 120 | 8.048070210265e-09 |
| end2end_main_chain | pypower | True | 120 | 1.0249396060157314e-09 |
| cuda_edge_ablation_operators | cuda_edge | True | 120 | 9.926894915656e-09 |
| cuda_edge_ablation_operators | cuda_fp64_edge | True | 120 | 1.024971697319e-09 |
| cuda_edge_ablation_operators | cuda_wo_cudss | True | 120 | 8.802585648874e-09 |
| cuda_edge_ablation_operators | cuda_wo_jacobian | True | 120 | 9.230374777014e-09 |
| jacobian_edge_vs_vertex | cuda_edge | True | 120 | 7.772284593699e-09 |
| jacobian_edge_vs_vertex | cuda_vertex | True | 120 | 8.781567393476e-09 |
