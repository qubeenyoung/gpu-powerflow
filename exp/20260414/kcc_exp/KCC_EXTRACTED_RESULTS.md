# KCC Extracted Results

- Result root: `/workspace/exp/20260414/kcc_exp/results`
- Dataset: `/workspace/datasets/cuPF_benchmark_dumps`
- Cases: `case30_ieee`, `case118_ieee`, `case793_goc`, `case1354_pegase`, `case2746wop_k`, `case4601_goc`, `case8387_pegase`, `case9241_pegase`
- Measurement: warmup 1, repeats 10
- CUDA runs: `CUDA_VISIBLE_DEVICES=1`, cuDSS MT enabled, host threads `AUTO`
- ND level: CLI option was not passed for the regenerated GPU runs

## PYPOWER Newtonpf Operator Profile
| case | mismatch ms | mismatch % | jacobian ms | jacobian % | solve ms | solve % | update ms | update % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case30_ieee | 0.160 | 2.0% | 7.167 | 91.3% | 0.452 | 5.8% | 0.061 | 0.8% |
| case118_ieee | 0.175 | 2.0% | 7.597 | 85.7% | 1.004 | 11.3% | 0.081 | 0.9% |
| case793_goc | 0.244 | 1.3% | 11.05 | 58.1% | 7.503 | 39.5% | 0.208 | 1.1% |
| case1354_pegase | 0.426 | 1.2% | 17.82 | 50.5% | 16.58 | 47.0% | 0.430 | 1.2% |
| case2746wop_k | 0.367 | 0.7% | 20.44 | 41.2% | 28.21 | 56.9% | 0.532 | 1.1% |
| case4601_goc | 0.640 | 0.5% | 37.80 | 28.9% | 91.17 | 69.7% | 1.135 | 0.9% |
| case8387_pegase | 1.293 | 0.6% | 76.64 | 33.7% | 146.8 | 64.6% | 2.601 | 1.1% |
| case9241_pegase | 1.613 | 0.5% | 97.08 | 32.3% | 197.9 | 65.9% | 3.835 | 1.3% |

Note: this table excludes `init_index`; the full runpf/newtonpf breakdown is in `tables/pypower_operator_pie.csv`.

## End-To-End Chain
| case | pypower ms | cpp naive ms | cpp optimized ms | cuda edge ms | cuda vs pypower | cuda vs cpp opt |
| --- | --- | --- | --- | --- | --- | --- |
| case30_ieee | 10.01 | 0.472 | 0.072 | 11.37 | 0.88x | 0.01x |
| case118_ieee | 12.02 | 1.074 | 0.231 | 11.75 | 1.02x | 0.02x |
| case793_goc | 21.95 | 9.220 | 2.947 | 14.88 | 1.47x | 0.20x |
| case1354_pegase | 38.39 | 19.38 | 9.811 | 17.62 | 2.18x | 0.56x |
| case2746wop_k | 54.78 | 33.36 | 11.47 | 19.93 | 2.75x | 0.58x |
| case4601_goc | 139.8 | 102.0 | 42.37 | 27.05 | 5.17x | 1.57x |
| case8387_pegase | 240.8 | 177.1 | 57.04 | 45.15 | 5.33x | 1.26x |
| case9241_pegase | 316.9 | 241.1 | 75.09 | 40.82 | 7.76x | 1.84x |

## CUDA Edge Ablation
| case | profile | elapsed mean ms | solve mean ms | elapsed / full | solve / full |
| --- | --- | --- | --- | --- | --- |
| case30_ieee | full | 11.30 | 0.749 | 1.00x | 1.00x |
| case30_ieee | w/o cuDSS | 0.729 | 0.510 | 0.06x | 0.68x |
| case30_ieee | w/o Jacobian | 11.30 | 0.868 | 1.00x | 1.16x |
| case30_ieee | w/o mixed precision | 12.37 | 0.870 | 1.09x | 1.16x |
| case118_ieee | full | 11.67 | 0.861 | 1.00x | 1.00x |
| case118_ieee | w/o cuDSS | 1.273 | 1.018 | 0.11x | 1.18x |
| case118_ieee | w/o Jacobian | 11.94 | 1.177 | 1.02x | 1.37x |
| case118_ieee | w/o mixed precision | 25.97 | 1.005 | 2.23x | 1.17x |
| case793_goc | full | 14.62 | 1.550 | 1.00x | 1.00x |
| case793_goc | w/o cuDSS | 9.144 | 8.679 | 0.63x | 5.60x |
| case793_goc | w/o Jacobian | 16.62 | 3.709 | 1.14x | 2.39x |
| case793_goc | w/o mixed precision | 14.68 | 1.577 | 1.00x | 1.02x |
| case1354_pegase | full | 44.10 | 2.155 | 1.00x | 1.00x |
| case1354_pegase | w/o cuDSS | 18.21 | 17.52 | 0.41x | 8.13x |
| case1354_pegase | w/o Jacobian | 28.19 | 6.351 | 0.64x | 2.95x |
| case1354_pegase | w/o mixed precision | 17.44 | 2.720 | 0.40x | 1.26x |
| case2746wop_k | full | 19.76 | 2.418 | 1.00x | 1.00x |
| case2746wop_k | w/o cuDSS | 32.13 | 31.08 | 1.63x | 12.85x |
| case2746wop_k | w/o Jacobian | 25.72 | 8.486 | 1.30x | 3.51x |
| case2746wop_k | w/o mixed precision | 20.60 | 3.369 | 1.04x | 1.39x |
| case4601_goc | full | 31.94 | 4.238 | 1.00x | 1.00x |
| case4601_goc | w/o cuDSS | 98.81 | 97.03 | 3.09x | 22.89x |
| case4601_goc | w/o Jacobian | 56.32 | 23.51 | 1.76x | 5.55x |
| case4601_goc | w/o mixed precision | 30.32 | 6.519 | 0.95x | 1.54x |
| case8387_pegase | full | 43.22 | 6.137 | 1.00x | 1.00x |
| case8387_pegase | w/o cuDSS | 160.7 | 156.6 | 3.72x | 25.51x |
| case8387_pegase | w/o Jacobian | 78.06 | 44.44 | 1.81x | 7.24x |
| case8387_pegase | w/o mixed precision | 51.75 | 9.086 | 1.20x | 1.48x |
| case9241_pegase | full | 71.92 | 7.410 | 1.00x | 1.00x |
| case9241_pegase | w/o cuDSS | 222.8 | 217.2 | 3.10x | 29.31x |
| case9241_pegase | w/o Jacobian | 98.84 | 62.34 | 1.37x | 8.41x |
| case9241_pegase | w/o mixed precision | 47.78 | 12.34 | 0.66x | 1.67x |

## CUDA Ablation Operator Snapshot
| case | profile | jacobian ms | linear solve ms | factor ms | refactor ms | cudss solve ms | cpu naive J ms | cpu SuperLU ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case4601_goc | full | 0.062 | 3.574 | 0.559 | 1.804 | 1.074 | - | - |
| case4601_goc | w/o cuDSS | 0.069 | 96.40 | - | - | - | - | 92.10 |
| case4601_goc | w/o Jacobian | 16.83 | 4.762 | 1.014 | 1.913 | 1.146 | 11.53 | - |
| case4601_goc | w/o mixed precision | 0.173 | 5.761 | 0.856 | 3.012 | 1.759 | - | - |
| case8387_pegase | full | 0.089 | 5.116 | 0.664 | 2.785 | 1.445 | - | - |
| case8387_pegase | w/o cuDSS | 0.099 | 155.6 | - | - | - | - | 146.3 |
| case8387_pegase | w/o Jacobian | 37.47 | 5.232 | 0.665 | 2.766 | 1.445 | 25.55 | - |
| case8387_pegase | w/o mixed precision | 0.370 | 7.880 | 0.989 | 4.477 | 2.179 | - | - |
| case9241_pegase | full | 0.115 | 6.290 | 0.708 | 3.582 | 1.693 | - | - |
| case9241_pegase | w/o cuDSS | 0.149 | 215.5 | - | - | - | - | 200.5 |
| case9241_pegase | w/o Jacobian | 51.63 | 8.050 | 0.756 | 3.649 | 1.755 | 34.68 | - |
| case9241_pegase | w/o mixed precision | 0.435 | 10.14 | 1.081 | 5.882 | 2.571 | - | - |

Full operator trace for all cases/profiles is in `tables/cuda_edge_ablation_operator_breakdown.csv`.

## Edge Vs Vertex Jacobian Update
| case | edge J ms | vertex J ms | edge speedup | edge per call ms | vertex per call ms |
| --- | --- | --- | --- | --- | --- |
| case30_ieee | 0.040 | 0.042 | 1.05x | 0.010 | 0.010 |
| case118_ieee | 0.044 | 0.045 | 1.02x | 0.011 | 0.011 |
| case793_goc | 0.055 | 0.071 | 1.29x | 0.011 | 0.015 |
| case1354_pegase | 0.063 | 0.074 | 1.18x | 0.013 | 0.015 |
| case2746wop_k | 0.063 | 0.060 | 0.94x | 0.016 | 0.015 |
| case4601_goc | 0.062 | 0.088 | 1.42x | 0.012 | 0.018 |
| case8387_pegase | 0.090 | 0.145 | 1.62x | 0.015 | 0.024 |
| case9241_pegase | 0.118 | 0.201 | 1.70x | 0.017 | 0.029 |

## Validation
| run | profile | success | rows | final mismatch max |
| --- | --- | --- | --- | --- |
| pypower_operator_profile | pypower | True | 80 | 5.425149218751812e-10 |
| end2end_main_chain | cpp | True | 80 | 5.425693783145e-10 |
| end2end_main_chain | cpp_naive | True | 80 | 5.425077054255e-10 |
| end2end_main_chain | cuda_edge | True | 80 | 3.688385903991e-09 |
| end2end_main_chain | pypower | True | 80 | 5.425149218751812e-10 |
| cuda_edge_ablation_operators | cuda_edge | True | 80 | 4.537929587656e-09 |
| cuda_edge_ablation_operators | cuda_fp64_edge | True | 80 | 5.425495608335e-10 |
| cuda_edge_ablation_operators | cuda_wo_cudss | True | 80 | 9.77692817851e-09 |
| cuda_edge_ablation_operators | cuda_wo_jacobian | True | 80 | 8.022681158293e-09 |
| jacobian_edge_vs_vertex | cuda_edge | True | 80 | 3.177430540902e-09 |
| jacobian_edge_vs_vertex | cuda_vertex | True | 80 | 9.688938207297e-09 |
