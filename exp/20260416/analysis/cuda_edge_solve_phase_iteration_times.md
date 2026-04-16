# CUDA edge solve phase time per NR iteration

Source: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/results/cuda_edge_ablation_operators/raw/operators/cuda_edge`

Values are means over successful `run_*.json` files, normalized by `NR.iteration.total.count`. Unit: ms / NR iteration. Raw timing separates `CUDA.solve.rhsPrepare`, `CUDA.solve.factorization32`, `CUDA.solve.refactorization32`, and `CUDA.solve.solve32`.

| case | buses | iters | rhs prep | factor | refactor | factor+refactor | cudss solve | subphase sum | NR linear_solve |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | 200 | 3.0 | 0.0085 | 0.0367 | 0.0344 | 0.0711 | 0.0516 | 0.1312 | 0.1322 |
| case_ACTIVSg500 | 500 | 4.0 | 0.0120 | 0.0642 | 0.0607 | 0.1249 | 0.0637 | 0.2006 | 0.2019 |
| MemphisCase2026_Mar7 | 993 | 3.0 | 0.0126 | 0.1509 | 0.0694 | 0.2202 | 0.0879 | 0.3208 | 0.3220 |
| case_ACTIVSg2000 | 2000 | 4.0 | 0.0201 | 0.1218 | 0.1786 | 0.3004 | 0.1386 | 0.4590 | 0.4606 |
| Base_Florida_42GW | 5658 | 5.0 | 0.0219 | 0.1151 | 0.2802 | 0.3953 | 0.1798 | 0.5970 | 0.5994 |
| Texas7k_20220923 | 6717 | 4.0 | 0.0225 | 0.1608 | 0.2691 | 0.4298 | 0.1767 | 0.6291 | 0.6314 |
| Base_Texas_66GW | 7336 | 5.0 | 0.0255 | 0.1397 | 0.3640 | 0.5037 | 0.2118 | 0.7409 | 0.7437 |
| Base_MIOHIN_76GW | 10189 | 5.0 | 0.0322 | 0.1748 | 0.4537 | 0.6285 | 0.2375 | 0.8983 | 0.9018 |
| Base_West_Interconnect_121GW | 20758 | 5.7 | 0.1450 | 0.3389 | 0.7461 | 1.0850 | 0.3615 | 1.5915 | 1.5979 |
| case_ACTIVSg25k | 25000 | 5.0 | 0.0596 | 0.2514 | 0.6846 | 0.9361 | 0.3820 | 1.3777 | 1.3851 |
| case_ACTIVSg70k | 70000 | 7.0 | 0.2027 | 0.4144 | 2.0178 | 2.4322 | 0.8493 | 3.4842 | 3.5069 |
| Base_Eastern_Interconnect_515GW | 78478 | 6.7 | 0.2287 | 0.5126 | 2.3169 | 2.8296 | 0.8656 | 3.9239 | 3.9527 |
