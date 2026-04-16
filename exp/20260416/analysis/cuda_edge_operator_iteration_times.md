# CUDA edge operator time per NR iteration

Source: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/results/cuda_edge_ablation_operators/raw/operators/cuda_edge`

Values are the mean over successful `run_*.json` files. Each operator time is normalized as `NR.iteration.<op>.total_sec / NR.iteration.total.count`, in milliseconds. Percentages are relative to normalized `NR.iteration.total`.

| case | buses | iters | mismatch ms (%) | jacobian ms (%) | linear_solve ms (%) | voltage_update ms (%) | non-solve ms (%) | iter_total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | 200 | 3.0 | 0.0263 (14.5) | 0.0087 (4.8) | 0.1322 (72.7) | 0.0134 (7.4) | 0.0484 (26.6) | 0.1819 |
| case_ACTIVSg500 | 500 | 4.0 | 0.0301 (11.6) | 0.0099 (3.8) | 0.2019 (78.0) | 0.0159 (6.1) | 0.0559 (21.6) | 0.2589 |
| MemphisCase2026_Mar7 | 993 | 3.0 | 0.0352 (9.2) | 0.0087 (2.3) | 0.3220 (84.5) | 0.0139 (3.6) | 0.0579 (15.2) | 0.3810 |
| case_ACTIVSg2000 | 2000 | 4.0 | 0.0425 (8.0) | 0.0099 (1.9) | 0.4606 (86.9) | 0.0159 (3.0) | 0.0682 (12.9) | 0.5300 |
| Base_Florida_42GW | 5658 | 5.0 | 0.0414 (6.2) | 0.0112 (1.7) | 0.5994 (89.6) | 0.0159 (2.4) | 0.0686 (10.2) | 0.6692 |
| Texas7k_20220923 | 6717 | 4.0 | 0.0473 (6.7) | 0.0104 (1.5) | 0.6314 (89.5) | 0.0151 (2.1) | 0.0728 (10.3) | 0.7056 |
| Base_Texas_66GW | 7336 | 5.0 | 0.0481 (5.8) | 0.0117 (1.4) | 0.7437 (90.2) | 0.0200 (2.4) | 0.0798 (9.7) | 0.8249 |
| Base_MIOHIN_76GW | 10189 | 5.0 | 0.0665 (6.6) | 0.0146 (1.5) | 0.9018 (90.1) | 0.0168 (1.7) | 0.0979 (9.8) | 1.0011 |
| Base_West_Interconnect_121GW | 20758 | 5.7 | 0.2240 (12.0) | 0.0238 (1.3) | 1.5979 (85.4) | 0.0250 (1.3) | 0.2728 (14.6) | 1.8720 |
| case_ACTIVSg25k | 25000 | 5.0 | 0.1093 (7.1) | 0.0189 (1.2) | 1.3851 (90.1) | 0.0233 (1.5) | 0.1515 (9.9) | 1.5379 |
| case_ACTIVSg70k | 70000 | 7.0 | 0.3154 (8.0) | 0.0545 (1.4) | 3.5069 (89.5) | 0.0406 (1.0) | 0.4105 (10.5) | 3.9188 |
| Base_Eastern_Interconnect_515GW | 78478 | 6.7 | 0.3591 (8.1) | 0.0673 (1.5) | 3.9527 (89.3) | 0.0466 (1.1) | 0.4729 (10.7) | 4.4272 |
