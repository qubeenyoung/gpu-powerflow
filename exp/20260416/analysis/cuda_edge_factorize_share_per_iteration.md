# CUDA edge factorize share per Newton iteration

Source: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/results/cuda_edge_ablation_operators/raw/operators/cuda_edge`

`factorize` here means `CUDA.solve.factorization32 + CUDA.solve.refactorization32`. Values are means over successful `run_*.json` files. Time is normalized by `NR.iteration.total.count`; share is relative to `NR.iteration.total.total_sec`.

| case | buses | iters | factorize ms / NR iter | share of NR iter | NR iter total ms | factor calls | refactor calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | 200 | 3.0 | 0.0711 | 39.1% | 0.1819 | 1.0 | 1.0 |
| case_ACTIVSg500 | 500 | 4.0 | 0.1249 | 48.3% | 0.2589 | 1.0 | 2.0 |
| MemphisCase2026_Mar7 | 993 | 3.0 | 0.2202 | 55.7% | 0.3810 | 1.0 | 1.0 |
| case_ACTIVSg2000 | 2000 | 4.0 | 0.3004 | 56.7% | 0.5300 | 1.0 | 2.0 |
| Base_Florida_42GW | 5658 | 5.0 | 0.3953 | 59.1% | 0.6692 | 1.0 | 3.0 |
| Texas7k_20220923 | 6717 | 4.0 | 0.4298 | 60.9% | 0.7056 | 1.0 | 2.0 |
| Base_Texas_66GW | 7336 | 5.0 | 0.5037 | 61.1% | 0.8249 | 1.0 | 3.0 |
| Base_MIOHIN_76GW | 10189 | 5.0 | 0.6285 | 62.8% | 1.0011 | 1.0 | 3.0 |
| Base_West_Interconnect_121GW | 20758 | 5.7 | 1.0850 | 57.0% | 1.8720 | 1.0 | 3.7 |
| case_ACTIVSg25k | 25000 | 5.0 | 0.9361 | 60.9% | 1.5379 | 1.0 | 3.0 |
| case_ACTIVSg70k | 70000 | 7.0 | 2.4322 | 62.2% | 3.9188 | 1.0 | 5.0 |
| Base_Eastern_Interconnect_515GW | 78478 | 6.7 | 2.8296 | 64.0% | 4.4272 | 1.0 | 4.7 |
