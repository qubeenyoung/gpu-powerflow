# cuPF vs PYPOWER End-to-End Analysis

- Baseline: PYPOWER end-to-end `elapsed_ms_mean` from `tables/end2end_main_chain.csv`.
- cuPF: `cuda_edge` end-to-end. For 25K+ cases, CUDA values use `results/end2end_cuda_edge_25k_plus_warmup3/aggregates_end2end.csv`.
- Speedup is `pypower_ms / cupf_ms`; larger is faster.
- `cuPF no-analyze ms` is `cuPF total ms - cuPF analyze ms`.

| case | buses | PYPOWER ms | cuPF total ms | total speedup | cuPF analyze ms | analyze % | cuPF no-analyze ms | no-analyze speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case_ACTIVSg200 | 200 | 7.837 | 12.01 | 0.65x | 11.44 | 95.3% | 0.568 | 13.80x |
| Base_Florida_42GW | 5658 | 130.2 | 38.68 | 3.37x | 35.09 | 90.7% | 3.584 | 36.33x |
| Texas7k_20220923 | 6717 | 118.1 | 62.88 | 1.88x | 58.44 | 92.9% | 4.438 | 26.62x |
| Base_MIOHIN_76GW | 10189 | 276.4 | 53.97 | 5.12x | 48.50 | 89.9% | 5.470 | 50.53x |
| Base_West_Interconnect_121GW | 20758 | 491.6 | 98.85 | 4.97x | 82.88 | 83.8% | 15.97 | 30.79x |
| Base_Eastern_Interconnect_515GW | 78478 | 2652.1 | 223.6 | 11.86x | 191.9 | 85.8% | 31.65 | 83.79x |

CSV: `tables/cupf_pypower_end2end_analyze_removed_speedup.csv`
