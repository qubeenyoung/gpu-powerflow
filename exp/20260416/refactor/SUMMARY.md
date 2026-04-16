# Refactorization Reorder Summary

- Dataset root: `/workspace/datasets/texas_univ_cases/cuPF_datasets`
- Profile: `cuda_edge` (`cuda_mixed_edge`, standard algorithm)
- Measurement mode: `operators`
- Algorithms: `DEFAULT`, `ALG_1`, `ALG_2`, `ALG_3`
- cuDSS MT: enabled, host threads `AUTO`

## Run Directories

- `DEFAULT`: `/workspace/exp/20260416/modified/results/texas_gpu3_mt_auto_r10`
- `ALG_1`: `/workspace/exp/20260416/refactor/results/texas_standard_reorder_alg1_operators_r10`
- `ALG_2`: `/workspace/exp/20260416/refactor/results/texas_standard_reorder_alg2_operators_r10`
- `ALG_3`: `/workspace/exp/20260416/refactor/results/texas_standard_reorder_alg3_operators_r10`

## Raw Operator Directories

- `DEFAULT`: `/workspace/exp/20260416/modified/results/texas_gpu3_mt_auto_r10/raw/operators/cuda_edge`
- `ALG_1`: `/workspace/exp/20260416/refactor/results/texas_standard_reorder_alg1_operators_r10/raw/operators/cuda_edge`
- `ALG_2`: `/workspace/exp/20260416/refactor/results/texas_standard_reorder_alg2_operators_r10/raw/operators/cuda_edge`
- `ALG_3`: `/workspace/exp/20260416/refactor/results/texas_standard_reorder_alg3_operators_r10/raw/operators/cuda_edge`

## Refactorization Focus

| case | DEFAULT refactor ms | ALG_1 refactor ms | ALG_2 refactor ms | ALG_3 refactor ms | DEFAULT factor ms | ALG_1 factor ms | ALG_2 factor ms | ALG_3 factor ms | DEFAULT solve32 ms | ALG_1 solve32 ms | ALG_2 solve32 ms | ALG_3 solve32 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base_Eastern_Interconnect_515GW | 3.285 | 55.32 | 55.32 | 73.78 | 3.453 | 930.8 | 931.9 | 73.15 | 1.236 | 92.73 | 92.93 | 6.927 |
| Base_Florida_42GW | 0.470 | 3.702 | 3.697 | 3.724 | 0.567 | 65.46 | 65.38 | 3.877 | 0.236 | 5.037 | 5.028 | 0.493 |
| Base_MIOHIN_76GW | 0.762 | 8.079 | 8.070 | 7.648 | 0.853 | 121.1 | 121.2 | 7.623 | 0.322 | 12.84 | 12.82 | 0.854 |
| Base_Texas_66GW | 0.595 | 4.506 | 4.502 | 5.560 | 0.690 | 84.72 | 85.41 | 5.585 | 0.280 | 7.088 | 7.076 | 0.667 |
| Base_West_Interconnect_121GW | 1.087 | 13.19 | 13.19 | 14.08 | 1.208 | 239.7 | 240.0 | 14.20 | 0.436 | 22.97 | 23.00 | 1.371 |
| MemphisCase2026_Mar7 | 0.208 | 0.825 | 0.823 | 0.771 | 0.307 | 10.07 | 9.989 | 1.164 | 0.131 | 0.938 | 0.933 | 0.182 |
| Texas7k_20220923 | 0.565 | 4.556 | 4.551 | 4.327 | 0.657 | 75.44 | 75.56 | 5.518 | 0.251 | 6.775 | 6.764 | 0.595 |
| case_ACTIVSg200 | 0.103 | 0.274 | 0.279 | 0.158 | 0.110 | 2.185 | 2.267 | 0.166 | 0.078 | 0.197 | 0.202 | 0.099 |
| case_ACTIVSg2000 | 0.351 | 1.884 | 1.881 | 1.428 | 0.439 | 20.76 | 20.73 | 1.565 | 0.173 | 1.749 | 1.744 | 0.263 |
| case_ACTIVSg25k | 1.133 | 12.65 | 12.65 | 17.55 | 1.279 | 270.6 | 270.5 | 17.21 | 0.485 | 22.72 | 22.75 | 1.668 |
| case_ACTIVSg500 | 0.116 | 0.465 | 0.463 | 0.266 | 0.205 | 5.293 | 5.229 | 0.291 | 0.079 | 0.389 | 0.386 | 0.134 |
| case_ACTIVSg70k | 2.803 | 39.28 | 39.28 | 53.95 | 2.920 | 788.7 | 788.5 | 55.86 | 1.172 | 72.52 | 72.66 | 5.856 |

## Reorder Ratios vs DEFAULT

### ALG_1 / DEFAULT

| case | elapsed ratio | analyze ratio | solve ratio | refactor ratio | factor ratio | cudss solve ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Base_Eastern_Interconnect_515GW | 8.02x | 0.78x | 43.89x | 16.84x | 269.54x | 75.03x |
| Base_Florida_42GW | 3.82x | 0.52x | 23.32x | 7.88x | 115.51x | 21.39x |
| Base_MIOHIN_76GW | 5.14x | 0.61x | 30.59x | 10.60x | 142.01x | 39.86x |
| Base_Texas_66GW | 4.34x | 0.65x | 24.68x | 7.57x | 122.87x | 25.35x |
| Base_West_Interconnect_121GW | 6.44x | 0.66x | 41.46x | 12.14x | 198.37x | 52.67x |
| MemphisCase2026_Mar7 | 1.16x | 0.32x | 11.22x | 3.97x | 32.77x | 7.13x |
| Texas7k_20220923 | 3.88x | 0.52x | 26.44x | 8.06x | 114.76x | 27.04x |
| case_ACTIVSg200 | 0.52x | 0.28x | 4.82x | 2.66x | 19.96x | 2.54x |
| case_ACTIVSg2000 | 1.97x | 0.37x | 14.54x | 5.38x | 47.30x | 10.10x |
| case_ACTIVSg25k | 5.48x | 0.65x | 35.61x | 11.16x | 211.60x | 46.83x |
| case_ACTIVSg500 | 0.87x | 0.30x | 7.76x | 4.00x | 25.86x | 4.89x |
| case_ACTIVSg70k | 7.55x | 0.61x | 37.07x | 14.01x | 270.08x | 61.88x |

- Geomean `ALG_1/DEFAULT` elapsed_ms: 3.08x
- Geomean `ALG_1/DEFAULT` analyze_ms: 0.50x
- Geomean `ALG_1/DEFAULT` solve_ms: 20.87x
- Geomean `ALG_1/DEFAULT` cuda_refactorization_ms: 7.60x

### ALG_2 / DEFAULT

| case | elapsed ratio | analyze ratio | solve ratio | refactor ratio | factor ratio | cudss solve ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Base_Eastern_Interconnect_515GW | 7.96x | 0.71x | 43.88x | 16.84x | 269.85x | 75.19x |
| Base_Florida_42GW | 3.76x | 0.48x | 23.17x | 7.87x | 115.37x | 21.35x |
| Base_MIOHIN_76GW | 5.07x | 0.56x | 30.46x | 10.59x | 142.22x | 39.79x |
| Base_Texas_66GW | 4.22x | 0.51x | 24.67x | 7.56x | 123.87x | 25.31x |
| Base_West_Interconnect_121GW | 6.52x | 0.65x | 42.09x | 12.13x | 198.62x | 52.73x |
| MemphisCase2026_Mar7 | 1.13x | 0.30x | 11.07x | 3.96x | 32.50x | 7.10x |
| Texas7k_20220923 | 3.81x | 0.46x | 26.33x | 8.05x | 114.94x | 27.00x |
| case_ACTIVSg200 | 0.53x | 0.29x | 5.03x | 2.71x | 20.71x | 2.60x |
| case_ACTIVSg2000 | 1.93x | 0.34x | 14.43x | 5.37x | 47.21x | 10.08x |
| case_ACTIVSg25k | 5.35x | 0.52x | 35.38x | 11.16x | 211.52x | 46.90x |
| case_ACTIVSg500 | 0.85x | 0.29x | 7.63x | 3.98x | 25.54x | 4.85x |
| case_ACTIVSg70k | 7.57x | 0.60x | 37.24x | 14.01x | 270.03x | 62.00x |

- Geomean `ALG_2/DEFAULT` elapsed_ms: 3.04x
- Geomean `ALG_2/DEFAULT` analyze_ms: 0.45x
- Geomean `ALG_2/DEFAULT` solve_ms: 20.87x
- Geomean `ALG_2/DEFAULT` cuda_refactorization_ms: 7.60x

### ALG_3 / DEFAULT

| case | elapsed ratio | analyze ratio | solve ratio | refactor ratio | factor ratio | cudss solve ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Base_Eastern_Interconnect_515GW | 4.44x | 3.04x | 11.34x | 22.46x | 21.18x | 5.60x |
| Base_Florida_42GW | 2.10x | 1.71x | 4.36x | 7.92x | 6.84x | 2.09x |
| Base_MIOHIN_76GW | 2.55x | 2.04x | 5.45x | 10.04x | 8.94x | 2.65x |
| Base_Texas_66GW | 2.51x | 2.04x | 5.11x | 9.34x | 8.10x | 2.39x |
| Base_West_Interconnect_121GW | 3.16x | 2.34x | 8.10x | 12.95x | 11.76x | 3.14x |
| MemphisCase2026_Mar7 | 1.23x | 1.14x | 2.29x | 3.71x | 3.79x | 1.38x |
| Texas7k_20220923 | 5.55x | 5.74x | 4.29x | 7.66x | 8.39x | 2.37x |
| case_ACTIVSg200 | 1.03x | 1.01x | 1.29x | 1.54x | 1.52x | 1.28x |
| case_ACTIVSg2000 | 1.47x | 1.31x | 2.78x | 4.07x | 3.57x | 1.52x |
| case_ACTIVSg25k | 3.30x | 2.67x | 7.23x | 15.48x | 13.46x | 3.44x |
| case_ACTIVSg500 | 1.21x | 1.17x | 1.66x | 2.29x | 1.42x | 1.69x |
| case_ACTIVSg70k | 4.36x | 3.19x | 9.31x | 19.25x | 19.13x | 5.00x |

- Geomean `ALG_3/DEFAULT` elapsed_ms: 2.38x
- Geomean `ALG_3/DEFAULT` analyze_ms: 2.01x
- Geomean `ALG_3/DEFAULT` solve_ms: 4.33x
- Geomean `ALG_3/DEFAULT` cuda_refactorization_ms: 7.38x


## Files

- `operator_refactor_comparison.csv`: per-case operator timing means for each algorithm
- `reorder_alg_comparison.csv`: pairwise candidate vs DEFAULT ratios
- `combined_summary_operators.csv`: raw operator rows with algorithm labels
