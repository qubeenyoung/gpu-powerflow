# End-to-End CUDA Edge 25K+ Warmup 3

- Source run: `/workspace/exp/20260414/kcc_exp/texas_pf_gpu3/results/end2end_cuda_edge_25k_plus_warmup3`
- Baseline for comparison: `KCC_EXTRACTED_RESULTS.md` / `tables/end2end_main_chain.csv` warmup 1, repeats 10
- New measurement: end-to-end only, `cuda_edge` only, warmup 3, repeats 10
- CUDA: `CUDA_VISIBLE_DEVICES=3`, cuDSS MT host threads `AUTO`
- ND level: CLI option was not passed (`--cudss-nd-nlevels` absent from manifest commands)

## Comparison
| case | old elapsed ms | old stdev | new elapsed ms | new stdev | delta | new analyze ms | new solve ms | iter mean | success | max mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_ACTIVSg25k | 92.06 | 16.17 | 71.43 | 3.002 | -22.4% | 62.96 | 8.471 | 5.000 | True | 6.388e-09 |
| case_ACTIVSg70k | 214.6 | 19.15 | 210.1 | 28.85 | -2.1% | 180.1 | 29.99 | 7.000 | True | 4.451e-09 |
| Base_Eastern_Interconnect_515GW | 272.7 | 22.59 | 223.6 | 9.122 | -18.0% | 191.9 | 31.65 | 6.700 | True | 9.360e-09 |

## New Repeats
| case | elapsed ms per repeat |
| --- | --- |
| case_ACTIVSg25k | 70.71, 70.10, 71.40, 79.71, 69.71, 71.33, 69.56, 71.35, 69.55, 70.92 |
| case_ACTIVSg70k | 186.1, 226.9, 181.8, 274.8, 191.5, 209.0, 201.1, 194.7, 197.0, 238.6 |
| Base_Eastern_Interconnect_515GW | 222.6, 215.7, 242.3, 205.8, 226.6, 224.5, 222.7, 225.8, 223.8, 225.8 |

## Takeaway
- `case_ACTIVSg25k`: 92.06 ms -> 71.43 ms (-22.4%).
- `case_ACTIVSg70k`: 214.6 ms -> 210.1 ms (-2.1%).
- `Base_Eastern_Interconnect_515GW`: 272.7 ms -> 223.6 ms (-18.0%).
