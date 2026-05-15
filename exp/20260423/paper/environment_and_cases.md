# Experiment Environment and Test Cases

Date: 2026-04-23

## Hardware and Software

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| GPU memory | 24 GiB / 24576 MiB |
| CPU | AMD Ryzen 5 5600 6-Core Processor |
| CPU cores / threads | 6 cores / 12 threads |
| System memory | 31.26 GiB visible to OS |
| NVIDIA driver | 580.126.09 |
| CUDA shown by driver | 13.0 |
| CUDA Toolkit / nvcc | 12.8.93 |
| CUDA compiler path used by cuPF builds | `/usr/local/cuda/bin/nvcc` |

Notes:

- For paper environment tables, use CUDA Toolkit `12.8` unless the table separately reports the driver-supported CUDA API version.
- The driver reports CUDA `13.0`, while the CUDA compiler used for the cuPF build is nvcc `12.8.93`.

## Test Case Counts

| dataset | experiment | total cases | used in final aggregate | note |
|---|---|---:|---:|---|
| MATPOWER | speed comparison | 78 | 77 | `case16am` was excluded from the speed aggregate. |
| MATPOWER | precision comparison | 78 | 78 | FP32, Mixed, and FP64 were measured, giving 234 case-profile rows. |
| Texas | speed comparison | 12 | 12 | Partial GPU run: batch 1 and 4 completed; batch 16 did not produce results. |

## MATPOWER Cases

Count: 78

- `case10ba`
- `case118`
- `case118zh`
- `case1197`
- `case12da`
- `case1354pegase`
- `case13659pegase`
- `case136ma`
- `case14`
- `case141`
- `case145`
- `case15da`
- `case15nbr`
- `case16am`
- `case16ci`
- `case17me`
- `case18`
- `case1888rte`
- `case18nbr`
- `case1951rte`
- `case22`
- `case2383wp`
- `case24_ieee_rts`
- `case2736sp`
- `case2737sop`
- `case2746wop`
- `case2746wp`
- `case2848rte`
- `case2868rte`
- `case2869pegase`
- `case28da`
- `case30`
- `case300`
- `case3012wp`
- `case30Q`
- `case30pwl`
- `case3120sp`
- `case3375wp`
- `case33bw`
- `case33mg`
- `case34sa`
- `case38si`
- `case39`
- `case4_dist`
- `case4gs`
- `case5`
- `case51ga`
- `case51he`
- `case533mt_hi`
- `case533mt_lo`
- `case57`
- `case59`
- `case60nordic`
- `case6468rte`
- `case6470rte`
- `case6495rte`
- `case6515rte`
- `case69`
- `case6ww`
- `case70da`
- `case74ds`
- `case8387pegase`
- `case85`
- `case89pegase`
- `case9`
- `case9241pegase`
- `case94pi`
- `case9Q`
- `case9target`
- `case_ACTIVSg10k`
- `case_ACTIVSg200`
- `case_ACTIVSg2000`
- `case_ACTIVSg25k`
- `case_ACTIVSg500`
- `case_ACTIVSg70k`
- `case_RTS_GMLC`
- `case_SyntheticUSA`
- `case_ieee30`

## Texas Cases

Count: 12

- `Base_Eastern_Interconnect_515GW`
- `Base_Florida_42GW`
- `Base_MIOHIN_76GW`
- `Base_Texas_66GW`
- `Base_West_Interconnect_121GW`
- `MemphisCase2026_Mar7`
- `Texas7k_20220923`
- `case_ACTIVSg200`
- `case_ACTIVSg2000`
- `case_ACTIVSg25k`
- `case_ACTIVSg500`
- `case_ACTIVSg70k`

## Source Files

- MATPOWER speed cases: `paper/speed_matpower_final/matpower_gpu_mt_b1_w3_r10_20260423/manifest.json`
- MATPOWER speed aggregate: `paper/speed_matpower_final/matpower_comparison_mt_20260423/solve_wide.csv`
- MATPOWER precision aggregate: `paper/precision_matpower_final/matpower_precision_fp32_mixed_fp64_fair_analysis_20260423/fair_precision_per_case.csv`
- Texas cases: `paper/speed_texas_partial/texas_cpu_ref_cpu_b1_w0_r10_20260423/manifest.json`
- Texas speed aggregate: `paper/speed_texas_partial/texas_comparison_mt_partial_20260423/solve_wide.csv`
