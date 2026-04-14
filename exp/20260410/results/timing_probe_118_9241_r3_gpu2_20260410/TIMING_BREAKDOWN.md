# Timing Breakdown

- run: `timing_probe_118_9241_r3_gpu2_20260410`
- benchmark binary: `/workspace/cuPF/build/bench-cuda-timing/cupf_case_benchmark`
- units: `ms`
- warmup: see `manifest.json`
- measured repeats: see `manifest.json`
- gpu: `CUDA_VISIBLE_DEVICES=2`

## CUDA Analyze

| case | impl | analyze mean (ms) | cudssAnalysis (ms) | cudssAnalysis share | cudssCreate (ms) | cusparseSetup (ms) | initialFactorization (ms) | uploadJacobianMaps (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 118_ieee | cuda edge | 12.739 | 9.032 | 70.9% | 1.471 | 1.713 | 0.0863 | 0.0597 |
| 118_ieee | cuda vertex | 12.622 | 8.962 | 71.0% | 1.476 | 1.672 | 0.0850 | 0.0570 |
| 9241_pegase | cuda edge | 55.971 | 46.797 | 83.6% | 1.485 | 1.621 | 0.7407 | 0.3777 |
| 9241_pegase | cuda vertex | 56.265 | 46.827 | 83.2% | 1.482 | 1.614 | 0.7377 | 0.3897 |

## Whole Solve

| case | impl | solve mean (ms) |
|---|---:|---:|
| 118_ieee | cpu optimized | 0.5403 |
| 118_ieee | cuda edge | 1.125 |
| 118_ieee | cuda vertex | 1.113 |
| 9241_pegase | cpu optimized | 62.407 |
| 9241_pegase | cuda edge | 7.203 |
| 9241_pegase | cuda vertex | 7.228 |

## Linear Solve Per Iteration

| case | impl | solveLinearSystem avg (ms) | rhsPrepare avg (ms) | refactorization avg (ms) | solve avg (ms) | refac share | solve share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 118_ieee | cpu optimized | 0.0535 | n/a | 0.0306 | 0.0016 | 57.2% | 3.0% |
| 118_ieee | cuda edge | 0.1648 | 0.0080 | 0.0772 | 0.0693 | 46.8% | 42.1% |
| 118_ieee | cuda vertex | 0.1644 | 0.0081 | 0.0771 | 0.0691 | 46.9% | 42.0% |
| 9241_pegase | cpu optimized | 7.454 | n/a | 7.137 | 0.3079 | 95.7% | 4.1% |
| 9241_pegase | cuda edge | 0.8607 | 0.0084 | 0.5922 | 0.2498 | 68.8% | 29.0% |
| 9241_pegase | cuda vertex | 0.8545 | 0.0082 | 0.5867 | 0.2490 | 68.7% | 29.1% |
