# Timing Breakdown

- run: `timing_probe_118_9241_r3_20260410`
- benchmark binary: `/workspace/cuPF/build/bench-cuda-timing/cupf_case_benchmark`
- units: `ms`
- warmup: see `manifest.json`
- measured repeats: see `manifest.json`
- gpu: `CUDA_VISIBLE_DEVICES=3`

## CUDA Analyze

| case | impl | analyze mean (ms) | cudssAnalysis (ms) | cudssAnalysis share | cudssCreate (ms) | cusparseSetup (ms) | initialFactorization (ms) | uploadJacobianMaps (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 118_ieee | cuda edge | 55.273 | 34.811 | 63.0% | 2.598 | 2.438 | 1.611 | 5.005 |
| 118_ieee | cuda vertex | 12.556 | 8.966 | 71.4% | 1.490 | 1.581 | 0.0853 | 0.0567 |
| 9241_pegase | cuda edge | 144.596 | 123.038 | 85.1% | 3.373 | 3.340 | 1.554 | 2.941 |
| 9241_pegase | cuda vertex | 100.591 | 82.316 | 81.8% | 3.256 | 3.163 | 0.8057 | 1.329 |

## Whole Solve

| case | impl | solve mean (ms) |
|---|---:|---:|
| 118_ieee | cpu optimized | 0.5203 |
| 118_ieee | cuda edge | 46.279 |
| 118_ieee | cuda vertex | 1.174 |
| 9241_pegase | cpu optimized | 62.015 |
| 9241_pegase | cuda edge | 62.773 |
| 9241_pegase | cuda vertex | 32.744 |

## Linear Solve Per Iteration

| case | impl | solveLinearSystem avg (ms) | rhsPrepare avg (ms) | refactorization avg (ms) | solve avg (ms) | refac share | solve share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 118_ieee | cpu optimized | 0.0500 | n/a | 0.0276 | 0.0017 | 55.2% | 3.3% |
| 118_ieee | cuda edge | 4.887 | 1.584 | 1.647 | 1.644 | 33.7% | 33.6% |
| 118_ieee | cuda vertex | 0.1800 | 0.0083 | 0.0922 | 0.0691 | 51.2% | 38.4% |
| 9241_pegase | cpu optimized | 7.408 | n/a | 7.092 | 0.3069 | 95.7% | 4.1% |
| 9241_pegase | cuda edge | 5.190 | 1.014 | 2.166 | 1.998 | 41.7% | 38.5% |
| 9241_pegase | cuda vertex | 3.122 | 0.1980 | 1.555 | 1.358 | 49.8% | 43.5% |
