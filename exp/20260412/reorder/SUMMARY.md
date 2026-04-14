# cuDSS Reordering Benchmark 20260412_2

- Profile: `cuda_edge` (`cuda_mixed_edge`, FP32 edge Jacobian + cuDSS32 solve)
- Modes: `end2end`, `operators`
- Reordering algorithms: `DEFAULT`, `ALG_1`, `ALG_2`
- Cases: case30_ieee, case118_ieee, case793_goc, case1354_pegase, case2746wop_k, case4601_goc, case8387_pegase, case9241_pegase
- Warmup/repeats: 1 / 10 per case, per mode, per algorithm
- Dataset root: `/workspace/datasets/cuPF_benchmark_dumps`

Lower elapsed/analyze/solve times are better. Speedup is computed against `DEFAULT` for the same case and measurement mode.

## end2end Elapsed Summary

| algorithm | sum elapsed ms | geomean speedup vs DEFAULT | case wins |
|---|---:|---:|---:|
| `DEFAULT` | 236.657 | 1.000x | 6 |
| `ALG_1` | 581.181 | 0.706x | 1 |
| `ALG_2` | 575.318 | 0.714x | 1 |

## operators Elapsed Summary

| algorithm | sum elapsed ms | geomean speedup vs DEFAULT | case wins |
|---|---:|---:|---:|
| `DEFAULT` | 234.663 | 1.000x | 6 |
| `ALG_1` | 585.110 | 0.702x | 0 |
| `ALG_2` | 577.409 | 0.712x | 2 |

## end2end Per-Case Elapsed

| case | DEFAULT ms | ALG_1 ms | ALG_1 speedup | ALG_2 ms | ALG_2 speedup | best |
|---|---:|---:|---:|---:|---:|---|
| case30_ieee | 10.931 | 4.267 | 2.56x | 4.142 | 2.64x | `ALG_2` |
| case118_ieee | 11.487 | 5.701 | 2.02x | 5.711 | 2.01x | `ALG_1` |
| case793_goc | 15.479 | 17.993 | 0.86x | 18.015 | 0.86x | `DEFAULT` |
| case1354_pegase | 19.224 | 25.754 | 0.75x | 25.207 | 0.76x | `DEFAULT` |
| case2746wop_k | 25.181 | 48.882 | 0.52x | 47.942 | 0.53x | `DEFAULT` |
| case4601_goc | 37.270 | 99.161 | 0.38x | 98.571 | 0.38x | `DEFAULT` |
| case8387_pegase | 55.869 | 171.215 | 0.33x | 168.247 | 0.33x | `DEFAULT` |
| case9241_pegase | 61.217 | 208.207 | 0.29x | 207.483 | 0.30x | `DEFAULT` |

## operators Per-Case Elapsed

| case | DEFAULT ms | ALG_1 ms | ALG_1 speedup | ALG_2 ms | ALG_2 speedup | best |
|---|---:|---:|---:|---:|---:|---|
| case30_ieee | 11.300 | 4.385 | 2.58x | 4.346 | 2.60x | `ALG_2` |
| case118_ieee | 11.848 | 5.805 | 2.04x | 5.788 | 2.05x | `ALG_2` |
| case793_goc | 15.883 | 18.342 | 0.87x | 17.823 | 0.89x | `DEFAULT` |
| case1354_pegase | 19.367 | 25.912 | 0.75x | 25.302 | 0.77x | `DEFAULT` |
| case2746wop_k | 25.374 | 49.003 | 0.52x | 48.064 | 0.53x | `DEFAULT` |
| case4601_goc | 37.495 | 100.491 | 0.37x | 98.540 | 0.38x | `DEFAULT` |
| case8387_pegase | 53.916 | 169.979 | 0.32x | 170.308 | 0.32x | `DEFAULT` |
| case9241_pegase | 59.480 | 211.194 | 0.28x | 207.239 | 0.29x | `DEFAULT` |

## Operator Timer Focus

| case | DEFAULT cuDSS analysis ms | ALG_1 ms | ALG_1 speedup | ALG_2 ms | ALG_2 speedup | best |
|---|---:|---:|---:|---:|---:|---|
| case30_ieee | 8.530 | 1.275 | 6.69x | 1.270 | 6.72x | `ALG_2` |
| case118_ieee | 8.921 | 1.344 | 6.64x | 1.333 | 6.69x | `ALG_2` |
| case793_goc | 12.047 | 1.928 | 6.25x | 1.814 | 6.64x | `ALG_2` |
| case1354_pegase | 14.628 | 2.529 | 5.78x | 2.313 | 6.32x | `ALG_2` |
| case2746wop_k | 19.869 | 3.913 | 5.08x | 3.505 | 5.67x | `ALG_2` |
| case4601_goc | 29.040 | 6.719 | 4.32x | 6.034 | 4.81x | `ALG_2` |
| case8387_pegase | 42.656 | 10.694 | 3.99x | 9.632 | 4.43x | `ALG_2` |
| case9241_pegase | 46.556 | 12.357 | 3.77x | 10.742 | 4.33x | `ALG_2` |

## Operator Timer Diagnosis

`ALG_1`/`ALG_2` reduce cuDSS analysis time, but on large cases they make numeric factorization and solve much more expensive. Representative rows:

| case | algorithm | cuDSS analysis ms | factorization ms | refactorization ms | solve ms | NR iteration linear solve ms |
|---|---|---:|---:|---:|---:|---:|
| case30_ieee | `DEFAULT` | 8.530 | 0.072 | 0.064 | 0.057 | 0.135 |
| case30_ieee | `ALG_1` | 1.275 | 0.324 | 0.102 | 0.065 | 0.235 |
| case30_ieee | `ALG_2` | 1.270 | 0.324 | 0.102 | 0.064 | 0.235 |
| case118_ieee | `DEFAULT` | 8.921 | 0.082 | 0.074 | 0.066 | 0.155 |
| case118_ieee | `ALG_1` | 1.344 | 1.196 | 0.157 | 0.130 | 0.560 |
| case118_ieee | `ALG_2` | 1.333 | 1.195 | 0.157 | 0.130 | 0.559 |
| case9241_pegase | `DEFAULT` | 46.556 | 0.686 | 0.592 | 0.242 | 0.890 |
| case9241_pegase | `ALG_1` | 12.357 | 97.085 | 4.865 | 8.514 | 26.763 |
| case9241_pegase | `ALG_2` | 10.742 | 97.031 | 4.891 | 8.515 | 26.657 |

## Files

- `cases.txt`: benchmark case list
- `results/cuda_edge_reorder_default/`: raw benchmark output for `DEFAULT`
- `results/cuda_edge_reorder_alg1/`: raw benchmark output for `ALG_1`
- `results/cuda_edge_reorder_alg2/`: raw benchmark output for `ALG_2`
- `combined_aggregates.csv`: all aggregate rows with algorithm labels
- `reorder_comparison.csv`: elapsed/analyze/solve comparison and DEFAULT ratios
- `operator_timer_comparison.csv`: focused operator timer comparison from `summary_operators.csv`
