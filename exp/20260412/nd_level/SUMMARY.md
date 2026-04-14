# cuDSS ND_NLEVELS Sweep

- Profile: `cuda_edge`
- Baseline: `auto` (do not set `CUDSS_CONFIG_ND_NLEVELS`)
- Completed values: `auto`, `8`, `9`, `10`, `11`

## Run Status

| label | status | note |
|---|---|---|
| `auto` | completed | manifest and aggregate results available |
| `nd_8` | completed | manifest and aggregate results available |
| `nd_9` | completed | manifest and aggregate results available |
| `nd_10` | completed | manifest and aggregate results available |
| `nd_11` | completed | manifest and aggregate results available |

## end2end Elapsed Summary

| label | sum elapsed ms | geomean speedup vs baseline | case wins |
|---|---:|---:|---:|
| `auto` | 230.927 | 1.000x | 0 |
| `nd_8` | 211.507 | 1.064x | 7 |
| `nd_9` | 222.172 | 1.029x | 1 |
| `nd_10` | 234.268 | 0.985x | 0 |
| `nd_11` | 242.442 | 0.964x | 0 |

## end2end Per-Case Elapsed

| case | `auto` ms | `nd_8` ms | `nd_9` ms | `nd_10` ms | `nd_11` ms | best |
|---|---:|---:|---:|---:|---:|---|
| case30_ieee | 10.933 | 10.893 | 10.878 | 11.151 | 11.232 | `nd_9` |
| case118_ieee | 11.502 | 11.434 | 11.500 | 11.701 | 11.561 | `nd_8` |
| case793_goc | 15.473 | 14.842 | 15.400 | 15.709 | 15.450 | `nd_8` |
| case1354_pegase | 18.850 | 17.445 | 17.889 | 19.216 | 19.053 | `nd_8` |
| case2746wop_k | 24.611 | 24.352 | 24.419 | 25.166 | 26.543 | `nd_8` |
| case4601_goc | 36.466 | 33.080 | 34.583 | 36.388 | 38.691 | `nd_8` |
| case8387_pegase | 53.810 | 46.762 | 50.022 | 53.764 | 57.211 | `nd_8` |
| case9241_pegase | 59.283 | 52.698 | 57.480 | 61.173 | 62.701 | `nd_8` |

## operators Elapsed Summary

| label | sum elapsed ms | geomean speedup vs baseline | case wins |
|---|---:|---:|---:|
| `auto` | 231.767 | 1.000x | 0 |
| `nd_8` | 211.071 | 1.071x | 8 |
| `nd_9` | 220.952 | 1.033x | 0 |
| `nd_10` | 238.483 | 0.973x | 0 |
| `nd_11` | 243.854 | 0.962x | 0 |

## operators Per-Case Elapsed

| case | `auto` ms | `nd_8` ms | `nd_9` ms | `nd_10` ms | `nd_11` ms | best |
|---|---:|---:|---:|---:|---:|---|
| case30_ieee | 11.029 | 10.952 | 10.976 | 11.293 | 11.084 | `nd_8` |
| case118_ieee | 11.572 | 11.518 | 11.527 | 11.899 | 11.671 | `nd_8` |
| case793_goc | 15.501 | 14.871 | 15.823 | 15.954 | 15.654 | `nd_8` |
| case1354_pegase | 18.956 | 17.510 | 18.089 | 19.350 | 19.127 | `nd_8` |
| case2746wop_k | 24.706 | 23.042 | 23.849 | 25.383 | 27.208 | `nd_8` |
| case4601_goc | 36.540 | 33.259 | 34.733 | 37.410 | 38.837 | `nd_8` |
| case8387_pegase | 53.961 | 46.991 | 50.194 | 55.710 | 57.290 | `nd_8` |
| case9241_pegase | 59.502 | 52.927 | 55.762 | 61.485 | 62.982 | `nd_8` |

## Operator Timer Focus

| case | label | cuDSS analysis ms | factorization ms | refactorization ms | solve ms | iteration linear solve ms |
|---|---|---:|---:|---:|---:|---:|
| case30_ieee | `auto` | 8.474 | 0.067 | 0.060 | 0.051 | 0.124 |
| case30_ieee | `nd_8` | 8.414 | 0.066 | 0.058 | 0.051 | 0.121 |
| case30_ieee | `nd_9` | 8.438 | 0.066 | 0.059 | 0.051 | 0.122 |
| case30_ieee | `nd_10` | 8.524 | 0.071 | 0.064 | 0.056 | 0.135 |
| case30_ieee | `nd_11` | 8.538 | 0.066 | 0.059 | 0.051 | 0.122 |
| case118_ieee | `auto` | 8.908 | 0.076 | 0.069 | 0.061 | 0.143 |
| case118_ieee | `nd_8` | 8.838 | 0.077 | 0.070 | 0.061 | 0.144 |
| case118_ieee | `nd_9` | 8.852 | 0.076 | 0.070 | 0.061 | 0.144 |
| case118_ieee | `nd_10` | 8.957 | 0.083 | 0.075 | 0.068 | 0.158 |
| case118_ieee | `nd_11` | 8.958 | 0.076 | 0.070 | 0.062 | 0.144 |
| case793_goc | `auto` | 11.999 | 0.220 | 0.114 | 0.084 | 0.235 |
| case793_goc | `nd_8` | 11.242 | 0.228 | 0.121 | 0.099 | 0.257 |
| case793_goc | `nd_9` | 11.758 | 0.259 | 0.133 | 0.107 | 0.284 |
| case793_goc | `nd_10` | 12.042 | 0.253 | 0.120 | 0.090 | 0.257 |
| case793_goc | `nd_11` | 12.073 | 0.219 | 0.114 | 0.084 | 0.232 |
| case1354_pegase | `auto` | 14.565 | 0.283 | 0.175 | 0.123 | 0.336 |
| case1354_pegase | `nd_8` | 13.205 | 0.274 | 0.167 | 0.117 | 0.321 |
| case1354_pegase | `nd_9` | 13.752 | 0.273 | 0.168 | 0.115 | 0.320 |
| case1354_pegase | `nd_10` | 14.615 | 0.310 | 0.181 | 0.128 | 0.359 |
| case1354_pegase | `nd_11` | 14.912 | 0.266 | 0.162 | 0.107 | 0.305 |
| case2746wop_k | `auto` | 19.717 | 0.383 | 0.275 | 0.153 | 0.475 |
| case2746wop_k | `nd_8` | 17.982 | 0.383 | 0.275 | 0.159 | 0.483 |
| case2746wop_k | `nd_9` | 18.808 | 0.383 | 0.272 | 0.161 | 0.482 |
| case2746wop_k | `nd_10` | 19.833 | 0.418 | 0.281 | 0.159 | 0.510 |
| case2746wop_k | `nd_11` | 21.642 | 0.419 | 0.288 | 0.164 | 0.519 |
| case4601_goc | `auto` | 28.893 | 0.554 | 0.451 | 0.215 | 0.713 |
| case4601_goc | `nd_8` | 25.359 | 0.587 | 0.481 | 0.217 | 0.746 |
| case4601_goc | `nd_9` | 27.014 | 0.560 | 0.454 | 0.217 | 0.720 |
| case4601_goc | `nd_10` | 28.980 | 0.584 | 0.462 | 0.221 | 0.760 |
| case4601_goc | `nd_11` | 30.928 | 0.570 | 0.462 | 0.219 | 0.732 |
| case8387_pegase | `auto` | 42.659 | 0.657 | 0.558 | 0.242 | 0.853 |
| case8387_pegase | `nd_8` | 35.403 | 0.692 | 0.595 | 0.257 | 0.908 |
| case8387_pegase | `nd_9` | 38.880 | 0.674 | 0.575 | 0.220 | 0.849 |
| case8387_pegase | `nd_10` | 42.954 | 0.690 | 0.575 | 0.247 | 0.919 |
| case8387_pegase | `nd_11` | 46.253 | 0.632 | 0.533 | 0.226 | 0.814 |
| case9241_pegase | `auto` | 46.573 | 0.691 | 0.593 | 0.242 | 0.893 |
| case9241_pegase | `nd_8` | 39.082 | 0.812 | 0.718 | 0.264 | 1.034 |
| case9241_pegase | `nd_9` | 42.890 | 0.695 | 0.596 | 0.234 | 0.886 |
| case9241_pegase | `nd_10` | 46.909 | 0.717 | 0.620 | 0.247 | 0.967 |
| case9241_pegase | `nd_11` | 50.380 | 0.641 | 0.545 | 0.248 | 0.850 |

## Files

- `combined_aggregates.csv`
- comparison CSV
- `operator_timer_comparison.csv`
- raw benchmark outputs under `results/`
