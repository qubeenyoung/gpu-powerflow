# cuDSS HOST_NTHREADS Sweep

- Profile: `cuda_edge`
- Baseline: `no_mt`
- MT workaround: `CUDSS_THREADING_LIB` is also prepended to `LD_PRELOAD` for MT runs.
- Re-run status: all planned MT values completed with the preload workaround.

## Run Status

| label | status | host threads | LD_PRELOAD |
|---|---|---:|---|
| `no_mt` | completed | AUTO | `` |
| `mt_auto` | completed | AUTO | `/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0` |
| `mt_1` | completed | 1 | `/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0` |
| `mt_2` | completed | 2 | `/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0` |
| `mt_4` | completed | 4 | `/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0` |
| `mt_8` | completed | 8 | `/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0` |
| `mt_16` | completed | 16 | `/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0` |
| `mt_32` | completed | 32 | `/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0` |

## end2end Elapsed Summary

| label | sum elapsed ms | geomean speedup vs no_mt | case wins |
|---|---:|---:|---:|
| `no_mt` | 236.476 | 1.000x | 0 |
| `mt_auto` | 194.885 | 1.082x | 5 |
| `mt_1` | 231.032 | 1.010x | 0 |
| `mt_2` | 202.348 | 1.110x | 2 |
| `mt_4` | 190.378 | 1.158x | 0 |
| `mt_8` | 182.754 | 1.189x | 0 |
| `mt_16` | 186.817 | 1.168x | 0 |
| `mt_32` | 202.052 | 1.076x | 1 |

## end2end Per-Case Elapsed

| case | `no_mt` ms | `mt_auto` ms | `mt_1` ms | `mt_2` ms | `mt_4` ms | `mt_8` ms | `mt_16` ms | `mt_32` ms | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case30_ieee | 10.941 | 11.111 | 11.003 | 10.940 | 10.995 | 11.018 | 11.252 | 11.502 | `mt_2` |
| case118_ieee | 11.505 | 28.650 | 11.495 | 11.459 | 11.464 | 11.483 | 11.698 | 13.846 | `mt_2` |
| case793_goc | 15.492 | 14.705 | 15.465 | 15.162 | 15.012 | 14.724 | 14.725 | 14.935 | `mt_auto` |
| case1354_pegase | 18.864 | 16.552 | 18.856 | 17.708 | 17.498 | 17.045 | 17.280 | 16.558 | `mt_auto` |
| case2746wop_k | 24.577 | 19.735 | 24.643 | 21.608 | 20.349 | 20.272 | 20.474 | 26.708 | `mt_auto` |
| case4601_goc | 36.360 | 26.907 | 36.414 | 30.123 | 27.422 | 27.584 | 27.368 | 36.460 | `mt_auto` |
| case8387_pegase | 53.735 | 37.113 | 53.811 | 45.983 | 42.944 | 38.549 | 41.582 | 36.538 | `mt_32` |
| case9241_pegase | 65.002 | 40.113 | 59.346 | 49.366 | 44.694 | 42.080 | 42.439 | 45.506 | `mt_auto` |

## operators Elapsed Summary

| label | sum elapsed ms | geomean speedup vs no_mt | case wins |
|---|---:|---:|---:|
| `no_mt` | 235.402 | 1.000x | 0 |
| `mt_auto` | 195.192 | 1.105x | 2 |
| `mt_1` | 231.768 | 1.018x | 1 |
| `mt_2` | 203.962 | 1.111x | 0 |
| `mt_4` | 194.017 | 1.143x | 0 |
| `mt_8` | 183.300 | 1.199x | 1 |
| `mt_16` | 178.097 | 1.222x | 3 |
| `mt_32` | 187.017 | 1.156x | 1 |

## operators Per-Case Elapsed

| case | `no_mt` ms | `mt_auto` ms | `mt_1` ms | `mt_2` ms | `mt_4` ms | `mt_8` ms | `mt_16` ms | `mt_32` ms | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case30_ieee | 11.272 | 11.625 | 11.028 | 11.048 | 11.039 | 11.045 | 11.197 | 11.784 | `mt_1` |
| case118_ieee | 11.796 | 11.667 | 11.551 | 11.588 | 11.561 | 11.517 | 11.629 | 11.661 | `mt_8` |
| case793_goc | 15.940 | 23.451 | 15.519 | 15.904 | 15.466 | 14.848 | 14.569 | 19.256 | `mt_16` |
| case1354_pegase | 19.395 | 16.773 | 18.985 | 17.798 | 17.607 | 17.125 | 16.916 | 17.104 | `mt_auto` |
| case2746wop_k | 25.301 | 24.749 | 24.754 | 21.692 | 22.941 | 20.241 | 19.970 | 20.750 | `mt_16` |
| case4601_goc | 36.513 | 26.554 | 36.500 | 30.275 | 29.075 | 27.410 | 26.660 | 26.519 | `mt_32` |
| case8387_pegase | 55.787 | 40.252 | 53.942 | 46.135 | 41.508 | 38.802 | 36.804 | 36.829 | `mt_16` |
| case9241_pegase | 59.399 | 40.121 | 59.490 | 49.522 | 44.822 | 42.312 | 40.352 | 43.115 | `mt_auto` |

## Operator Timer Focus

| case | label | cuDSS analysis ms | factorization ms | refactorization ms | solve ms | iteration linear solve ms |
|---|---|---:|---:|---:|---:|---:|
| case30_ieee | `no_mt` | 8.528 | 0.072 | 0.064 | 0.056 | 0.135 |
| case30_ieee | `mt_auto` | 9.081 | 0.066 | 0.059 | 0.051 | 0.122 |
| case30_ieee | `mt_1` | 8.473 | 0.066 | 0.059 | 0.051 | 0.122 |
| case30_ieee | `mt_2` | 8.485 | 0.066 | 0.059 | 0.051 | 0.123 |
| case30_ieee | `mt_4` | 8.508 | 0.065 | 0.058 | 0.050 | 0.121 |
| case30_ieee | `mt_8` | 8.521 | 0.066 | 0.058 | 0.051 | 0.121 |
| case30_ieee | `mt_16` | 8.566 | 0.066 | 0.059 | 0.055 | 0.127 |
| case30_ieee | `mt_32` | 9.208 | 0.066 | 0.059 | 0.050 | 0.121 |
| case118_ieee | `no_mt` | 8.922 | 0.081 | 0.074 | 0.067 | 0.156 |
| case118_ieee | `mt_auto` | 8.989 | 0.076 | 0.070 | 0.061 | 0.144 |
| case118_ieee | `mt_1` | 8.882 | 0.077 | 0.070 | 0.061 | 0.144 |
| case118_ieee | `mt_2` | 8.859 | 0.076 | 0.069 | 0.062 | 0.144 |
| case118_ieee | `mt_4` | 8.866 | 0.076 | 0.069 | 0.061 | 0.144 |
| case118_ieee | `mt_8` | 8.857 | 0.076 | 0.069 | 0.061 | 0.142 |
| case118_ieee | `mt_16` | 8.944 | 0.076 | 0.069 | 0.061 | 0.142 |
| case118_ieee | `mt_32` | 8.971 | 0.076 | 0.070 | 0.062 | 0.144 |
| case793_goc | `no_mt` | 12.055 | 0.248 | 0.119 | 0.090 | 0.255 |
| case793_goc | `mt_auto` | 19.759 | 0.219 | 0.114 | 0.084 | 0.232 |
| case793_goc | `mt_1` | 12.006 | 0.219 | 0.114 | 0.084 | 0.234 |
| case793_goc | `mt_2` | 11.996 | 0.251 | 0.120 | 0.090 | 0.258 |
| case793_goc | `mt_4` | 11.522 | 0.247 | 0.120 | 0.089 | 0.254 |
| case793_goc | `mt_8` | 11.242 | 0.218 | 0.114 | 0.084 | 0.233 |
| case793_goc | `mt_16` | 11.027 | 0.221 | 0.114 | 0.084 | 0.236 |
| case793_goc | `mt_32` | 15.559 | 0.225 | 0.114 | 0.085 | 0.235 |
| case1354_pegase | `no_mt` | 14.640 | 0.314 | 0.180 | 0.129 | 0.360 |
| case1354_pegase | `mt_auto` | 12.301 | 0.281 | 0.178 | 0.123 | 0.338 |
| case1354_pegase | `mt_1` | 14.580 | 0.282 | 0.175 | 0.124 | 0.336 |
| case1354_pegase | `mt_2` | 13.390 | 0.279 | 0.178 | 0.123 | 0.337 |
| case1354_pegase | `mt_4` | 13.156 | 0.284 | 0.177 | 0.123 | 0.337 |
| case1354_pegase | `mt_8` | 12.690 | 0.281 | 0.175 | 0.123 | 0.334 |
| case1354_pegase | `mt_16` | 12.459 | 0.283 | 0.175 | 0.127 | 0.338 |
| case1354_pegase | `mt_32` | 12.626 | 0.278 | 0.174 | 0.122 | 0.332 |
| case2746wop_k | `no_mt` | 19.822 | 0.415 | 0.280 | 0.159 | 0.507 |
| case2746wop_k | `mt_auto` | 19.478 | 0.383 | 0.275 | 0.157 | 0.481 |
| case2746wop_k | `mt_1` | 19.730 | 0.383 | 0.274 | 0.153 | 0.475 |
| case2746wop_k | `mt_2` | 16.648 | 0.382 | 0.273 | 0.152 | 0.472 |
| case2746wop_k | `mt_4` | 16.543 | 0.450 | 0.285 | 0.176 | 0.575 |
| case2746wop_k | `mt_8` | 15.077 | 0.382 | 0.273 | 0.152 | 0.474 |
| case2746wop_k | `mt_16` | 14.707 | 0.381 | 0.282 | 0.152 | 0.487 |
| case2746wop_k | `mt_32` | 15.397 | 0.383 | 0.277 | 0.152 | 0.486 |
| case4601_goc | `no_mt` | 28.826 | 0.555 | 0.451 | 0.215 | 0.716 |
| case4601_goc | `mt_auto` | 18.327 | 0.554 | 0.450 | 0.214 | 0.712 |
| case4601_goc | `mt_1` | 28.793 | 0.556 | 0.451 | 0.215 | 0.716 |
| case4601_goc | `mt_2` | 22.341 | 0.556 | 0.451 | 0.219 | 0.717 |
| case4601_goc | `mt_4` | 21.016 | 0.558 | 0.452 | 0.215 | 0.715 |
| case4601_goc | `mt_8` | 19.403 | 0.555 | 0.451 | 0.215 | 0.712 |
| case4601_goc | `mt_16` | 18.538 | 0.557 | 0.455 | 0.214 | 0.720 |
| case4601_goc | `mt_32` | 18.291 | 0.552 | 0.451 | 0.214 | 0.713 |
| case8387_pegase | `no_mt` | 43.033 | 0.691 | 0.561 | 0.249 | 0.936 |
| case8387_pegase | `mt_auto` | 28.224 | 0.657 | 0.560 | 0.241 | 0.853 |
| case8387_pegase | `mt_1` | 42.675 | 0.656 | 0.562 | 0.243 | 0.858 |
| case8387_pegase | `mt_2` | 34.342 | 0.656 | 0.556 | 0.242 | 0.853 |
| case8387_pegase | `mt_4` | 29.444 | 0.656 | 0.558 | 0.242 | 0.854 |
| case8387_pegase | `mt_8` | 26.719 | 0.658 | 0.559 | 0.241 | 0.853 |
| case8387_pegase | `mt_16` | 24.802 | 0.657 | 0.557 | 0.250 | 0.861 |
| case8387_pegase | `mt_32` | 24.761 | 0.657 | 0.562 | 0.246 | 0.861 |
| case9241_pegase | `no_mt` | 46.547 | 0.686 | 0.591 | 0.242 | 0.886 |
| case9241_pegase | `mt_auto` | 26.547 | 0.687 | 0.592 | 0.241 | 0.887 |
| case9241_pegase | `mt_1` | 46.509 | 0.691 | 0.593 | 0.242 | 0.891 |
| case9241_pegase | `mt_2` | 36.120 | 0.704 | 0.591 | 0.242 | 0.887 |
| case9241_pegase | `mt_4` | 31.046 | 0.689 | 0.594 | 0.247 | 0.894 |
| case9241_pegase | `mt_8` | 28.563 | 0.689 | 0.596 | 0.244 | 0.894 |
| case9241_pegase | `mt_16` | 26.856 | 0.689 | 0.595 | 0.242 | 0.889 |
| case9241_pegase | `mt_32` | 29.432 | 0.686 | 0.592 | 0.242 | 0.884 |

## Files

- `mt_status.csv`
- `combined_aggregates.csv`
- `mt_comparison.csv`
- `operator_timer_comparison.csv`
- raw benchmark outputs under `results/`
