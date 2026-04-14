# cuDSS ND_NLEVELS Follow-up Sweep

- Profile: `cuda_edge`
- Reordering: `CUDSS_ALG_DEFAULT`
- MT mode: disabled
- New values: `5`, `6`, `7` under `/workspace/exp/20260413/nd_level`
- Comparison includes previous `/workspace/exp/20260412/nd_level` values: `auto`, `8`, `9`, `10`, `11`
- Note: this is a cross-day comparison, so small differences should be treated cautiously.

## Run Status

| label | source | status | note |
|---|---|---|---|
| `auto` | `20260412` | completed | manifest and aggregate results available |
| `nd_5` | `20260413` | completed | manifest and aggregate results available |
| `nd_6` | `20260413` | completed | manifest and aggregate results available |
| `nd_7` | `20260413` | completed | manifest and aggregate results available |
| `nd_8` | `20260412` | completed | manifest and aggregate results available |
| `nd_9` | `20260412` | completed | manifest and aggregate results available |
| `nd_10` | `20260412` | completed | manifest and aggregate results available |
| `nd_11` | `20260412` | completed | manifest and aggregate results available |

## end2end Elapsed Summary

| label | sum elapsed ms | geomean speedup vs auto | case wins |
|---|---:|---:|---:|
| `auto` | 230.927 | 1.000x | 0 |
| `nd_5` | 222.119 | 1.054x | 4 |
| `nd_6` | 210.922 | 1.077x | 1 |
| `nd_7` | 206.960 | 1.091x | 1 |
| `nd_8` | 211.506 | 1.064x | 2 |
| `nd_9` | 222.172 | 1.029x | 0 |
| `nd_10` | 234.268 | 0.985x | 0 |
| `nd_11` | 242.442 | 0.964x | 0 |

## end2end Per-Case Elapsed

| case | `auto` ms | `nd_5` ms | `nd_6` ms | `nd_7` ms | `nd_8` ms | `nd_9` ms | `nd_10` ms | `nd_11` ms | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case118_ieee | 11.502 | 11.412 | 11.693 | 11.435 | 11.434 | 11.500 | 11.701 | 11.561 | `nd_5` |
| case1354_pegase | 18.850 | 16.299 | 16.579 | 16.946 | 17.445 | 17.889 | 19.216 | 19.053 | `nd_5` |
| case2746wop_k | 24.611 | 22.469 | 21.459 | 21.779 | 24.352 | 24.419 | 25.166 | 26.543 | `nd_6` |
| case30_ieee | 10.933 | 10.870 | 11.071 | 10.872 | 10.893 | 10.878 | 11.151 | 11.232 | `nd_5` |
| case4601_goc | 36.466 | 34.669 | 32.958 | 31.874 | 33.080 | 34.583 | 36.388 | 38.691 | `nd_7` |
| case793_goc | 15.473 | 14.011 | 14.483 | 14.426 | 14.842 | 15.400 | 15.709 | 15.450 | `nd_5` |
| case8387_pegase | 53.810 | 50.612 | 48.763 | 46.804 | 46.762 | 50.022 | 53.764 | 57.211 | `nd_8` |
| case9241_pegase | 59.283 | 61.776 | 53.914 | 52.823 | 52.698 | 57.480 | 61.173 | 62.701 | `nd_8` |

## operators Elapsed Summary

| label | sum elapsed ms | geomean speedup vs auto | case wins |
|---|---:|---:|---:|
| `auto` | 231.767 | 1.000x | 0 |
| `nd_5` | 218.160 | 1.063x | 3 |
| `nd_6` | 211.167 | 1.085x | 2 |
| `nd_7` | 207.865 | 1.090x | 2 |
| `nd_8` | 211.071 | 1.071x | 1 |
| `nd_9` | 220.952 | 1.033x | 0 |
| `nd_10` | 238.483 | 0.973x | 0 |
| `nd_11` | 243.854 | 0.962x | 0 |

## operators Per-Case Elapsed

| case | `auto` ms | `nd_5` ms | `nd_6` ms | `nd_7` ms | `nd_8` ms | `nd_9` ms | `nd_10` ms | `nd_11` ms | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case118_ieee | 11.572 | 11.480 | 11.616 | 11.491 | 11.518 | 11.527 | 11.899 | 11.671 | `nd_5` |
| case1354_pegase | 18.956 | 16.430 | 16.602 | 17.091 | 17.510 | 18.089 | 19.350 | 19.127 | `nd_5` |
| case2746wop_k | 24.706 | 22.614 | 21.512 | 21.941 | 23.042 | 23.849 | 25.383 | 27.208 | `nd_6` |
| case30_ieee | 11.029 | 11.000 | 10.938 | 10.952 | 10.952 | 10.976 | 11.293 | 11.084 | `nd_6` |
| case4601_goc | 36.540 | 34.849 | 33.091 | 32.010 | 33.259 | 34.733 | 37.410 | 38.837 | `nd_7` |
| case793_goc | 15.501 | 14.095 | 14.189 | 14.492 | 14.871 | 15.823 | 15.954 | 15.654 | `nd_5` |
| case8387_pegase | 53.961 | 50.752 | 49.008 | 46.846 | 46.991 | 50.194 | 55.710 | 57.290 | `nd_7` |
| case9241_pegase | 59.502 | 56.941 | 54.212 | 53.043 | 52.927 | 55.762 | 61.485 | 62.982 | `nd_8` |

## Operator Analysis Timer Summary

| label | cuDSS analysis sum ms | analysis speedup vs auto | factorization sum ms | iteration linear solve sum ms |
|---|---:|---:|---:|---:|
| `auto` | 181.788 | 1.000x | 2.930 | 3.771 |
| `nd_5` | 160.940 | 1.130x | 3.917 | 5.052 |
| `nd_6` | 155.794 | 1.167x | 3.517 | 4.677 |
| `nd_7` | 154.765 | 1.175x | 3.292 | 4.281 |
| `nd_8` | 159.525 | 1.140x | 3.119 | 4.014 |
| `nd_9` | 170.392 | 1.067x | 2.987 | 3.807 |
| `nd_10` | 182.813 | 0.994x | 3.126 | 4.064 |
| `nd_11` | 193.684 | 0.939x | 2.889 | 3.720 |

## Notes

- `end2end` total best: `nd_7` at 206.960 ms.
- `operators` total best: `nd_7` at 207.865 ms.
- The lower values improve the total versus `auto`; in this combined comparison `nd_7` is the best total, while `nd_8` remains best for `case9241_pegase`.
- The main movement is still in `CUDA.analyze.cudss32.analysis`; factorization and iterative solve do not improve monotonically as ND_NLEVELS decreases.

## Files

- `combined_aggregates.csv`
- `nd_level_comparison.csv`
- `operator_timer_comparison.csv`
- `operator_analysis_summary.csv`
- raw benchmark outputs under `results/`
