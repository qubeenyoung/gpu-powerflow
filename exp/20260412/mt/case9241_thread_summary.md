# case9241_pegase HOST_NTHREADS Summary

- Source: `/workspace/exp/20260412_2/mt`
- Profile: `cuda_edge`
- MT workaround: `CUDSS_THREADING_LIB` + `LD_PRELOAD`
- Baseline: `no_mt`

Best end2end: `mt_auto` (auto threads), 40.113 ms, 1.62x vs no_mt.
Best operators: `mt_auto` (auto threads), 40.121 ms, 1.48x vs no_mt.

| label | threads | end2end ms | end2end speedup | operators ms | operators speedup | cuDSS analysis ms | iteration linear solve ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_mt` | off | 65.002 | 1.00x | 59.399 | 1.00x | 46.547 | 0.886 |
| `mt_auto` | auto | 40.113 | 1.62x | 40.121 | 1.48x | 26.547 | 0.887 |
| `mt_1` | 1 | 59.346 | 1.10x | 59.490 | 1.00x | 46.509 | 0.891 |
| `mt_2` | 2 | 49.366 | 1.32x | 49.522 | 1.20x | 36.120 | 0.887 |
| `mt_4` | 4 | 44.694 | 1.45x | 44.822 | 1.33x | 31.046 | 0.894 |
| `mt_8` | 8 | 42.080 | 1.54x | 42.312 | 1.40x | 28.563 | 0.894 |
| `mt_16` | 16 | 42.439 | 1.53x | 40.352 | 1.47x | 26.856 | 0.889 |
| `mt_32` | 32 | 45.506 | 1.43x | 43.115 | 1.38x | 29.432 | 0.884 |

## Notes

- The main gain comes from `CUDA.analyze.cudss32.analysis`; factorization/refactorization/solve stay nearly flat.
- `mt_auto` is best for both end2end and operators on this case; `mt_16` is a close second in operators.
- `mt_32` regresses versus `mt_16`, so more host threads are not monotonically better.

- CSV: `/workspace/exp/20260412_2/mt/case9241_thread_summary.csv`
