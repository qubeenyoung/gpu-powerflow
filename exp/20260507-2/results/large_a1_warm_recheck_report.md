# Large A1 Warm Recheck
## Result
- Both modes were run with full-J cuDSS analyze before the NR loop. Per-iteration full-J work is factorize+solve only.
- With this warm comparison, A1 hybrid does not beat pure cuDSS on any of the five large cases, either in total time or NR-loop-only time.
- The previous apparent A1 advantage came from comparing against a larger pure cuDSS timing baseline; in this recheck, pure warm cuDSS is much smaller.

## Case Summary
| case | pure loop ms | pure setup ms | pure total ms | A1 loop ms | A1 setup ms | A1 total ms | loop speedup | total speedup | A1 calls | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case8387pegase | 4.907 | 39.270 | 44.177 | 10.209 | 319.650 | 329.860 | 0.481 | 0.134 | 2 | 0 |
| case_ACTIVSg10k | 6.001 | 42.216 | 48.217 | 21.351 | 379.629 | 400.979 | 0.281 | 0.120 | 7 | 0 |
| case_ACTIVSg25k | 12.616 | 90.733 | 103.349 | 28.342 | 958.459 | 986.802 | 0.445 | 0.105 | 3 | 1 |
| case_ACTIVSg70k | 41.834 | 240.145 | 281.979 | 96.346 | 2738.583 | 2834.928 | 0.434 | 0.099 | 7 | 1 |
| case_SyntheticUSA | 42.420 | 296.183 | 338.603 | 126.641 | 3328.668 | 3455.309 | 0.335 | 0.098 | 7 | 2 |

## Per-Call Medians
| case | full warm cuDSS factor+solve median ms | A1 middle median ms | winner |
|---|---:|---:|---|
| case8387pegase | 1.222 | 1.612 | pure cuDSS |
| case_ACTIVSg10k | 1.097 | 1.667 | pure cuDSS |
| case_ACTIVSg25k | 2.544 | 2.881 | pure cuDSS |
| case_ACTIVSg70k | 5.821 | 6.298 | pure cuDSS |
| case_SyntheticUSA | 5.867 | 6.845 | pure cuDSS |

## Answers
1. In this recheck, pure cuDSS `total_seconds` is NR-loop-only; full-J analyze is separated in `setup ms`. No per-iteration full analyze was called.
2. Warm NR-loop 기준에서도 A1은 pure cuDSS보다 빠르지 않습니다. 모든 케이스에서 loop speedup < 1입니다.
3. Case별 full warm cuDSS 1회 median은 위 표에 있습니다.
4. Case별 A1 middle 1회 median도 위 표에 있습니다.
5. A1이 per-middle 기준으로 이긴 케이스: `none`. 진 케이스: `case8387pegase, case_ACTIVSg10k, case_ACTIVSg25k, case_ACTIVSg70k, case_SyntheticUSA`.
6. Total 기준에서는 A1 setup(METIS/block metadata + J11/J22 analyze)이 커서 전부 pure cuDSS가 이깁니다.
