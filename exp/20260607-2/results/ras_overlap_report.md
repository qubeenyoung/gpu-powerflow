# RAS overlap=1 pilot

## Verdict

- RAS overlap=1 is implemented and device-resident, but this pilot does **not** justify continuing it as-is.
- Standalone dx error improves on several cases, especially block size 16, but true residual becomes much worse.
- In hybrid NR, RAS does not reduce fallback versus block-Jacobi; it usually increases fallback or leaves it unchanged.
- RAS setup/apply cost is too high for the current dense-inverse pilot, especially larger cases.

## Symbolic overlap cost

| case | bs | avg owned | avg omega | max omega | growth | padded dense MB | risk |
|---|---:|---:|---:|---:|---:|---:|---|
| case2383wp | 8 | 7.9 | 49.9 | 200 | 6.3 | 86.1 | ok |
| case2383wp | 16 | 15.9 | 108.8 | 350 | 6.8 | 130.4 | ok |
| case3120sp | 8 | 7.9 | 46.4 | 168 | 5.9 | 81.7 | ok |
| case3120sp | 16 | 15.9 | 103.4 | 336 | 6.5 | 161.9 | ok |
| case9241pegase | 8 | 7.9 | 60.0 | 280 | 7.6 | 646.0 | ok |
| case9241pegase | 16 | 15.8 | 123.0 | 448 | 7.8 | 824.6 | ok |
| case13659pegase | 8 | 7.7 | 65.1 | 313 | 8.5 | 1130.9 | ok |
| case13659pegase | 16 | 15.8 | 131.0 | 447 | 8.3 | 1118.2 | ok |
| case6468rte | 8 | 7.9 | 58.5 | 248 | 7.4 | 375.9 | ok |
| case6468rte | 16 | 15.9 | 122.5 | 502 | 7.7 | 763.3 | ok |

## Standalone J1/F1 quality gate

| block | rel-res improved | dx-error improved | cosine improved | cases |
|---:|---:|---:|---:|---:|
| 8 | 0/5 | 3/5 | 3/5 | 5 |
| 16 | 0/5 | 4/5 | 4/5 | 5 |

## Hybrid summary

| case | mode | conv | NR | pure NR | cuDSS | middle | accept | fallback | total ms | pure ms | linear ms | pure linear ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | pure_cudss | true | 6 | 6 | 6 | 0 | 0 | 0 | 48.31 | 48.31 | 40.42 | 40.42 |
| case3120sp | pure_cudss | true | 6 | 6 | 6 | 0 | 0 | 0 | 25.40 | 25.40 | 23.48 | 23.48 |
| case9241pegase | pure_cudss | true | 6 | 6 | 6 | 0 | 0 | 0 | 53.05 | 53.05 | 50.75 | 50.75 |
| case13659pegase | pure_cudss | true | 5 | 5 | 5 | 0 | 0 | 0 | 63.65 | 63.65 | 61.41 | 61.41 |
| case6468rte | pure_cudss | true | 3 | 3 | 3 | 0 | 0 | 0 | 34.20 | 34.20 | 33.08 | 33.08 |
| case2383wp | bj_bs8 | true | 7 | 6 | 5 | 3 | 2 | 1 | 52.87 | 48.31 | 45.11 | 40.42 |
| case3120sp | bj_bs8 | true | 8 | 6 | 6 | 2 | 2 | 0 | 28.08 | 25.40 | 25.51 | 23.48 |
| case9241pegase | bj_bs8 | true | 7 | 6 | 5 | 2 | 2 | 0 | 56.54 | 53.05 | 53.75 | 50.75 |
| case13659pegase | bj_bs8 | true | 7 | 5 | 5 | 2 | 2 | 0 | 69.98 | 63.65 | 66.75 | 61.41 |
| case6468rte | bj_bs8 | true | 3 | 3 | 2 | 2 | 1 | 1 | 36.94 | 34.20 | 36.47 | 33.08 |
| case2383wp | ras_bs8 | true | 6 | 6 | 6 | 4 | 0 | 4 | 106.33 | 48.31 | 100.10 | 40.42 |
| case3120sp | ras_bs8 | true | 6 | 6 | 6 | 4 | 0 | 4 | 71.25 | 25.40 | 71.35 | 23.48 |
| case9241pegase | ras_bs8 | true | 6 | 6 | 4 | 2 | 2 | 0 | 357.47 | 53.05 | 355.16 | 50.75 |
| case13659pegase | ras_bs8 | true | 5 | 5 | 5 | 4 | 0 | 4 | 1315.04 | 63.65 | 1317.90 | 61.41 |
| case6468rte | ras_bs8 | true | 3 | 3 | 3 | 1 | 0 | 1 | 106.58 | 34.20 | 106.12 | 33.08 |
| case2383wp | bj_bs16 | true | 7 | 6 | 5 | 3 | 2 | 1 | 50.74 | 48.31 | 43.16 | 40.42 |
| case3120sp | bj_bs16 | true | 8 | 6 | 6 | 3 | 2 | 1 | 26.87 | 25.40 | 25.05 | 23.48 |
| case9241pegase | bj_bs16 | true | 7 | 6 | 5 | 3 | 2 | 1 | 55.96 | 53.05 | 54.44 | 50.75 |
| case13659pegase | bj_bs16 | true | 7 | 5 | 5 | 2 | 2 | 0 | 66.46 | 63.65 | 63.55 | 61.41 |
| case6468rte | bj_bs16 | true | 4 | 3 | 3 | 2 | 1 | 1 | 35.84 | 34.20 | 35.07 | 33.08 |
| case2383wp | ras_bs16 | true | 6 | 6 | 6 | 4 | 0 | 4 | 194.00 | 48.31 | 187.76 | 40.42 |
| case3120sp | ras_bs16 | true | 6 | 6 | 6 | 4 | 0 | 4 | 166.24 | 25.40 | 166.26 | 23.48 |
| case9241pegase | ras_bs16 | true | 6 | 6 | 4 | 2 | 2 | 0 | 625.61 | 53.05 | 623.50 | 50.75 |
| case13659pegase | ras_bs16 | true | 5 | 5 | 5 | 4 | 0 | 4 | 1678.97 | 63.65 | 1681.77 | 61.41 |
| case6468rte | ras_bs16 | true | 3 | 3 | 3 | 1 | 0 | 1 | 334.16 | 34.20 | 333.70 | 33.08 |

## Answers

1. Standalone dx quality: mixed. RAS bs16 improves dx error/cosine on several cases, but residual quality worsens everywhere.
2. Overlap size: avg omega grows about 6-8x over owned block size; max omega reaches 502 at bs16.
3. Fallback: RAS does not reduce fallback compared with block-Jacobi.
4. NR iterations: RAS generally converges only because fallback/full cuDSS still carries the trajectory.
5. Cost: dense local inverse setup dominates; current RAS middle is not competitive with warm full cuDSS.
6. Better block size: bs16 has better standalone dx direction on some cases, but hybrid fallback/cost remains bad.
7. Decision: do not continue this dense-inverse RAS overlap=1 path without a different cheaper local solve strategy.
