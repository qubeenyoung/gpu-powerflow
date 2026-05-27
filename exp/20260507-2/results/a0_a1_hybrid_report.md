# A0/A1 Hybrid NR Report

## Standalone J1 quality

| variant | dx error after | theta error after | |V| error after | full residual after | P residual after | Q residual after | correction/full | wall ms | serial sum ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.06633 | 0.05914 | 0.7491 | 0.1592 | 0.3339 | 0.1237 | 0.949 | 0.736 | 1.099 |
| A1 | 0.04999 | 0.0461 | 0.1097 | 0.04608 | 0.1993 | 0.03206 | 1.289 | 1.177 | 1.485 |

## Hybrid averages

| block | mode | NR iters | full cuDSS calls | middle calls | accepted | fallback | hybrid ms | pure ms | speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | BJ | 6.8 | 4.4 | 5.2 | 2.4 | 2.8 | 48.6 | 45.8 | 1.005 |
| 8 | A0 | 8.4 | 2.8 | 6.6 | 5.6 | 1.0 | 59.6 | 43.5 | 0.768 |
| 8 | A1 | 7.0 | 2.8 | 5.2 | 4.2 | 1.0 | 57.0 | 43.5 | 0.788 |
| 16 | BJ | 7.0 | 4.6 | 5.4 | 2.4 | 3.0 | 49.1 | 44.7 | 0.969 |
| 16 | A0 | 8.0 | 3.2 | 6.0 | 4.8 | 1.2 | 59.5 | 43.4 | 0.796 |
| 16 | A1 | 6.4 | 2.6 | 4.6 | 3.8 | 0.8 | 54.4 | 43.5 | 0.837 |

## Best individual runs

| case | block | mode | NR iters | full cuDSS calls | fallback | hybrid ms | pure ms | speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| case2383wp | 8 | BJ | 7 | 5 | 4 | 29.1 | 47.4 | 1.628 |
| case2383wp | 16 | BJ | 8 | 5 | 4 | 30.4 | 47.2 | 1.553 |
| case2383wp | 16 | A1 | 7 | 2 | 1 | 34.6 | 42.0 | 1.214 |
| case2383wp | 16 | A0 | 9 | 3 | 1 | 35.6 | 42.0 | 1.181 |
| case2383wp | 8 | A1 | 11 | 2 | 0 | 41.8 | 41.9 | 1.002 |
| case2383wp | 8 | A0 | 13 | 2 | 0 | 43.1 | 42.1 | 0.978 |

## Field correction timing

| mode | middle rows | wall ms mean | wall ms max | serial cuDSS event sum ms | J11 factor ms | J11 solve ms | J22 factor ms | J22 solve ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 52 | 1.546 | 2.761 | 0.139 | 0.040 | 0.035 | 0.034 | 0.030 |
| A1 | 40 | 1.932 | 3.612 | 0.213 | 0.047 | 0.066 | 0.040 | 0.059 |

## Answers

1. A0/A1 now launch J11 and J22 factorize/solve on separate non-blocking CUDA streams in the hybrid path and standalone quality tool.
2. A1 reuses the round-0 J11/J22 factors; round 1 performs solve-only launches for cross residuals.
3. Standalone J1 still shows A1 improves residual and |V| quality, but its correction/full cost remains above A0.
4. Hybrid fallback decreases versus plain BJ in several cases, but NR iterations usually grow and total time is not lower on most cases.
5. The event serial sum is lower than before due overlap, but end-to-end field correction wall time still includes host residual extraction, RHS rebuild, H2D/D2H copies, and cross residual products.
6. Decision remains: A1 is too costly as default; A0 is cheaper but does not satisfy the overall keep rule except selected cases such as case2383wp.
