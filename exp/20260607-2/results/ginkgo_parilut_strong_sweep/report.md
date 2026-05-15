# Ginkgo ParILUT Strong Fill Sweep
## Setup
- Cases: `case2383wp`, `case3120sp`, `case6468rte`, `case9241pegase`, `case13659pegase` where runs fit in memory.
- Common NR policy: bootstrap full cuDSS 1, iterative only after `mismatch_inf < 1`, polish `1e-4`, accept `0.9`, reject `1.05`, fallback immediate.
- Middle solver: Ginkgo ParILUT + Ginkgo BiCGSTAB fixed iterations.
- Baseline row: `fill=2`, ParILUT iterations `5`, BiCGSTAB iterations `2`. Strong rows use `fill=5/10`, ParILUT iterations `10/20`, BiCGSTAB iterations `4/8`.
- Note: current implementation rebuilds Ginkgo matrix/preconditioner from host-side values on every middle attempt, so wall time is not final GPU performance. This sweep primarily tests correction quality and cuDSS-call reduction.

## Aggregate Result
| setting | cases | accepted middle steps | rejected/fallback | pure full cuDSS calls | hybrid full cuDSS calls | call reduction | max total ms | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_fill2_p5_b2` | 5 | 0 | 9 | 26 | 26 | 0 | 202.9 |  |
| `fill10_p10_b8` | 4 | 0 | 7 | 21 | 21 | 0 | 1922.0 | partial; larger cases hit OOM in some fill=10 runs |
| `fill10_p20_b4` | 4 | 0 | 7 | 21 | 21 | 0 | 3872.2 | partial; larger cases hit OOM in some fill=10 runs |
| `fill10_p20_b8_case2383_only` | 1 | 0 | 2 | 6 | 6 | 0 | 1048.4 | partial; larger cases hit OOM in some fill=10 runs |
| `fill5_p10_b4` | 5 | 0 | 9 | 26 | 26 | 0 | 907.5 |  |
| `fill5_p10_b8` | 5 | 0 | 9 | 26 | 26 | 0 | 1783.6 |  |
| `fill5_p20_b4` | 5 | 0 | 9 | 26 | 26 | 0 | 2045.8 |  |
| `fill5_p20_b8` | 5 | 0 | 9 | 26 | 26 | 0 | 2156.2 |  |

## Case-Level Rows
| setting | case | pure calls | hybrid calls | accepted | fallback | total ms | pure total ms | final mismatch inf |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_fill2_p5_b2` | case2383wp | 6 | 6 | 0 | 2 | 119.5 | 25.68 | 4.708e-12 |
| `baseline_fill2_p5_b2` | case3120sp | 6 | 6 | 0 | 2 | 53.3 | 5.74 | 7.532e-12 |
| `baseline_fill2_p5_b2` | case6468rte | 3 | 3 | 0 | 1 | 49.4 | 3.67 | 1.127e-11 |
| `baseline_fill2_p5_b2` | case9241pegase | 6 | 6 | 0 | 2 | 152.0 | 9.30 | 2.129e-09 |
| `baseline_fill2_p5_b2` | case13659pegase | 5 | 5 | 0 | 2 | 202.9 | 8.95 | 2.289e-09 |
| `fill5_p10_b4` | case2383wp | 6 | 6 | 0 | 2 | 281.5 | 25.18 | 5.650e-12 |
| `fill5_p10_b4` | case3120sp | 6 | 6 | 0 | 2 | 277.8 | 5.03 | 1.135e-11 |
| `fill5_p10_b4` | case6468rte | 3 | 3 | 0 | 1 | 345.1 | 3.20 | 7.789e-12 |
| `fill5_p10_b4` | case9241pegase | 6 | 6 | 0 | 2 | 730.9 | 8.75 | 2.129e-09 |
| `fill5_p10_b4` | case13659pegase | 5 | 5 | 0 | 2 | 907.5 | 8.83 | 2.289e-09 |
| `fill5_p10_b8` | case2383wp | 6 | 6 | 0 | 2 | 320.4 | 25.07 | 5.442e-12 |
| `fill5_p10_b8` | case3120sp | 6 | 6 | 0 | 2 | 387.2 | 5.03 | 7.942e-12 |
| `fill5_p10_b8` | case6468rte | 3 | 3 | 0 | 1 | 476.5 | 3.20 | 1.160e-11 |
| `fill5_p10_b8` | case9241pegase | 6 | 6 | 0 | 2 | 1130.3 | 8.67 | 2.129e-09 |
| `fill5_p10_b8` | case13659pegase | 5 | 5 | 0 | 2 | 1783.6 | 8.82 | 2.289e-09 |
| `fill5_p20_b4` | case2383wp | 6 | 6 | 0 | 2 | 421.2 | 25.22 | 4.659e-12 |
| `fill5_p20_b4` | case3120sp | 6 | 6 | 0 | 2 | 454.1 | 5.04 | 7.581e-12 |
| `fill5_p20_b4` | case6468rte | 3 | 3 | 0 | 1 | 625.1 | 3.21 | 1.352e-11 |
| `fill5_p20_b4` | case9241pegase | 6 | 6 | 0 | 2 | 1471.8 | 8.73 | 2.129e-09 |
| `fill5_p20_b4` | case13659pegase | 5 | 5 | 0 | 2 | 2045.8 | 8.82 | 2.289e-09 |
| `fill5_p20_b8` | case2383wp | 6 | 6 | 0 | 2 | 469.0 | 25.13 | 7.513e-12 |
| `fill5_p20_b8` | case3120sp | 6 | 6 | 0 | 2 | 482.9 | 5.06 | 1.084e-11 |
| `fill5_p20_b8` | case6468rte | 3 | 3 | 0 | 1 | 679.9 | 3.22 | 8.138e-12 |
| `fill5_p20_b8` | case9241pegase | 6 | 6 | 0 | 2 | 1877.1 | 8.78 | 2.129e-09 |
| `fill5_p20_b8` | case13659pegase | 5 | 5 | 0 | 2 | 2156.2 | 8.88 | 2.289e-09 |
| `fill10_p10_b8` | case2383wp | 6 | 6 | 0 | 2 | 701.5 | 25.28 | 5.834e-12 |
| `fill10_p10_b8` | case3120sp | 6 | 6 | 0 | 2 | 692.7 | 5.07 | 7.433e-12 |
| `fill10_p10_b8` | case6468rte | 3 | 3 | 0 | 1 | 1870.9 | 3.23 | 7.858e-12 |
| `fill10_p10_b8` | case9241pegase | 6 | 6 | 0 | 2 | 1922.0 | 8.79 | 2.129e-09 |
| `fill10_p20_b4` | case2383wp | 6 | 6 | 0 | 2 | 841.0 | 25.16 | 5.435e-12 |
| `fill10_p20_b4` | case3120sp | 6 | 6 | 0 | 2 | 1116.9 | 5.06 | 7.607e-12 |
| `fill10_p20_b4` | case6468rte | 3 | 3 | 0 | 1 | 2008.9 | 3.23 | 1.132e-11 |
| `fill10_p20_b4` | case9241pegase | 6 | 6 | 0 | 2 | 3872.2 | 8.73 | 2.129e-09 |
| `fill10_p20_b8_case2383_only` | case2383wp | 6 | 6 | 0 | 2 | 1048.4 | 25.81 | 7.138e-12 |

## OOM / Stability Notes
- `fill=10` triggered CUDA out-of-memory in the Ginkgo ParILUT path during larger-case runs (`out of memory` occurrences in log: 3).
- In all completed CSV rows, accepted middle steps remained zero; hybrid full cuDSS calls matched pure full cuDSS calls.
- Increasing fill/ParILUT iterations/outer BiCGSTAB iterations therefore did not reproduce the old CPU Eigen ILUT behavior in this implementation path.

## Interpretation
The earlier CPU ILUT result was not just “more fill”; it used Eigen `IncompleteLUT` with `fill_factor=10`, `drop_tol=1e-4`, and much more flexible outer iterations. The current Ginkgo ParILUT path is a parallel approximate ILU rebuilt through a host-side integration path. Raising fill to 5 did not improve NR acceptance, and fill 10 became memory-heavy without reliable acceptance.
