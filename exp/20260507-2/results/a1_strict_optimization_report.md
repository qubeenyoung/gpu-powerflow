# A1 Strict Optimization Report

## Scope
Fixed math/policy: BiCGSTAB(2) + METIS block-Jacobi + device A1, block_size=16, strict accept=0.5, max A1 accepts=2, bootstrap full cuDSS=1, polish=1e-4, immediate fallback. Four implementation modes only were run.

## Mode Median Timing
| mode | BJ setup ms | BiCGSTAB dot ms | A1 field wall ms | middle/full warm cuDSS | middle rows < full | total speedup cases | linear speedup cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_current | 1.198 | 0.177 | 1.270 | 2.397 | 0/9 | 1/5 | 1/5 |
| optimized_dag_only | 1.231 | 0.185 | 1.115 | 2.381 | 0/9 | 1/5 | 1/5 |
| optimized_reuse_value | 1.217 | 0.144 | 1.135 | 2.345 | 0/9 | 1/5 | 1/5 |
| optimized_numeric_reuse | 0.007 | 0.140 | 1.119 | 0.572 | 8/9 | 4/5 | 1/5 |

## Best Mode: optimized_numeric_reuse
| case | NR iters / pure | full cuDSS calls | A1 middle calls | total speedup | linear speedup | median middle ms | median full warm cuDSS ms | median middle/full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 6 / 6 | 4 | 2 | 1.560 | 1.422 | 2.806 | 0.944 | 2.972 |
| case3120sp | 6 / 6 | 4 | 2 | 0.992 | 0.858 | 0.795 | 1.181 | 0.673 |
| case9241pegase | 6 / 6 | 4 | 2 | 1.053 | 0.862 | 1.177 | 2.894 | 0.407 |
| case13659pegase | 6 / 5 | 4 | 2 | 1.008 | 0.840 | 1.385 | 3.826 | 0.362 |
| case6468rte | 3 / 3 | 2 | 1 | 1.005 | 0.899 | 1.126 | 1.970 | 0.572 |

## Answers
1. Block-Jacobi setup decreased only in `optimized_numeric_reuse`: median BJ setup fell from about 1.20 ms to 0.007 ms per middle step. `value_update_only` did not reduce setup because it still rebuilds the block inverse each middle step.
2. Full A1 DAG reduced A1 field wall modestly: median wall went from 1.270 ms to about 1.115 ms. The field path is still around 1.1 ms, so the stream DAG alone is not enough.
3. Fused BiCGSTAB reduced dot/reduction median from 0.177 ms to about 0.140 ms. This is real but smaller than the A1 field wall and setup effects.
4. `middle_total_ms < full warm cuDSS` happened only with `optimized_numeric_reuse`, and in 8 of 9 A1 middle rows. `case2383wp` remained slower per middle step because its full warm cuDSS reference is especially small.
5. NR trajectory stayed close enough: all modes converged; `optimized_numeric_reuse` kept NR iters within pure cuDSS + 1 for all five cases. `case13659pegase` used 6 iterations versus pure cuDSS 5.
6. Best implementation mode is `optimized_numeric_reuse` by middle-step cost and total-time count, but it only improves linear solve time on 1/5 cases.
7. A1 strict should not be treated as a solved speedup path yet. The only promising piece is numeric reuse; the full linear solve total is still worse than pure cuDSS on 4/5 cases, so continued work would need to attack A1 field wall and total NR linear accounting, not policy tuning.

## Files
- `results/a1_strict_optimization_summary.csv`
- `results/a1_strict_optimization_iters.csv`
- `results/a1_strict_optimization_timing.csv`
- `results/a1_strict_optimization_report.md`
