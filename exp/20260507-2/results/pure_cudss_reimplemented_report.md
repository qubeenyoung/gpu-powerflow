# Pure cuDSS Reimplemented Timing

Full-J cuDSS `analyze()` is now executed before `total_start`, i.e. outside the NR loop, when `--full-cudss-analyze-before-loop true` is used. Iteration rows contain factorize + solve only for cuDSS linear work.

## Warmed run (`--warmup 1`)
| case | NR iters | analyze setup ms | NR loop total ms | factor+solve sum ms | avg factor ms | avg solve ms | avg factor+solve ms | median factor+solve ms | first factor+solve ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 5 | 52.393 | 10.117 | 7.883 | 1.131 | 0.445 | 1.577 | 1.559 | 1.650 |
| case2383wp | 6 | 17.171 | 4.978 | 3.095 | 0.326 | 0.190 | 0.516 | 0.503 | 0.579 |
| case3120sp | 6 | 19.107 | 5.738 | 3.833 | 0.412 | 0.226 | 0.639 | 0.627 | 0.702 |
| case6468rte | 3 | 30.442 | 3.642 | 2.540 | 0.560 | 0.287 | 0.847 | 0.821 | 0.899 |
| case9241pegase | 6 | 42.628 | 9.960 | 7.671 | 0.922 | 0.356 | 1.278 | 1.261 | 1.364 |

## Cold library run, for reference
| case | analyze setup ms | first factor+solve ms | median factor+solve ms |
|---|---:|---:|---:|
| case13659pegase | 52.908 | 1.730 | 1.562 |
| case2383wp | 21.906 | 15.464 | 0.504 |
| case3120sp | 19.727 | 0.705 | 0.623 |
| case6468rte | 30.458 | 0.906 | 0.832 |
| case9241pegase | 42.836 | 1.397 | 1.259 |

## Correct comparison
Use the warmed `avg/median factor+solve ms` columns for comparing one full-J cuDSS Newton linear solve against an A1 middle step. The cold library first factorization is a separate startup effect and should not be used as the per-iteration cuDSS cost.
