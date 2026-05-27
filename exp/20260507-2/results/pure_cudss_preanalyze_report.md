# Pure cuDSS pre-analyze report

Full-J cuDSS analysis is performed once before the NR loop. The NR-loop timing below contains Jacobian/mismatch/update work plus cuDSS factorize+solve, but not cuDSS analyze.

| case | NR iters | analyze setup ms | NR loop total ms | linear loop total ms | avg factor ms | avg solve ms | avg factor+solve ms | median factor+solve ms | final mismatch inf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 6 | 21.906 | 25.547 | 17.983 | 2.324 | 0.673 | 2.997 | 0.504 | 4.824e-12 |
| case3120sp | 6 | 19.727 | 5.714 | 3.823 | 0.411 | 0.226 | 0.637 | 0.623 | 1.097e-11 |
| case9241pegase | 6 | 42.836 | 9.949 | 7.687 | 0.926 | 0.355 | 1.281 | 1.259 | 2.129e-09 |
| case13659pegase | 5 | 52.908 | 10.194 | 7.979 | 1.152 | 0.444 | 1.596 | 1.562 | 2.289e-09 |
| case6468rte | 3 | 30.458 | 3.675 | 2.556 | 0.557 | 0.295 | 0.852 | 0.832 | 7.986e-12 |

Correct comparison for a replacement middle step is `avg/median factor+solve ms`, not cold total with analyze.
