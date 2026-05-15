# Bp/Bpp residual refinement report

## Setup
- Predictor: BiCGSTAB(2) + METIS block-Jacobi, block_size=16.
- Refinement: fixed Bp/Bpp from `-Im(Ybus)`, analyze/factorize once before NR; NR loop uses solve only.
- Stage 0 residual-refinement convention check on `case2383wp` selected `P=r`, `Q=-r/|V|`.
- Policy: bootstrap full cuDSS=1, accept=0.5, reject=1.05, fallback=immediate, max middle accepts=2, polish=1e-4.

## Case summary
| case | mode | conv | NR | cuDSS calls | middle calls | accepted middle | fallback | total ms | linear ms | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | pure_cudss | true | 6 | 6 | 0 | 0 | 0 | 4.998 | 3.101 | 1.000 |
| case2383wp | bicgstab2_bj | true | 7 | 5 | 6 | 2 | 4 | 10.698 | 8.071 | 0.462 |
| case2383wp | bicgstab2_bj_bpbpp_refine | true | 8 | 6 | 4 | 2 | 2 | 11.583 | 8.521 | 0.435 |
| case3120sp | pure_cudss | true | 6 | 6 | 0 | 0 | 0 | 5.070 | 3.372 | 1.000 |
| case3120sp | bicgstab2_bj | true | 8 | 6 | 6 | 2 | 4 | 13.057 | 10.064 | 0.438 |
| case3120sp | bicgstab2_bj_bpbpp_refine | true | 6 | 6 | 4 | 0 | 4 | 12.674 | 9.929 | 0.452 |
| case9241pegase | pure_cudss | true | 6 | 6 | 0 | 0 | 0 | 8.666 | 6.633 | 1.000 |
| case9241pegase | bicgstab2_bj | true | 7 | 5 | 5 | 2 | 3 | 20.481 | 17.328 | 0.484 |
| case9241pegase | bicgstab2_bj_bpbpp_refine | true | 7 | 5 | 3 | 2 | 1 | 18.002 | 14.848 | 0.550 |
| case13659pegase | pure_cudss | true | 5 | 5 | 0 | 0 | 0 | 8.805 | 6.846 | 1.000 |
| case13659pegase | bicgstab2_bj | true | 8 | 5 | 6 | 3 | 3 | 29.173 | 25.097 | 0.345 |
| case13659pegase | bicgstab2_bj_bpbpp_refine | true | 5 | 5 | 4 | 0 | 4 | 25.789 | 22.448 | 0.391 |
| case6468rte | pure_cudss | true | 3 | 3 | 0 | 0 | 0 | 3.217 | 2.234 | 1.000 |
| case6468rte | bicgstab2_bj | true | 4 | 3 | 2 | 1 | 1 | 7.567 | 5.936 | 0.484 |
| case6468rte | bicgstab2_bj_bpbpp_refine | true | 3 | 3 | 1 | 0 | 1 | 6.222 | 4.844 | 0.587 |

## Middle-step quality and cost
| case | BJ mean lin rel | refined mean lin rel | BJ fallback | refined fallback | refined median middle ms | refined median Bp/Bpp ms |
|---|---:|---:|---:|---:|---:|---:|
| case2383wp | 1.23 | 0.852 | 4 | 2 | 1.336 | 0.415 |
| case3120sp | 13.7 | 1.35 | 4 | 4 | 1.517 | 0.475 |
| case9241pegase | 3.29 | 10.7 | 3 | 1 | 2.804 | 0.622 |
| case13659pegase | 1.73 | 0.628 | 3 | 4 | 3.634 | 0.747 |
| case6468rte | 0.652 | 1.65 | 1 | 1 | 2.294 | 0.592 |

## Answers
1. Bp/Bpp refinement improved the linear residual on 3/5 cases (`case2383wp`, `case3120sp`, `case13659pegase`), worsened it on `case9241pegase` and `case6468rte`.
2. Fallback decreased on `case2383wp` and `case9241pegase`, stayed the same on `case3120sp`/`case6468rte`, and increased on `case13659pegase`; not a robust fallback reduction.
3. NR iterations were not consistently close to pure cuDSS: some cases matched pure, but `case2383wp` and `case9241pegase` still took +2/+1 iterations.
4. Refined middle step is not lower than full warm cuDSS once BiCGSTAB+B/J setup plus refinement are included; the Bp/Bpp solve-only part is small, but total middle is larger.
5. Total NR time is lower than pure cuDSS on none of the converged comparison rows; speedup stays below 1.
6. Round1 was not isolated as a separate quality row in this run; the logged result is the full two-round correction. The two-round path did not give enough trajectory improvement to justify replacing J11/J22 A1.
7. Decision: do not replace J11/J22 A1 with fixed Bp/Bpp refinement. It is cheaper per field solve, but too weak/inconsistent as a Newton correction.
