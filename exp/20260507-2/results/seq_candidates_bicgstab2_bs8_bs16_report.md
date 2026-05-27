# Sequential candidates for BiCGSTAB(2) block-Jacobi

Common policy: bootstrap cuDSS 1, polish `1e-4`, accept `0.9`, reject `1.05`, fallback immediate. Cases: `case2383wp`, `case3120sp`, `case9241pegase`, `case13659pegase`, `case6468rte`.

Baseline: BiCGSTAB(2), METIS block-Jacobi, FP32 `inverse_gemv`, no coarse, no scaling, no warm start. Candidates are tested one at a time.

## Aggregate first-pass metrics

| block | variant | NR | cuDSS | middle | accepted | rejected | fallback | accepted middle ratio | agg speedup | avg middle ms | fallback delta | verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | baseline | 34 | 22 | 26 | 12 | 14 | 14 | 0.343 | 0.812 | 0.295 | +0 | baseline |
| 8 | field | 34 | 22 | 26 | 12 | 14 | 14 | 0.363 | 0.764 | 0.301 | +0 | fail fallback check |
| 8 | coarse2 | 32 | 24 | 22 | 8 | 14 | 14 | 0.400 | 0.376 | 3.643 | +0 | fail fallback check |
| 8 | prevdx | 33 | 29 | 24 | 4 | 20 | 20 | 0.308 | 0.818 | 0.304 | +6 | fail fallback check |
| 16 | baseline | 35 | 23 | 27 | 12 | 15 | 15 | 0.365 | 0.808 | 0.294 | +0 | baseline |
| 16 | field | 35 | 23 | 27 | 12 | 15 | 15 | 0.364 | 0.754 | 0.303 | +0 | fail fallback check |
| 16 | coarse2 | 31 | 24 | 22 | 7 | 15 | 15 | 0.351 | 0.594 | 2.109 | +0 | fail fallback check |
| 16 | prevdx | 29 | 26 | 21 | 3 | 18 | 18 | 0.254 | 0.834 | 0.306 | +3 | fail fallback check |

## Case-level fallback and NR changes

### block size 8

| case | candidate | fallback baseline -> candidate | NR baseline -> candidate | middle baseline -> candidate | accepted ratio baseline -> candidate | speedup baseline -> candidate |
|---|---|---:|---:|---:|---:|---:|
| case2383wp | field | 4 -> 4 | 7 -> 7 | 6 -> 6 | 0.253 -> 0.253 | 0.784 -> 0.739 |
| case3120sp | field | 4 -> 4 | 9 -> 9 | 7 -> 7 | 0.420 -> 0.530 | 0.727 -> 0.684 |
| case9241pegase | field | 2 -> 2 | 7 -> 7 | 5 -> 5 | 0.203 -> 0.201 | 0.860 -> 0.803 |
| case13659pegase | field | 3 -> 3 | 8 -> 8 | 6 -> 6 | 0.365 -> 0.356 | 0.773 -> 0.720 |
| case6468rte | field | 1 -> 1 | 3 -> 3 | 2 -> 2 | 0.473 -> 0.474 | 0.916 -> 0.889 |
| case2383wp | coarse2 | 4 -> 3 | 7 -> 9 | 6 -> 7 | 0.253 -> 0.501 | 0.784 -> 0.505 |
| case3120sp | coarse2 | 4 -> 4 | 9 -> 6 | 7 -> 4 | 0.420 -> 0.000 | 0.727 -> 0.564 |
| case9241pegase | coarse2 | 2 -> 3 | 7 -> 7 | 5 -> 5 | 0.203 -> 0.279 | 0.860 -> 0.366 |
| case13659pegase | coarse2 | 3 -> 3 | 8 -> 6 | 6 -> 4 | 0.365 -> 0.313 | 0.773 -> 0.307 |
| case6468rte | coarse2 | 1 -> 1 | 3 -> 4 | 2 -> 2 | 0.473 -> 0.504 | 0.916 -> 0.390 |
| case2383wp | prevdx | 4 -> 5 | 7 -> 8 | 6 -> 6 | 0.253 -> 0.107 | 0.784 -> 0.757 |
| case3120sp | prevdx | 4 -> 6 | 9 -> 10 | 7 -> 8 | 0.420 -> 0.710 | 0.727 -> 0.687 |
| case9241pegase | prevdx | 2 -> 4 | 7 -> 7 | 5 -> 5 | 0.203 -> 0.108 | 0.860 -> 0.816 |
| case13659pegase | prevdx | 3 -> 4 | 8 -> 5 | 6 -> 4 | 0.365 -> 0.000 | 0.773 -> 0.848 |
| case6468rte | prevdx | 1 -> 1 | 3 -> 3 | 2 -> 1 | 0.473 -> 0.000 | 0.916 -> 0.940 |

### block size 16

| case | candidate | fallback baseline -> candidate | NR baseline -> candidate | middle baseline -> candidate | accepted ratio baseline -> candidate | speedup baseline -> candidate |
|---|---|---:|---:|---:|---:|---:|
| case2383wp | field | 4 -> 4 | 8 -> 8 | 7 -> 7 | 0.414 -> 0.412 | 0.757 -> 0.697 |
| case3120sp | field | 4 -> 4 | 8 -> 8 | 6 -> 6 | 0.290 -> 0.289 | 0.768 -> 0.713 |
| case9241pegase | field | 3 -> 3 | 7 -> 7 | 5 -> 5 | 0.125 -> 0.125 | 0.846 -> 0.788 |
| case13659pegase | field | 3 -> 3 | 8 -> 8 | 6 -> 6 | 0.436 -> 0.416 | 0.782 -> 0.725 |
| case6468rte | field | 1 -> 1 | 4 -> 4 | 3 -> 3 | 0.558 -> 0.578 | 0.873 -> 0.835 |
| case2383wp | coarse2 | 4 -> 3 | 8 -> 9 | 7 -> 7 | 0.414 -> 0.479 | 0.757 -> 0.609 |
| case3120sp | coarse2 | 4 -> 4 | 8 -> 6 | 6 -> 4 | 0.290 -> 0.000 | 0.768 -> 0.696 |
| case9241pegase | coarse2 | 3 -> 3 | 7 -> 7 | 5 -> 5 | 0.125 -> 0.125 | 0.846 -> 0.589 |
| case13659pegase | coarse2 | 3 -> 4 | 8 -> 6 | 6 -> 5 | 0.436 -> 0.449 | 0.782 -> 0.501 |
| case6468rte | coarse2 | 1 -> 1 | 4 -> 3 | 3 -> 1 | 0.558 -> 0.000 | 0.873 -> 0.767 |
| case2383wp | prevdx | 4 -> 5 | 8 -> 9 | 7 -> 7 | 0.414 -> 0.371 | 0.757 -> 0.724 |
| case3120sp | prevdx | 4 -> 4 | 8 -> 6 | 6 -> 4 | 0.290 -> 0.000 | 0.768 -> 0.838 |
| case9241pegase | prevdx | 3 -> 4 | 7 -> 6 | 5 -> 5 | 0.125 -> 0.136 | 0.846 -> 0.801 |
| case13659pegase | prevdx | 3 -> 4 | 8 -> 5 | 6 -> 4 | 0.436 -> 0.000 | 0.782 -> 0.852 |
| case6468rte | prevdx | 1 -> 1 | 4 -> 3 | 3 -> 1 | 0.558 -> 0.000 | 0.873 -> 0.948 |

## Judgment

- `field`: total fallback delta across block 8/16 is `+0`, total NR delta `+0`. It fails the fallback-first check.
- `coarse2`: total fallback delta across block 8/16 is `+0`, total NR delta `-6`. It fails the fallback-first check.
- `prevdx`: total fallback delta across block 8/16 is `+9`, total NR delta `-7`. It fails the fallback-first check.
- Speedup is not used as the primary gate here, but 2-variable coarse is much slower because coarse solve overhead dominates.
- `previous dx warm start` reduces NR iterations in a few cases, but aggregate fallback increases; it fails the fallback-first gate.
