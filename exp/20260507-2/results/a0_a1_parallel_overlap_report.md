# A0/A1 Parallel J11-J22 Correction Report

## What Changed

- J11 and J22 now use separate non-blocking CUDA streams in both standalone quality and hybrid NR paths.
- A0 launches J11/J22 factorization concurrently, then J11/J22 solve concurrently.
- A1 launches the same concurrent round 0, then reuses the same factors and launches a second concurrent solve-only round for the cross residuals.
- J11/J22 analyze is done once before NR middle iterations in hybrid. It is not repeated inside the NR middle loop.

## Main Finding

The cuDSS factor/solve GPU work is now small and overlapped. The remaining end-to-end correction wall time is dominated by glue around the solves: copying J/F/dx to host, building J11/J22 values and RHS vectors, computing residual/cross residuals on host, and copying the corrected dx back to device. The column `non-cuDSS-event overhead` below is `field correction wall - serial cuDSS event sum`; it is an approximate lower-bound view of this glue cost.

- A0: mean wall `1.596 ms`, mean serial cuDSS event sum `0.156 ms`, mean non-cuDSS-event overhead `1.440 ms`.
- A1: mean wall `2.217 ms`, mean serial cuDSS event sum `0.248 ms`, mean non-cuDSS-event overhead `1.969 ms`.

## Per-Case Correction Timing

| case | block | mode | middle rows | wall ms | serial cuDSS event sum ms | non-cuDSS-event overhead ms | overhead share | J11 factor | J11 solve | J22 factor | J22 solve |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 8 | A0 | 11 | 0.840 | 0.121 | 0.719 | 85.9% | 0.031 | 0.033 | 0.027 | 0.030 |
| case2383wp | 8 | A1 | 9 | 1.171 | 0.184 | 0.987 | 84.5% | 0.034 | 0.064 | 0.028 | 0.058 |
| case2383wp | 16 | A0 | 6 | 0.875 | 0.133 | 0.741 | 85.2% | 0.038 | 0.033 | 0.032 | 0.029 |
| case2383wp | 16 | A1 | 5 | 1.210 | 0.196 | 1.014 | 84.1% | 0.040 | 0.064 | 0.034 | 0.058 |
| case3120sp | 8 | A0 | 5 | 0.989 | 0.135 | 0.854 | 86.5% | 0.039 | 0.033 | 0.034 | 0.029 |
| case3120sp | 8 | A1 | 4 | 1.389 | 0.200 | 1.189 | 85.7% | 0.042 | 0.063 | 0.038 | 0.057 |
| case3120sp | 16 | A0 | 4 | 0.990 | 0.143 | 0.846 | 85.7% | 0.043 | 0.034 | 0.037 | 0.029 |
| case3120sp | 16 | A1 | 6 | 1.373 | 0.193 | 1.180 | 86.0% | 0.036 | 0.063 | 0.033 | 0.061 |
| case9241pegase | 8 | A0 | 6 | 1.974 | 0.135 | 1.839 | 93.2% | 0.039 | 0.034 | 0.033 | 0.029 |
| case9241pegase | 8 | A1 | 4 | 2.700 | 0.209 | 2.491 | 92.3% | 0.046 | 0.067 | 0.038 | 0.059 |
| case9241pegase | 16 | A0 | 4 | 1.986 | 0.146 | 1.840 | 92.7% | 0.044 | 0.035 | 0.037 | 0.030 |
| case9241pegase | 16 | A1 | 4 | 2.741 | 0.223 | 2.519 | 92.0% | 0.052 | 0.067 | 0.044 | 0.059 |
| case13659pegase | 8 | A0 | 5 | 2.555 | 0.160 | 2.395 | 93.8% | 0.044 | 0.045 | 0.037 | 0.034 |
| case13659pegase | 8 | A1 | 3 | 3.440 | 0.228 | 3.212 | 93.4% | 0.055 | 0.069 | 0.046 | 0.059 |
| case13659pegase | 16 | A0 | 8 | 2.441 | 0.127 | 2.315 | 94.8% | 0.035 | 0.034 | 0.029 | 0.029 |
| case13659pegase | 16 | A1 | 3 | 3.339 | 0.223 | 3.116 | 93.4% | 0.052 | 0.068 | 0.043 | 0.059 |
| case6468rte | 8 | A0 | 1 | 1.709 | 0.273 | 1.436 | 84.0% | 0.115 | 0.040 | 0.090 | 0.029 |
| case6468rte | 8 | A1 | 1 | 2.419 | 0.391 | 2.028 | 83.8% | 0.144 | 0.077 | 0.108 | 0.062 |
| case6468rte | 16 | A0 | 2 | 1.601 | 0.189 | 1.412 | 88.4% | 0.068 | 0.035 | 0.057 | 0.030 |
| case6468rte | 16 | A1 | 1 | 2.386 | 0.431 | 1.955 | 81.9% | 0.124 | 0.096 | 0.131 | 0.080 |

## Standalone J1 Quality And Cost

| case | block | mode | dx error after | theta error after | |V| error after | full residual after | P residual after | Q residual after | correction/full | correction wall ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 8 | A0 | 0.03251 | 0.02884 | 1.552 | 0.03979 | 0.2654 | 0.03886 | 0.169 | 0.457 |
| case2383wp | 8 | A1 | 0.04308 | 0.04307 | 0.09133 | 0.009779 | 0.2821 | 0.003371 | 0.234 | 0.688 |
| case2383wp | 16 | A0 | 0.03435 | 0.03106 | 1.517 | 0.03866 | 0.1289 | 0.03846 | 0.168 | 0.452 |
| case2383wp | 16 | A1 | 0.04265 | 0.04264 | 0.0888 | 0.00826 | 0.2502 | 0.001393 | 0.233 | 0.680 |
| case3120sp | 8 | A0 | 0.08629 | 0.07669 | 1.791 | 0.3412 | 0.6478 | 0.3219 | 1.230 | 0.513 |
| case3120sp | 8 | A1 | 0.1064 | 0.1063 | 0.1796 | 0.1225 | 0.5649 | 0.04626 | 1.721 | 0.791 |
| case3120sp | 16 | A0 | 0.08744 | 0.07677 | 1.896 | 0.3052 | 0.2581 | 0.307 | 1.231 | 0.514 |
| case3120sp | 16 | A1 | 0.1123 | 0.1123 | 0.1549 | 0.1089 | 0.5238 | 0.02753 | 1.715 | 0.789 |
| case9241pegase | 8 | A0 | 0.0817 | 0.08239 | 0.07034 | 0.05819 | 0.06032 | 0.01279 | 0.993 | 0.827 |
| case9241pegase | 8 | A1 | 0.03103 | 0.02405 | 0.08279 | 0.01041 | 0.004046 | 0.03587 | 1.331 | 1.367 |
| case9241pegase | 16 | A0 | 0.01993 | 0.01775 | 0.04071 | 0.02145 | 0.02213 | 0.008876 | 0.994 | 0.831 |
| case9241pegase | 16 | A1 | 0.01536 | 0.01262 | 0.03754 | 0.003745 | 0.0005871 | 0.01376 | 1.377 | 1.493 |
| case13659pegase | 8 | A0 | 0.06884 | 0.06881 | 0.1425 | 0.08784 | 0.0931 | 0.07597 | 1.041 | 1.122 |
| case13659pegase | 8 | A1 | 0.03995 | 0.03989 | 0.1557 | 0.02191 | 0.009709 | 0.03561 | 1.354 | 1.780 |
| case13659pegase | 16 | A0 | 0.06594 | 0.06591 | 0.1443 | 0.08565 | 0.09009 | 0.07579 | 1.033 | 1.106 |
| case13659pegase | 16 | A1 | 0.03849 | 0.03842 | 0.155 | 0.02082 | 0.008875 | 0.03403 | 1.362 | 1.780 |
| case6468rte | 8 | A0 | 0.08993 | 0.05794 | 0.1908 | 0.3024 | 0.9075 | 0.1603 | 1.318 | 0.769 |
| case6468rte | 8 | A1 | 0.02704 | 0.02222 | 0.04643 | 0.06462 | 0.1452 | 0.05154 | 1.781 | 1.202 |
| case6468rte | 16 | A0 | 0.09635 | 0.08528 | 0.146 | 0.3118 | 0.8652 | 0.1967 | 1.318 | 0.769 |
| case6468rte | 16 | A1 | 0.04364 | 0.01946 | 0.1051 | 0.08983 | 0.2036 | 0.07121 | 1.780 | 1.206 |

## Hybrid NR Case Results

| case | block | mode | converged | NR iters | full cuDSS calls | middle calls | accepted | fallback | hybrid ms | pure cuDSS ms | speedup | final mismatch inf |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 8 | BJ | true | 7 | 5 | 6 | 2 | 4 | 29.1 | 47.4 | 1.628 | 4.496e-10 |
| case2383wp | 8 | A0 | true | 13 | 2 | 11 | 11 | 0 | 43.1 | 42.1 | 0.978 | 1.618e-10 |
| case2383wp | 8 | A1 | true | 11 | 2 | 9 | 9 | 0 | 41.8 | 41.9 | 1.002 | 2.128e-10 |
| case2383wp | 16 | BJ | true | 8 | 5 | 7 | 3 | 4 | 30.4 | 47.2 | 1.553 | 6.235e-10 |
| case2383wp | 16 | A0 | true | 9 | 3 | 7 | 6 | 1 | 35.6 | 42.0 | 1.181 | 3.991e-12 |
| case2383wp | 16 | A1 | true | 7 | 2 | 6 | 5 | 1 | 34.6 | 42.0 | 1.214 | 6.148e-09 |
| case3120sp | 8 | BJ | true | 9 | 6 | 7 | 3 | 4 | 33.4 | 25.4 | 0.760 | 7.580e-12 |
| case3120sp | 8 | A0 | true | 9 | 4 | 7 | 5 | 2 | 39.4 | 25.1 | 0.637 | 4.910e-12 |
| case3120sp | 8 | A1 | true | 8 | 4 | 6 | 4 | 2 | 39.5 | 25.0 | 0.633 | 7.849e-12 |
| case3120sp | 16 | BJ | true | 8 | 6 | 6 | 2 | 4 | 32.4 | 25.4 | 0.786 | 7.930e-12 |
| case3120sp | 16 | A0 | true | 8 | 4 | 6 | 4 | 2 | 37.1 | 25.0 | 0.675 | 1.133e-11 |
| case3120sp | 16 | A1 | true | 9 | 3 | 7 | 6 | 1 | 41.5 | 25.0 | 0.603 | 7.572e-12 |
| case9241pegase | 8 | BJ | true | 7 | 4 | 5 | 3 | 2 | 61.5 | 55.8 | 0.907 | 2.616e-10 |
| case9241pegase | 8 | A0 | true | 9 | 3 | 7 | 6 | 1 | 78.6 | 52.8 | 0.672 | 9.819e-12 |
| case9241pegase | 8 | A1 | true | 7 | 3 | 5 | 4 | 1 | 73.1 | 52.9 | 0.724 | 1.085e-11 |
| case9241pegase | 16 | BJ | true | 7 | 5 | 5 | 2 | 3 | 62.8 | 53.0 | 0.844 | 6.998e-12 |
| case9241pegase | 16 | A0 | true | 8 | 4 | 6 | 4 | 2 | 75.5 | 53.0 | 0.702 | 6.762e-12 |
| case9241pegase | 16 | A1 | true | 7 | 3 | 5 | 4 | 1 | 74.4 | 53.1 | 0.714 | 1.587e-11 |
| case13659pegase | 8 | BJ | true | 8 | 5 | 6 | 3 | 3 | 81.8 | 66.2 | 0.809 | 7.080e-12 |
| case13659pegase | 8 | A0 | true | 8 | 3 | 7 | 5 | 2 | 100.3 | 63.3 | 0.631 | 2.244e-09 |
| case13659pegase | 8 | A1 | true | 6 | 3 | 5 | 3 | 2 | 92.4 | 63.5 | 0.687 | 5.447e-10 |
| case13659pegase | 16 | BJ | true | 8 | 5 | 6 | 3 | 3 | 81.2 | 63.6 | 0.783 | 5.448e-12 |
| case13659pegase | 16 | A0 | true | 11 | 3 | 9 | 8 | 1 | 109.1 | 63.0 | 0.577 | 1.806e-10 |
| case13659pegase | 16 | A1 | true | 6 | 3 | 4 | 3 | 1 | 83.6 | 63.2 | 0.756 | 3.036e-10 |
| case6468rte | 8 | BJ | true | 3 | 2 | 2 | 1 | 1 | 37.1 | 34.1 | 0.918 | 3.463e-09 |
| case6468rte | 8 | A0 | true | 3 | 2 | 1 | 1 | 0 | 36.9 | 34.1 | 0.924 | 3.607e-10 |
| case6468rte | 8 | A1 | true | 3 | 2 | 1 | 1 | 0 | 38.2 | 34.1 | 0.894 | 4.570e-11 |
| case6468rte | 16 | BJ | true | 4 | 2 | 3 | 2 | 1 | 38.8 | 34.2 | 0.881 | 8.587e-09 |
| case6468rte | 16 | A0 | true | 4 | 2 | 2 | 2 | 0 | 40.3 | 34.3 | 0.849 | 1.317e-11 |
| case6468rte | 16 | A1 | true | 3 | 2 | 1 | 1 | 0 | 37.9 | 34.1 | 0.901 | 1.359e-10 |

## Case Notes

- case2383wp: best A0/A1 is A1 block 16: speedup 1.214, NR 7, cuDSS calls 2, fallback 1. best BJ baseline is block 8: speedup 1.628, cuDSS calls 5, fallback 4.
- case3120sp: best A0/A1 is A0 block 16: speedup 0.675, NR 8, cuDSS calls 4, fallback 2. best BJ baseline is block 16: speedup 0.786, cuDSS calls 6, fallback 4.
- case9241pegase: best A0/A1 is A1 block 8: speedup 0.724, NR 7, cuDSS calls 3, fallback 1. best BJ baseline is block 8: speedup 0.907, cuDSS calls 4, fallback 2.
- case13659pegase: best A0/A1 is A1 block 16: speedup 0.756, NR 6, cuDSS calls 3, fallback 1. best BJ baseline is block 8: speedup 0.809, cuDSS calls 5, fallback 3.
- case6468rte: best A0/A1 is A0 block 8: speedup 0.924, NR 3, cuDSS calls 2, fallback 0. best BJ baseline is block 8: speedup 0.918, cuDSS calls 2, fallback 1.

## Interpretation

1. The intended concurrency is active: J11/J22 factorize and solve event timings are much smaller than the end-to-end correction wall time, and serial event sums are below the wall time.
2. The dominant remaining cost is not the cuDSS factor/solve kernels themselves. It is the CPU/device plumbing around them.
3. A1 improves standalone residual quality compared with A0, especially |V| and Q residual, but A1 has higher wall time and still does not produce broad hybrid speedup.
4. A0/A1 reduce fallback versus plain BJ in several cases, but NR iteration count usually grows enough that total time remains worse than pure cuDSS.
5. case2383wp remains the only clearly promising case. For the other four cases, A0/A1 reduce some direct calls but do not beat pure cuDSS end-to-end.

## Next Bottleneck If Continuing

To make A0/A1 a fair performance candidate, the next implementation target is not more cuDSS overlap. It is moving residual computation, J11/J22 value extraction, RHS construction, A1 cross residual products, and dx accumulation back to device-resident kernels so the middle step avoids full vector/matrix D2H/H2D traffic.
