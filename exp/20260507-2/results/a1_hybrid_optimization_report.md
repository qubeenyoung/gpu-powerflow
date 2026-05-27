# A1 Hybrid Optimization Pass

Fixed policy: bootstrap full cuDSS=1, polish=1e-4, accept=0.5, reject=1.05, fallback=immediate, max A1 accepts=2, block_size=16. Math and NR policy were not changed.

## every_middle
| case | NR | pure NR | full cuDSS | A1 calls | total ms | linear ms | lin speedup | full warm ms | BJ setup ms | BiCGSTAB ms | dot ms | A1 field ms | event wait ms | unacct ms | A1 middle ms | middle/full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 6 | 6 | 4 | 2 | 51.71 | 44.66 | 0.90x | 0.503 | 1.107 | 0.483 | 0.319 | 3.054 | 0.318 | 0.098 | 4.644 | 9.23x |
| case3120sp | 6 | 6 | 4 | 2 | 27.79 | 26.45 | 0.89x | 0.624 | 0.569 | 0.232 | 0.113 | 0.870 | 0.413 | 0.094 | 1.671 | 2.68x |
| case9241pegase | 6 | 6 | 4 | 2 | 57.33 | 56.16 | 0.90x | 1.257 | 1.646 | 0.299 | 0.139 | 1.188 | 0.725 | 0.100 | 3.133 | 2.49x |
| case13659pegase | 6 | 5 | 4 | 2 | 70.99 | 69.85 | 0.88x | 1.551 | 2.289 | 0.322 | 0.152 | 1.487 | 1.021 | 0.097 | 4.098 | 2.64x |
| case6468rte | 3 | 3 | 2 | 1 | 36.41 | 35.72 | 0.93x | 0.822 | 1.162 | 0.276 | 0.142 | 1.252 | 0.717 | 0.098 | 2.691 | 3.27x |

## reuse_after_full_cudss
| case | NR | pure NR | full cuDSS | A1 calls | total ms | linear ms | lin speedup | full warm ms | BJ setup ms | BiCGSTAB ms | dot ms | A1 field ms | event wait ms | unacct ms | A1 middle ms | middle/full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 6 | 6 | 4 | 2 | 50.93 | 44.05 | 0.92x | 0.503 | 0.805 | 0.483 | 0.321 | 3.043 | 0.319 | 0.101 | 4.330 | 8.61x |
| case3120sp | 6 | 6 | 4 | 2 | 27.27 | 25.95 | 0.90x | 0.624 | 0.302 | 0.233 | 0.114 | 0.870 | 0.409 | 0.093 | 1.405 | 2.25x |
| case9241pegase | 6 | 6 | 4 | 2 | 55.90 | 54.70 | 0.93x | 1.257 | 0.838 | 0.293 | 0.138 | 1.187 | 0.725 | 0.092 | 2.318 | 1.84x |
| case13659pegase | 6 | 5 | 4 | 2 | 68.75 | 67.62 | 0.91x | 1.551 | 1.178 | 0.318 | 0.151 | 1.495 | 1.027 | 0.094 | 2.991 | 1.93x |
| case6468rte | 3 | 3 | 2 | 1 | 36.67 | 35.97 | 0.92x | 0.822 | 1.232 | 0.276 | 0.143 | 1.278 | 0.715 | 0.096 | 2.786 | 3.39x |

## reuse_for_2_middle_steps
| case | NR | pure NR | full cuDSS | A1 calls | total ms | linear ms | lin speedup | full warm ms | BJ setup ms | BiCGSTAB ms | dot ms | A1 field ms | event wait ms | unacct ms | A1 middle ms | middle/full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 6 | 6 | 4 | 2 | 50.73 | 43.84 | 0.92x | 0.503 | 0.812 | 0.479 | 0.321 | 3.047 | 0.317 | 0.099 | 4.338 | 8.63x |
| case3120sp | 6 | 6 | 4 | 2 | 27.24 | 25.90 | 0.90x | 0.624 | 0.291 | 0.233 | 0.112 | 0.866 | 0.410 | 0.095 | 1.391 | 2.23x |
| case9241pegase | 6 | 6 | 4 | 2 | 55.72 | 54.55 | 0.93x | 1.257 | 0.840 | 0.295 | 0.138 | 1.194 | 0.730 | 0.095 | 2.329 | 1.85x |
| case13659pegase | 6 | 5 | 4 | 2 | 68.84 | 67.70 | 0.91x | 1.551 | 1.175 | 0.320 | 0.152 | 1.496 | 1.028 | 0.096 | 2.991 | 1.93x |
| case6468rte | 3 | 3 | 2 | 1 | 36.64 | 35.96 | 0.92x | 0.822 | 1.225 | 0.274 | 0.141 | 1.277 | 0.717 | 0.097 | 2.775 | 3.38x |

## Findings
- Best per-case A1 middle remained above full cuDSS warm factor+solve on 5/5 cases.
- Large-case success criterion was not met: 0/2 large cases had A1 middle < full cuDSS warm factor+solve.
- BJ setup reuse reduced the second middle setup cost and preserved the strict NR trajectory, but only two A1 middle steps are used in these cases, so total impact is limited.
- Fused BiCGSTAB removed scalar D2H inside the loop. Dot/reduction time is lower on most cases than the previous host-scalar path but remains a visible share of BiCGSTAB time.
- A1 phase-level cudaDeviceSynchronize was removed from J11/J22 factor/solve scheduling and event wait/unaccounted columns are now logged. The path is still not fully per-side pipelined because J12/J21 cross SpMV remains on the shared cuSPARSE/default-stream path.

## Decision
The optimization pass improves setup reuse and instrumentation, but this version does not satisfy the success criteria: A1 middle is still not below full warm cuDSS on the large cases, and linear solve time remains slower than pure cuDSS.