# Bp/Bpp FDLF 2-round report

## Convention
Stage 0 checked RHS sign/scaling on `case2383wp` for two NR steps. Best observed convention was `P=-r/|V|`, `Q=r/|V|` with final mismatch_inf `3.4077e+01` after two steps. This convention was fixed for the 5-case NR run.

## NR Result
| case | converged | NR iters | full-J cuDSS calls in FDLF path | full-J cuDSS analyze in FDLF path ms | final mismatch inf | pure NR iters | pure total ms | FDLF total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | false | 12 | 0 | 0.000 | inf | 6 | 4.998 | 9.283 |
| case3120sp | false | 12 | 0 | 0.000 | 7.58152644841e+278 | 6 | 5.070 | 8.846 |
| case9241pegase | false | 11 | 0 | 0.000 | inf | 6 | 8.666 | 10.315 |
| case13659pegase | false | 8 | 0 | 0.000 | 1.47962152284e+283 | 5 | 8.805 | 8.679 |
| case6468rte | false | 16 | 0 | 0.000 | 7.50473700168e+226 | 3 | 3.217 | 13.451 |

## Per-iteration Solve Cost
| case | pure cuDSS warm factor+solve median ms | Bp/Bpp 2-round median ms | Bp/Bpp / pure | Bp/Bpp setup analyze+factor ms |
|---|---:|---:|---:|---:|
| case2383wp | 0.504 | 0.392 | 0.78 | 29.671 |
| case3120sp | 0.544 | 0.394 | 0.72 | 30.171 |
| case9241pegase | 1.092 | 0.555 | 0.51 | 64.389 |
| case13659pegase | 1.352 | 0.636 | 0.47 | 83.936 |
| case6468rte | 0.716 | 0.484 | 0.68 | 53.293 |

## Answers
- Bp/Bpp 2-round did not converge on any of the 5 cases without full-J cuDSS fallback.
- The FDLF path made zero full-J cuDSS calls and zero full-J cuDSS analyze calls in the primary NR loop/path.
- Per-iteration Bp/Bpp solve-only wall time is lower than warm full-J cuDSS factor+solve on all 5 cases, roughly 47-79% of pure cuDSS median.
- Total NR time does not beat pure cuDSS in any meaningful sense because the nonlinear trajectory diverges and reaches bad dx/max failure before convergence.
- Cross round did not stabilize the trajectory in this implementation; mismatch usually decreases only in the first one or two steps and then grows rapidly.
- Decision: keep the timing result as useful evidence that fixed Bp/Bpp solves are cheap, but do not keep this Bp/Bpp-only 2-round path as a full-J cuDSS replacement.

## Files
- `results/fdlf_bpbpp_quality_summary.csv`
- `results/fdlf_bpbpp_quality_report.md`
- `results/fdlf_bpbpp_nr_summary.csv`
- `results/fdlf_bpbpp_nr_iters.csv`
- `results/fdlf_bpbpp_nr_timing.csv`
