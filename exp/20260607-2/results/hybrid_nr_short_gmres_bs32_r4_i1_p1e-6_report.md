# Hybrid NR Short GMRES Report

## Setting

- cases: case2383wp, case3120sp, case9241pegase, case13659pegase, case6468rte
- block_size: 32
- gmres_restart: 4
- gmres_fixed_iters: 1
- force_gmres_min_steps: 0
- polish_threshold: 1e-6
- accept_mismatch_ratio: 0.95
- reject_mismatch_ratio: 1.05
- fallback_policy: immediate

## Summary

| case | converged | NR iters | cuDSS calls | GMRES calls | accepted GMRES | rejected GMRES | fallback | polish | hybrid total (s) | pure cuDSS (s) | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | yes | 15 | 5 | 14 | 10 | 4 | 4 | 0 | 0.032075 | 0.021922 | 0.683x |
| case3120sp | yes | 11 | 6 | 9 | 5 | 4 | 4 | 1 | 0.031998 | 0.025225 | 0.788x |
| case9241pegase | yes | 16 | 4 | 15 | 12 | 3 | 3 | 0 | 0.062489 | 0.052292 | 0.837x |
| case13659pegase | yes | 12 | 5 | 11 | 7 | 4 | 4 | 0 | 0.072770 | 0.062331 | 0.857x |
| case6468rte | yes | 5 | 3 | 3 | 2 | 1 | 1 | 1 | 0.036849 | 0.034051 | 0.924x |

## Notes

- All five cases converged with this short GMRES policy.
- Compared with the previous permissive setting, case3120sp now converges instead of hitting max_nr_iters.
- Hybrid is still slower than pure cuDSS on all five cases, but the slowdown is much smaller than the previous long-accepting policy.
- Accepted GMRES steps have high linear residuals, but several still reduce NR mismatch enough to be accepted.

