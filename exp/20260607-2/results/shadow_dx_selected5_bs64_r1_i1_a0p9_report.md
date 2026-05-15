# Shadow Dx Diagnostic: Selected 5 Cases

## Setting

- cases: case2383wp, case3120sp, case9241pegase, case13659pegase, case6468rte
- block_size: 64
- gmres_restart: 1
- gmres_iters: 1
- gmres_fixed_iter_mode: true
- polish_threshold: 1e-4
- accept_mismatch_ratio: 0.9
- reject_mismatch_ratio: 1.05
- fallback_policy: immediate
- shadow_dx_diagnostic: true

## Output

- summary: `results/shadow_dx_selected5_bs64_r1_i1_a0p9.csv`
- iteration log: `results/shadow_dx_selected5_bs64_r1_i1_a0p9_iters.csv`
- requested shadow CSV: `results/shadow_dx_selected5_bs64_r1_i1_a0p9_requested.csv`

## Run Summary

| case | converged | NR iters | cuDSS | GMRES | accepted | rejected | fallback | polish | hybrid time | pure cuDSS | speedup | shadow diag s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | true | 12 | 5 | 10 | 7 | 3 | 3 | 1 | 0.031052 | 0.045674 | 1.471x | 0.043445 |
| case3120sp | true | 12 | 6 | 10 | 6 | 4 | 4 | 1 | 0.034546 | 0.025391 | 0.735x | 0.045787 |
| case9241pegase | true | 12 | 4 | 10 | 8 | 2 | 2 | 1 | 0.062619 | 0.052947 | 0.846x | 0.079185 |
| case13659pegase | true | 12 | 5 | 10 | 7 | 3 | 3 | 1 | 0.077226 | 0.063151 | 0.818x | 0.093271 |
| case6468rte | true | 5 | 2 | 3 | 3 | 0 | 0 | 1 | 0.036760 | 0.035127 | 0.956x | 0.040221 |

## Shadow Dx Aggregate

| case | shadow rows | avg dx cosine | avg dx norm ratio | avg GMRES nonlinear ratio inf | avg cuDSS nonlinear ratio inf | avg step efficiency inf | avg GMRES total ms | avg shadow cuDSS ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case13659pegase | 10 | 0.027 | 0.006 | 0.857 | 6.129e-02 | 0.152 | 1.122 | 6.710 |
| case2383wp | 10 | 0.516 | 0.022 | 0.754 | 1.289e+00 | -0.088 | 0.822 | 2.243 |
| case3120sp | 10 | 0.436 | 0.031 | 0.819 | 4.464e-01 | -0.380 | 0.741 | 2.553 |
| case6468rte | 3 | 0.456 | 0.144 | 0.503 | 3.173e-05 | 0.497 | 0.869 | 11.115 |
| case9241pegase | 10 | 0.115 | 0.061 | 0.686 | 5.259e-02 | 0.350 | 1.016 | 5.445 |

## Notes

- The requested shadow CSV contains 43 middle-iteration rows.
- Shadow diagnostic time is reported separately and is excluded from hybrid `total_seconds`.
- GMRES dx is generally much smaller than cuDSS dx, especially on case13659pegase and case2383wp.
- dx cosine is low on case13659pegase and case9241pegase, suggesting the one-step GMRES correction direction is often far from the direct correction.

