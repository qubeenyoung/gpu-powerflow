# Global post-correction experiment

## Implementation summary

Implemented a minimal global post-correction hook for `gmres_block_ilu0`. After the existing GMRES+block-ILU step, the code can apply `p = p_gmres + Z c`, where `c` solves `min ||r - A Z c||_2` by a small normal-equation least-squares problem. Basis vectors are empirical global error directions `p_full - p_gmres`.

Two basis sources were tested:

- `fallback`: collect a basis only when required fallback cuDSS is already called. This adds no diagnostic full cuDSS calls and is the fair performance mode.
- `diagnostic`: compute an extra full cuDSS solve after a GMRES step and use the resulting basis only from the next iteration. This avoids same-iteration oracle leakage, but diagnostic calls are not performance-fair.

## Command/config

Common settings: `middle_solver=gmres_block_ilu0`, `preconditioner=block_ilu0`, `block_size=16`, `gmres_max_iters=4`, `bootstrap=1`, `polish=1e-4`, `accept=0.9`, `reject=1.05`, `fallback=immediate`, `max_nr_iters=20`, `global_correction_acceptance=residual`. Ranks swept for four hard cases: 8, 16, 32. `case6468rte` was checked at rank 16 because pure cuDSS already converges in 3 iterations.

## Main comparison table

|case|method|converged|nr_iters|pure_full_cudss_calls|hybrid_required_full_cudss_calls|diagnostic_full_cudss_calls|total_full_cudss_calls|gmres_calls|final_basis_rank|corrections_accepted|final_residual_norm|total_time|delta_required_full_cudss_vs_pure|delta_nr_iters_vs_pure|delta_required_full_cudss_vs_hybrid_baseline|delta_nr_iters_vs_hybrid_baseline|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|case13659pegase|hybrid_baseline|true|15|5|5|0|5|12|0|0|6.10719973399e-12|5.39378362|0|10|0|0|
|case13659pegase|hybrid_global_post_diagnostic_rank16|true|13|5|5|11|16|11|11|1|7.99180779063e-12|4.996214363|0|8|0|-2|
|case13659pegase|hybrid_global_post_fallback_rank16|true|15|5|5|0|5|12|2|0|1.01756589266e-11|5.434236284|0|10|0|0|
|case2383wp|hybrid_baseline|true|19|6|4|0|4|17|0|0|5.37395566336e-12|1.0713951|-2|13|0|0|
|case2383wp|hybrid_global_post_diagnostic_rank16|true|14|6|3|12|15|12|12|4|1.20576700133e-10|0.768314673|-3|8|-1|-5|
|case2383wp|hybrid_global_post_fallback_rank16|false|20|6|2|0|2|19|1|14|0.000155710011442|1.205516511|-4|14|-2|1|
|case3120sp|hybrid_baseline|false|20|6|2|0|2|19|0|0|0.0124326036695|1.632312188|-4|14|0|0|
|case3120sp|hybrid_global_post_diagnostic_rank16|false|20|6|2|19|21|19|16|2|0.00982462207363|1.651359957|-4|14|0|0|
|case3120sp|hybrid_global_post_fallback_rank16|false|20|6|2|0|2|19|1|16|0.00541971463604|1.638264158|-4|14|0|0|
|case6468rte|hybrid_baseline|true|3|3|2|0|2|1|0|0|7.49841938907e-10|0.21715975|-1|0|0|0|
|case6468rte|hybrid_global_post_diagnostic_rank16|true|3|3|2|1|3|1|1|0|7.49811630519e-10|0.225889627|-1|0|0|0|
|case6468rte|hybrid_global_post_fallback_rank16|true|3|3|2|0|2|1|0|0|7.4975075475e-10|0.234529249|-1|0|0|0|
|case9241pegase|hybrid_baseline|true|8|6|3|0|3|6|0|0|1.69714153841e-09|1.813707494|-3|2|0|0|
|case9241pegase|hybrid_global_post_diagnostic_rank16|true|8|6|3|6|9|6|6|0|1.69711167342e-09|1.829275603|-3|2|0|0|
|case9241pegase|hybrid_global_post_fallback_rank16|true|8|6|3|0|3|6|1|0|1.69675506978e-09|1.824098717|-3|2|0|0|


## Best fallback-basis result per case

|case|method|converged|nr_iters|pure_full_cudss_calls|hybrid_required_full_cudss_calls|diagnostic_full_cudss_calls|total_full_cudss_calls|gmres_calls|final_basis_rank|corrections_accepted|final_residual_norm|total_time|delta_required_full_cudss_vs_pure|delta_nr_iters_vs_pure|delta_required_full_cudss_vs_hybrid_baseline|delta_nr_iters_vs_hybrid_baseline|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|case2383wp|hybrid_global_post_fallback_rank16|false|20|6|2|0|2|19|1|14|0.000155710011442|1.205516511|-4|14|-2|1|
|case3120sp|hybrid_global_post_fallback_rank16|false|20|6|2|0|2|19|1|16|0.00541971463604|1.638264158|-4|14|0|0|
|case6468rte|hybrid_global_post_fallback_rank16|true|3|3|2|0|2|1|0|0|7.4975075475e-10|0.234529249|-1|0|0|0|
|case9241pegase|hybrid_global_post_fallback_rank16|true|8|6|3|0|3|6|1|0|1.69675506978e-09|1.824098717|-3|2|0|0|
|case13659pegase|hybrid_global_post_fallback_rank16|true|15|5|5|0|5|12|2|0|1.01756589266e-11|5.434236284|0|10|0|0|


## Best diagnostic-basis result per case

|case|method|converged|nr_iters|pure_full_cudss_calls|hybrid_required_full_cudss_calls|diagnostic_full_cudss_calls|total_full_cudss_calls|gmres_calls|final_basis_rank|corrections_accepted|final_residual_norm|total_time|delta_required_full_cudss_vs_pure|delta_nr_iters_vs_pure|delta_required_full_cudss_vs_hybrid_baseline|delta_nr_iters_vs_hybrid_baseline|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|case2383wp|hybrid_global_post_diagnostic_rank16|true|14|6|3|12|15|12|12|4|1.20576700133e-10|0.768314673|-3|8|-1|-5|
|case3120sp|hybrid_global_post_diagnostic_rank16|false|20|6|2|19|21|19|16|2|0.00982462207363|1.651359957|-4|14|0|0|
|case6468rte|hybrid_global_post_diagnostic_rank16|true|3|3|2|1|3|1|1|0|7.49811630519e-10|0.225889627|-1|0|0|0|
|case9241pegase|hybrid_global_post_diagnostic_rank16|true|8|6|3|6|9|6|6|0|1.69711167342e-09|1.829275603|-3|2|0|0|
|case13659pegase|hybrid_global_post_diagnostic_rank16|true|13|5|5|11|16|11|11|1|7.99180779063e-12|4.996214363|0|8|0|-2|


## Case interpretation

- `case2383wp`: diagnostic basis improved baseline 19 NR iterations to 14, and required full cuDSS calls from 4 to 3. But it needed 12 diagnostic full solves and remains far from pure cuDSS 6 iterations.
- `case9241pegase`: correction did essentially nothing. The baseline was already 8 NR iterations and remains 8.
- `case13659pegase`: diagnostic basis shortened 15 to 13 NR iterations, but required full cuDSS calls stayed at 5, same as pure cuDSS.
- `case3120sp`: neither fallback nor diagnostic mode converged within 20 iterations. Diagnostic rank 8 reduced the final mismatch much more than baseline, but still failed the convergence target.
- `case6468rte`: no meaningful room for improvement; pure is already 3 iterations and hybrid is also 3.

## Full cuDSS call interpretation

`hybrid_required_full_cudss_calls` is the fair algorithmic count. `diagnostic_full_cudss_calls` is only oracle/basis collection cost. Diagnostic mode can improve direction quality but is not a speed result unless saved/replay basis is implemented and reused without diagnostic solves.

## Verdict

The empirical global post-correction is useful as a diagnostic: it confirms that missing global directions can improve some GMRES steps. However, the fair fallback-basis mode does not produce enough early basis information, and the diagnostic mode still does not keep NR iterations close to pure cuDSS. This experiment does not yet meet the key success condition: reduce required full cuDSS calls while preserving a pure-cuDSS-like NR trajectory.

## Next experiment suggestion

A saved/replay basis mode is the only logical next check for this idea. If a diagnostic run can build reusable case-specific `Z`, then a second run could test whether those global directions help without extra diagnostic full cuDSS calls. If saved/replay also fails, this low-rank empirical correction path should be deprioritized.
