# mismatch < 1 hybrid field-error check

## What changed

Added `--iterative-start-mismatch-threshold`. With `--iterative-start-mismatch-threshold 1.0`, the hybrid NR driver keeps using full-J cuDSS until `||F||_inf < 1`; only then does it attempt the iterative middle solver. Polish cuDSS is still used when `||F||_inf <= 1e-4`.

Shadow diagnostics were also fixed for `gmres_block_ilu0`: when block-ILU is selected, the shadow iterative dx now comes from the CPU block-ILU pilot path, not from the block-Jacobi GMRES path. For every middle step, the run computes both iterative dx and full cuDSS dx on the same Jacobian/RHS, then compares theta/Vm dx and P/Q mismatch after applying each step.

## NR summary

|case|method|converged|nr_iters|pure_full_cudss_calls|hybrid_full_cudss_calls|gmres_calls|fallback_calls|final_mismatch_inf|
|---|---|---|---|---|---|---|---|---|
|case2383wp|GMRES+block-Jacobi|true|8|6|6|4|2|5.56518546524e-12|
|case3120sp|GMRES+block-Jacobi|true|7|6|6|3|2|1.12863483645e-11|
|case6468rte|GMRES+block-Jacobi|true|3|3|2|2|1|7.5520904718e-09|
|case9241pegase|GMRES+block-Jacobi|true|10|6|6|5|1|7.21644966006e-12|
|case13659pegase|GMRES+block-Jacobi|true|8|5|5|4|1|3.06237257774e-11|
|case2383wp|GMRES+block-ILU0|true|10|6|5|6|1|8.15816973107e-12|
|case3120sp|GMRES+block-ILU0|false|20|6|3|17|0|0.000899447346416|
|case6468rte|GMRES+block-ILU0|true|3|3|2|1|0|7.49719928375e-10|
|case9241pegase|GMRES+block-ILU0|true|9|6|6|4|1|9.89246878857e-12|
|case13659pegase|GMRES+block-ILU0|true|11|5|5|7|1|1.494160351e-11|


## dx quality split by theta / Vm

Ratios are `||dx_iter_field|| / ||dx_cudss_field||`. Values far below 1 mean the iterative step is too small in that field.

|case|method|shadow_middle_steps|avg_theta_norm_ratio|avg_theta_cosine|avg_vmag_norm_ratio|avg_vmag_cosine|dx_weaker_field|
|---|---|---|---|---|---|---|---|
|case2383wp|GMRES+block-Jacobi|4|0.0146|0.525|0.464|0.784|theta|
|case3120sp|GMRES+block-Jacobi|3|0.00643|0.333|0.0706|0.353|theta|
|case6468rte|GMRES+block-Jacobi|2|0.167|0.444|0.296|0.772|theta|
|case9241pegase|GMRES+block-Jacobi|5|0.0324|0.154|0.27|0.747|theta|
|case13659pegase|GMRES+block-Jacobi|4|0.00223|0.0154|0.188|0.522|theta|
|case2383wp|GMRES+block-ILU0|6|0.313|0.951|0.581|0.837|theta|
|case3120sp|GMRES+block-ILU0|17|0.184|0.97|0.567|0.682|theta|
|case6468rte|GMRES+block-ILU0|1|0.407|0.572|0.861|0.976|theta|
|case9241pegase|GMRES+block-ILU0|4|0.0524|0.579|0.497|0.508|theta|
|case13659pegase|GMRES+block-ILU0|7|0.0032|0.206|0.407|0.323|theta|


## mismatch after one step split by P / Q

`avg_iter_*_after` is the average P/Q infinity mismatch after the iterative step. `avg_cudss_*_after` is the same diagnostic after full cuDSS from the same state.

|case|method|avg_iter_p_inf_after|avg_iter_q_inf_after|avg_cudss_p_inf_after|avg_cudss_q_inf_after|larger_iter_mismatch_field|avg_iter_inf_ratio|avg_cudss_inf_ratio|
|---|---|---|---|---|---|---|---|---|
|case2383wp|GMRES+block-Jacobi|0.0272|0.0474|0.00119|0.00235|Q|0.6|0.00766|
|case3120sp|GMRES+block-Jacobi|0.0647|0.248|0.00161|0.00329|Q|0.805|0.00492|
|case6468rte|GMRES+block-Jacobi|0.000122|0.000188|1.1e-08|2.46e-08|Q|0.609|4.77e-05|
|case9241pegase|GMRES+block-Jacobi|0.000288|0.000672|6.46e-05|1.59e-05|Q|0.627|0.00056|
|case13659pegase|GMRES+block-Jacobi|0.0132|0.0209|0.00011|0.0001|Q|0.546|0.00335|
|case2383wp|GMRES+block-ILU0|0.012|0.00363|0.000781|0.00153|P|0.658|0.0139|
|case3120sp|GMRES+block-ILU0|0.00701|0.00589|0.000198|0.000542|P|0.759|0.00663|
|case6468rte|GMRES+block-ILU0|9.49e-05|4.93e-05|1.87e-08|4.17e-08|P|0.128|5.62e-05|
|case9241pegase|GMRES+block-ILU0|0.000199|5.24e-05|8.07e-05|1.99e-05|P|0.577|0.000719|
|case13659pegase|GMRES+block-ILU0|0.00228|0.00172|6.32e-05|8.38e-05|P|0.633|0.0367|


## Main observations

- Block-Jacobi is overwhelmingly weak in the theta component. The theta norm ratio is often near zero, while Vm is less small. This explains why the Newton direction is not cuDSS-like even when Vm looks partially corrected.
- Block-ILU improves direction quality on several cases, especially `case2383wp`, `case3120sp`, and `case6468rte`, but it still leaves theta badly under-corrected on `case13659pegase` and `case9241pegase`.
- P/Q mismatch after the iterative step remains orders of magnitude larger than after cuDSS. The dominant residual field varies by case, but both P and Q are much worse than cuDSS.
- Starting the iterative method only after `||F||_inf < 1` makes the trajectory more stable, but it mostly removes the opportunity to reduce full cuDSS calls. It is useful as a diagnostic/safety gate, not as a speed win by itself.

## Files

- `block_jacobi_summary.csv`, `block_jacobi_shadow.csv`
- `block_ilu0_summary.csv`, `block_ilu0_shadow.csv`
- `preconditioner_field_error_summary.csv`
