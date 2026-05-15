# Ginkgo ParILUT NR Run

## Configuration

- Middle solver: `ginkgo_parilut_bicgstab`
- Preconditioner: Ginkgo `ParIlut` factorization + Ginkgo `Ilu` triangular preconditioner
- Outer solver: fixed BiCGSTAB(2)
- ParILUT options: `iterations=5`, `fill_in_limit=2.0`
- NR policy: bootstrap cuDSS 1, iterative only after `mismatch_inf < 1`, polish `1e-4`, accept `0.9`, reject `1.05`, fallback immediate
- Important implementation note: this is a first NR integration path that rebuilds the Ginkgo matrix/preconditioner from host-side values in each middle attempt. Use it for quality/trajectory, not final performance.

## Summary

| case_name       | converged   |   nr_iters |   pure_full_cudss_calls |   hybrid_required_full_cudss_calls |   gmres_calls |   accepted_gmres_steps |   rejected_gmres_steps |   fallback_calls |   pure_cudss_total_ms |   total_ms |   speedup_vs_pure_cudss |   final_mismatch_inf |
|:----------------|:------------|-----------:|------------------------:|-----------------------------------:|--------------:|-----------------------:|-----------------------:|-----------------:|----------------------:|-----------:|------------------------:|---------------------:|
| case2383wp      | True        |          6 |                       6 |                                  6 |             2 |                      0 |                      2 |                2 |                25.8   |     128.7  |                 0.2004  |            5.436e-12 |
| case3120sp      | True        |          6 |                       6 |                                  6 |             2 |                      0 |                      2 |                2 |                 5.754 |      49.37 |                 0.1165  |            7.571e-12 |
| case6468rte     | True        |          3 |                       3 |                                  3 |             1 |                      0 |                      1 |                1 |                 3.173 |      48.06 |                 0.06602 |            7.779e-12 |
| case9241pegase  | True        |          6 |                       6 |                                  6 |             2 |                      0 |                      2 |                2 |                 8.724 |     145.9  |                 0.05979 |            2.129e-09 |
| case13659pegase | True        |          5 |                       5 |                                  5 |             2 |                      0 |                      2 |                2 |                 8.878 |     204.5  |                 0.04341 |            2.289e-09 |

## Rejected Middle Attempts

Note: `mismatch_inf_after` in the table is the state after immediate cuDSS fallback, not the rejected ParILUT trial mismatch. The current iteration log does not persist the rejected trial mismatch separately.

| case_name       |   nr_iter |   mismatch_inf_before |   mismatch_inf_after |   gmres_trial_setup_ms |   gmres_trial_solve_ms |   block_ilu_factor_ms |   bicgstab_total_ms |   middle_solver_total_ms | fallback_used   | stop_reason                            |
|:----------------|----------:|----------------------:|---------------------:|-----------------------:|-----------------------:|----------------------:|--------------------:|-------------------------:|:----------------|:---------------------------------------|
| case2383wp      |         3 |             0.5573    |            0.007804  |                  61.33 |                 37.59  |                 56.2  |              37.59  |                    98.92 | True            | gmres_rejected:ginkgo_parilut_bicgstab |
| case2383wp      |         4 |             0.007804  |            1.499e-06 |                  16.19 |                  2.926 |                 15.58 |               2.926 |                    19.11 | True            | gmres_rejected:ginkgo_parilut_bicgstab |
| case3120sp      |         3 |             0.9076    |            0.007526  |                  17.13 |                  4.184 |                 16.66 |               4.184 |                    21.32 | True            | gmres_rejected:ginkgo_parilut_bicgstab |
| case3120sp      |         4 |             0.007526  |            7.16e-07  |                  17.19 |                  4.054 |                 16.91 |               4.054 |                    21.24 | True            | gmres_rejected:ginkgo_parilut_bicgstab |
| case6468rte     |         1 |             0.000741  |            4.168e-08 |                  38.35 |                  5.314 |                 37.54 |               5.314 |                    43.67 | True            | gmres_rejected:ginkgo_parilut_bicgstab |
| case9241pegase  |         4 |             0.1226    |            0.0003228 |                  55.5  |                  9.088 |                 54.53 |               9.088 |                    64.58 | True            | gmres_rejected:ginkgo_parilut_bicgstab |
| case9241pegase  |         5 |             0.0003228 |            2.129e-09 |                  57.62 |                  9.301 |                 56.69 |               9.301 |                    66.92 | True            | gmres_rejected:ginkgo_parilut_bicgstab |
| case13659pegase |         3 |             0.1998    |            0.0003946 |                  85.05 |                 11.43  |                 83.77 |              11.43  |                    96.48 | True            | gmres_rejected:ginkgo_parilut_bicgstab |
| case13659pegase |         4 |             0.0003946 |            2.289e-09 |                  85.38 |                 10.93  |                 84.17 |              10.93  |                    96.31 | True            | gmres_rejected:ginkgo_parilut_bicgstab |

## Interpretation

- ParILUT was invoked in the NR middle phase, but no middle step was accepted in this run.
- Every ParILUT+BiCGSTAB attempt triggered immediate cuDSS fallback, so hybrid required full cuDSS calls equal pure cuDSS calls for all cases.
- Because rejected ParILUT trials are extra work, total time is worse than pure cuDSS in every case.
- Relaxing accept from `0.9` to `1.0` did not change this behavior; the attempted middle steps still fell back.
- This result says Ginkgo ParILUT, in this direct fixed-2 BiCGSTAB middle form, is not yet a useful NR replacement step. A separate dx/shadow diagnostic would be needed to see whether the failure is scale, direction, or nonlinear trajectory.