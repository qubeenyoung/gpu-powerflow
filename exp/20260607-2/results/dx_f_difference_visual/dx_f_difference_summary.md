# dx/F Difference Visual Summary

## Inputs

- J1/F1 localization CSV: `results/dx_f_difference_visual/gmres_j1_bs16`
- NR shadow CSV: `results/field_gain_j11/A_baseline_shadow.csv`
- J1 means the second Newton linear system in the existing dump naming.

## What The Figures Show

- `fig_case13659pegase_j1_bus_error_map.*` maps where the iterative `dx` differs from cuDSS and where that difference appears in the linear equation rows `J(dx_iter-dx_cuDSS)`.
- `fig_case13659pegase_j1_top_dx_pairs.*` shows direct-vs-iterative correction values on the buses with the largest theta error.
- `fig_iteration_dx_f_gap.*` shows how the dx scale gap and nonlinear mismatch reduction gap evolve across middle NR iterations.

## Case Selection

- Main visual case: `case13659pegase`. It has the largest J1 block-ILU0 dx error among the selected cases, so it is the clearest failure-mode picture.
- The comparison set keeps `case2383wp`, `case3120sp`, `case6468rte`, `case9241pegase`, and `case13659pegase` so mild, failed, and severe cases are all visible.

## J1/F1 Block-ILU0 Summary

| case            |   dx_norm_ratio_iter_to_cudss |   dx_error_ratio |   dx_cosine |   theta_norm_ratio_iter_to_cudss |   theta_error_ratio |   theta_cosine |   vmag_norm_ratio_iter_to_cudss |   vmag_error_ratio |   vmag_cosine |   top_dx_error_bus |   top_f_error_bus |
|:----------------|------------------------------:|-----------------:|------------:|---------------------------------:|--------------------:|---------------:|--------------------------------:|-------------------:|--------------:|-------------------:|------------------:|
| case13659pegase |                       0.01579 |           0.9995 |     0.03985 |                         0.006726 |              0.9996 |        0.05756 |                          0.8111 |            0.3107  |        0.9625 |               9967 |             12217 |
| case9241pegase  |                       0.2777  |           0.9583 |     0.2858  |                         0.0906   |              0.9938 |        0.1137  |                          0.9958 |            0.02127 |        0.9998 |               2762 |              4919 |
| case3120sp      |                       0.1166  |           0.9106 |     0.7911  |                         0.1154   |              0.9101 |        0.8019  |                          0.7199 |            1.603   |       -0.7295 |               1225 |                69 |
| case2383wp      |                       0.1419  |           0.8773 |     0.8826  |                         0.1413   |              0.8773 |        0.8858  |                          1.193  |            1.218   |        0.3936 |               1849 |               156 |
| case6468rte     |                       0.4803  |           0.7952 |     0.6228  |                         0.3736   |              0.857  |        0.5423  |                          0.8613 |            0.2449  |        0.9763 |               3516 |              2046 |

## NR Shadow Summary

| case_name       | converged   |   nr_iters |   pure_full_cudss_calls |   hybrid_required_full_cudss_calls |   gmres_calls |   fallback_calls |   final_mismatch_inf |
|:----------------|:------------|-----------:|------------------------:|-----------------------------------:|--------------:|-----------------:|---------------------:|
| case2383wp      | True        |         10 |                       6 |                                  5 |             6 |                1 |            8.115e-12 |
| case3120sp      | False       |         20 |                       6 |                                  3 |            17 |                0 |            0.0008994 |
| case6468rte     | True        |          3 |                       3 |                                  2 |             1 |                0 |            7.499e-10 |
| case9241pegase  | True        |          9 |                       6 |                                  6 |             4 |                1 |            5.795e-12 |
| case13659pegase | True        |         11 |                       5 |                                  5 |             7 |                1 |            1.516e-11 |

| case            |   middle_steps |   avg_theta_norm_ratio |   avg_theta_cosine |   avg_vmag_norm_ratio |   avg_vmag_cosine |   avg_iterative_nonlinear_ratio_inf |   avg_cudss_nonlinear_ratio_inf |
|:----------------|---------------:|-----------------------:|-------------------:|----------------------:|------------------:|------------------------------------:|--------------------------------:|
| case2383wp      |              6 |               0.3129   |             0.9508 |                0.5811 |            0.8372 |                              0.6575 |                       0.0139    |
| case3120sp      |             17 |               0.1841   |             0.9697 |                0.5674 |            0.6821 |                              0.759  |                       0.006634  |
| case6468rte     |              1 |               0.407    |             0.5724 |                0.8611 |            0.9763 |                              0.1281 |                       5.625e-05 |
| case9241pegase  |              4 |               0.05232  |             0.578  |                0.4981 |            0.507  |                              0.5771 |                       0.0007194 |
| case13659pegase |              7 |               0.003203 |             0.206  |                0.4067 |            0.3224 |                              0.6323 |                       0.03675   |

## Key Observations

- Worst J1 case is `case13659pegase`: dx error ratio is 0.999, theta norm ratio is 0.00673, and theta cosine is 0.0576.
- Smallest theta scale appears in `case13659pegase`: iterative theta norm is 0.00673 of cuDSS.
- Weakest nonlinear middle-step reduction is `case3120sp`: average iterative mismatch ratio is 0.759, while the shadow cuDSS ratio is 0.00663.
- `J(dx_iter-dx_cuDSS)` is the linear F-row difference; large P/Q spikes identify where the missed correction reappears in the mismatch equations.