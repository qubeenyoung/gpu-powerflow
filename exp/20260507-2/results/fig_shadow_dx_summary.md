# Shadow Dx Figure Summary

- Input CSV: `/workspace/gpu-powerflow/exp/20260607-2/results/shadow_dx_selected5_bs64_r1_i1_a0p9_requested.csv`
- Rows used: 43 / 43 (excluded non-finite rows: 0)

## Figures

- `fig_dx_scale_by_field`: case-averaged GMRES dx scale relative to cuDSS, split into overall, theta, and |V| components.
- `fig_step_effectiveness`: case-averaged one-step mismatch ratio for GMRES versus cuDSS, with theta ratio shown beside each case.

## Key Observations

- GMRES corrections are much smaller than cuDSS corrections across all five cases.
- The theta component is usually smaller than the |V| component, which weakens the Newton direction.
- GMRES one-step mismatch reduction is generally much weaker than cuDSS.
- `case6468rte` is the closest to break-even because its theta ratio is relatively large.

## Case Notes

- `case6468rte`: theta ratio=0.095, GMRES mismatch ratio=0.503, cuDSS mismatch ratio=3.17e-05.
- `case2383wp`: theta ratio=0.011, GMRES mismatch ratio=0.754, cuDSS mismatch ratio=1.29. The cuDSS shadow step can overshoot here.

## Columns Used

- Figure 1: `dx_norm_ratio`, `theta_norm_ratio`, `vmag_norm_ratio`
- Figure 2: `gmres_nonlinear_ratio_inf`, `cudss_nonlinear_ratio_inf`, `theta_norm_ratio`
