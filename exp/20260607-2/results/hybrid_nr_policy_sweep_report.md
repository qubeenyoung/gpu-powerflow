# Hybrid NR Policy Sweep Report

Cases: `case1197`, `case2736sp`, `case3375wp`, `case6468rte`, `case_ACTIVSg10k`.

Base policy: cuDSS bootstrap once, GMRES block-Jacobi middle solver, block size 64, restart 16, fixed iteration mode.

- summary CSV: `results/hybrid_nr_policy_sweep.csv`
- iteration CSV: `results/hybrid_nr_policy_sweep_iters.csv`
- total runs: 175
- converged runs: 173/175
- runs faster than pure cuDSS: 29

## Best Converged Run Per Case

| case | experiment | params | speedup | total s | pure cuDSS s | NR iters | cuDSS | GMRES | accepted | rejected | fallback | polish | final inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case1197 | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.99, reject=1.1, damp=False, fallback=immediate | 1.120 | 1.275554e-02 | 1.428419e-02 | 3 | 3 | 0 | 0 | 0 | 0 | 2 | 4.645e-12 |
| case2736sp | C_gmres_fixed_iterations | polish=1e-6, force=0, iters=1, accept=0.95, reject=1.05, damp=False, fallback=immediate | 0.760 | 2.846280e-02 | 2.162871e-02 | 11 | 4 | 10 | 7 | 3 | 3 | 0 | 7.825e-12 |
| case3375wp | A_polish_threshold | polish=1e-6, force=0, iters=8, accept=0.95, reject=1.05, damp=False, fallback=immediate | 1.090 | 2.009081e-02 | 2.189409e-02 | 2 | 2 | 0 | 0 | 0 | 0 | 1 | 1.239e-11 |
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.99, reject=1.1, damp=False, fallback=immediate | 1.014 | 3.419361e-02 | 3.467302e-02 | 3 | 2 | 1 | 1 | 0 | 0 | 1 | 7.637e-10 |
| case_ACTIVSg10k | C_gmres_fixed_iterations | polish=1e-6, force=0, iters=1, accept=0.95, reject=1.05, damp=False, fallback=immediate | 0.820 | 5.966499e-02 | 4.891293e-02 | 10 | 4 | 9 | 6 | 3 | 3 | 0 | 1.351e-11 |

## Best Runs With Accepted GMRES Steps

| case | experiment | params | speedup | total s | pure cuDSS s | accepted | rejected | fallback | ratio mean | rel mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.99, reject=1.1, damp=False, fallback=immediate | 1.014 | 3.419361e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.99, reject=1.05, damp=False, fallback=immediate | 1.013 | 3.422842e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.99, reject=1.01, damp=False, fallback=immediate | 0.992 | 3.494507e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.97, reject=1.1, damp=False, fallback=immediate | 0.974 | 3.559082e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.01, damp=False, fallback=immediate | 0.969 | 3.577337e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | F_fallback_policy | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.05, damp=False, fallback=after_two_failures | 0.966 | 3.589666e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | F_fallback_policy | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.05, damp=False, fallback=immediate | 0.962 | 3.604358e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | B_forced_middle | polish=1e-4, force=1, iters=8, accept=0.95, reject=1.05, damp=False, fallback=immediate | 0.962 | 3.605764e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.97, reject=1.05, damp=False, fallback=immediate | 0.957 | 3.621216e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.05, damp=False, fallback=immediate | 0.954 | 3.635532e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | A_polish_threshold | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.05, damp=False, fallback=immediate | 0.953 | 3.637735e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.1, damp=False, fallback=immediate | 0.953 | 3.638742e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | D_accept_reject_policy | polish=1e-4, force=0, iters=8, accept=0.97, reject=1.01, damp=False, fallback=immediate | 0.953 | 3.639599e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | F_fallback_policy | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.05, damp=False, fallback=off | 0.951 | 3.647000e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |
| case6468rte | E_damping_before_fallback | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.05, damp=True, fallback=immediate | 0.950 | 3.648259e-02 | 3.467302e-02 | 1 | 0 | 0 | 0.046 | 0.123 |

## Experiment-Level Aggregate

| experiment | converged | avg speedup | max speedup | avg total s | avg cuDSS | avg GMRES | avg accepted | avg rejected | avg fallback | avg polish | rel mean | rel max | mismatch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_polish_threshold | 25/25 | 0.756 | 1.090 | 4.053097e-02 | 3.00 | 5.64 | 4.20 | 1.44 | 1.44 | 0.56 | 0.638 | 0.986 | 0.583 |
| B_forced_middle | 20/20 | 0.753 | 1.000 | 3.951292e-02 | 3.00 | 4.35 | 3.35 | 1.00 | 1.00 | 1.00 | 0.594 | 0.986 | 0.545 |
| C_gmres_fixed_iterations | 50/50 | 0.763 | 1.085 | 4.013674e-02 | 3.00 | 5.20 | 3.48 | 1.72 | 1.72 | 0.28 | 0.602 | 0.992 | 0.552 |
| D_accept_reject_policy | 60/60 | 0.845 | 1.120 | 3.689911e-02 | 3.00 | 3.50 | 2.70 | 0.80 | 0.80 | 1.20 | 0.445 | 0.986 | 0.390 |
| E_damping_before_fallback | 5/5 | 0.821 | 1.003 | 3.808125e-02 | 3.00 | 3.60 | 2.80 | 0.80 | 0.80 | 1.20 | 0.449 | 0.986 | 0.382 |
| F_fallback_policy | 13/15 | 0.869 | 1.033 | 3.555606e-02 | 2.60 | 3.07 | 2.13 | 0.93 | 0.53 | 1.07 | 0.373 | 0.986 | 0.310 |

## A: Polish Threshold

| polish_threshold | converged | avg speedup | max speedup | avg total s | avg cuDSS | avg GMRES | avg accepted | avg rejected | avg fallback | avg polish | rel mean | rel max | mismatch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 5/5 | 0.825 | 1.000 | 3.772966e-02 | 3.00 | 3.60 | 2.80 | 0.80 | 0.80 | 1.20 | 0.449 | 0.986 | 0.382 |
| 1e-5 | 5/5 | 0.802 | 1.089 | 3.799473e-02 | 3.00 | 4.60 | 3.40 | 1.20 | 1.20 | 0.80 | 0.654 | 0.986 | 0.596 |
| 1e-6 | 5/5 | 0.806 | 1.090 | 3.760707e-02 | 3.00 | 4.80 | 3.60 | 1.20 | 1.20 | 0.80 | 0.676 | 0.986 | 0.614 |
| 1e-8 | 5/5 | 0.682 | 0.888 | 4.414430e-02 | 3.00 | 7.60 | 5.60 | 2.00 | 2.00 | 0.00 | 0.673 | 0.986 | 0.625 |
| disabled | 5/5 | 0.666 | 0.890 | 4.517911e-02 | 3.00 | 7.60 | 5.60 | 2.00 | 2.00 | 0.00 | 0.673 | 0.986 | 0.625 |

## B: Forced Middle

| force_gmres_min_steps | converged | avg speedup | max speedup | avg total s | avg cuDSS | avg GMRES | avg accepted | avg rejected | avg fallback | avg polish | rel mean | rel max | mismatch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5/5 | 0.819 | 1.000 | 3.812667e-02 | 3.00 | 3.60 | 2.80 | 0.80 | 0.80 | 1.20 | 0.449 | 0.986 | 0.382 |
| 1 | 5/5 | 0.781 | 0.962 | 3.834375e-02 | 3.00 | 4.00 | 3.20 | 0.80 | 0.80 | 1.20 | 0.538 | 0.986 | 0.494 |
| 2 | 5/5 | 0.731 | 0.893 | 3.984328e-02 | 3.00 | 4.60 | 3.60 | 1.00 | 1.00 | 1.00 | 0.667 | 0.986 | 0.621 |
| 3 | 5/5 | 0.681 | 0.828 | 4.173799e-02 | 3.00 | 5.20 | 3.80 | 1.40 | 1.40 | 0.60 | 0.664 | 0.986 | 0.618 |

## C: GMRES Fixed Iterations

| polish_threshold / gmres_iters | converged | avg speedup | max speedup | avg total s | avg cuDSS | avg GMRES | avg accepted | avg rejected | avg fallback | avg polish | rel mean | rel max | mismatch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-6 / 1 | 5/5 | 0.925 | 1.085 | 3.211137e-02 | 3.00 | 5.40 | 3.80 | 1.60 | 1.60 | 0.40 | 0.687 | 0.992 | 0.629 |
| 1e-6 / 16 | 5/5 | 0.648 | 0.998 | 4.676161e-02 | 3.00 | 3.20 | 2.00 | 1.20 | 1.20 | 0.80 | 0.467 | 0.906 | 0.477 |
| 1e-6 / 2 | 5/5 | 0.866 | 0.999 | 3.371496e-02 | 3.00 | 4.40 | 2.80 | 1.60 | 1.60 | 0.40 | 0.591 | 0.944 | 0.485 |
| 1e-6 / 4 | 5/5 | 0.849 | 0.997 | 3.475773e-02 | 3.00 | 4.00 | 2.40 | 1.60 | 1.60 | 0.40 | 0.542 | 0.983 | 0.486 |
| 1e-6 / 8 | 5/5 | 0.738 | 0.998 | 4.089239e-02 | 3.00 | 4.80 | 3.60 | 1.20 | 1.20 | 0.80 | 0.676 | 0.986 | 0.614 |
| disabled / 1 | 5/5 | 0.821 | 0.900 | 3.484161e-02 | 3.00 | 6.20 | 4.20 | 2.00 | 2.00 | 0.00 | 0.680 | 0.992 | 0.630 |
| disabled / 16 | 5/5 | 0.470 | 0.724 | 6.249043e-02 | 3.00 | 6.80 | 4.80 | 2.00 | 2.00 | 0.00 | 0.634 | 0.955 | 0.608 |
| disabled / 2 | 5/5 | 0.836 | 0.904 | 3.438186e-02 | 3.00 | 5.00 | 3.00 | 2.00 | 2.00 | 0.00 | 0.543 | 0.944 | 0.455 |
| disabled / 4 | 5/5 | 0.811 | 0.900 | 3.560646e-02 | 3.00 | 4.60 | 2.60 | 2.00 | 2.00 | 0.00 | 0.502 | 0.983 | 0.462 |
| disabled / 8 | 5/5 | 0.662 | 0.891 | 4.580900e-02 | 3.00 | 7.60 | 5.60 | 2.00 | 2.00 | 0.00 | 0.673 | 0.986 | 0.625 |

## D: Accept/Reject Policy

| accept_mismatch_ratio / reject_mismatch_ratio | converged | avg speedup | max speedup | avg total s | avg cuDSS | avg GMRES | avg accepted | avg rejected | avg fallback | avg polish | rel mean | rel max | mismatch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9 / 1.01 | 5/5 | 0.861 | 0.999 | 3.471400e-02 | 3.00 | 2.40 | 1.60 | 0.80 | 0.80 | 1.20 | 0.348 | 0.912 | 0.282 |
| 0.9 / 1.05 | 5/5 | 0.862 | 1.000 | 3.468422e-02 | 3.00 | 2.40 | 1.60 | 0.80 | 0.80 | 1.20 | 0.348 | 0.912 | 0.282 |
| 0.9 / 1.1 | 5/5 | 0.863 | 1.001 | 3.465849e-02 | 3.00 | 2.40 | 1.60 | 0.80 | 0.80 | 1.20 | 0.348 | 0.912 | 0.282 |
| 0.95 / 1.01 | 5/5 | 0.842 | 1.037 | 3.717269e-02 | 3.00 | 3.60 | 2.80 | 0.80 | 0.80 | 1.20 | 0.449 | 0.986 | 0.382 |
| 0.95 / 1.05 | 5/5 | 0.826 | 1.001 | 3.763445e-02 | 3.00 | 3.60 | 2.80 | 0.80 | 0.80 | 1.20 | 0.449 | 0.986 | 0.382 |
| 0.95 / 1.1 | 5/5 | 0.825 | 1.002 | 3.771422e-02 | 3.00 | 3.60 | 2.80 | 0.80 | 0.80 | 1.20 | 0.449 | 0.986 | 0.382 |
| 0.97 / 1.01 | 5/5 | 0.815 | 1.002 | 3.878468e-02 | 3.00 | 4.00 | 3.20 | 0.80 | 0.80 | 1.20 | 0.492 | 0.986 | 0.448 |
| 0.97 / 1.05 | 5/5 | 0.817 | 1.002 | 3.870376e-02 | 3.00 | 4.00 | 3.20 | 0.80 | 0.80 | 1.20 | 0.492 | 0.986 | 0.448 |
| 0.97 / 1.1 | 5/5 | 0.827 | 1.021 | 3.824537e-02 | 3.00 | 4.00 | 3.20 | 0.80 | 0.80 | 1.20 | 0.492 | 0.986 | 0.448 |
| 0.99 / 1.01 | 5/5 | 0.845 | 1.064 | 3.742624e-02 | 3.00 | 4.00 | 3.20 | 0.80 | 0.80 | 1.20 | 0.492 | 0.986 | 0.448 |
| 0.99 / 1.05 | 5/5 | 0.877 | 1.096 | 3.616269e-02 | 3.00 | 4.00 | 3.20 | 0.80 | 0.80 | 1.20 | 0.492 | 0.986 | 0.448 |
| 0.99 / 1.1 | 5/5 | 0.874 | 1.120 | 3.688845e-02 | 3.00 | 4.00 | 3.20 | 0.80 | 0.80 | 1.20 | 0.492 | 0.986 | 0.448 |

## E: Damping Before Fallback

| damping_enabled / damping_factors | converged | avg speedup | max speedup | avg total s | avg cuDSS | avg GMRES | avg accepted | avg rejected | avg fallback | avg polish | rel mean | rel max | mismatch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| True / 1.0,0.5,0.25 | 5/5 | 0.821 | 1.003 | 3.808125e-02 | 3.00 | 3.60 | 2.80 | 0.80 | 0.80 | 1.20 | 0.449 | 0.986 | 0.382 |

## F: Fallback Policy

| fallback_policy | converged | avg speedup | max speedup | avg total s | avg cuDSS | avg GMRES | avg accepted | avg rejected | avg fallback | avg polish | rel mean | rel max | mismatch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| after_two_failures | 5/5 | 0.821 | 1.033 | 3.932023e-02 | 3.00 | 4.40 | 2.80 | 1.60 | 0.80 | 1.20 | 0.449 | 0.986 | 0.382 |
| immediate | 5/5 | 0.835 | 1.024 | 3.727008e-02 | 3.00 | 3.60 | 2.80 | 0.80 | 0.80 | 1.20 | 0.449 | 0.986 | 0.382 |
| off | 3/5 | 0.950 | 1.000 | 3.007787e-02 | 1.80 | 1.20 | 0.80 | 0.40 | 0.00 | 0.80 | 0.223 | 0.843 | 0.166 |

## Failed Runs

| case | experiment | params | stop reason | total s | final inf |
|---|---|---|---|---:|---:|
| case2736sp | F_fallback_policy | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.05, damp=False, fallback=off | gmres_step_rejected | 2.453345e-02 | 1.167e-01 |
| case_ACTIVSg10k | F_fallback_policy | polish=1e-4, force=0, iters=8, accept=0.95, reject=1.05, damp=False, fallback=off | gmres_step_rejected | 5.317445e-02 | 1.255e-01 |
