# NR Iteration Reduction Sweep Report

## Sweep

- cases: case2383wp, case3120sp, case9241pegase, case13659pegase, case6468rte
- fixed: bootstrap_iters=1, force_gmres_min_steps=0, fallback=immediate, reject=1.05
- fixed: gmres_fixed_iter_mode=true, middle_solver=gmres_block_jacobi
- fixed: block_precision=fp32, block_apply=inverse_gemv
- sweep: polish_threshold=1e-4
- sweep: block_size in [32, 64]
- sweep: (gmres_restart, gmres_iters) in [(1, 1), (2, 2)]
- sweep: accept_mismatch_ratio in [0.7, 0.85, 0.9, 0.95]

`avg_fallback_wasted_time` is the average GMRES setup+solve trial time spent before a fallback cuDSS solve.

## Output

- full report CSV: `results/nr_iteration_reduction_sweep.csv`
- per-config summary CSVs: `results/nr_iter_reduce_*.csv`
- per-config iteration CSVs: `results/nr_iter_reduce_*_iters.csv`

## Overall

- runs: 80
- converged: 80
- speedup > 1.0: 0
- rows with fewer cuDSS calls than pure cuDSS NR: 40 / 80

Best aggregate setting by total pure/hybrid time:

| block | restart | gmres iters | accept | converged | total NR iters | cuDSS | GMRES | accepted | rejected | fallback | polish | hybrid total | pure total | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 1 | 0.7 | 5/5 | 37 | 23 | 28 | 14 | 14 | 14 | 4 | 0.212522 | 0.194997 | 0.918x |

## Best Per Case By Speedup

| case | converged | NR iters | cuDSS | GMRES | accepted | rejected | fallback | polish | hybrid time | pure cuDSS | speedup | avg accepted mismatch ratio | max accepted mismatch ratio | avg GMRES trial ms | avg fallback wasted ms | setting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case13659pegase | true | 6 | 5 | 4 | 1 | 3 | 3 | 1 | 0.065007 | 0.061685 | 0.949x | 0.413 | 0.413 | 0.750 | 0.755 | bs=32, r=1, i=1, a=0.7 |
| case2383wp | true | 9 | 5 | 8 | 4 | 4 | 4 | 0 | 0.027011 | 0.021895 | 0.811x | 0.381 | 0.667 | 0.547 | 0.552 | bs=32, r=1, i=1, a=0.7 |
| case3120sp | true | 9 | 6 | 7 | 3 | 4 | 4 | 1 | 0.030043 | 0.024883 | 0.828x | 0.522 | 0.671 | 0.558 | 0.560 | bs=32, r=1, i=1, a=0.7 |
| case6468rte | true | 5 | 2 | 3 | 3 | 0 | 0 | 1 | 0.036570 | 0.035722 | 0.977x | 0.503 | 0.664 | 0.868 | 0.000 | bs=64, r=1, i=1, a=0.9 |
| case9241pegase | true | 9 | 4 | 7 | 5 | 2 | 2 | 1 | 0.054675 | 0.052449 | 0.959x | 0.531 | 0.801 | 0.685 | 0.681 | bs=32, r=1, i=1, a=0.7 |

## Interpretation

- The sweep reduced cuDSS calls in many rows, but did not reduce total NR iterations versus pure cuDSS.
- No setting was faster than pure cuDSS on these five cases.
- The best aggregate setting was the most conservative GMRES work setting: block_size=32, restart=1, gmres_iters=1, accept=0.7.
- `case6468rte` is closest to break-even and can use only two cuDSS calls, but still does not beat pure cuDSS in this run.
- Fallback waste is small per event, but frequent fallback plus extra NR iterations removes the benefit of fewer direct solves.

