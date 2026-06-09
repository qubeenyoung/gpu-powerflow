# 2606012 Lab Meeting

Front-size tracking workspace for the lab meeting.

## Scope

Track two distributions for the selected power-grid cases:

- Case-level front-size distribution.
- Per-level front-size distribution along the panel etree.

Target cases:

| band | case |
| --- | --- |
| 1xxx | `case1197` |
| 2xxx | `case_ACTIVSg2000` |
| 3xxx | `case3012wp` |
| 6xxx | `case6468rte` |
| 8xxx | `case8387pegase` |
| 10K | `case_ACTIVSg10k` |
| 13K | `case13659pegase` |
| 25K | `case_ACTIVSg25k` |
| 70K | `case_ACTIVSg70k` |
| usa | `case_SyntheticUSA` |

## Files

- `data/fronts/*.csv`: raw per-front dumps, copied from the existing B=1 TC overhead report.
- `data/case_front_summary.csv`: one summary row per case.
- `data/front_size_histogram.csv`: front-size bucket histogram per case.
- `data/front_size_pow2_histogram.csv`: powers-of-two histogram per case (`<=4`, `5-8`, `9-16`, ...).
- `data/level_front_summary.csv`: one summary row per `(case, level)`.
- `data/level_front_size_histogram.csv`: front-size bucket histogram per `(case, level)`.
- `SUMMARY.md`: generated readable summary.
- `PLOTS.md`: generated matplotlib plot index.
- `figures/`: generated PNG plots.
- `CUPF_MIXED_CUDSS_FP32_MT_AUTO_SCALE1_MAXITER10.md`: merged cuPF mixed + cuDSS FP32/MT-auto timing for `B=1,16,64,256`; batch cases use identical replicas (`scale=1.0`) and `max_iter=10`.
- `data/cupf_mixed_cudss_fp32_mt_auto_scale1_maxiter10_init_solve.csv`: merged initial/solve timing.
- `data/cupf_mixed_cudss_fp32_mt_auto_scale1_maxiter10_ops_ms.csv`: merged solve-stage operator timing in milliseconds.

## Rebuild

From the repository root:

```bash
python3 gpu-powerflow/custom_linear_solver/docs/2606012_lab_meeting/scripts/summarize_fronts.py
python3 gpu-powerflow/custom_linear_solver/docs/2606012_lab_meeting/scripts/plot_front_stats.py
```

The bucket and tier definitions live in the script so later meeting snapshots use the same accounting.
