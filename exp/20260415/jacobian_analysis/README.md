# 2026-04-15 Jacobian Block Norm Analysis

This experiment measures the norms of the four cuPF Newton Jacobian blocks:

```text
J = [ J11  J12 ] = [ dP/dtheta  dP/dVm ]
    [ J21  J22 ]   [ dQ/dtheta  dQ/dVm ]
```

The layout follows cuPF's reduced Jacobian:

- `theta` columns and P rows use `pvpq = pv || pq`
- `Vm` columns and Q rows use `pq`
- `dim(J) = len(pvpq) + len(pq)`

Run all dumped cases:

```bash
python3 exp/20260415/jacobian_analysis/scripts/analyze_jacobian_blocks.py
```

Outputs:

- `results/block_norms.csv`: one row per case and Jacobian block
- `results/block_norm_summary.csv`: one row per case with diagonal/off-diagonal ratios
- `results/voltage_scale_stats.csv`: Vm/theta distribution statistics by case, variable, and subset
- `results/voltage_scale_overview.csv`: compact Newton-variable Vm/theta scale summary by case

Useful options:

```bash
python3 exp/20260415/jacobian_analysis/scripts/analyze_jacobian_blocks.py \
  --dataset-root exp/20260414/amgx/cupf_dumps \
  --output-dir exp/20260415/jacobian_analysis/results \
  --case case_ACTIVSg200
```

Voltage magnitude and angle scale:

```bash
python3 exp/20260415/jacobian_analysis/scripts/analyze_voltage_scales.py
```

This reports both raw `theta = atan2(V.imag, V.real)` and slack-relative
`theta - theta_slack` wrapped to `[-pi, pi]`.  The reduced Newton subsets are
`theta` over `pv || pq` and `Vm` over `pq`.
