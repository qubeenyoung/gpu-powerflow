# PyPower vs cuPF Jacobian Sparsity

- PyPower CSV: `exp/20260525/results/pypower_jacobian_sparsity.csv`
- cuPF CSV: `/workspace/gpu-powerflow/exp/20260525/results/jacobian_sparsity.csv`

| case | PyPower nnz | cuPF nnz | nnz diff | PyPower sparsity (%) | cuPF sparsity (%) | sparsity diff (pp) |
|---|---:|---:|---:|---:|---:|---:|
| case_SyntheticUSA | 1,052,085 | 1,052,085 | 0 | 99.995690936 | 99.995690936 | 0.000000000 |
| case_ACTIVSg70k | 900,558 | 900,558 | 0 | 99.994992416 | 99.994992416 | 0.000000000 |
| case_ACTIVSg25k | 318,672 | 318,672 | 0 | 99.985723761 | 99.985723761 | 0.000000000 |
| case13659pegase | 173,948 | 174,703 | -755 | 99.967751613 | 99.967611643 | 0.000139970 |
| case_ACTIVSg10k | 125,174 | 125,174 | 0 | 99.963599505 | 99.963599505 | 0.000000000 |
| case9241pegase | 128,698 | 129,412 | -714 | 99.955655830 | 99.955409814 | 0.000246016 |
| case300 | 3,736 | 3,736 | 0 | 98.669989320 | 98.669989320 | 0.000000000 |
