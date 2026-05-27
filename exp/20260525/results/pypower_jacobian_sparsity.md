# PyPower Jacobian Sparsity

- Date: 2026-05-25
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Tool path: `pypower.dSbus_dV`
- Matrix: MATPOWER/PyPower Newton four-block Jacobian `[J11 J12; J21 J22]`
- Structural density = stored nnz / (rows * cols)
- Structural sparsity = 1 - structural density

| case | buses | J dim | stored nnz | density (%) | sparsity (%) | explicit zeros |
|---|---:|---:|---:|---:|---:|---:|
| case_SyntheticUSA | 82,000 | 156,255 | 1,052,085 | 0.004309064 | 99.995690936 | 0 |
| case_ACTIVSg70k | 70,000 | 134,104 | 900,558 | 0.005007584 | 99.994992416 | 0 |
| case_ACTIVSg25k | 25,000 | 47,246 | 318,672 | 0.014276239 | 99.985723761 | 0 |
| case13659pegase | 13,659 | 23,225 | 173,948 | 0.032248387 | 99.967751613 | 755 |
| case_ACTIVSg10k | 10,000 | 18,544 | 125,174 | 0.036400495 | 99.963599505 | 0 |
| case9241pegase | 9,241 | 17,036 | 128,698 | 0.044344170 | 99.955655830 | 714 |
| case300 | 300 | 530 | 3,736 | 1.330010680 | 98.669989320 | 0 |
