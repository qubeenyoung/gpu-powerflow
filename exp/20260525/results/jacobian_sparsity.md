# MATPOWER cuPF Jacobian Sparsity

- Date: 2026-05-25
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- cuPF binary: `/workspace/gpu-powerflow/build/20260525-jacobian-sparsity/benchmarks/cupf_case_benchmark`
- Profile: `cuda_mixed_edge`
- Dumped matrix: `jacobian_iter0.txt` from repeat 00
- Structural density = stored nnz / (rows * cols)
- Structural sparsity = 1 - structural density

| case | buses | J dim | stored nnz | density (%) | sparsity (%) | numeric nnz | returncode |
|---|---:|---:|---:|---:|---:|---:|---:|
| case_SyntheticUSA | 82,000 | 156,255 | 1,052,085 | 0.004309064 | 99.995690936 | 1,052,085 | 0 |
| case_ACTIVSg70k | 70,000 | 134,104 | 900,558 | 0.005007584 | 99.994992416 | 900,558 | 0 |
| case_ACTIVSg25k | 25,000 | 47,246 | 318,672 | 0.014276239 | 99.985723761 | 318,672 | 0 |
| case13659pegase | 13,659 | 23,225 | 174,703 | 0.032388357 | 99.967611643 | 174,703 | 0 |
| case_ACTIVSg10k | 10,000 | 18,544 | 125,174 | 0.036400495 | 99.963599505 | 125,174 | 0 |
| case9241pegase | 9,241 | 17,036 | 129,412 | 0.044590186 | 99.955409814 | 129,412 | 0 |
| case300 | 300 | 530 | 3,736 | 1.330010680 | 98.669989320 | 3,736 | 0 |

## Raw Dumps

- `case_SyntheticUSA`: `/workspace/gpu-powerflow/exp/20260525/raw/jacobian_dumps/case_SyntheticUSA/repeat_00/jacobian_iter0.txt`
- `case_ACTIVSg70k`: `/workspace/gpu-powerflow/exp/20260525/raw/jacobian_dumps/case_ACTIVSg70k/repeat_00/jacobian_iter0.txt`
- `case_ACTIVSg25k`: `/workspace/gpu-powerflow/exp/20260525/raw/jacobian_dumps/case_ACTIVSg25k/repeat_00/jacobian_iter0.txt`
- `case13659pegase`: `/workspace/gpu-powerflow/exp/20260525/raw/jacobian_dumps/case13659pegase/repeat_00/jacobian_iter0.txt`
- `case_ACTIVSg10k`: `/workspace/gpu-powerflow/exp/20260525/raw/jacobian_dumps/case_ACTIVSg10k/repeat_00/jacobian_iter0.txt`
- `case9241pegase`: `/workspace/gpu-powerflow/exp/20260525/raw/jacobian_dumps/case9241pegase/repeat_00/jacobian_iter0.txt`
- `case300`: `/workspace/gpu-powerflow/exp/20260525/raw/jacobian_dumps/case300/repeat_00/jacobian_iter0.txt`
