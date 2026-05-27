# MATPOWER cuPF Jacobian Sparsity

This experiment dumps the first Newton Jacobian from cuPF for selected
MATPOWER cases and measures the CSR structural sparsity.

## Command

```bash
python3 exp/20260525/scripts/run_jacobian_sparsity.py
```

The script builds:

```text
build/20260525-jacobian-sparsity/benchmarks/cupf_case_benchmark
```

with `ENABLE_DUMP=ON`, then runs `cuda_mixed_edge` with
`--max-iter 1 --tolerance 0` so `jacobian_iter0.txt` is produced for each
case. The `success=false` field in the CSV only means the run was intentionally
stopped after one Newton iteration, not that the Jacobian dump failed.

## Outputs

- `results/jacobian_sparsity.csv`: machine-readable sparsity metrics.
- `results/jacobian_sparsity.md`: compact report table.
- `results/pypower_jacobian_sparsity.csv`: PyPower `dSbus_dV` sparsity metrics.
- `results/pypower_jacobian_sparsity.md`: PyPower compact report table.
- `results/pypower_vs_cupf_jacobian_sparsity.md`: PyPower/cuPF comparison.
- `raw/jacobian_dumps/<case>/repeat_00/jacobian_iter0.txt`: raw cuPF CSR dumps.
- `logs/<case>.stdout.txt` and `logs/<case>.stderr.txt`: benchmark logs.

## cuPF Summary

| case | J dim | stored nnz | density (%) | sparsity (%) |
|---|---:|---:|---:|---:|
| case_SyntheticUSA | 156,255 | 1,052,085 | 0.004309064 | 99.995690936 |
| case_ACTIVSg70k | 134,104 | 900,558 | 0.005007584 | 99.994992416 |
| case_ACTIVSg25k | 47,246 | 318,672 | 0.014276239 | 99.985723761 |
| case13659pegase | 23,225 | 174,703 | 0.032388357 | 99.967611643 |
| case_ACTIVSg10k | 18,544 | 125,174 | 0.036400495 | 99.963599505 |
| case9241pegase | 17,036 | 129,412 | 0.044590186 | 99.955409814 |
| case300 | 530 | 3,736 | 1.330010680 | 98.669989320 |

## PyPower Summary

PyPower uses `pypower.dSbus_dV` and builds the standard four-block Newton
Jacobian `[J11 J12; J21 J22]` from the same `dump_Ybus.mtx`, `dump_V.txt`,
`dump_pv.txt`, and `dump_pq.txt` inputs. Explicit zero entries are removed
before reporting `stored nnz`.

| case | J dim | stored nnz | density (%) | sparsity (%) |
|---|---:|---:|---:|---:|
| case_SyntheticUSA | 156,255 | 1,052,085 | 0.004309064 | 99.995690936 |
| case_ACTIVSg70k | 134,104 | 900,558 | 0.005007584 | 99.994992416 |
| case_ACTIVSg25k | 47,246 | 318,672 | 0.014276239 | 99.985723761 |
| case13659pegase | 23,225 | 173,948 | 0.032248387 | 99.967751613 |
| case_ACTIVSg10k | 18,544 | 125,174 | 0.036400495 | 99.963599505 |
| case9241pegase | 17,036 | 128,698 | 0.044344170 | 99.955655830 |
| case300 | 530 | 3,736 | 1.330010680 | 98.669989320 |

The two PEGASE cases differ from cuPF only by PyPower explicit-zero cleanup:
`case13659pegase` removes 755 entries and `case9241pegase` removes 714 entries.
