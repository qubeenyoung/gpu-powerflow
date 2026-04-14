# J SPD Experiment Summary

Date: 2026-04-13

Command:

```bash
python3 /workspace/exp/20260413/j_spd_exp/check_j_spd.py --all
```

Output:

```text
/workspace/exp/20260413/j_spd_exp/results/j_spd_summary.csv
```

## Aggregate Results

| metric | result |
|---|---:|
| cases checked | 67 |
| dumped cuPF J matrices checked | 67 |
| Ybus structural-asymmetry nonzero cases | 0 |
| Python-rebuilt cuPF pattern structural-asymmetry nonzero cases | 0 |
| symmetric Ybus-lift pattern differing from cuPF pattern | 0 |
| dumped J structural-asymmetry nonzero cases | 0 |
| dumped J pattern differing from Python-rebuilt cuPF pattern | 0 |
| numerically symmetric dumped J cases | 1 (`case3_lmbd`) |
| dense eig cases (`dimF <= 300`) | 12 |
| raw SPD among dense eig cases | 1 (`case3_lmbd`) |
| symmetric-part SPD among dense eig cases | 11 |
| normal-equation SPD among dense eig cases | 12 |

The only non-SPD symmetric part in the dense-eig subset was:

| case | `min eig((J + J.T) / 2)` |
|---|---:|
| `case60_c` | -48.12920543291343 |

## Interpretation

The current four-block cuPF Jacobian layout does not force structural
asymmetry for the checked cases. Because all checked Ybus patterns were
structurally symmetric, the resulting cuPF Jacobian patterns were also
structurally symmetric: J11/J22 are symmetric blocks, and J12/J21 are
structural transposes.

The actual numeric cuPF Jacobian is still generally nonsymmetric. Therefore the
raw Newton matrix should not be treated as SPD. The only raw SPD case in the
small dense-eigenvalue subset was the trivial 2-by-2 `case3_lmbd`.

The proposed symmetric Ybus-lift pattern,

```text
pattern(Ybus) OR pattern(Ybus.T) OR I
```

matched the current cuPF pattern for all checked cases. It remains useful as a
guardrail if a future dataset has structurally asymmetric Ybus.

`J.T @ J` was SPD for the small checked cases, but that is a normal-equation
formulation. It changes sparsity and squares conditioning, so it is not a
drop-in replacement for using an SPD solver on the raw Newton Jacobian.
