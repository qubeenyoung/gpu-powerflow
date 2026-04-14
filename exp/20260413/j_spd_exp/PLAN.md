# J SPD Experiment Plan

Date: 2026-04-13

## Purpose

This experiment checks whether the current cuPF Newton Jacobian can be treated
as symmetric positive definite (SPD), and whether the current four-block
assembly is structurally symmetric.

The target questions are:

- Does the current cuPF four-block Jacobian pattern force structural
  asymmetry?
- If it is not structurally symmetric, what mapping could produce a symmetric
  pattern from the Ybus sparsity?
- For the resulting matrices, are they numerically symmetric or SPD?

## Code Reading Notes

Main files:

- `/workspace/cuPF/cpp/src/newton_solver/core/jacobian_builder.cpp`
- `/workspace/cuPF/cpp/inc/newton_solver/core/jacobian_types.hpp`
- `/workspace/cuPF/cpp/src/newton_solver/ops/jacobian/cpu_f64.cpp`
- `/workspace/cuPF/cpp/src/newton_solver/ops/jacobian/cuda_edge_fp64.cu`
- `/workspace/cuPF/cpp/src/newton_solver/ops/jacobian/cuda_vertex_fp64.cu`

Current cuPF structure:

```text
pvpq = pv || pq
dimF = n_pvpq + n_pq

J = [ J11  J12 ] = [ dP/dtheta  dP/dVm ]
    [ J21  J22 ]   [ dQ/dtheta  dQ/dVm ]

J11 rows pvpq, cols pvpq
J12 rows pvpq, cols pq
J21 rows pq,   cols pvpq
J22 rows pq,   cols pq
```

`JacobianBuilder::analyzeEdgeBased()` builds the pattern by iterating over
Ybus nonzeros and registering up to four coordinates:

```text
(i in pvpq, j in pvpq) -> J11
(i in pq,   j in pvpq) -> J21
(i in pvpq, j in pq)   -> J12
(i in pq,   j in pq)   -> J22
```

`analyzeVertexBased()` reuses the same pattern and only changes the fill kernel.

Important preliminary conclusion: the four-block layout does not by itself
force structural asymmetry. If Ybus has a structurally symmetric pattern, then
J11/J22 are structurally symmetric and J12/J21 are structural transposes. The
numeric Jacobian values are still generally nonsymmetric because the polar
power-flow partial derivatives are not transpose pairs.

## Symmetric Pattern Candidate

Use the same state/equation slots as cuPF:

```text
state slots:    theta[pvpq], Vm[pq]
equation slots: P[pvpq],     Q[pq]
```

Then build the pattern from a structural Ybus closure:

```text
Ypat_sym = pattern(Ybus) OR pattern(Ybus.T) OR I
```

For each `(i, j)` in `Ypat_sym`, add the same four possible block coordinates
as cuPF. This guarantees structural symmetry under the current slot ordering.

Expected relationship:

- If `pattern(Ybus)` is already symmetric, this candidate should match the
  current cuPF Jacobian pattern.
- If `pattern(Ybus)` is not symmetric, this candidate adds the missing
  transpose-side coordinates and may increase nnz.

This only addresses structural symmetry. It does not make the Newton Jacobian
numerically symmetric or SPD.

## SPD Checks

For actual dumped cuPF matrices:

```text
raw_spd = numeric_symmetric(J) AND min_eig(J) > 0
```

If `J != J.T`, it is not SPD regardless of Cholesky behavior in libraries that
may read only one triangle.

Diagnostic alternatives:

- `sympart = (J + J.T) / 2`
  - symmetric by construction
  - not the Newton matrix
  - may be indefinite
- `normal = J.T @ J`
  - symmetric positive semidefinite by construction
  - SPD if J has full column rank
  - solves the same square nonsingular Newton step through normal equations in
    exact arithmetic, but squares the condition number and changes sparsity

## Python Experiment

Script:

```text
/workspace/exp/20260413/j_spd_exp/check_j_spd.py
```

Inputs:

```text
/workspace/datasets/nr_dataset/<case>/Ybus.npz
/workspace/datasets/nr_dataset/<case>/pv.npy
/workspace/datasets/nr_dataset/<case>/pq.npy
/workspace/exp/20260413/iterative/pf_dumps/<short_case>/cuda_mixed_edge/iter_000/J.csr
```

Primary command:

```bash
python3 /workspace/exp/20260413/j_spd_exp/check_j_spd.py --all
```

Smoke command:

```bash
python3 /workspace/exp/20260413/j_spd_exp/check_j_spd.py \
  --cases case14_ieee case60_c case118_ieee case9241_pegase
```

Default output:

```text
/workspace/exp/20260413/j_spd_exp/results/j_spd_summary.csv
```

Metrics:

| column | meaning |
|---|---|
| `ybus_struct_asym_nnz` | structural asymmetry count for Ybus |
| `cupf_pattern_struct_asym_nnz` | structural asymmetry count for Python-rebuilt cuPF pattern |
| `lift_pattern_struct_asym_nnz` | structural asymmetry count for Ybus structural-closure pattern |
| `dump_pattern_struct_asym_nnz` | structural asymmetry count for actual dumped cuPF J |
| `dump_pattern_vs_cupf_xor_nnz` | mismatch between dumped J pattern and Python-rebuilt cuPF pattern |
| `dump_value_asym_inf` | `max(abs(J - J.T))` |
| `raw_spd` | true only when J is numerically symmetric and positive definite |
| `sympart_min_eig` | minimum eigenvalue of `(J + J.T) / 2` for small cases |
| `normal_min_eig` | minimum eigenvalue of `J.T @ J` for small cases |

## Initial Smoke Observations

From a quick read of the existing dumped systems:

- All 67 dumped cuPF matrices under
  `/workspace/exp/20260413/iterative/pf_dumps` were structurally symmetric.
- Their numeric values were generally nonsymmetric; only the 2-by-2
  `case3_lmbd` dump was exactly symmetric in this smoke check.
- Among dumped cases with `dim <= 300`, the symmetric part was indefinite for
  `case60_c`, so `(J + J.T) / 2` cannot be assumed SPD.
- For the same small set, `J.T @ J` had positive minimum eigenvalues, but its
  condition number was much larger than the original solve should imply.

The experiment script records these findings in a reproducible CSV rather than
leaving them as manual notes.
