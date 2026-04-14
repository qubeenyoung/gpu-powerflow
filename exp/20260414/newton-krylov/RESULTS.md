# Newton-Krylov Results

Date: 2026-04-14

## Implemented

Built a CPU FP64 Jacobian-free Newton-Krylov probe:

```text
/workspace/exp/20260414/newton-krylov/build/newton_krylov_probe
```

The nonlinear loop uses only:

```text
CpuMismatchOpF64 -> JFNK linear_solve -> CpuVoltageUpdateF64
```

The main solve path passes empty `JacobianMaps` and an empty
`JacobianStructure` to `CpuFp64Storage::prepare()`. It does not instantiate or
call a Jacobian op.

Implemented matrix-free Krylov solvers:

| solver flag | meaning |
| --- | --- |
| `--solver bicgstab_none` | BiCGSTAB with no preconditioner |
| `--solver gmres_none` | restarted GMRES with no preconditioner |
| `--solver bicgstab` | BiCGSTAB with selected preconditioner |
| `--solver fgmres` | restarted flexible GMRES with selected preconditioner |

Both use finite-difference `Jv`:

```text
Jv ~= (F(V + eps*v) - F(V)) / eps
```

`V + eps*v` is applied through `CpuVoltageUpdateF64` on a scratch CPU storage.

Implemented preconditioners:

| preconditioner flag | meaning |
| --- | --- |
| `--preconditioner none` | no preconditioner |
| `--preconditioner bus_block_jacobi_fd` | finite-difference bus-local block Jacobi: PV buses use 1x1 `P/theta`, PQ buses use 2x2 `[P,Q] x [theta, Vm]` |
| `--preconditioner ilut_fd` | ILUT built from finite-difference `Jv` columns |
| `--preconditioner ilu0_fd` | ILU(0) built from finite-difference `Jv` columns |
| `--preconditioner amg_fd` | HYPRE BoomerAMG built from finite-difference `Jv` columns |

These preconditioner setups still do not call a Jacobian op. They probe columns
through the same matrix-free `Jv` callback, which is intentionally simple but
expensive on CPU.

## Build

```bash
cmake -S /workspace/exp/20260414/newton-krylov \
  -B /workspace/exp/20260414/newton-krylov/build \
  -GNinja

cmake --build /workspace/exp/20260414/newton-krylov/build \
  --target newton_krylov_probe -j
```

## Data Root

Available benchmark dumps in this workspace:

```text
/workspace/datasets/pglib-opf/cuPF_benchmark_dumps
```

Cases:

```text
case30_ieee
case118_ieee
case793_goc
case1354_pegase
case2746wop_k
case4601_goc
case8387_pegase
case9241_pegase
```

## Corrected Nonlinear JFNK Solver Grid

This is the corrected experiment for the requested solver/preconditioner
combinations. It uses the raw PF dump cases:

```text
/workspace/datasets/pglib-opf/cuPF_benchmark_dumps
```

Cases:

```text
case30_ieee
case118_ieee
case793_goc
case1354_pegase
case2746wop_k
case4601_goc
case8387_pegase
case9241_pegase
```

The nonlinear loop is still:

```text
mismatch -> JFNK linear_solve -> voltage_update
```

For `ILUT`, `ILU(0)`, and `AMG`, the preconditioner matrix is assembled inside
`linear_solve` from finite-difference `Jv` calls. It does not read a saved
matrix file and does not call a Jacobian op. The suffix `FD` below means
"preconditioner built from the current nonlinear state through
`voltage_update + mismatch` Jv".

Run options:

```text
nonlinear tolerance = 1e-8
max outer iterations = 20
linear tolerance = 1e-2
max inner iterations = 2000
fd eps = auto
FGMRES restart = 30
ILUT drop_tol = 1e-4
ILUT fill_factor = 10
```

Output CSVs:

```text
/workspace/exp/20260414/newton-krylov/results/jfnk_corrected_grid
```

| solver | preconditioner | converged | mean outer | total Jv calls | total inner iters | total sec | failed cases |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BiCGSTAB | ILUT-FD | 8 / 8 | 6.50 | 341561 | 87 | 150.705 | none |
| BiCGSTAB | ILU(0)-FD | 6 / 8 | 5.50 | 194279 | 4759 | 54.139 | `case8387_pegase`, `case9241_pegase` |
| FGMRES(30) | ILU(0)-FD | 6 / 8 | 5.88 | 221544 | 10489 | 72.749 | `case8387_pegase`, `case9241_pegase` |
| BiCGSTAB | AMG-FD | 6 / 8 | 5.25 | 155162 | 4613 | 46.624 | `case8387_pegase`, `case9241_pegase` |
| FGMRES(30) | AMG-FD | 7 / 8 | 6.12 | 241734 | 6148 | 94.405 | `case9241_pegase` |

Case-level highlights:

| combo | `case8387_pegase` | `case9241_pegase` |
| --- | --- | --- |
| BiCGSTAB + ILUT-FD | converged, 8 outer, `6.959e-11` | converged, 9 outer, `1.224e-10` |
| BiCGSTAB + ILU(0)-FD | failed, linear max inner | failed, rho breakdown |
| FGMRES(30) + ILU(0)-FD | failed, linear max inner | failed, linear max inner |
| BiCGSTAB + AMG-FD | failed, linear max inner | failed, linear max inner |
| FGMRES(30) + AMG-FD | converged, 8 outer, `3.600e-09` | failed, linear max inner |

Interpretation for the corrected nonlinear JFNK run:

- `BiCGSTAB + ILUT-FD` is the strongest baseline: 8/8 converged.
- Among non-ILUT candidates, `FGMRES(30) + AMG-FD` is the only one that rescued
  `case8387_pegase`, so it is the best robustness candidate in this corrected
  JFNK run.
- `BiCGSTAB + ILU(0)-FD`, `FGMRES(30) + ILU(0)-FD`, and `BiCGSTAB + AMG-FD`
  all landed at 6/8 on the same two failures.
- `ILUT-FD` is much more robust, but it is expensive here because the
  preconditioner matrix is reassembled from FD `Jv` at each nonlinear state.

## Interpretation

The corrected nonlinear JFNK run shows that matrix-free `Jv` can drive the full
outer PF solve on all 8 benchmark dumps when paired with the strongest tested
preconditioner, `BiCGSTAB + ILUT-FD`.

For non-ILUT candidates, `FGMRES(30) + AMG-FD` is the most robust result so far:
it reaches 7/8 and is the only non-ILUT combination that converges
`case8387_pegase`. `case9241_pegase` remains the hard failure outside ILUT-FD.

## Next Recommendation

Keep `BiCGSTAB + none` as the pure JFNK baseline, but do not expect it to be the
main result. For the corrected nonlinear JFNK comparison, carry forward:

1. `BiCGSTAB + ILUT-FD` as the robustness baseline
2. `FGMRES(30) + AMG-FD` as the strongest non-ILUT robustness candidate
3. `BiCGSTAB + ILU(0)-FD` as the lightweight sparse factor candidate, but not
   as the current best robustness candidate

Next implement a GPU-friendly lightweight preconditioner before moving to CUDA:

1. replace the finite-difference bus-local `2x2` block setup with analytic
   bus-local blocks from current `V`, `Ibus`, and `Ybus`
2. fast-decoupled `B' / B''` relaxed preconditioner
3. evaluate stale or lagged `ILUT-FD` setup, since fresh ILUT-FD converges 8/8
   but spends most of its cost rebuilding the preconditioner
