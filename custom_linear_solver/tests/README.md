# custom_linear_solver / tests

Test harnesses and analysis scripts for the solver. **`run_custom_solver.cu` is the
canonical runner** — build it and use it to exercise / time / validate the solver. The
Python files are offline analysis tooling (not part of the build).

## Files

| File | Built target | Role |
|------|--------------|------|
| `run_custom_solver.cu` | `custom_linear_solver_run` | **Canonical runner**: analyze → factorize → solve on a MatrixMarket case, prints timings + residual. The B=1 single-system and B>1 batched paths both run from here. |
| `io.cpp` / `io.hpp` | (linked into the runners) | MatrixMarket CSR / dense-vector read & write. Shared. |
| `*.py` | — | Offline analysis (elimination-tree depth, GEMM-fraction, front-size distributions, no-pivot stability proof). Standalone; run with `python3 <script>.py`. Not part of CMake. |

## Build & run (canonical)

```sh
# from custom_linear_solver/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCLS_CUDA_ARCHITECTURES=86
cmake --build build -j

# A case directory holds J.mtx (the matrix) and F.mtx (the RHS).
./build/custom_linear_solver_run ../exp/cases/case_ACTIVSg25k \
    --precision tf32 --repeat 7 --warmup 3
```

## Test matrices

A case directory holds two MatrixMarket files:
- `J.mtx` — the matrix, `coordinate real general` (sparse COO, unsymmetric storage).
- `F.mtx` — the RHS, `array real general` (dense `n×1` vector).

You may also pass `--matrix J.mtx --rhs F.mtx` directly instead of a case directory.

**`../exp/cases/` — power-flow Jacobians (the primary target domain).** Newton–Raphson
Jacobians; tiny fronts (panel cap nc ≤ 8), the latency-bound regime the solver targets.
`n` is the matrix (Jacobian) dimension = `2·npq + npv`, NOT the bus count.

| case | bus | n (= J dim) | nnz |
|------|-----|-------------|-----|
| `case118` | 118 | 181 | 1 051 |
| `case1354pegase` | 1 354 | 2 447 | 15 803 |
| `case3120sp` | 3 120 | 5 890 | 37 040 |
| `case6468rte` | 6 468 | 12 293 | 84 465 |
| `case8387pegase` | 8 387 | 14 908 | 110 572 |
| `case13659pegase` | 13 659 | 23 225 | 174 703 |
| `case_ACTIVSg25k` | 25 000 | 46 764 | 314 914 |
| `case_ACTIVSg70k` | 70 000 | 132 918 | 891 438 |
| `case_SyntheticUSA` | 82 000 | 154 703 | 1 040 157 |

All nine are canonical NR linear systems — the Jacobian at NR iteration 2 with the power
mismatch as RHS, each with a `metadata.json` — generated the SAME way (so the set is
uniform). They solve at fp64 ≈ 1e-13..1e-16. tf32 accuracy depends on conditioning
(small cases ≈ 1e-5; the larger RTE/PEGASE/ACTIVSg cases are stiffer — `case_ACTIVSg70k`
≈ 6e-3), which is the TF32/Ozaki floor for ill-conditioned grids, not a bug.

Regenerate or add more from MATPOWER `.m` cases (no MATLAB/Julia needed — pure
pandapower + scipy; `--cases` takes case names or `.m` paths):

```sh
# from repo root.
python3 -m python.prepare.convert_linear_system \
    --dataset-root /opt/matpower/data --output-root exp/cases \
    --cases case118 case1354pegase case3120sp case6468rte case8387pegase \
            case13659pegase case_ACTIVSg25k case_ACTIVSg70k case_SyntheticUSA
```

**`../exp/cases_ss_full/` — SuiteSparse generalization (out-of-domain stress).** Spans
tiny-front → large-front classes to probe where the multifrontal path generalizes. NOT
the target domain — expect worse behavior.

| case | n | nnz | class | status |
|------|----|-----|-------|--------|
| `cant` | 62 451 | 4.0 M | 3D FEM | solves, but **relres ≈ 7e-7** — no-pivot LU accuracy limit on stiff fronts, not a bug |
| `parabolic_fem` | 525 825 | 3.67 M | 2D FEM | solves (relres ≈ 1.6e-11) |
| `bmwcra_1` | 148 770 | 10.6 M | 3D structural | **fails at analyze** — front arena exceeds the 1G-double (8 GB) cap (huge fill) |
| `G3_circuit` | 1 585 478 | 7.66 M | 2D circuit/PDE | **fails at analyze** — same arena-cap bail |

The two analyze failures are a deliberate symbolic-phase guard (`layout_front_arena`
returns false past 1G doubles), not a regression — they fail identically with the batched
and single-system paths. `cant`'s low accuracy is the no-pivot algorithm, also expected.

## Flags (run_custom_solver)

```
<case-dir>                 directory holding J.mtx + F.mtx (or use --matrix/--rhs)
--matrix PATH --rhs PATH   matrix + RHS as MatrixMarket files
--precision fp64|fp32|tf32 batched/factor precision (tf32 = TF32 Ozaki mma). Default fp64.
--single-precision fp64|fp32   single-system input dtype (the non-batch run)
--batch B                  also run a uniform-batch experiment with B systems
--batch-only               skip the single-system run
--repeat N                 timed trials, median reported (default 1)
--warmup N                 untimed iters discarded before timing (default 0)
--max-panel-width N        analyze: max supernode amalgamation width (1..64, default 8)
--serial-nd                deterministic serial METIS NodeND instead of parallel ND
--matching                 structural row/column matching before factorization
--pivot-strategy none|shift static-diagonal-shift pivoting (default shift)
--pivot-epsilon X          shift threshold/magnitude
--analyze-only             stop after analyze (pair with --dump-fronts)
--analyze-info             print front-size / subtree summary
--dump-fronts PATH         write per-front CSV after analyze
--solution-out PATH        write recovered x as MatrixMarket
```

## Output keys (the machine-readable contract)

The runner prints `key=value` lines. These keys are STABLE — scripts parse them, so do
not rename them:

```
n, nnz, analyze_ms, factorize_ms, solve_ms          # single-system (B=1) timings
relative_residual_l2, residual_l2, residual_inf      # accuracy (B=1 solution)
setup_ms                                             # batch setup (one-time)
batch_factor_per_sys_ms, batch_solve_per_sys_ms      # per-system batched timings
batch_relres                                         # max relative residual over all B systems
```

`factorize_ms` / `solve_ms` are wall-clock medians over `--repeat` (each trial syncs
inside the timed region). `batch_*_per_sys_ms` is the batch median divided by B.

## Gotchas (avoid the recurring mistakes)

- **B=1 vs batch are different code paths.** The default (no `--batch`) run is the
  single-system path; `--batch B` runs the batched path. Both must stay correct — check
  `relative_residual_l2` (B=1) AND `batch_relres` (batch).
- **Accuracy expectations (conditioning-dependent):** fp64 ≈ 1e-13..1e-16; fp32 ≈ 1e-4;
  tf32 ≈ 1e-4 on small/well-conditioned cases up to ≈ 1e-2 on the large stiff grids
  (e.g. `case_ACTIVSg70k` ≈ 3e-3..6e-3). That 1e-3..1e-2 on big ill-conditioned cases is
  the TF32/Ozaki floor, **not** a bug. Ozaki actually being off looks far worse —
  relres ≈ 1e-1 or outright divergence (>1), not 1e-3.
- **`exp/cases/...` is relative to the repo, not to `tests/`.** Run from
  `custom_linear_solver/` (or pass absolute paths).
