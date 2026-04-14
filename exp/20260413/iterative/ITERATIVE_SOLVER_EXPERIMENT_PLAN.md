# Iterative Solver Experiment Plan

Date: 2026-04-13

## Purpose

This experiment compares iterative linear solvers for the Newton step linear
system:

```text
J dx = -F
```

The first target is not yet to replace cuDSS inside the full cuPF Newton loop.
The target here is to use the dumped PF linear systems and compare which
iterative solver/preconditioner combinations solve the same systems cheaply and
robustly.

## Dataset

Input snapshots:

```text
/workspace/exp/20260413/iterative/pf_dumps/<case>/cuda_mixed_edge/iter_000/
```

Each snapshot has:

```text
J.csr
rhs.txt
x_direct.txt
meta.txt
```

Summary metadata:

```text
/workspace/exp/20260413/iterative/results/pf_dataset_snapshot_summary.csv
```

This CSV records whether the original Newton solve converged within
`max_iter=10`. That flag must be carried into all iterative-solver result tables
so we can classify each linear solve result by:

- iterative solver solved / did not solve
- original Newton case converged / did not converge

## Baseline

The baseline is:

```text
BiCGSTAB + ILUT
```

Current baseline parameters:

```text
solver: BiCGSTAB
preconditioner: ILUT
tolerance: 1e-8
max_iter: 1000
ILUT drop_tol: 1e-4
ILUT fill_factor: 10
```

Existing baseline result over all 67 PF snapshots:

```text
raw result:
/workspace/exp/20260413/iterative/results/pf_iterative_bicgstab_ilut.csv

merged with Newton convergence status:
/workspace/exp/20260413/iterative/results/pf_iterative_vs_newton_summary.csv
```

Observed baseline:

| Criterion | Result |
| --- | ---: |
| Eigen success flag | 67 / 67 |
| Measured `relative_residual_inf <= 1e-8` | 64 / 67 |
| Iterative solved and Newton converged | 32 |
| Iterative solved and Newton did not converge | 32 |
| Iterative not solved and Newton converged | 0 |
| Iterative not solved and Newton did not converge | 3 |

The 3 strict residual misses were only slightly above the `1e-8` measured
threshold:

| Case | Measured relative residual |
| --- | ---: |
| `case162_ieee_dtc` | 1.093e-08 |
| `case19402_goc` | 1.316e-08 |
| `case4020_goc` | 1.062e-08 |

This baseline must be kept. All other combinations should be judged by how much
cheaper they are and how much robustness they lose relative to this baseline.

## Candidate 1: BiCGSTAB + ILU(0)

Purpose:

```text
First lightweight ILU-family candidate.
```

Rationale:

ILUT is an ILU-family dual-threshold method. The most natural lighter version is
level-based ILU. Start with `ILU(0)` only:

```text
solver: BiCGSTAB
preconditioner: ILU(0)
tolerance: 1e-8
max_iter: 1000
```

Do not sweep `k` initially. If `ILU(0)` collapses badly relative to the ILUT
baseline, add only one follow-up:

```text
ILU(1)
```

Implementation note:

`ILU(0)` means incomplete LU with the original matrix sparsity pattern. In PETSc
terms this corresponds to:

```text
-ksp_type bcgs
-pc_type ilu
-pc_factor_levels 0
```

If we implement this inside the local C++ harness rather than via PETSc, the
preconditioner must be level-of-fill ILU(0), not ILUT with different thresholds.

## Candidate 2: BiCGSTAB + Block-Jacobi

Purpose:

```text
Lightweight GPU-friendly preconditioner candidate.
```

Rationale:

Block-Jacobi is stronger than point Jacobi but much simpler than ILU-family
preconditioners. It avoids global triangular solves, which makes it useful as a
GPU-oriented direction.

Initial configuration:

```text
solver: BiCGSTAB
preconditioner: Block-Jacobi
tolerance: 1e-8
max_iter: 1000
```

Block choices should be conservative:

1. Natural contiguous row blocks as a first harness baseline.
2. If needed, PF-structure-aware blocks can be tested later, but not in the
   first comparison.

PETSc equivalent direction:

```text
-ksp_type bcgs
-pc_type bjacobi
```

## Metrics

Collect these for every case and every solver configuration:

| Metric | Reason |
| --- | --- |
| `iterative_success_flag` | Library-reported solve status |
| `relative_residual_inf` | Main solve-quality criterion |
| `residual_inf` | Absolute residual |
| `iterations` | Robustness and convergence speed |
| `setup_sec` | Preconditioner setup cost |
| `solve_sec` | Iteration solve cost |
| `total_sec = setup_sec + solve_sec` | End-to-end linear solve cost |
| `x_delta_direct_inf` | Difference from cuDSS direct solution |
| `direct_residual_inf` | cuDSS direct solution residual on dumped system |
| `newton_converged` | Original Newton case classification |

The main pass/fail criterion should match the baseline summary:

```text
iterative_solved = success flag AND relative_residual_inf <= 1e-8
```

## Result Tables

For each solver configuration, write one raw CSV and one merged CSV:

```text
results/pf_iterative_<solver>_<preconditioner>.csv
results/pf_iterative_<solver>_<preconditioner>_vs_newton.csv
```

The merged CSV must include:

```text
case
iterative_solved
iterative_success_flag
iterative_iterations
iterative_relative_residual_inf
iterative_residual_inf
setup_sec
solve_sec
newton_converged
newton_iterations
newton_final_mismatch
n_bus
n_pv
n_pq
snapshot
```

## Comparison Order

Run in this order:

1. Keep `BiCGSTAB + ILUT` as the baseline.
2. Run `BiCGSTAB + ILU(0)`.
3. Run `BiCGSTAB + Block-Jacobi`.
4. Add `BiCGSTAB + ILU(1)` only if `ILU(0)` is too weak and still looks close
   enough to be worth rescuing.

## Expected Interpretation

Expected outcomes:

- If `ILU(0)` retains most of the ILUT success rate with lower setup/solve time,
  it becomes the strongest immediate replacement candidate.
- If `ILU(0)` loses too much robustness, keep it as a reference and try `ILU(1)`.
- If `Block-Jacobi` is weaker but much cheaper, it may still be useful for a GPU
  path or as part of a later flexible/preconditioned Krylov design.
- If both lightweight candidates fail on cases that ILUT solves, the next step
  should not be a wide solver sweep. First inspect the failed cases by matrix
  size, conditioning proxy, and Newton convergence class.
