# Milestone M0 — Baseline & Harness

**Completed:** 2026-05-26 (cycle 1)
**Tag:** `m0`

## Goal
Stand up the autorun loop and benchmark infrastructure end-to-end with the
solver body delegated to KLU, so later milestones replace internals against a
working harness.

## What landed
- `mysolver` public API (`src/mysolver/solver.{hpp,cpp}`) exposing the cuDSS
  phase contract `analyze → factorize → solve` (PLAN §1.2). M0 backend delegates
  numeric work to SuiteSparse KLU behind that contract.
- `--solver mysolver` registered via `src/benchmark/solver_registry.{hpp,cpp}`;
  each phase timed independently into `analysis_ms / factor_ms / solve_ms`.
- `smoke` matrix set aligned to PLAN §2/§4: `memplus, rajat27, wang3, case30,
  case118, case1197`.
- Unit test `tests/mysolver/test_solver.cpp` (5×5 system; verifies solve
  correctness + AnalyzeResult reuse across factorizations). 2/2 pass.
- Autorun harness `src/autorun/{state,analyze_csv,discord_report,run_cycle}.py`.

## Exit criteria (PLAN §M0)
| Criterion | Result |
|---|---|
| `success=true` on all smoke matrices | ✅ 6/6 |
| `berr`/`absolute_error` identical to KLU | ✅ bit-for-bit on all 6 |
| Automation cycle completes (`cycle_count >= 1`) | ✅ cycle 1 |
| Discord baseline report delivered | ✅ change + benchmark reports |

## Accuracy gate decision (option A, confirmed by user 2026-05-26)
KLU-fallback milestones (M0–M2) gate accuracy as **"mysolver berr ≤ KLU berr"**
(PLAN §8: a high berr that KLU also produces is a matrix property, not a
regression). Strict absolute `berr ≤ 1e-10` applies from M3, when native numeric
factorization replaces the fallback.

- `rajat27`: KLU (and therefore mysolver) yields berr=1.53e-5; umfpack=8.4e-16,
  cudss=8.3e-12. Target 1e-10 is achievable (umfpack/cudss prove it) and is
  deferred to M3.
- `wang3`: KLU fill-in is pathological (factor 3.3 s vs cudss 34 ms); mysolver
  inherits it. M0 has no performance gate.

## Baseline numbers (smoke, factor_ms — mysolver / klu / cudss-gpu)
```
case30    0.015 / 0.016 / 0.087
case118   0.039 / 0.038 / 0.102
case1197  0.198 / 0.198 / 0.224
memplus   2.28  / 2.84  / 1.15
rajat27   3.05  / 3.01  / 1.00
wang3     3444  / 3329  / 34
```
Artifacts: `report/benchmark/baseline_reference.csv`,
`report/benchmark/baseline_mysolver.csv`, `report/benchmark/cycle_0001.csv`.
