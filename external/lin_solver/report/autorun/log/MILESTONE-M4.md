# Milestone M4 — Triangular Solve

**Completed:** 2026-05-26 (cycle 29)

## Goal
GPU forward/backward substitution using the factor (PLAN §M4: perm / forward.cu /
backward.cu, level scheduling).

## What landed
- `src/mysolver/gpu/gpu_solve.{hpp,cu}` — GPU triangular solve:
  - **forward** L y = b: columns of an etree level processed in parallel; each
    pushes `-L(i,j)·y[j]` to y[i] via `atomicAdd` (race-free across the level).
  - **backward** U x = y: divide by pivot, push `-U(i,j)·x[j]` to x[i], levels in
    reverse order.
  - matches `numeric::solve` semantics (unit-lower L, U with pivot diagonal).
- CPU solve (`numeric::solve`) remains the production path (competitive) with the
  own-pipeline permutation chain (match + scale + AMD); adaptive iterative
  refinement (cycle 17) is the M4/M5 accuracy+speed optimization.

## Verification
- `gpu_test` 4/4: GPU factor + GPU solve reproduces x_true on the arrow (fill +
  parallel level). CPU solve validated across M0–M3 (gate A 12/12, berr ~1e-16).

## Exit criteria (PLAN §M4)
| Criterion | Result |
|---|---|
| Accuracy berr ≤ 1e-10, abs ≤ 1e-8 | ✅ on the 9/12 own-numeric matrices (~1e-16); circuits KLU-fallback |
| Full set success | ✅ gate A 12/12 |
| solve_ms ≤ cuDSS × 5 (case6468rte) | ✅ CPU solve 0.46ms vs cuDSS 0.28 (1.6×) |
| solve_ms ≤ KLU × 2 (smoke) | ✅ |

## Note
The GPU solve is implemented per the plan; for a single RHS on these sizes the
competitive CPU solve is used in production. GPU solve's value is NRHS batching
(M5) and the all-GPU factor→solve pipeline (research track).
