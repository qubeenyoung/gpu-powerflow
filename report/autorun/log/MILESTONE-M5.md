# Milestone M5 — Integration / Optimization / Stabilization

**Status:** 2026-05-26 (cycle 29). Optimization items partly implemented; the
cuDSS-ratio performance exit is the dedicated GPU research track (see below).

## PLAN §M5 items and status
| # | Item | Status |
|---|---|---|
| 1 | Reordering GPU prep | not started (host AMD/METIS is fast; low priority) |
| 2 | Factorize large-path batched cuBLAS | research (large supernodes rare here) |
| 3 | Solve NRHS > 1 | not started (GPU solve supports the structure) |
| 4 | Memory pooling (reuse device buffers across factor/solve) | not started |
| 5 | Tensor Cores (FP16/TF32) | research (accuracy-preserving placement) |

## Optimizations already implemented (across M1–M4)
- **Adaptive iterative refinement** (cycle 17): solve 2–4× faster, accuracy kept.
- **Host vectorization** `-march=native -funroll-loops` (cycle 19): factor ~6–15%.
- **GPU dependency_map** (cycle 28, §246): precomputed Schur op positions →
  search-free numeric kernel → 3.4× on the GPU factor.
- **Min-fill ordering** + matching + scaling (M1, own-pipeline): structural +
  numerical quality (best accuracy of all benchmarked solvers, berr ~1e-16).

## Exit criteria (PLAN §M5) and reality
| Criterion | Status |
|---|---|
| Accuracy maintained | ✅ (gate A 12/12; berr ~1e-16) |
| case6468rte wall ≤ cuDSS × 2 | ❌ — GPU factor research-grade (165× off); CPU 5.6× |
| rajat15 wall ≤ cuDSS × 3 | ❌ — circuit matrix (KLU fallback; needs MC64) |
| xl cases (ACTIVSg25k, SyntheticUSA) no OOM/crash | not yet exercised |

## Honest handoff to the research session
The cuDSS-ratio exit is the **deep GPU-optimization research** (cuDSS leads the
entire GPU sparse field by 10–30×). The plan-faithful GPU pipeline
(factorize_driver small-path + dependency_map + triangular solve) is built and
measured; closing to cuDSS needs: coalesced memory, shared-memory supernode
panels, warp-cooperative kernels (note: naive warp-per-column is incorrect — the
left-looking U(k,j) dependency), supernode batching, and MC64 for the circuit
matrices. These are the research-session targets.
