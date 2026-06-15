# Mixed-precision power-flow study (rationale for cuPF Mixed)

Controlled NumPy/SciPy NR power-flow harness with exact per-operation precision control,
used to establish the rationale in `docs/mixed-precision-findings.md` (theory: inexact Newton +
mixed-precision iterative refinement). No GPU / cuPF build needed — accuracy is precision-determined.

- `harness.py` — case loader (pypower/pandapower), NR with per-op precision config, true-‖F‖ scoring,
  true FP32 sparse LU (`splu`), 1-norm condition estimate.
- `driver_A.py` — FP64 vs Mixed vs FP32 across the case suite: iters, true ‖F‖, forcing term η, κ.
- `driver_B.py` — per-operation precision ablation: decomposes the full-FP32 ~1e-3 floor.

Run: `cd tests/precision_study && python3 driver_A.py && python3 driver_B.py`
(requires numpy, scipy, pypower, pandapower). Results: `resA.json`, `resB.json`.
