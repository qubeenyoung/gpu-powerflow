# Full A / F / S / E2E latency + error: mysolver vs cuDSS (cycle 106, 2026-05-26)

Measured: `benchmark --solver mysolver,cudss-gpu` (warm). **A**=Analysis (reorder+
symbolic), **F**=Factor, **S**=Solve, **E2E**=A+F+S, all ms. **mysolver = production
CPU path** (own-numeric, MC64+scaling for circuits, residual-gated). berr = componentwise
backward error. Lower is better.

## All 13 matrices (9 power-grid + 4 circuit)

| matrix | n | solver | A | F | S | E2E | berr | E2E winner |
|---|---:|---|---:|---:|---:|---:|---|---|
| case30 | 53 | mysolver | 0.05 | 0.01 | 0.00 | **0.07** | 1.5e-16 | my 124× |
|  |  | cuDSS | 8.54 | 0.09 | 0.06 | 8.69 | 2.1e-16 | |
| case118 | 181 | mysolver | 0.09 | 0.03 | 0.01 | **0.14** | 2.1e-16 | my 64× |
|  |  | cuDSS | 8.80 | 0.10 | 0.07 | 8.97 | 2.1e-16 | |
| case1197 | 2 392 | mysolver | 0.72 | 0.13 | 0.09 | **0.94** | 1.7e-16 | my 13× |
|  |  | cuDSS | 11.61 | 0.22 | 0.11 | 11.95 | 5.7e-15 | |
| case_ACTIVSg2000 | 3 607 | mysolver | 1.82 | 2.12 | 0.19 | **4.13** | 4.5e-16 | my 3.6× |
|  |  | cuDSS | 13.89 | 0.64 | 0.27 | 14.81 | 5.2e-16 | |
| case3012wp | 5 725 | mysolver | 2.95 | 1.31 | 0.56 | **4.82** | 1.9e-16 | my 3.4× |
|  |  | cuDSS | 15.55 | 0.49 | 0.24 | 16.28 | 1.0e-14 | |
| case6468rte | 12 643 | mysolver | 8.33 | 3.23 | 0.56 | **12.12** | 5.6e-16 | my 2.1× |
|  |  | cuDSS | 24.05 | 0.61 | 0.29 | 24.94 | 2.8e-14 | |
| case8387pegase | 14 908 | mysolver | 7.78 | 4.12 | 1.23 | **13.13** | 2.8e-16 | my 2.3× |
|  |  | cuDSS | 28.09 | 1.10 | 0.36 | 29.55 | 3.5e-11 | |
| case_ACTIVSg25k | 47 246 | mysolver | 26.3 | 22.1 | 7.9 | **56.4** | 2.4e-16 | my 1.1× |
|  |  | cuDSS | 61.9 | 1.88 | 0.67 | 64.4 | 1.1e-14 | |
| case_SyntheticUSA | 156 255 | mysolver | 100.2 | 98.1 | 33.1 | 231.3 | 2.7e-16 | cuDSS 1.2× |
|  |  | cuDSS | 190.0 | 5.24 | 1.43 | **196.7** | 5.7e-14 | |
| memplus | 17 758 | mysolver | 8.95 | 2.56 | 0.37 | **11.9** | 1.0e-15 | my 8.4× |
|  |  | cuDSS | 97.7 | 1.14 | 0.43 | 99.3 | 9.3e-16 | |
| rajat27 | 20 640 | mysolver | 11.4 | 3.15 | 0.56 | **15.1** | 1.0e-13 | my 3.1× |
|  |  | cuDSS | 45.3 | 1.00 | 0.39 | 46.7 | 8.3e-12 | |
| onetone2 | 36 057 | mysolver | 22.6 | 148.8 | 6.6 | 178.0 | 4.1e-16 | cuDSS 1.9× |
|  |  | cuDSS | 82.5 | 7.95 | 1.53 | **92.0** | 1.9e-10 | |
| rajat15 | 37 261 | mysolver | 48.0 | 103.3 | 3.5 | 154.8 | 5.5e-13 | cuDSS 1.3× |
|  |  | cuDSS | 118.4 | 3.58 | 0.87 | **122.9** | 1.3e-13 | |

**Production (CPU) E2E: mysolver wins 10/13.** Loses only SyntheticUSA (156k),
onetone2, rajat15 — all because the **CPU factor** is slow on the largest power-grid
and the high-fill circuits (onetone2 F=149ms, rajat15 F=103ms). mysolver's analysis
is far cheaper than cuDSS's (cuDSS A = 8.5–190ms host reorder + GPU setup), and its
accuracy is best/comparable everywhere.

## With the GPU multifrontal factor+solve (cy70–104 research path)

The GPU MF factor+solve replaces the slow CPU F/S on exactly the 3 matrices mysolver
loses. Projected E2E = (CPU analysis) + (GPU MF F) + (GPU MF S):

| matrix | CPU-path E2E | with GPU MF F/S | cuDSS E2E | result |
|---|---:|---:|---:|---|
| case_SyntheticUSA | 231.3 | 100.2 + 6.55 + 2.28 = **109** | 196.7 | **my 1.8×** (was losing) |
| onetone2 | 178.0 | 22.6 + 36.2 + 4.18 = **63** | 92.0 | **my 1.5×** (was losing) |
| rajat15 | 154.8 | 48.0 + 3.74 + 1.92 = **53.7** | 122.9 | **my 2.3×** (was losing) |

**=> With the GPU MF factor+solve integrated, mysolver would win E2E on all 13.**
The GPU MF F/S also beats cuDSS's factor on case6468rte, case8387pegase, rajat27,
memplus and ties on rajat15 (see COMPETITIVE_SUMMARY). Remaining: onetone2 GPU factor
(36 vs cuDSS 8, inherent deep+high-fill); production integration of the GPU path.
