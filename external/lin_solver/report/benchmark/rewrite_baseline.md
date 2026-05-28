# PRE-REWRITE BASELINE (cycle 162, HOST-LOCKED 1395MHz)

Frozen baseline before the cuDSS-internals A/S rewrite (user-directed). All warm-kernel ms,
locked clock (cross-process variance <0.2%), gpu_test 4/4. Goal of the rewrite: improve A
and S toward cuDSS, KEEP F (must not regress vs these numbers). cuDSS refs from cy145.

## Power-grid A / F / S (ms)
| matrix | A ser | A parND | F ser | F parND | S ser | S parND | cuDSS A | cuDSS F | cuDSS S |
|---|---|---|---|---|---|---|---|---|---|
| case6468rte | 33.8 | 34.0 | 0.524 | 0.492 | 0.548 | 0.517 | 25.8 | 0.734 | 0.339 |
| case8387pegase | 44.1 | 43.7 | 0.711 | 0.656 | 0.696 | 0.668 | 31.4 | 1.338 | 0.433 |
| case_ACTIVSg25k | 152.4 | 101.2 | 1.787 | 1.535 | 1.584 | 1.359 | 61.7 | 2.225 | 0.807 |
| case_SyntheticUSA | 603.7 | 358.6 | 4.113 | 4.018 | 2.767 | 2.767 | 203.2 | 6.069 | 1.713 |

## Circuits F / S (ms, serial ND; berr)
| matrix | F | S | berr | cuDSS F | cuDSS S | cuDSS A |
|---|---|---|---|---|---|---|
| rajat27 | 0.843 | 0.981 | 4.1e-13 | 1.205 | 0.463 | 51.9 |
| memplus | 0.695 | 0.672 | 1.0e-15 | 1.369 | 0.503 | 55.3 |
| onetone2 | 11.17 | 4.540 | 4.8e-08(raw) | 9.829 | 1.826 | 90.6 |
| rajat15 | 3.287 | 2.514 | 2.7e-13 | 4.263 | 1.035 | 125.0 |

## Standing summary (baseline)
- **F: beats cuDSS 7/8** (only onetone2 1.14x). MUST be preserved by the rewrite.
- **A: gap 1.6-2.0x** with parallel ND (opt-in, non-deterministic); 2.5-3.1x serial.
- **S: 1.3-2.5x behind** (multifrontal kernel limit).
- Accuracy >= cuDSS everywhere (refined to 1e-13..1e-16).
- NOTE: parallel ND's ordering also improves F+S on some matrices (ACTIV F/S -14%) via lower
  fill -> the rewrite's deterministic-parallel-ordering could lock in F/S gains too.

## Rewrite targets (cuDSS-internals, multi-week)
1. **A**: deterministic parallel ordering (parallel ND works -40% but METIS RNG is
   thread-unsafe -> non-deterministic). Make it deterministic (ParMETIS / process-parallel /
   thread-safe ND) -> ship as default.
2. **S**: cuDSS-style solve (data layout + scheduling) to cut the atomic-scatter +
   scattered-x-gather + level-serialization bottlenecks.
