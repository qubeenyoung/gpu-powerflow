# A / F / S vs cuDSS — all benchmarks (cycle 151, HOST-LOCKED 1395MHz)

Methodology: same RTX 3090, clock host-locked at 1395MHz (cross-process variance <0.1%).
- **A (analysis)** = one-time setup. Ours: equilibrate+MC64+METIS ND ordering+symbolic+
  gpu_mf_analyze. cuDSS: its reorder+symbolic. Single-call (analysis runs once). Amortized
  over many factor/solve calls in a Newton iteration.
- **F (factor) / S (solve)** = WARM kernel (steady-state, kernel-only, the NR-amortized
  regime where structure is reused). Ours: gpu_mf_bench / circuit_mf_test warm median.
  cuDSS: factorization/solve phase. Single-call F/S are cold-start+upload-inflated (e.g.
  onetone2 single-call F=56ms vs warm 11.2ms) -> warm is the fair kernel-vs-kernel compare.

## A — Analysis / ordering (ms, one-time; lower=better)
| matrix | ours | cuDSS | verdict |
|---|---|---|---|
| case6468rte | 33.1 | 25.8 | cuDSS 1.28x |
| case8387pegase | 43.8 | 31.4 | cuDSS 1.39x |
| case_ACTIVSg25k | 155.3 | 61.7 | cuDSS 2.52x |
| case_SyntheticUSA | 630.8 | 203.2 | cuDSS 3.10x |
| rajat27 | 67.2 | 51.9 | cuDSS 1.29x |
| memplus | 62.1 | 55.3 | cuDSS 1.12x |
| onetone2 | 241.3 | 90.6 | cuDSS 2.66x |
| rajat15 | 318.6 | 125.0 | cuDSS 2.38x |

cuDSS analysis is faster on all (1.1-3.1x) -- our METIS ND ordering dominates A. This is
one-time and AMORTIZES across the many factor/solve calls of a Newton solve.

## F — Factorization (warm kernel ms; lower=better)
| matrix | ours | cuDSS | verdict |
|---|---|---|---|
| case6468rte | 0.524 | 0.734 | **ours 1.40x** |
| case8387pegase | 0.711 | 1.338 | **ours 1.88x** |
| case_ACTIVSg25k | 1.786 | 2.225 | **ours 1.25x** |
| case_SyntheticUSA | 4.111 | 6.069 | **ours 1.48x** |
| rajat27 | 0.843 | 1.205 | **ours 1.43x** |
| memplus | 0.695 | 1.369 | **ours 1.97x** |
| onetone2 | 11.17 | 9.829 | cuDSS 1.14x |
| rajat15 | 3.288 | 4.263 | **ours 1.30x** |

**FACTOR: ours beats cuDSS on 7/8** (only onetone2 behind, 1.14x). The cy147/148 multi-
block big-front factor is the key.

## S — Solve (warm kernel ms; lower=better)
| matrix | ours | cuDSS | verdict |
|---|---|---|---|
| case6468rte | 0.544 | 0.339 | cuDSS 1.61x |
| case8387pegase | 0.689 | 0.433 | cuDSS 1.59x |
| case_ACTIVSg25k | 1.570 | 0.807 | cuDSS 1.95x |
| case_SyntheticUSA | 2.752 | 1.713 | cuDSS 1.61x |
| rajat27 | 0.968 | 0.463 | cuDSS 2.09x |
| memplus | 0.665 | 0.503 | cuDSS 1.32x |
| onetone2 | 4.521 | 1.826 | cuDSS 2.48x |
| rajat15 | 2.472 | 1.035 | cuDSS 2.39x |

**SOLVE: cuDSS faster on all (1.3-2.5x)** -- memory-bandwidth + level-serialization bound,
at the multifrontal kernel-tuning limit (all levers tested cy142-150).

## Summary
- **F: we WIN 7/8** (factor superiority, the repeated NR cost).
- **A: cuDSS wins (1.1-3.1x)** = ordering, but one-time / NR-amortized.
- **S: cuDSS wins (1.3-2.5x)** = research-grade solve scheduling to close.
- **Accuracy: ours >= cuDSS everywhere** (berr 1e-13..1e-16; refines to ~1e-16).
- Combined F+S (warm): ours beat/tie 4/8 (case6468 tie, case8387, SyntheticUSA, memplus).

## cy158 — CONSOLIDATED standing after the research-grade push (cy155-158)

The research-grade challenge (beat/match cuDSS on A/F/S, keep F) outcome:

| phase | result vs cuDSS | how |
|---|---|---|
| **F** (factor) | **beat 7/8** | multi-block big-front factor (cy147/148) + occupancy tuning (cy140-144) |
| **A** (analysis) | gap **2.5-3.1x -> 1.6-2.0x** | parallel nested dissection (cy155-156), opt-in PAR_ND, -26..-41% |
| **S** (solve) | 1.3-2.5x (kernel limit) | exhaustively tuned (cy142-154); research-grade rewrite needed for more |
| accuracy | **>= cuDSS everywhere** | berr 1e-13..1e-16 |

Parallel-ND A (production, integrated, verified post-rebuild): ACTIVSg25k 169->125ms (-26%),
SyntheticUSA 633->384ms (-39%). Kernel-only A (gpu_mf_bench): Synth 605->358 (-41%). Opt-in
(PAR_ND) so kernel benchmarks stay reproducible; production enables it. F preserved (parallel
ND fill ~= serial; gpu_test 4/4; integrated all-success).

NET: F won decisively (7/8), A gap roughly halved (parallel ND), S at the multifrontal kernel
limit. The realized research-grade breakthroughs: multi-block big-front FACTOR + parallel-ND
ANALYSIS. S remains the open multi-week item (cuDSS-internals solve rewrite).
