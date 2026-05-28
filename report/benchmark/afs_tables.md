# A / F / S vs cuDSS — current standing (cycle 174, HOST-LOCKED 1395MHz, RTX 3090)

gpu_test 4/4; all 14 matrices on GPU (gate-safe). **F/S are at the PRODUCTION cap=8** (cy169
corrected a benchmark bug where circuit_mf_test ran cap=16, making the cy168 circuit numbers
non-production). A = production (parallel ND default, deterministic under
`LD_PRELOAD=libdet_rand.so`). F/S = warm kernel (NR-amortized); ours + cuDSS read from the
same run (consistent). cuDSS via benchmark --solver cudss-gpu.

## A — Analysis / ordering (ms; lower=better) — production parND
| matrix | ours (parND) | cuDSS | verdict |
|---|---|---|---|
| case6468rte | 32.9 | 25.8 | cuDSS 1.28x |
| case8387pegase | 43.7 | 31.4 | cuDSS 1.39x |
| case_ACTIVSg25k | 110.7 | 61.7 | cuDSS 1.79x |
| case_SyntheticUSA | 385.2 | 203.2 | cuDSS 1.90x |
| rajat27 | 49.4 | 51.9 | **ours 1.05x** |
| memplus | 61.1 | 55.3 | cuDSS 1.10x |
| onetone2 | 185.0 | 90.6 | cuDSS 2.04x |
| rajat15 | 226.4 | 125.0 | cuDSS 1.81x |

Parallel ND (cy155) closed the gap from 2.5-3.1x to ~1.1-2.0x (rajat27 beats); deterministic
via LD_PRELOAD (cy168). Still behind on the large (cuDSS's ordering is faster). A is one-time
(NR-amortized). Further gains need GPU-accelerated ordering (multi-week research).

## F — Factorization (warm kernel ms; lower=better) — cap8 production
| matrix | ours | cuDSS | verdict |
|---|---|---|---|
| case6468rte | 0.524 | 0.612 | **ours 1.17x** |
| case8387pegase | 0.710 | 1.113 | **ours 1.57x** |
| case_ACTIVSg25k | 1.788 | 1.876 | **ours 1.05x** |
| case_SyntheticUSA | 4.111 | 5.081 | **ours 1.24x** |
| rajat27 | 0.650 | 1.002 | **ours 1.54x** |
| memplus | 0.763 | 1.146 | **ours 1.50x** |
| onetone2 | 12.95 | 8.325 | cuDSS 1.56x |
| rajat15 | 3.130 | 3.495 | **ours 1.12x** |

**F: ours beats cuDSS 7/8.** Only onetone2 behind (1.56x at production cap8; it improves to
~1.34x at cap16 but cap16 regresses other matrices' F/S and has no safe per-matrix selector —
cy169). The win drivers: multi-block big-front factor (cy147), count-guard occupancy (cy143/4),
adaptive per-level block sizes.

## S — Solve (warm kernel ms; lower=better) — cap8 production
| matrix | ours | cuDSS | verdict |
|---|---|---|---|
(updated cy193 — reflects the cy188 __ldg + cy189 forward-pivot-staging memory-access wins, -5..-9.4% S)
| matrix | ours | cuDSS | verdict |
|---|---|---|---|
| case6468rte | 0.516 | 0.286 | cuDSS 1.80x |
| case8387pegase | 0.650 | 0.363 | cuDSS 1.79x |
| case_ACTIVSg25k | 1.450 | 0.674 | cuDSS 2.15x |
| case_SyntheticUSA | 2.540 | 1.443 | cuDSS 1.76x |
| rajat27 | 0.624 | 0.393 | cuDSS 1.59x |
| memplus | 0.615 | 0.421 | cuDSS 1.46x |
| onetone2 | 3.528 | 1.528 | cuDSS 2.31x |
| rajat15 | 1.778 | 0.871 | cuDSS 2.04x |

**S: behind 1.46-2.31x on all** (was 1.5-2.5x; cy188-189 memory-access wins narrowed it ~7-9%).
Accuracy >= cuDSS everywhere (berr 1e-13..1e-16; onetone2 2.5e-7, ill-conditioned, still < cuDSS).

### S — comprehensive structural conclusion (cy169-174, all measured)
The solve is **latency/dependency-bound, NOT fill/bandwidth-bound**: it runs at only ~14% of
the RTX3090's ~936 GB/s peak (cy171) — the GPU is ~85% idle waiting on the serial etree levels.
Every in-structure lever was tested and ruled out for single-RHS:
- **cap/amalgamation** (cy169): cap8 Pareto-optimal; bigger cap = fewer levels but MORE padded work.
- **FP32 solve** (cy170): only -1.4..-8.8% raw, and berr fails the cuDSS-level bar (1e-4..1e-8;
  onetone2 blows up to 0.228); refinement to recover negates the saving.
- **multi-block backward** (cy154): solve per-front work is O(fsz·nc), no fsz² term -> marginal.
- **concurrency / multi-subtree** (cy172-173): the GPU has ~3.6x throughput headroom, but it's
  INTER-solve (batched/multi-RHS); a single solve's etree is dominated by one heavy separator
  chain (97-100% of work in one subtree) -> a finer/multi-subtree schedule gives ~1.0x. The
  level-set already exploits the (leaf) parallelism a single solve has.
- **big-front block size** (cy174): more warps don't hide the scattered-gather latency -> flat.

**BUT the structural sweep missed the MEMORY-ACCESS class (cy188-190, 193) -- these gave the real
S wins:** cy188 __ldg the backward scattered gather (read-only cache, -0.7%); **cy189 stage the
forward-solve pivots in shared (-4..-9%, avoids the fr[k]->y[] double-indirection per CB row) -- the
biggest S win**; cy190 back-solve staging / cy192 F-read coalescing / cy193 #pragma unroll all
neutral-or-negative (reverted). Net S -5..-9.4% from baseline; gap 1.5-2.5x -> 1.46-2.31x.

So single-RHS S is bound by (1) the sequential separator chain (critical path) + (2) scattered-
gather latency. cuDSS's single-solve edge is kernel micro-efficiency (thoroughly explored), not
a missing parallel structure. A lower-fill ordering (GPU-ordering research) would help A a lot
but S little (S isn't bandwidth-bound).

## Summary
**F 7/8 win; A −40% + deterministic (1.05-2.0x, rajat27 beats); S behind 1.5-2.5x but rigorously
characterized as structurally bound; accuracy ≥ cuDSS everywhere.** Remaining gains require
multi-week research (GPU-accelerated ordering for A; cuDSS-level kernel micro-tuning for S) with
uncertain payoff. Open fork awaiting direction.
