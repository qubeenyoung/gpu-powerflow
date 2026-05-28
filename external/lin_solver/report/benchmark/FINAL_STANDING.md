# mysolver vs cuDSS — FINAL STANDING (cycle 338, 2026-05-28)

Single consolidated picture (supersedes everything below). gpu_test 4/4, accuracy >= cuDSS
throughout. ncu (cap'd restart) + the cy335/336 partitioned-inverse + cy338 adaptive cap broke
the long-standing S plateau.

## Per-phase vs cuDSS (warm kernel ms, multifrontal)
| phase | result | how / status |
|---|---|---|
| **F** factorization | **BEATS cuDSS 8/8 (ALL matrices)** | case6468 0.55/case8387 0.74/ACTIVSg25k 1.66/SyntheticUSA 3.74 vs 0.61/1.11/1.88/5.08; circuits rajat27 0.77/memplus 0.66/onetone2 7.91/rajat15 2.79 vs 1.00/1.15/8.33/3.50. (F rose from cy335/336 pivot inversion + cy338 bigger amalgamation cap, but still wins everywhere — user OK'd F dropping as long as it beats cuDSS.) |
| **S** solve | **~cuDSS broadly: BEATS on 2, ~tied on 3, 1.17-1.34x on 3** | cy335/336 partitioned-inverse (invert each front's nc×nc pivot block at factor → solve triangular solves become parallel GEMVs; [[partitioned-inverse-solve]]) + cy338 size-adaptive amalgamation cap (shallower panel-etree → fewer serialized solve levels; onetone2 plev 110→79). S −36..−45% over the session. vs cuDSS S: memplus 0.82x & rajat27 0.93x **WIN**; SyntheticUSA 1.02x / case6468 1.04x / case8387 1.05x ~tied; rajat15 1.17x, ACTIVSg25k 1.20x, onetone2 1.34x. Residual = inherent deep-spine latency (research-grade). |
| **F+S** (cuDSS's NR use case) | **BEATS cuDSS 7/8** | case6468 0.94x, case8387 0.76x, ACTIVSg25k 0.97x, SyntheticUSA 0.80x, rajat27 0.76x, memplus 0.68x, rajat15 0.86x all WIN; onetone2 1.01x the lone (close) holdout — F-limited (cap16 would break its F). |
| **A** analysis | **size-dependent: WIN small, 1.2-2.3x behind large** | METIS-ND-ordering-bound; research-grade GPU-ordering to close. De-prioritized by user in favor of S. |
| accuracy | **>= cuDSS everywhere** | berr 5e-16..5e-13 (onetone2 raw 2.5e-7 -> refinement). |

## Headline
**F beats cuDSS 8/8, F+S beats cuDSS 7/8, S ~cuDSS broadly (2 wins, 3 ties).** The partitioned-inverse (cy335/336) + size-adaptive cap (cy338) closed the long-standing S gap. The user's "match/beat cuDSS broadly" goal is strongly met. Residual: onetone2 F+S 1.01x (F-limited) and the large-A GPU ordering — both research-grade.

---
# (superseded) FINAL STANDING (cycle 167, after the research-grade A/S rewrite)

Single consolidated picture (supersedes scattered docs). HOST-LOCKED 1395MHz, gpu_test 4/4,
all 14 matrices on GPU (gate-safe, CPU fallback never triggered in validation).

## Per-phase vs cuDSS
| phase | result | how / status |
|---|---|---|
| **F** factorization | **BEATS cuDSS 7/8** (warm, serial-measured) | multi-block big-front factor (cy147/148) + count-guard occupancy (cy140-144). onetone2 the only holdout (1.14x; sometimes 8/8 under parND). MUST stay -- the F-regression gate. |
| **A** analysis | gap **2.5-3.1x -> 1.6-2.0x** | parallel nested dissection (cy155), SHIPPED to production default (cy164): -40% kernel / -25-39% integrated. ACTIVSg25k 169->126ms, SyntheticUSA 637->389ms. |
| **S** solve | 1.3-2.5x behind | at the multifrontal DEPENDENCY limit (etree-depth critical path + read/compute). Every lever tested-negative: block size, multi-block, fusion, FP32, CB-tile, gather-forward, coalescing -- most via cheap proxies. |
| accuracy | **>= cuDSS everywhere** | berr 1e-13..1e-16 (refined). |

## What the research-grade rewrite delivered (cy162-166)
- **A: realized -40%** (parallel ND, F kept competitive). The cuDSS A-gap is roughly halved.
- **S: rigorously bounded** -- multifrontal-dependency-bound; cuDSS's edge is tuning of the
  same approach + a shallower-etree ordering, not a droppable algorithm.

## Open items (need a decision / are research-grade)
1. **Exact determinism of parND** (so the A win is a clean deterministic default): BLOCKED by
   clean means -- METIS's RNG is a thread-unsafe glibc rand() global; AMD-leaf alternative
   fails on fill (+30-72%); rand-override is versioned-symbol-fragile (~2% residual); no
   static libmetis.a to link cleanly. Only the GKlib __thread RNG rebuild (high-risk system
   surgery, uncertain it fully resolves) remains -> awaiting explicit user buy-in on the risk.
   (parND is correct + run-fixed in NR; non-determinism is run-to-run only.)
2. **A beating cuDSS** (not just halving the gap): needs a GPU-accelerated ordering -- multi-week.
3. **S beating cuDSS**: needs matching cuDSS's solve internals -- structural, multi-week, uncertain.

## Bottom line
Factor decisively won (7/8), analysis gap halved (-40%, shipped), accuracy superior, solve at
its honest structural limit. The realizable research-grade wins are banked; the rest is
research-grade or risk-gated. Strong, well-documented position; awaiting user direction.
