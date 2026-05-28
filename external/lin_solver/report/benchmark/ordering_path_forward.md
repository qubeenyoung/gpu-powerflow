# Path forward to fully match cuDSS on A/S/F (cy197 decision-support)

## Where we stand (28-cycle campaign cy169-196, all gate-safe, gpu_test 4/4)
| metric | status vs cuDSS |
|---|---|
| **F** (factorize) | **7/8 BEAT** (case6468/8387/ACTIV/Synth/rajat27/memplus/rajat15); onetone2 the holdout |
| **A** (analyze) | **small matrices MATCHED** (case6468 ~26.7 vs 25.8, case8387 ~31.6 vs 31.4); big matrices behind (Synth ~1.9x, ACTIV ~1.8x); rajat27 beats |
| **S** (solve) | behind **1.46-2.31x** (was 1.5-2.5x; cy188-189 memory-access wins narrowed it -5..-9.4%) |
| accuracy | **>= cuDSS everywhere** (berr 1e-13..1e-16) |

## All remaining gaps converge on ONE lever: the ORDERING
- big-matrix A: dominated by parND-METIS ordering (Synth 148ms, onetone2 88.7ms ~= cuDSS's WHOLE analysis 99.8).
- onetone2 A: same (its parND ordering ~= cuDSS whole-A); and onetone2's CPU-fallback / F gap is downstream of it.
- S: latency-bound by the **etree depth** (cy171); a shallower-etree ordering = fewer serial solve levels = less latency.
- parND is already at its tractable optimum: depth d=4 (cy187), per-size base threshold (cy178/184), adj_build + symbolic phases parallelized (cy177/181/185/186). No further tractable ordering speedup.

## Why this is NOT deliverable in incremental cron cycles
A GPU nested-dissection needs coarsening + separator + **refinement (FM/KL)**. Without *complete* refinement, the
ordering has WORSE fill -> regresses the working parND's F and S. So it is **all-or-nothing (multi-month)**: any
partial increment is counterproductive (breaks the strong standing). The cron loop (per-cycle, gate-safe) cannot
build it without regressing what works. This is why cy190-196 held rather than churn.

## Options for the deliberate investment (the decision)
1. **GPU nested-dissection** (from scratch): closes big-A + onetone2-A + S-via-depth simultaneously. Highest payoff,
   highest risk/effort (METIS is decades-tuned; matching its fill quality on GPU is research-grade, multi-month).
2. **ParMETIS** (MPI parallel ND, a mature library): proven quality, parallel ordering. Infra-heavy (adds MPI to a
   non-MPI build) + uncertain gain over our 12-thread parND, but it's integration (not research). Lower risk than #1.
3. **Accept the strong standing**: F 7/8, small-matrix A matched, S narrowed, accuracy superior -- a strong
   research-grade result. The unmatched gaps are A (one-time, NR-amortized) on big matrices + S (1.5-2.3x).

## Recommendation
The cron-incremental optimization space is comprehensively captured. A genuine further leap requires a deliberate
ordering investment (#1 or #2) in a different working mode than incremental cron cycles. Pending that decision, the
loop holds the strong standing and implements any fresh tractable idea the instant it surfaces (as cy188-189 did).
