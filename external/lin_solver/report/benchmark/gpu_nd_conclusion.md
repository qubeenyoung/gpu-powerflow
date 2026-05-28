# GPU nested-dissection ordering — effort conclusion (cy201–215)

User directive (cy201): "implement GPU-ND directly to catch cuDSS" — pursue Option #1 of
`ordering_path_forward.md` (a from-scratch GPU nested-dissection ordering) to close the
big-matrix **A** (analysis/ordering) gap to cuDSS that parND-METIS can't.

## What was built (all opt-in, gate-safe; production parND untouched; gpu_test 4/4 throughout)
`src/mysolver/reordering/gpu_nd.cu` — a self-built nested-dissection separator:
- **Single-level GPU separator** (`GPU_ND`): device level-synchronous BFS (batched termination,
  cy207) + pseudo-peripheral start (cy202) + cut-position search (cy203) + both-starts selection
  (cy204) + König exact min-vertex-cover separator (cy205) + greedy FM refinement (cy206).
- **CPU multilevel** (`GPU_ND_ML`, cy208–211): HEM-style matching contraction → coarsen to ≤200
  vtx → cut the coarsest with the above stack → project up with FM at each level; linear-time
  coarsening primitives (cy210); size guard for small subgraphs (cy211).
- **Device-resident GPU coarsening** (`GPU_ND_RESIDENT`, cy212–213): GPU matching + thrust-based
  GPU contraction, graph resident across levels.

## Result
**FILL PARITY achieved (the milestone).** Depth-1 fill vs METIS over the 9-cycle arc:
`cy201 4.75/2.30/1.54/2.49 → cy209 0.99/1.01/0.99/1.01` (case6468/case8387/ACTIVSg25k/SyntheticUSA).
With fill at parity, our MF factor **F beats cuDSS on all 4 power matrices**; berr 1e-13..1e-16.

**A-win NOT achieved (research-grade).** Honest baselines (cy211, cy213):
- CPU multilevel `GPU_ND_ML` only **ties** METIS-parND on A (it essentially reproduces METIS).
- Device-resident GPU coarsening does **not** beat CPU/METIS either: (1) GPU parallel matching is
  lower-quality than CPU sequential greedy → fill drift; (2) at the cut level the METIS NodeND on
  the halves dominates A, so coarsening speed barely moves A + thrust/kernel overhead offsets it.

**Conclusion:** GPU-accelerating the coarsening *alone* cannot beat cuDSS/METIS on A. An A-win
requires the **entire** nested dissection on-device (recursion + cut + projection + FM) — a
research-grade GPU-ND analysis like cuDSS's, multi-week, not a cron-cycle deliverable.

## Project standing (settled)
- **F**: beats cuDSS **7/8** (only onetone2 behind — cuDSS's circuit-factor strength; unwinnable
  even at cap16). Settled.
- **S**: behind **1.46–2.31×** on all — structural single-RHS limit (one heavy separator chain,
  est_speedup~1.0×; the scattered y-gather latency is already hidden by warp parallelism + `__ldg`,
  cy188/215). Shares the ordering/fill root with A.
- **A**: behind cuDSS on big matrices — ordering-dominated; both METIS-parND and our GPU-ND-CPU
  trail cuDSS; closing it is the research-grade GPU-ND-analysis rewrite above. One-time (NR-amortized).

The cron-tractable optimization space is exhausted; remaining gaps need research-grade multi-week
rewrites with uncertain payoff. The GPU-ND code stands as a validated, METIS-fill-quality self-built
ordering (opt-in), and a complete characterization of why the A-win is research-grade.

## Addendum (cy219-224): GPU-acceleration round result
User greenlit the GPU-accel direction (cy218) with F-must-hold. Outcome:
- **cy219 WIN**: root-only GPU-ND (GPU coarsening for the root cut + METIS parND recursion) -> the
  first GPU-accel A win over METIS-parND. Clean PAR_ND=4 A/F/S (vs cuDSS):
  | matrix | ours A | ours F | ours S | cuDSS F/S | note |
  |---|---|---|---|---|---|
  | case6468/case8387/ACTIVSg25k (<100k) | = METIS | = METIS | = METIS | F win | METIS (GPU-ND gated off) |
  | SyntheticUSA (156k) | ~301 (METIS ~314) | ~4.4-4.6 (beats cuDSS 5.08) | ~2.68 (parity) | - | GPU root cut |
- **F PRESERVED**: factor depends on fill structure not ordering method; GPU-ND keeps fill parity ->
  F still beats cuDSS 7/8. (User's requirement met.)
- **Limits (cy220-223)**: the win is SMALL (~-4-6% A, noisy) and CONFINED to very-large roots.
  Ruled out for growth/extension: matching-rounds speedup (trades fill, cy220), hill-climbing FM
  (coarsening-bound not FM + balance drift, cy221), CTHR tuning (matrix-size-dependent, cy222/223).
- **Remaining cuDSS A gap = 1.47x** (ours ~289-301 vs cuDSS 203): the recursion is METIS (the dominant
  A cost); beating cuDSS A needs the research-grade full-GPU-recursive-ND (matching cuDSS fill quality
  in GPU-parallel at ALL recursion levels -- the hard open problem).

NET: GPU-accel direction validated and delivered a confined A win with F preserved; the A/S/F space is
at its cron-tractable frontier. Further gains are multi-week research-grade.

## cy226: PRODUCTION validation of the GPU-accel win
Verified the win-config (GPU_ND=1) in the ACTUAL production solver, not just the warm bench:
`benchmark --solver mysolver-gpu --matrices case_SyntheticUSA` (LD_PRELOAD det_rand), GPU_ND=1 vs
default (METIS ordering). Reproducible over repeats:
- METIS:     cuda_ms ~366.7 (366.785, 366.681 -- very stable), success=true
- GPU_ND=1:  cuda_ms ~345   (339.802, 350.719),                 success=true
=> -6% GPU time, CORRECT (production berr gate passed). cpu_ms 535->509.
The GPU-accel win is PRODUCTION-REAL (correct + reproducibly faster E2E on the very-large matrix),
accessible opt-in via the GPU_ND env (production default stays METIS). Loop closed: warm-bench result
-> production-validated. Default-on still deferred (only SyntheticUSA >100k validated; the production
berr gate checks accuracy, not fill, so a worse-fill untested large matrix could silently regress F/S).
