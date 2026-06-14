# 14 — Tailored ordering for power-grid + GPU multifrontal: COMPLETE, and the honest verdict

**Date:** 2026-06-13. **Goal:** design an ordering/ND algorithm tailored to our power-flow Jacobian + GPU multifrontal factorize (primary B=1 / Newton). **Verdict: no ordering algorithm we built beats the METIS envelope. METIS's fill objective is empirically near-optimal for this problem.** The real gains in this project were kernel-side (panel −11% B=64, Ozaki-TC −17% B=1), not ordering.

## What was built and tested (all env-gated, default off)
A custom ND owning recursion + objective + stopping + a genuinely-new separator primitive:
- `gpu_nd_rec` / `gpu_sep_refine` (metis_nd.cpp) — recursion + per-node candidate selection + FM separator refinement toward a GPU critical-path objective (TC-discounted separator + balance).
- `gpu_nd_weighted_*` (metis_nd.cpp) — **electrical-weighted bisection**: |J_ij|-weighted graph → `METIS_PartGraphRecursive` edge-weighted cut (cuts weak tie-lines — the one thing METIS_NodeND structurally cannot do, no adjwgt) → derived vertex separator → FM. Wired in pipeline.cpp (`CLS_GPU_ND` / `CLS_GPU_ND_EW`), fill-gated.

## Stage 0 — GPU-objective recursion: REFUTED (and a prior over-claim corrected)
Measured **vs the best-of-k METIS envelope** (the doc-12 mistake was a *single-seed* baseline):
- The critical-path **objective contributes nothing**: FM=0 = FM=1; λ barely matters.
- The apparent "−2–5% win" (doc 12) was **two artifacts**: (1) single-seed baseline, (2) an escalating-UFACTOR candidate schedule that got lucky on USA (−8%) and **exploded at CAND≥6 (+20–130%)**. With robust (seed-varied, capped-imbalance) candidates, **gpu_nd ≈ METIS** (USA 2.06–2.15 vs 2.10; 8387 0.31 vs 0.32) — inside the noise band.
- → The "own-recursion + METIS-separator + GPU-objective" direction does **not** beat METIS.

## Stage 2 — electrical-weighted bisection: REFUTED (worse, mechanism-explained)
The genuinely-new, power-grid-tailored lever. **Worse than METIS at every depth** (B=1 fp32, best-of-4-seeds):
| | METIS | EW d=1 | d=2 | d=3 |
|---|---|---|---|---|
| case8387 | 0.317 | +16% | +7% | +17% |
| SyntheticUSA | 2.106 | +9% | +56% | +73% |
Cause: cutting **electrically weak** ties (small |Y_ij|/|J_ij|) minimizes coupling, **not** separator size or balance → larger/raggeder separators → **+28% fill** (case8387 1.21×, USA 1.28×) → bigger fronts → slower B=1. The intuition ("areas connect via thin tie-lines → good separators") is false here: administrative/electrical weakness ≠ graph-small-and-balanced. Correct (fp64 1e-11..1e-16), just a loss.

## Why METIS is hard to beat here (the honest synthesis)
- **fill ≈ work**, and for the multifrontal front sizes, the fill-minimal separators are also the **small, balanced** separators that minimize the B=1 critical-path front cost. METIS optimizes exactly that. The B=1 "different objective" (tree height) is, for near-planar power grids, *also* served by small balanced separators → little divergence from fill.
- The total B=1 ordering ceiling is measured-small (docs 04: 13K 1.19×, 25K 1.05×, 70K 1.08×), and METIS already sits near it.
- The literature agrees: balanced-ND/etree-height work gives bounded gains with a fill cost; cuDSS ships "customized ND based on METIS" (no separator-objective change). Nobody reports a large GPU-factor speedup from re-objecting ND.

## The one real, actionable finding → IMPLEMENTED ([doc 15](15-measured-best-of-k-ordering.md))
**The production best-of-k selection is broken:** `CLS_ORDER_K=16` (tail_cube proxy) picks a *worse* ordering than a single good seed (case8387: K16 0.330 vs single-seed 0.316; the proxy mis-ranks, confirmed by the doc-12 seed-7-vs-13 case). Fixing the *selection* — measured pick over a few seeds, or just a validated fixed seed instead of the proxy — recovers a few % for free. This, not a new ND objective, is the ordering lever worth taking.

**Done in [doc 15](15-measured-best-of-k-ordering.md):** implemented `CLS_ORDER_MEASURE_K` — a measured best-of-k that *times a real factorize* per candidate seed and keeps the fastest (proxy can't mis-rank a direct measurement). Confirmed the proxy is **anti-informative** (its #1 pick is 8% slower than the trivial default; fill is seed-invariant) and the measured selector recovers **6–13% at B=1** (8387 −8%, 25K −11%, USA −8.4%) plus determinism. Cost is ND-build-dominated (= the broken proxy's cost), so it's a near-free drop-in replacement, worth it under factorize reuse (time-series / N-1).

## Status
All custom-ND code kept env-gated (`CLS_GPU_ND*`, default off; production = METIS, unchanged). Doc 12's headline is corrected here (the win was artifact). Net: **ordering is bounded and METIS-near-optimal for this problem; the tailored-ND and electrical-bisection hypotheses are empirically refuted; the actionable win is fixing the broken proxy selection.**
