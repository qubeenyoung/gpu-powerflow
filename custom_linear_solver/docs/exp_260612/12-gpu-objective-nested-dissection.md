# 12 — Custom GPU-objective nested dissection (balance > fill)

> **⚠️ CORRECTION (2026-06-13, see [doc 14](14-tailored-ordering-conclusion.md)): the "win" below is NOT real.** It was measured vs a *single* METIS seed, and the candidate scheme had an escalating-UFACTOR bug that got lucky on USA (and exploded at CAND≥6). Measured **vs the best-of-k METIS envelope with robust candidates, gpu_nd ≈ METIS** (inside the seed-noise band). The critical-path objective contributes nothing (FM on=off). Treat this doc as the (flawed) first pass; the honest verdict is doc 14.


**Date:** 2026-06-13
**Regime:** B=1/16/64, fp32 + Ozaki-TC. RTX 3090.
**Result:** A custom ND that owns the recursion + objective (reusing METIS's separator primitive) and optimizes a **GPU critical-path cost (balanced splits)** instead of fill **beats METIS by 2–5% (up to −10.6% with leaf tuning) at *lower* fill**, correct to fp64 1e-13. The first ordering-side structural win — and it confirms the thesis: METIS optimizes the wrong objective.

## Why (the misalignment, now proven actionable)
METIS minimizes **fill (separator size)**; GPU wall-clock wants **balanced subtrees** (parallelism/occupancy). The proxy/best-of-k selection can't fix this (it only filters METIS's fill-optimal outputs). The fix is to own the dissection objective.

## Design — `gpu_nd` (`src/analyze/reorder/metis_nd.cpp`)
Deterministic recursion; at each subgraph: generate C candidate vertex separators via `METIS_ComputeVertexSeparator` with varied `UFACTOR` (spanning balanced↔imbalanced cuts), and keep the one minimizing
`score = sep_front_cost(|S|, tc_g) · (1 + λ·imbalance)`
- `sep_front_cost` = |S|³, **TC-discounted** for TF32/Ozaki (doc 11) so big top separators aren't over-penalized;
- `imbalance` = |n0−n1|/(n0+n1) → **λ penalizes unbalanced splits** (the GPU lever METIS lacks);
- stop at `leaf_target` → `base_nodend` (full METIS on the leaf), shaping the leaf front tier.
perm convention = `par_nd_rec` (separator-last, `perm[new_pos]=old`). Env: `CLS_GPU_ND`(on) `_CAND`(4) `_LEAF`(2000) `_LAMBDA`(0.5) `_TC_G`(1.0). Fill-gated in `build_plan_from_csr` (reject if front_total/METIS > `CLS_GPU_ND_FILL`=1.3; never triggers — gpu_nd fill is *lower*).

## Mechanism — it's BALANCE, not fill or candidate-selection (decisive)
SyntheticUSA B=1 fp32, λ decomposition:
| config | factor | vs METIS (2.092) |
|---|---|---|
| λ=0 (pick smallest separator = fill objective) | 2.200 | **+5% WORSE** |
| λ=0.5 (imbalance penalty) | 1.975 | **−5.5%** |
**Picking the smallest separator (METIS's objective) is GPU-suboptimal; penalizing imbalance is the win.** And it wins at *lower* fill (front_total 0.95–0.98× METIS) — so this is not a fill trade, it's a genuinely better tree shape for the GPU. λ must be strong enough: λ=0.3 regresses ACTIVSg25k B=1 to 0.83 (vs METIS 0.68); λ=0.5–0.7 is the robust sweet spot.

## Measured A/B (gpu_nd default vs METIS, factor per-sys ms)
| fp32 | B=1 | B=16 | B=64 |
|---|---|---|---|
| case8387pegase | −2.7% | −1.6% | −2.3% |
| case_ACTIVSg25k | −2.3% | ~0 | ~0 |
| case_SyntheticUSA | **−5.5%** (−10.6% @leaf=4000) | −2.1% | −2.6% |

- **Ozaki-TC B=1 stacks:** ACTIVSg25k METIS 0.613 → gpu_nd 0.607 → +tc_g=1.25 **0.604**; SyntheticUSA 1.712 → 1.697 → 1.695. The TC-discount adds a small extra on top of the balance win.
- **Fill:** gpu_nd front_total is 0.95–0.98× METIS — *less* fill, despite a different objective (the per-node best-of-candidates).
- **Correctness:** fp64 relres 4.7e-16 / 3.2e-13 (ordering is exact); fp32/Ozaki relres within atomic noise.
- **Leaf granularity** is a second real lever (case-dependent optimum; USA B=1 best at leaf=4000 = −10.6%).

## Cost / status
gpu_nd runs C=4 separator calls per node + builds a METIS baseline plan for the fill gate → **analyze is ~3–5× slower** (one-time; amortized over a Newton loop's many factorizations). Env-gated (`CLS_GPU_ND`, default off) — it's a validated, strictly-better-or-neutral ordering (faster + less fill + correct); flipping the default is a deployment decision (the only cost is analyze time).

## The honest headline
The bounded-ceiling pessimism was wrong on the mechanism: owning the ND objective and **balancing for parallelism instead of minimizing fill** is a real, consistent 2–5% (up to −10%) win with no fill cost. METIS's fill objective genuinely leaves GPU performance on the table, and the custom recursion captures it. This is the structurally-right ordering lever — exactly what "implement ND ourselves and optimize for our problem" was meant to find.

## Reproduce
`CLS_GPU_ND=1 [CLS_GPU_ND_LAMBDA=0.5 CLS_GPU_ND_LEAF=2000 CLS_GPU_ND_TC_G=1.25] build/custom_linear_solver_run <case> --precision {fp32,tf32} --single-precision fp64 --batch {1,16,64} --repeat 20 --warmup 6 --serial-nd --metis-seed 7`. build-ozaki for fp32-accurate TC. Analyze-time fill diff: add `--analyze-info`, grep `gpu_nd front_total`.
