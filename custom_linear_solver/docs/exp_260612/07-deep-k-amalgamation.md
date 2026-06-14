# 07 — Deep-K amalgamation (Stage 0 of the gather-favorable redesign): NEGATIVE

**Date:** 2026-06-13
**Regime:** batch fp32 factorize, B=16/64, RTX 3090 (sm_86). Cases: case8387pegase (13K), case_ACTIVSg25k (25K), case_SyntheticUSA (70K).
**Verdict:** **FAIL.** Deep-K supernode amalgamation regresses factorize at every cap, on every case — even though it *reduces* fill and *halves* level count. The premise (thicken K → cross the fp32 roofline ridge → compute-bound trailing) is empirically refuted for power-grid + METIS-ND fronts. Stages 1–2 (front-buffer reorder, phase-batched GEMM) are **not justified** — their entire payoff depended on this gate passing.

## Why this was the decisive cheap test
Per the redesign plan, the only lever that raises arithmetic intensity is thicker K (nc):
AI ≈ uc·nc/(nc+uc) ≈ **nc** when uc≫nc. Front reorder + phase-batched GEMM only cut launch/assembly overhead — they don't raise per-GEMM AI. So if thickening nc doesn't help the *existing* (already TC-capable) kernels, no amount of reorder/batching downstream can make the trailing compute-bound. The test is also cheap: a deep-K partition injected via the existing `forced_panels` param needs **zero kernel changes**, and the go/no-go is mostly decidable at analyze time.

## What was built (kept, env-gated, default OFF)
- `symbolic::deep_k_panels(n, parent, colcount, cap_nc)` — `src/analyze/symbolic/supernode.cpp`. Builds on `relaxed_panels` (chain merge) and additionally absorbs whole child **subtrees** into a parent panel, bottom-up, while (a) the merged column range stays postorder-contiguous (closest-child-first adjacency), (b) nc ≤ `cap_nc` (clamped to 32 = the `kTensorCorePivotColumnCap` + single-warp solve-substitution ceiling). Validity proven: contiguous-suffix absorption keeps member columns as the leading front rows, and a column's CB nests in its parent's structure (elimination-tree theorem) → single parent front. Re-validated by `multifrontal_symbolic`'s `asm_idx==-1` flag (always 0 in tests).
- `maybe_amalgamate(...)` — `src/analyze/pipeline.cpp`. Gated on `CLS_AMALG_K` (cap) + `CLS_AMALG_FILL` (fill-growth budget, default 1.30). Builds the partition, rejects on any nesting violation or fill over budget, and (with `--analyze-info`) prints panel/fill/nc-histogram diagnostics. When `CLS_AMALG_K` is unset the default ordering is unchanged.

## Analyze-side results (the thickening that is achievable)
`CLS_AMALG_K=32`, serial-ND seed 7:

| case | panels | levels | fill ratio | nc work-wt mean (base→amalg) | fronts reaching nc 17–32 |
|---|---|---|---|---|---|
| case8387pegase | 7406→7046 | 29→16 | **0.868** | 2.6 → 4.6 | 0 → 48 |
| case_ACTIVSg25k | 22731→21931 | — | **0.882** | 3.1 → 4.6 | 0 → 159 |

Key facts: amalgamation is **fill-negative** (0.87×) and **level-halving** (29→16) — the merges it finds are structurally "free" (nested CBs). But the thickening is **modest and confined**: the work-weighted mean nc only ~doubles to ~4.6 (still far below the ridge), and the **vast majority of panels stay nc 1–2** (6747/7046). Power-grid + ND etrees are inherently thin-supernode; cap=32 cannot broadly reach nc~32 without the merges either not existing (contiguity) or — if forced — exploding CB fill.

## Timing A/B (best-vs-best, B=64, `batch_factor_per_sys_ms`)
Monotonic regression with cap, every case. (`CLS_AMALG_FILL=2.0` to let all caps build.)

| case | base | K=8 | K=16 | K=24 | K=32 |
|---|---|---|---|---|---|
| case8387pegase | 0.02380 | 0.02531 (+6.3%) | 0.02532 | 0.02625 | 0.02692 (+13%) |
| case_ACTIVSg25k | 0.09448 | — | 0.09583 (+1.4%) | 0.09786 | 0.10190 (+7.9%) |
| case_SyntheticUSA | 0.39312 | — | 0.40238 (+2.4%) | 0.43212 | 0.47761 (+21.6%) |

Solve also regresses slightly at every cap. Accuracy (`relative_residual_l2`) is noisier and trends **worse** with thicker fronts (case_SyntheticUSA: 0.0051 → 0.013 at K=32; denser fp32 fronts accumulate more rounding). There is **no cap that improves factor** — even the gentlest (K=8/16) is net-negative.

## Root cause (why thicker-K hurts here)
Even with fill *down* and levels *halved*, factor is slower → the bottleneck is **not** fill or launch count. It is per-front kernel efficiency / occupancy:
1. The achievable thickening is too small (work-wt nc ~4.6) to change the AI regime — AI≈nc≈4.6 is still deep in the memory-bound zone, well below the fp32 ridge (~10). Tensor cores still cannot fire usefully.
2. Fewer-but-bigger fronts **reduce the parallelism the batch provides** and **push work out of the efficient warp-packed small-front tier** (many (front,batch) per warp) into the block-per-front mid/big tiers. That occupancy loss dominates any AI gain.
3. Merging heterogeneous subtrees into one level widens the per-level size spread → worse tier batching.

This is fully consistent with prior findings (doc 06: AI≈3.5 wall, TC pipe 0%, occupancy is the lever the batch already maximizes).

## Consequence for the redesign
- **Stages 1–2 are contraindicated, not merely unjustified.** Their premise was "thicker-K fronts become compute-bound; reorder + phase-batched GEMM maximize that." Stage 0 shows the thickening that is structurally available (a) is small (nc plateaus ~4.6) and (b) *already* makes the existing kernels slower. A phase-batched GEMM over nc≈4.6 fronts is still memory-bound; it cannot recover a compute-bound regime that the front structure does not admit.
- The thin-K ceiling is a **property of power-grid Jacobians under fill-reducing ND**, not of the kernels. No assembly/layout/batching change downstream removes it.

## Reproduce
```
# analyze-side (free): fill ratio + nc histogram
CLS_AMALG_K=32 build/custom_linear_solver_run <case> --precision fp32 --single-precision fp32 \
  --batch 16 --repeat 1 --warmup 0 --serial-nd --metis-seed 7 --analyze-info 2>&1 | grep amalg
# timing A/B
CLS_AMALG_K=32 CLS_AMALG_FILL=2.0 build/custom_linear_solver_run <case> --precision fp32 \
  --single-precision fp32 --batch 64 --repeat 20 --warmup 6 --serial-nd --metis-seed 7
```
Default solver (no `CLS_AMALG_K`) is unchanged and verified identical to baseline.
