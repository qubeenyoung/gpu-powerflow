# 08 — Gather-favorable front buffer + phase-batched assembly: NEGATIVE (fair test)

**Date:** 2026-06-13
**Regime:** batch fp32 factorize, B=64, RTX 3090. Cases: case8387pegase (13K), case_ACTIVSg25k (25K), case_SyntheticUSA (70K).
**Question (user):** the prior fused-gather variants reused scatter's layout, so the A/B was unfair. Build the front buffer to *favor* gather — front-buffer reorder + **phase-batched** assembly (assembly hoisted to its own kernel) — so gather competes on its own terms.
**Verdict:** **FAIL, but now fairly.** Phase-batching the gather makes it *worse*, not better. Scatter's advantage over gather is **structural**, and phase-separation only adds a global round-trip. All four gather variants lose to scatter.

## Setup: localizing the gap first (so the redesign targets the real cost)
`CLS_SKIP_ASSEMBLE` decomposition, case_ACTIVSg25k B=64 (`batch_factor_per_sys_ms`):
- full scatter = 0.0941; no-assembly floor = 0.0774; **scatter assembly = 0.0167** (memset 0.0061 + scatter 0.0100).
- The no-assembly floor is **~18% below scatter** → if gather's assembly were near-free, gather would *win* by ~18%. Real upside exists; the bar is "assemble for < 0.0167".

ncu (factor_mid_blocked, scatter vs fused gather): **identical 60 reg/thread, 8 blocks/SM, same ~46–49% occupancy** — the fused gather does NOT lose to register pressure. It loses because the memory-bound gather runs *inside* the factor kernel at the factor's low occupancy: the gather-variant kernel is ~2× the scatter-variant duration (23.97 vs 12.99 µs). That pointed to phase-batching (hoist assembly to a lean high-occupancy kernel) as the fix.

## What was built (kept, env-gated; default `scatter` unchanged & verified)
`CLS_ASM_MODE` gains two modes (atomic + output-centric):
- `gather_pb` / `gather_pb_oc` — **phase-batched**: a separate `assemble_level_gather` kernel (256 threads, lean) gathers every front in a level into the **global** arena (matrix + pull children CBs from already-factored deeper levels via the parent-grouped CB buffer); the factor kernels then stage in clean (no in-kernel gather), and write the CB back. Injected per level in the leaf→root walk (`schedule.cuh issue_assemble_gather`), so children CBs are ready. One whole-arena `cudaMemsetAsync` pre-zeros (streaming), assemble writes only occupied positions.
- New code: `assemble_level_gather` + `make_gather_args` (front_ops.cuh), `phase_batched` flag (GatherArgs + State), stage-in/skip branches in small/mid/big kernels, driver wiring (factorize.cu), schedule injection (schedule.cuh). Correctness: `gather_pb` relres matches scatter within fp32 FMA-reorder noise across case30→USA.

## Timing A/B (B=64, `batch_factor_per_sys_ms`, scatter = 1.00×)
| case | scatter | gather (fused) | gather_oc (fused) | **gather_pb** | gather_pb_oc |
|---|---|---|---|---|---|
| case8387pegase | 0.0238 | 0.0272 (+14%) | 0.0269 | 0.0477 (+100%) | 0.0431 |
| case_ACTIVSg25k | 0.0942 | 0.1094 (+16%) | 0.1090 | 0.1694 (+80%) | 0.1530 |
| case_SyntheticUSA | 0.3930 | 0.4864 (+24%) | 0.5107 | 0.6983 (+78%) | 0.6683 |

**Fused gather is the best gather variant (−15%); phase-batching is far worse (−78–100%).**

## Why phase-batching backfires (the structural reason)
ncu confirms `assemble_level_gather` *is* lean + high-occupancy (34 reg/thread, 57–67% warps active vs the factor's 25–49%) — the design worked. But it's **bandwidth-bound**, and phase-separation forces a **global round-trip the fused gather avoided**:
- Fused gather assembles in **shared** → no global write of the assembled input; loses only the in-kernel-occupancy tax.
- Phase-batched must **materialize the assembled front in global** (so the separate factor kernel can read it) → +1 full write +1 full read of the dense fsz² arena (which is ~80% structural zeros), plus the children-CB extend-add becomes a separate **global atomic** pass instead of being fused into the factor.

Scatter pays the *same* memset + stage-in round-trip, but its "assemble" is uniquely cheap: streaming `cudaMemsetAsync`, an **atomic-free** unique scatter (`a_pos_unique=true` → direct `F[a_pos[q]]=v`, no atomics), and the extend-add **fused into the factor** (overlapped). Gather's pulls are indirected (`o2c`) and atomic, and phase-separation un-fuses the extend-add. A front-buffer reorder cannot help: the round-trip is bandwidth (volume), already coalesced; reordering changes access pattern, not volume.

## Conclusion
This is the fair comparison the redesign asked for: gather was given its own phase-batched, high-occupancy assemble pre-pass on a pre-zeroed arena. It still loses, and loses *more* than the fused gather, because:
1. scatter's assembly is structurally cheap (streaming memset + atomic-free unique scatter + factor-fused extend-add), and
2. any phase-separation of gather requires a global round-trip of the dense (80%-zero) front that the fused-shared path avoids.

The best assembly remains **scatter** (default). The best *gather* is the fused-shared variant at −15%. No gather layout/phasing closes that gap, because the gap is not occupancy (ncu: equal) — it is scatter's cheaper assembly primitives. Phase-batched GEMM for the *compute* (trailing) is moot here: the kernel is memory-bound (doc 06), not GEMM-bound.

## Reproduce
```
CLS_ASM_MODE={scatter|gather|gather_oc|gather_pb|gather_pb_oc} \
  build/custom_linear_solver_run <case> --precision fp32 --single-precision fp32 \
  --batch 64 --repeat 20 --warmup 6 --serial-nd --metis-seed 7
# scatter assembly decomposition:
CLS_SKIP_ASSEMBLE={1 none|2 scatter-only|3 memset-only} ... (numerically WRONG, timing only)
```
Default (`scatter`) is unchanged and verified.
