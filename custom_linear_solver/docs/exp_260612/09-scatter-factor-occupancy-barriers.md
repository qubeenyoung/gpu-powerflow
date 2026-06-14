# 09 — Scatter factorize: register + barrier reductions (default-on, ~2–3%)

**Date:** 2026-06-13
**Regime:** batch fp32 factorize, RTX 3090. Default scatter path (gather concluded, doc 08).
**Result:** robust, no-regression, **default-on** speedups: case_ACTIVSg25k −2.7%, case_SyntheticUSA −2.5%, case8387pegase −1.3% (B=64). B=1 and small cases unchanged; FP64 relres still 1e-13/1e-16; fp32 relres preserved.

## Bottleneck (ncu, scatter factor_mid_blocked)
- **Not bandwidth** (l1tex ~6–13%, DRAM low) and **not compute** (TC pipe ~0%, fp32 has no TC).
- It is **latency + barrier bound at low occupancy**: long_scoreboard 18–25%, barrier (`__syncthreads`) 12–17%.
- Occupancy is co-limited by **registers** (60/thread → 8 blocks/SM for small-mid) and **shared memory** (whole front in shared → 2–4 blocks/SM for big-mid); achieved 18–54%. Many deep levels are also under-filled (waves/SM < 1) — latency-bound regardless of occupancy.

## Changes (all default-on)
1. **Lean kernels — compile gather out of the hot path** (`CLS_FACTOR_GATHER`, default OFF). The concluded gather A/B code was compiled *into* the factor kernels via `if (ga.active)`, inflating their register count. Removing it: `factor_mid_blocked<float,false>` **60→51 regs**, `factor_big<float>` **48→40 regs** (small unchanged at 48). Lifts the register-limited mid launches 8→10 blocks/SM. Gather modes still reproducible via `-DCLS_FACTOR_GATHER`.
2. **Sync-free U-panel solve** (`u_panel_solve_fewsync`, front_ops.cuh). Each thread owns one U column and forward-substitutes all rows in-thread, collapsing the `nc` Phase-2 block barriers to **one**. Same column parallelism and arithmetic (bit-identical). Default in mid/big/tiled.
3. **Conditional row-fused panel LU** (lu_panel_factor). Row-fused pays 1 barrier/pivot vs the two-phase form's 2. Extended the cutoff from nc≤12 to **nc≤16 on short fronts (fsz≤96)** — covers the panel-width=16 fronts of the large (n≥16K) cases. Gated on fsz because a *blanket* nc≤16 cutoff regressed SyntheticUSA +2.6% (row-fused serializes the per-thread inner panel loop, costly on tall fronts); the fsz≤96 gate keeps the ACTIVSg25k win without it.

## Timing (B=64 `batch_factor_per_sys_ms`, default)
| case | before | after | Δ |
|---|---|---|---|
| case8387pegase | 0.0238 | 0.0235 | −1.3% |
| case_ACTIVSg25k | 0.0945 | 0.0919 | −2.7% |
| case_SyntheticUSA | 0.393 | 0.383 | −2.5% |

## What did NOT help (measured, rejected)
- **mid/big boundary** (`CLS_MID_FSZ_MAX` routing big-mid → global big tier): worse (+48% at 64). Shared-residency beats the big tier's global trailing traffic even at 2–4 blocks/SM. Boundary already well-tuned.
- **blocked-fp32 LU** (`CLS_MID_BLOCKED_FP32`): neutral.
- **Tensor cores**: N/A to the fp32 path (UseTC=false); and the kernel is memory-latency-bound, not GEMM-bound, so TC cannot help here even on TF32.

## Honest ceiling
The fp32 factorize is near its efficiency limit for this structure: memory-latency + barrier + under-fill bound. The tractable micro-levers (register, barrier) each yield ~1%, stacking to ~2–3%. Larger gains would require attacking the deep-level under-fill (structural) or changing the algorithm, not kernel micro-tuning.

## Reproduce
Default build = all three changes on. `cmake --build build -j`; run `--precision fp32 --single-precision fp32 --batch {16,64} --repeat 20 --warmup 6 --serial-nd --metis-seed 7`. Gather A/B: rebuild `-DCLS_FACTOR_GATHER`.
