# 10 — Panel-resident mid kernel (STRUCTURAL): bandwidth saturation on big fronts

**Date:** 2026-06-13
**Regime:** batch fp32 factorize, RTX 3090. Default scatter path.
**Result:** structural win on the big-front cases — **SyntheticUSA (70K) −9.3% @B=64, −2.1% @B=16**; ACTIVSg25k −1.4% @B=64. No regressions (gated). Default-on.

## The real bottleneck (correcting the "underfill" framing)
Decomposing *every* factor launch at B=64 (ACTIVSg25k): **84% of factorize time is in FILLED launches (waves/SM ≥ 1)** — multi-batch is NOT underfilled. But those filled mid launches run at **8–32% occupancy and only 35–40% of peak DRAM** (the deepest big-front launches: 2–3% DRAM). The work is abundant; the *mapping* starves bandwidth.

Root cause: `factor_mid_blocked` stages the **whole front (fsz²)** into shared → big fronts are shared-limited to 1–2 blocks/SM → can't issue enough in-flight loads to hide latency → memory-bound kernel stuck far below bandwidth. This is structural, not a micro-tuning target.

## The redesign: only the panels live in shared
`factor_mid_panel<float>` (mid.cuh) stages just the L/U **panels**, leaving the contribution block (uc², the bulk) in global:
- `Lpan` = left panel rows[0,fsz)×cols[0,nc) (stride nc) — pivot block + L strip
- `Upan` = U panel rows[0,nc)×cols[nc,fsz) (stride uc)
- Shared shrinks **fsz² → nc(fsz+uc)** (~3× for nc=16). Phase 1 (row-fused panel LU) + Phase 2 (sync-free U-solve) run in shared; Phase 3 (single-pass trailing) reads Lpan/Upan from shared + the assembled CB from global and **fuses the extend-add into the parent** (atomicAdd) — so global CB traffic is the same one pass the whole-front kernel already paid at stage-in.

## Mechanism confirmed (ncu, SyntheticUSA B=64)
| | whole-front | panel |
|---|---|---|
| DRAM throughput (big-front launches) | **2–3%** … 28–32% | **55–65%** |
| registers/thread | 48 | 40 |

Memory throughput on the bandwidth-starved launches jumps ~20×. That is the −9.3%.

## Gating (default-on, no regressions)
Panel wins only when the whole-front kernel is BOTH shared-limited AND occupancy-bound. The benefit scales with how starved the whole-front kernel is = front size; medium fronts (fsz 64–96) only help at very high block counts, and at thin batch the extra CB pass isn't repaid. Gate: **fsz_cap ≥ 112 AND level_size·B ≥ 2·SMs**, fp32 scatter only (`CLS_MID_PANEL`, `CLS_MID_PANEL_MIN`). At fsz≥112 the win is clean across B=16/64 and small cases stay neutral (their fronts never hit 112). Correctness: fp32 relres matches whole-front within FMA/atomic noise; FP64 untouched (uses whole-front kernel) → 1e-13.

## Timing (B=64 `batch_factor_per_sys_ms`, panel-on vs panel-off)
| case | off | panel | Δ |
|---|---|---|---|
| case8387pegase | 0.0235 | 0.0235 | ~0 (fronts < 112) |
| case_ACTIVSg25k | 0.0921 | 0.0909 | −1.4% |
| case_SyntheticUSA | 0.383 | 0.347 | **−9.3%** |
(B=16: SyntheticUSA −2.1%, others neutral.)

## Extensions (① medium fronts, ② FP64/TF32) — all default-on, no regressions
The kernel is templated on T and the gate is two-tier: **BIG** (fsz≥112, blocks≥2·SMs) helps at moderate occupancy; **MEDIUM** (fsz≥64, blocks≥16·SMs) only under heavy occupancy pressure (the medium block gate avoids the thin-batch regression). Applied to every precision (FP64 front is 2× bytes → even more shared-starved → biggest relative gain). `CLS_MID_PANEL_MED` / `CLS_MID_PANEL_MED_BLK` tune the medium tier.

Final matrix (panel-on vs -off, B=64 `batch_factor_per_sys_ms`):
| precision | ACTIVSg25k | SyntheticUSA | SyntheticUSA B=16 |
|---|---|---|---|
| fp32 | −4.3% | −11.0% | −2.9% |
| fp64 | −7.6% | −7.8% | −3.3% |
| tf32 | −1.0% | −7.9% | — |
Medium-front capture (①) lifted ACTIVSg25k fp32 B=64 from −1.4% (big-tier only) to −4.3%. FP64/TF32 (②) win −7–8% on the big-front case; TF32 uses the panel's scalar (fp32) trailing — TC barely fires on thin-K, so the occupancy win dominates and relres is unchanged (≥ TC accuracy). Correctness: fp64 1e-13, fp32 1e-4, tf32 ~0.05 all preserved; small cases / thin batches neutral (within ±1% noise).

## TF32 tensor-core trailing variant — tried, REGRESSES (CLS_MID_PANEL_TC, default off)
Added a TF32 mma (m16n8k8) trailing that reads Lpan/Upan directly from shared (no extra staging) with the CB-subtract + fused extend in the store path. **Net regression, mechanism-explained (ncu, SyntheticUSA TF32 B=64):**
| | scalar (default) | TC |
|---|---|---|
| registers/thread | 40 | 112 |
| occupancy | **94%** | 28% |
| DRAM throughput | **73%** | 43% |
| tensor pipe | 0% | 0.9% |
| factor | 0.345 | 0.380 (+10%) |
The mma accumulators blow registers 40→112 → occupancy crashes 94%→28% → DRAM 73%→43%, while the tensor pipe fires at 0.9% (thin-K nc≤16 → the mma is mostly padding waste). The panel kernel's advantage IS occupancy/bandwidth; TC trades it away for compute the memory-bound kernel doesn't need. The scalar panel trailing already saturates (94% occ, 73% DRAM) — near-optimal. Default stays scalar; TC kept env-gated for the record. Confirms once more: on thin-K power-grid fronts TC is not the lever (docs 06/09).

## Why this is the structural lever (vs doc 09 micro-opts)
doc 09 (register/barrier) gave ~1% each by squeezing the existing mapping. This changes the **mapping**: it removes the whole-front shared residency that was the occupancy/bandwidth ceiling, so the memory-bound kernel actually saturates DRAM. The bigger the fronts, the bigger the win — SyntheticUSA (largest fronts) gains most. Remaining headroom is the same idea pushed to FP64/TF32 and to the medium fronts (a cooperative-block / tiled trailing for fsz 64–112).

## Reproduce
Default on. Disable: `CLS_MID_PANEL=0`. Threshold: `CLS_MID_PANEL_MIN=<fsz>`. Run `--precision fp32 --single-precision fp32 --batch {16,64} --repeat 20 --warmup 6 --serial-nd --metis-seed 7`.
