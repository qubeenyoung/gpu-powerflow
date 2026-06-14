# 11 — B=1: tensor cores ARE the lever (regime reversal), via Ozaki-TC at fp32 accuracy

**Date:** 2026-06-13
**Regime:** single-system factorize (B=1), RTX 3090. Complements doc 10 (B=64 panel-resident win).
**Result:** B=1 is the *opposite* regime from B=64. The panel-resident win does NOT transfer. The B=1 lever is the **TC trailing**, and **Ozaki-TC** captures it at full fp32 accuracy: **ACTIVSg25k −8.5%, SyntheticUSA −17.5%** (B=1 factor).

## B=1 is block-starved, not occupancy-starved
ncu (SyntheticUSA B=1): the time-dominant deep big-front launches run **1 block each → waves/SM 0.02–0.24** (80+ of 82 SMs idle) but per-block **occupancy 65%**. So the GPU is starved of *independent blocks*, not of occupancy-per-block. Consequences:
- **Panel-resident kernel is neutral at B=1** (measured default ≈ panel-off: SyntheticUSA 2.142 vs 2.139). Its lever — free shared → more blocks/SM — buys nothing when there aren't enough blocks, while its cost (CB pushed to global) is real. Forcing panel on with a B=1-friendly gate makes it *worse* than the whole-front kernel (USA +4%). The default gate correctly keeps panel OFF at B=1.
- Time split at B=1: big 50% / mid 42% / small 8%.

## TC is effective at B=1 — the regime reversal
Isolation (same panel kernel, only the trailing path TC↔scalar, B=1):
| TF32 B=1 | panel-scalar | panel-TC |
|---|---|---|
| ACTIVSg25k | 0.823 | 0.799 (−2.9%) |
| SyntheticUSA | 2.20 | **1.82 (−17.4%)** |

TC trailing is clearly faster at B=1 — the **opposite** of B=64 (doc 10, where TC's 112 registers crashed occupancy 94%→28% and DRAM 73%→43%). The reversal:
- **B=64 = bandwidth-bound.** TC's register cost crashes occupancy → loses. Compute isn't the bottleneck.
- **B=1 = latency-bound + block-starved.** Occupancy is moot (80 SMs idle), so the register cost is **free**; the mma collapses the trailing's serial FMA chain into a few instructions → shortens the per-front critical path → wins. The win scales with front size (USA's big trailing → −17%).

Resolving the apparent contradiction with "TC pipe ≤2%": low TC-pipe% means the tensor units are busy only ~2% of cycles **because the trailing is a small fraction of the kernel** (most of it is panel LU + memory latency) — not because TC is idle. Within that fraction, the mma finishes fast. Low pipe% ≠ TC unused.

## Ozaki-TC: the fp32-accurate B=1 lever
Plain TF32 is fastest but drops accuracy to ~1e-2 (risky for a final solve). Ozaki-TC splits fp32 into TF32 head/tail and runs 3–4 mma passes to **recover fp32 accuracy** (`CLS_TF32_OZAKI_TC2`, `build-ozaki`, run `--precision tf32`). B=1 factor:
| B=1 | fp32 scalar | plain TF32-TC | **Ozaki-TC** |
|---|---|---|---|
| ACTIVSg25k | 0.709 (relres 1.4e-4) | 0.605 −15% (relres **5e-2**) | **0.649 −8.5% (relres 1.4e-4)** |
| SyntheticUSA | 2.140 (4.6e-3) | 1.649 −23% (4.4e-2) | **1.765 −17.5% (4.4e-3)** |

Ozaki keeps **fp32 accuracy** (relres matches fp32) and still wins −8.5%/−17.5%; the extra mma passes cost some of the raw TF32 win but B=1's free occupancy absorbs them. **Ozaki-TC is itself B=1-specific** — at B=64 the extra passes + registers would deepen the occupancy/bandwidth loss (doc 10).

## Practical
Ozaki is a compile flag, not a runtime switch. For a B=1-dominated (single-system Newton) workload: build with `CLS_TF32_OZAKI_TC2`, run `--precision tf32` → fp32 accuracy + −8.5–17.5% factor on the large cases. To share one binary across B=1 and batched regimes, Ozaki-TC would need a runtime toggle (compile-flag → env) so B=1 uses it and B≥16 doesn't.

## Two regimes, opposite optimal kernels (the unifying picture)
| | bottleneck | optimal mid kernel | trailing |
|---|---|---|---|
| **B≥16 (batched)** | bandwidth (occupancy) | **panel-resident** (free shared) | **scalar** (TC crashes occupancy) |
| **B=1 (single)** | latency (block-starved) | **whole-front** (CB in shared) | **TC / Ozaki-TC** (shortens critical path) |

The current defaults already split correctly: panel gated ON for high-block batched big-fronts, OFF for B=1 → whole-front + (Ozaki-)TC.
