# mysolver Competitive Summary

## ⭐ GPU multifrontal factor + solve vs cuDSS (cycle 277, 2026-05-28) — CURRENT

Warm phase-only kernel ms (the Newton-Raphson regime where cuDSS amortizes its one-time
analysis). **FACTOR now beats cuDSS 8/8 (all matrices)** after the cy140-275 kernel arc
(count-guard occupancy, multi-block big-front trailing+extend, tiled trailing). The biggest
recent step: cy273 multi-blocked the big-front extend-add (was the only one-block-per-front
big-front kernel) + cy275 tuned its tile count -> closed the long-standing onetone2 holdout
(was 4.49x behind at cy114).

**Power-grid Jacobians (METIS parND), warm MF kernel ms:**

| matrix | n | MF factor | cuDSS factor | MF solve | cuDSS solve | F+S |
|---|---:|---:|---:|---:|---:|:--|
| case6468rte | 12 643 | **0.52** | 0.61 | 0.52 | 0.29 | cuDSS (S) |
| case8387pegase | 14 908 | **0.71** | 1.11 | 0.65 | 0.36 | **WIN** |
| case_ACTIVSg25k | 47 246 | **1.52** | 1.88 | 1.45 | 0.67 | cuDSS (S) |
| case_SyntheticUSA | 156 255 | **3.10** | 5.08 | 2.54 | 1.44 | **WIN** |

**Circuit matrices (MC64 + METIS), warm MF kernel ms:**

| matrix | n | MF factor | cuDSS factor | MF solve | cuDSS solve | F+S |
|---|---:|---:|---:|---:|---:|:--|
| rajat27 | 20 640 | **0.65** | 1.00 | 0.62 | 0.39 | **WIN** |
| memplus | 17 758 | **0.65** | 1.15 | 0.61 | 0.42 | **WIN** |
| onetone2 | 36 057 | **7.84** | 8.33 | 3.52 | 1.53 | cuDSS (S) |
| rajat15 | 37 261 | **2.62** | 3.50 | 1.78 | 0.87 | ~tie |

**Bottom line:** FACTOR beats cuDSS 8/8; SOLVE still 1.3-2.5x behind (latency + fill-work
bound, research-grade — every incremental lever ruled out: multiblock cy154/274, cap cy169/277,
coalescing cy159/258, FP32 cy170, multi-subtree cy257). Combined F+S beats-or-ties cuDSS ~5/8;
the F+S losses are purely the SOLVE gap. ANALYSIS -13% via cy259-266 symbolic parallelization
(floor = METIS recursion, research-grade). berr 1e-13..1e-16 (onetone2 raw 2.5e-7 -> refinement).

## ⭐ GPU multifrontal factor + solve vs cuDSS (cycle 104, 2026-05-26) — SUPERSEDED

The GPU sparse-direct path (own multifrontal: relaxed dense panels, level-scheduled
GPU dense-front LU + extend-add, multifrontal triangular solve). Phase-only kernel
ms (cuDSS = `CUDSS_PHASE_FACTORIZATION`/`SOLVE`, the regime where cuDSS amortizes
its expensive one-time analysis — what matters in a Newton-Raphson loop). The arc:
the GPU factor began **29× behind** cuDSS (cycle 27, naive level-set) and is now at
parity or ahead on most matrices.

**Power-grid Jacobians (METIS), GPU MF kernel ms:**

| matrix | n | MF factor | cuDSS factor | MF solve | cuDSS solve |
|---|---:|---:|---:|---:|---:|
| case6468rte | 12 643 | **0.48** | 0.61 | 0.41 | 0.29 |
| case8387pegase | 14 908 | **0.64** | 1.11 | 0.62 | 0.36 |
| case_ACTIVSg25k | 47 246 | 2.11 | 1.88 | 1.5 | 0.67 |
| case_SyntheticUSA | 156 255 | 6.55 | 5.08 | 2.28 | 1.44 |

Factor **beats cuDSS on case6468rte + case8387pegase**, 1.1–1.3× behind on the two
largest. Solve within 1.6–2.2×. berr ~1e-13…1e-16.

**Circuit matrices (MC64 matching + scaling + METIS), GPU MF kernel ms:**

| matrix | n | MF factor | cuDSS factor | MF solve | cuDSS solve |
|---|---:|---:|---:|---:|---:|
| rajat27 | 20 640 | **0.81** | 1.00 | 0.83 | 0.39 |
| memplus | 17 758 | **0.95** | 1.15 | 0.56 | 0.42 |
| rajat15 | 37 261 | 3.74 | 3.50 | 1.92 | 0.87 |
| onetone2 | 36 057 | 36.2 | 8.33 | 4.18 | 1.53 |

Factor **beats cuDSS on rajat27 + memplus, ties on rajat15**; onetone2 is 4.4×
(inherent: deep etree + high fill). The cy71 right-looking kernel OOM'd these;
MC64 (diagonal matching) makes the no-pivot MF stable, METIS shallows the deep
circuit etrees, and a tiled backward handles the big circuit fronts.

**Key techniques:** right-looking → multifrontal redesign (cy70/81); 512-thread
dense-front blocks (cy83); relaxed panel_cap=8 (cy87); blocked rank-nc + fused
factor/extend (cy88/89); multifrontal solve with warp-shuffle + tiled backward
(cy90–103); MC64 + METIS for circuits (cy100–103). Kernels confirmed at their
multifrontal-level-set limit (cy94–98, 104). Production keeps the CPU path with a
residual-gated fallback; the GPU MF is the research factor+solve.

---

# (older) mysolver Competitive Summary (cycle 58, 2026-05-26 — CPU-era, pre-GPU-MF)

Honest, data-grounded standing of the mysolver own-numeric pipeline vs cuDSS (GPU)
and KLU (CPU). All numbers are **warm, end-to-end single-solve wall time (ms)** from
the production benchmark (`--warmup-gpu`, analyze+factor+solve in one call), measured
under quiet conditions (GPU idle 0%, load 0.05).

## End-to-end single-solve (ms, lower is better)

| matrix | n | mysolver | KLU | cuDSS-gpu | vs cuDSS | vs KLU |
|---|---:|---:|---:|---:|---|---|
| case118 | 181 | 0.19 | 0.12 | 49.7 | **260× faster** | 1.6× slower |
| case1197 | 2 392 | 1.49 | 0.47 | 14.1 | **9.5× faster** | 3.2× slower |
| case_ACTIVSg2000 | 3 607 | 4.65 | 2.68 | 16.9 | **3.6× faster** | 1.7× slower |
| case3012wp | 5 725 | 6.44 | 2.60 | 18.5 | **2.9× faster** | 2.5× slower |
| case6468rte | 12 643 | 14.9 | 6.09 | 26.4 | **1.8× faster** | 2.4× slower |
| case8387pegase | 14 908 | 16.7 | 8.74 | 32.6 | **2.0× faster** | 1.9× slower |
| case_ACTIVSg25k | 47 246 | 68.0 | 27.4 | 106.0 | **1.6× faster** | 2.5× slower |
| case_SyntheticUSA | 156 255 | 289.5 | 131.8 | 198.3 | 1.5× slower | 2.2× slower |
| rajat27 (circuit) | 20 640 | 20.3 | 237.2 | 51.7 | **2.5× faster** | **11.7× faster** |
| onetone2 (circuit) | 36 057 | 191.1 | 240.7 | 96.3 | 2.0× slower | **1.3× faster** |
| rajat15 (circuit) | 37 261 | 182.3 | 125.5 | 126.7 | 1.4× slower | 1.5× slower |

**vs cuDSS: mysolver wins 8/11** (all power-grid up to 47k + rajat27); loses only on
the 156k power-grid (1.5×) and two hard circuits (onetone2, rajat15). cuDSS's per-call
GPU setup/transfer overhead dominates its fast factor kernel on a single solve.
(NOTE: this cycle-58 line is SUPERSEDED — see the top section. The GPU factor kernel
was ~73ms here but the cy70–104 multifrontal redesign brought it to 0.48ms on
case6468rte, beating cuDSS. The end-to-end table below is the old CPU-path state.)

## Takeaways
- **vs cuDSS (single solve): mysolver wins on every matrix ≤ 20k** — cuDSS's
  per-call analyze + GPU malloc + H2D/D2H overhead dwarfs its fast factor kernel.
  cuDSS pulls ahead only on the largest (47k 1.4×, 156k 1.8×).
- **vs KLU**: KLU (highly tuned CPU BTF+AMD) is faster on power-grid, but mysolver
  **beats KLU 9.4× on the circuit matrix rajat27** (MC64 matching).
- **Accuracy**: mysolver has the best backward error of all (berr ~1e-16; 12/12 on
  the production gate, incl. all circuit matrices).
- **Coverage**: own-numeric solves 12/12 (power-grid + circuit) on the berr gate.
  rajat27 forward error (4e-8) is intrinsic conditioning (κ~4e5), not a solver gap.

## Repeated-factorization (Newton-Raphson) regime

In a power-flow NR loop the sparsity is fixed and only the values change, so the
analysis (matching + AMD + symbolic fill) is done once and reused via
`own_refactor` (cycle 59). Per-iteration time vs. the full first solve:

| matrix | n | first solve | per refactor | speedup |
|---|---:|---:|---:|---|
| case6468rte | 12 643 | 17.7 ms | 3.8 ms | **4.7×** |
| case_ACTIVSg25k | 47 246 | 76.5 ms | 28.1 ms | **2.7×** |
| case_SyntheticUSA | 156 255 | 290.8 ms | 128.1 ms | **2.3×** |

So mysolver is competitive in BOTH regimes: single-solve (beats cuDSS 8/11) and
the NR refactor loop (2.3–4.7× per iteration), with refactor berr ~2e-16.

## Where the time goes (large matrices)
SyntheticUSA (156k) ≈ analysis 154ms (SuiteSparse AMD-dominated) + factor 85ms
(memory-bandwidth-bound) + solve 32ms. Both remaining components are at their
practical floor for a single-threaded CPU scalar factor.

## Optimizations that landed
- MC64 max-product matching (circuit coverage 9/12 → 12/12), applied conditionally
  (only weak-natural-diagonal matrices) — 2.4–2.5× on large power-grid solves.
- cuDSS-style analyze/factorize split (40–80× less per-refactorization setup).
- AMD-vs-METIS-ND ordering cost model (GPU path); arena alignment fix; `__ldg`.
- Faster `build_symmetric_filled` (two-pass CSR) and reused symmetric pattern
  (`permute_pattern`) — large-matrix analysis/factor speedups.

## Directions closed with data (not worth pursuing for these matrices)
- **GPU factor kernel**: memory-latency-bound, ~100× off cuDSS on fine-grained
  power-grid matrices; slower than the CPU factor. (cuDSS leads the GPU sparse field.)
- **Supernodal / dense BLAS**: fundamental supernodes average ~2 columns — no
  BLAS-worthy blocks (cycle 41).
- **BTF**: reducible matrices are already won; the one we lose (rajat15) is
  irreducible (cycle 49).
- **CPU multithreading (level-set AND subtree-parallel)**: the scalar sparse
  factor is memory-bandwidth-bound (low arithmetic intensity: per op = 2 loads +
  1 store, little compute). Both level-set (cycle 48/52) and subtree task-parallel
  (cycle 61, bit-identical but 0.31–0.84×) give no speedup — the memory bus
  saturates with a few threads. No CPU parallel form helps these matrices.

## Remaining frontier (hardware-bandwidth-fundamental; goal substantially met)
- Repeated-factorization (NR) on the largest matrices: cuDSS's amortized factor
  is faster because GPU memory bandwidth >> CPU, on a well-engineered fine-grained
  kernel. Our CPU factor is bandwidth-limited (can't be parallelized away, cycle
  61); our GPU kernel is latency-bound. Closing this needs a cuDSS-grade
  fine-grained GPU kernel — the honest, research-grade frontier.
- High-fill non-power-grid matrices (wang3: ~10M fill) need a supernodal BLAS factor.
