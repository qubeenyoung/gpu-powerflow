# 13 — Literature review: METIS ND internals + prior art on GPU/parallel-objective ordering

**Date:** 2026-06-13. **Purpose:** before reimplementing ND for our problem, understand (a) how METIS ND actually works, (b) who has tried what we're trying (ordering objective ≠ fill), and (c) what it implies for the bounded ceiling we keep hitting.

## 1. How METIS nested dissection works (multilevel ND)
Three-phase **multilevel** scheme, recursively applied (find vertex separator → order each half → separator last):
1. **Coarsen** — heavy-edge matching + contraction; a sequence of smaller graphs. The heavy-edge heuristic makes the coarsest partition within a small factor of the final.
2. **Initial separator** — vertex separator computed on the coarsest graph.
3. **Uncoarsen + refine** — project the separator back up; at each level run **Fiduccia–Mattheyses (FM) node refinement**, focusing only on the boundary.

**The FM/separator objective = minimize separator SIZE (|S|) subject to a balance CONSTRAINT** (α-balanced, default uniform vertex weights). Balance is a *constraint* (within tolerance), not the objective. `vwgt` shifts the balance; there is **no edge-weight (`adjwgt`) path through NodeND** (confirmed in source, doc 12). → METIS optimizes fill; balance is just kept "good enough."

Sources: METIS manual / Karypis-Kumar; mt-Metis (LaSalle & Karypis, parallel ND + FM).

## 2. Prior art on ordering for PARALLELISM (our exact analog)
**This is the one objective with a literature basis that genuinely differs from fill:**
- **Elimination-tree HEIGHT = the critical-path length of parallel factorization.** Minimizing it exposes parallelism / shortens the serial dependency chain — precisely our **B=1** bottleneck (deep serial fronts).
- **Kayaaslan & Uçar (IPDPS 2015), "Reducing elimination tree height for parallel LU":** minimizing etree height is **NP-complete**; heuristic = order to **bordered block diagonal (BBD) with a small border + similar-size blocks** (= *balanced* nested dissection), then locally order. Reported **~28% tree-height reduction** vs standard tools.
- **Critical caveats the literature states plainly:**
  - "ND guarantees low fill **AND approximately optimal operation count**" — i.e. **fill ≈ total work**. So minimizing fill already ≈ minimizing the throughput-regime (B=64) cost. This is *why* our ordering ceiling is bounded.
  - "a **fundamental tradeoff between separator size/balance and fill-in**" — balancing harder *increases* fill.
  - 28% tree-HEIGHT reduction is **not** a 28% time speedup: height is the critical path, but per-node WORK still dominates total time. The literature does **not** claim proportional speedup from height reduction.

## 3. State of the art GPU solvers — what they actually do
- **cuDSS (NVIDIA production GPU sparse solver):** `CUDSS_ALG_DEFAULT` = **"a customized nested dissection algorithm based on METIS"**, plus an AMD option, plus `CUDSS_CONFIG_ND_NLEVELS` to **control ND depth**. → The state-of-the-art GPU solver does **exactly our `gpu_nd` recipe**: own the recursion on top of METIS's separator + control the depth/stopping. They do **not** replace METIS's separator objective.
- **STRUMPACK (GPU multifrontal):** relies on ND (METIS/SCOTCH) for fill + **subtree parallelism** (factor subtrees that fit in GPU memory); adds *recursive separator reordering* only for HSS rank structure (fill/rank, not a GPU-time objective).

## 4. What this means for us (honest)
1. **Our approach is the literature-standard one.** `gpu_nd` (custom recursion on METIS + leaf/depth control) = cuDSS's production design. We are not behind; we're at the state of the art. That also means the ~2% we saw is the expected envelope, not a missed breakthrough.
2. **fill ≈ work ⇒ the ordering ceiling is real and bounded.** The throughput regime (B=64) is already near-optimally served by METIS's fill objective. No literature reports a large GPU-factor speedup from changing the ND objective away from fill.
3. **The ONE objective that genuinely differs = critical-path / elimination-tree height** — and it maps to **B=1** (latency/serial), exactly where we already found TC helps. Our FM "imbalance penalty" is a crude proxy for tree-height minimization; the principled version is **minimize tree height via balanced BBD** (Kayaaslan-Uçar). Expected: **bounded** gains, **B=1-weighted**, with a **fill increase** to gate.
4. **The FM node-refinement we implemented IS the right mechanism** — FM with a modified gain is exactly how the etree-height line of work steers ND. So our `gpu_sep_refine` (FM toward a balance/critical-path objective) is literature-grounded; the honest task is to measure it **against the METIS seed envelope (best-of-k)**, framed as a critical-path objective, and accept the bounded result.

## 5. Recommendation
- Keep the FM separator refinement, but **frame and tune it as a tree-height/critical-path (B=1) objective** (where the literature says the gain lives), not a general speedup.
- **Measure vs the best-of-k METIS envelope** (not a single seed — doc-12 mistake), and report honestly; the literature predicts a small, B=1-weighted gain with a fill tradeoff.
- Do **not** expect to beat fill at B=64 — fill ≈ work is a theorem-level coupling; that ceiling is real.

## Sources
- METIS / mt-Metis: [METIS manual (Karypis)](https://www.jmlr.org/papers/volume23/21-0644/21-0644.pdf), [Efficient Nested Dissection for Multicore (LaSalle-Karypis)](https://dlasalle.github.io/publications/mtmetis2015nd.pdf)
- Tree height / critical path: [Reducing elimination tree height for parallel LU (Kayaaslan-Uçar, IEEE)](https://ieeexplore.ieee.org/document/7116880/)
- GPU solvers: [cuDSS docs](https://docs.nvidia.com/cuda/cudss/), [High performance sparse multifrontal solvers on modern GPUs (STRUMPACK)](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000059)
- Nested dissection / fill↔work: [Nested dissection (Wikipedia)](https://en.wikipedia.org/wiki/Nested_dissection)
