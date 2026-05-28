# M3 — Supernodal → GPU factorization: implementation spec

Status after cycle 19. Goal: cuDSS-competitive factor speed. This spec makes the
next (dedicated) build turnkey.

## Honest performance reality (cycle 17 data, total ms vs cuDSS)
- **Small (case30/118): mysolver already WINS** (CPU beats GPU launch overhead).
- **Solve: competitive** (case6468rte 0.47 vs cuDSS 0.28 after adaptive refinement).
- **Medium/large factor: the gap.** case6468rte 4.27 vs 0.58 (5.6×); wang3 catastrophic
  (scalar no-pivot LU 4.6 s vs cuDSS 34 ms) because of unblocked dense fill.

Two distinct regimes:
- **High-fill / large supernodes (wang3 snode 1446):** supernodal dense BLAS wins big.
- **Power-grid Jacobians (tiny supernodes, avg ~2, e.g. case6468rte):** dense blocking
  does NOT help; beating cuDSS needs fine-grained GPU sparse parallelism (research-grade).

## Step 1 — Postorder
Permute M (AMD-ordered, matched, scaled) by the etree postorder so supernodes are
contiguous column ranges. Compose into the existing solve permutation chain in
`own_pipeline`. Validate: `own_numeric_check` berr unchanged (9/12).

## Step 2 — Supernodal right-looking numeric (CPU, then BLAS)
Storage: symmetric fill `S` (Sp/Si/Sx), Sx holds L (i>j) and U (i≤j). Process
supernodes left→right (postordered, contiguous). For supernode J = cols [c0,c1),
sz=c1-c0, panel rows R = fill(col c0) (diag block c0..c1-1 first, then below rows):
1. Diagonal block D = Sx[R[0:sz], c0:c1] → dense LU (no pivot; matching+scaling
   already make it stable). Use `dense_lu` (cycle 12) or MKL `dgetrf`.
2. Below-L panel B = Sx[R[sz:], c0:c1] ← B · U_d⁻¹  (TRSM, `dtrsm`).
3. Right-U panel Ur = Sx[c0:c1, R[sz:]] ← L_d⁻¹ · Ur  (TRSM).
4. Schur update (the dominant cost, the BLAS win): for below rows i and right cols
   k, Sx(i,k) -= B(i,:)·Ur(:,k)  (GEMM, `dgemm`), scattered into S via
   `find_pos` (positions guaranteed present by the fill). This is the extend-add.
Validate: solve == `factor_nopiv` solve (berr match) on toys (dense 4×4 exercises
the block path; arrow exercises fill+updates) and on reals via `own_numeric_check`.

## Step 3 — GPU port
Swap the dense block ops (getrf/trsm/gemm) to cuBLAS; keep panels in device memory.
Launch independent supernodes (same `schedule` level) together. Expect wins on
large-supernode matrices; tiny-supernode matrices stay launch-overhead bound.

## Step 4 — Power-grid regime (research)
For tiny-supernode sparse matrices, cuDSS uses fine-grained GPU kernels over the
sparse structure (not dense blocks). This is the hard, open part of matching cuDSS
broadly — a sustained GPU R&D effort.

## Deferred
Circuit matrices (rajat27/onetone2/rajat15): optimal MC64 (max-product matching +
dual scaling). A greedy heuristic was tried and reverted (it regressed the
power-grid matrices). btf provides only structural maxtrans.

## Safety
The production adapter (`solver_registry.cpp`) accepts own-numeric only if its
componentwise berr ≤ 1e-10, else falls back to KLU. So any supernodal bug is
caught (KLU fallback) and never ships a wrong answer — develop boldly.
