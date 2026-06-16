# Contribution analysis — claim-based, falsifiable, critically argued

> **상태**: canonical   **갱신**: 2026-06-15
> **한 줄**: 솔버가 내거는 falsifiable 한 주장들을 하나씩 공격·판정한다 — **현재 `src/` 코드 재독 기준**.

This re-does the contribution question with a different method than the literature survey
[`01-orientation/02-related-work-and-novelty.md`](01-orientation/02-related-work-and-novelty.md)
(which this builds on). It exists because the survey concluded "no defensible contribution until a
head-to-head exists" — which conflates two different things. The narrative companion is
[`storyline.md`](storyline.md); this doc is the adversarial test of its claims.

## 0. Reframing (why the old verdict was wrong-shaped)

"We have not measured against cuDSS / KLU / STRUMPACK" disqualifies exactly one kind of claim — a
**competitive** claim ("we are faster than X"). It says nothing about whether the work makes
**characterization** or **method** claims that are novel and testable on their own. The right
question is therefore not *"did we win?"* but:

> **What falsifiable claims does this solver make, and which can be validated WITHOUT any other
> solver?**

A statement is a contribution here if it is (a) **not located in prior art** and (b) **testable**.
Most of our claims are testable with experiments we run entirely in-house (accuracy ablations,
occupancy/bandwidth profiling, per-kernel ablations). The external head-to-head upgrades a
contribution from "characterized method" to "characterized *and* competitive method" — it is a
**strengthener, not a gate** for everything except the literal speed claim.

Stance: **critical, not promotional.** Each candidate claim is stated, then attacked, then judged.
Several are rejected; two former "contributions" are **retired because the code that backed them was
deleted** (see §3.5).

---

## 1. What the solver actually is (code-grounded facts, re-verified against `src/`)

From this repo, not aspirations:

- **Multifrontal LU, no pivoting.** Nested-dissection (METIS) ordering, relaxed-panel amalgamation
  (`max_panel_width`, analyzer auto-sets 16 for n≥16k else 8), front-major batched arena. Partial
  pivoting is **reserved infrastructure, off by default** (`CLS_USE_PIVOTING`, `enable_shift_retry`);
  correctness rests on power-grid diagonal dominance
  ([`02-design-analysis/02-no-pivoting-proof.md`](02-design-analysis/02-no-pivoting-proof.md)). Single
  symbolic/analyze shared by the single-system (B=1) and uniform-batched paths. Code: `analyze/`,
  `factorize/front_ops.cuh`.
- **GPU-resident execution.** `analyze()` runs once; `factorize()`/`solve()` replay device pointers.
  In `CLS_INTERNAL_GRAPH` mode (default ON) the factor and the *full* solve (gather → solve levels →
  scatter) are captured into replayable CUDA graphs. In external/capturable mode (`CLS_INTERNAL_GRAPH`
  OFF + `set_stream`) the kernels issue onto a caller stream with no host sync, so **cuPF can capture
  the entire Newton iteration as one outer graph**. Code: `factorize/factorize.cu`, `solve/solve.cu`,
  `internal/runtime/setup.cu`.
- **Single shared symbolic across B systems.** One etree/amalgamation/arena; front-major arena
  `B * front_total`; one kernel launch per etree level with `blockIdx.y = batch`. Code:
  `internal/plan/`, `internal/runtime/state.hpp`.
- **Three precisions: FP64 / FP32 / TF32.** FP32 and TF32 both run an **all-float front (no FP64
  master)** — assembly, LU, U-solve, trailing, extend-add and solve all `float`. TF32 adds a TF32
  `mma.m16n8k8` trailing GEMM with optional Ozaki 2-component accuracy recovery. The FP64-master mixed
  path and an FP16 path **existed and were deleted** as regressions (`deprecated/precision_mixed/`,
  `deprecated/fp16/`). Code: `internal/runtime/state.hpp` (`Precision`), `factorize/{small,big,large}.cuh`.
- **Tier-specialized kernels.** Four tiers, routed **deterministically by front size** (boundaries in
  `internal/types.hpp`), not a runtime shared-fit/occupancy test. `tiny` (`fsz≤32`): one sub-group
  (8/16/32 lanes) per front, packed into warps. `small`: shared-resident blocked, whole front in
  shared. `big`: **panel-resident** — only the L/U panels in shared, the contribution block left in
  global (recovers batched occupancy on the larger shared-resident fronts). `large`: global-resident,
  only L/U panels staged. Code: `factorize/{tiny,small,big,large}.cuh`.
- **TF32 tensor-core trailing is integrated**, not a separate module — it lives inside the
  small/big/large kernels (`factor_small<float,true>`, `factor_big<float,true>`,
  `factor_large<float,true>`). There is **no `src/tc/`**.
- **Solve is forward/backward substitution** — warp-packed sub-group panel substitution + CB update,
  fixed-nc specialization, spine walk. It is **NOT a GEMV / partitioned-inverse solve** (that path was
  built and deleted; `deprecated/selinv/`). Code: `solve/{phases,kernels,dispatch}.cuh`.

---

## 2. The literature, only as it bounds each claim (cited)

Full citations and primary-text quotes: [`01-orientation/02-related-work-and-novelty.md`](01-orientation/02-related-work-and-novelty.md). The two facts that recur and matter:

- **The nearest power-grid neighbors are FP64 column-LU.** Wang/Fraunhofer [5] (arXiv:2101.02270) is
  **KLU column-LU refactorization**, batched, one shared symbolic, and **explicitly FP64** *"due to
  the high numerical stability requirement during the LU refactorization."* Zhou [6] is batched
  whole-system LU. Swirydowicz [8] (*"GPU-Resident Sparse Direct Linear Solvers for AC OPF"*) is KLU +
  refactorization, FP64. cuDSS [9b] exposes a uniform-batch primitive (FP32/FP64/double-double + IR,
  **no FP16/tensor-core factor mode**), vendor black box, no published power-grid Jacobian benchmark.
  STRUMPACK [1] is multifrontal but FP64-centric (MAGMA vbatched + a hand kernel for tiny fronts),
  general SuiteSparse.
- **Every GPU neighbor fights host↔device transfer rather than removing it.** Wang [5] and MDPI-2024
  are CPU–GPU hybrids that *hide* the H↔D transaction with streams; none captures the whole batched NR
  iteration on device. Spatula [7] (MICRO'23) documents that general multifrontal on near-planar
  (grid-like) matrices hits **0.004 % of V100 peak** and that *"batching … causes load imbalance …
  level-by-level traversal … incurs additional DRAM traffic."*

---

## 3. Candidate claims — each attacked, then judged

Legend for **Validation**: 🟢 = testable in-house, no other solver needed; 🔴 = needs an external baseline.

### Angle A — Power-flow (batched, shared-symbolic)

**A1. "Batched, single-shared-symbolic solving of many power-grid NR Jacobians."**
Prior art: Wang [5], Zhou [6]; uniform-batch-of-systems is now a **cuDSS primitive** [9b]. This is the
headline idea and it is **fully prior art**.
**Verdict: REJECT.** Not a contribution. Do not claim it.

**A2. "Doing the batched power-grid solve with a *multifrontal / dense-front* kernel rather than
column-LU refactorization."**
Prior art: the power-grid batched solvers [5][6] use column/whole-system LU; multifrontal-batched for
this target is not located. But this is an **implementation choice**, not a new algorithm. 🟢 (ablation
vs a column-LU batched baseline).
**Verdict: WEAK.** Frame as "an alternative realization," never as novelty on its own.

### Angle B — Precision (the strongest positive angle)

**B1. "An all-float front (no FP64 master) suffices for the Newton outer loop on power-grid
Jacobians."** i.e. an all-FP32/TF32 batched factor+solve keeps NR convergence (iteration count, final
residual) at parity with FP64, with no iterative refinement.
Prior art: the closest neighbor **explicitly chose FP64 "due to the high numerical stability
requirement during the LU refactorization"** [5]; the ACOPF winners are FP64 [8]; cuDSS power-grid is
FP64/double-double [9b]. **No all-float batched power-grid solver is located, and the field's stated
assumption is the opposite.**
Critical view: "FP32 sparse LU" is old (GLU3.0 [4], but circuit, single-system, Maxwell workaround);
absence-of-evidence is not proof; and float *does* fail at the top of the size range (70k diverges at
a conditioning floor), so the claim must be **scoped**. Note the *recommended* path is TF32+Ozaki,
which recovers FP32-band accuracy (relres ~1e-4..1e-5) — so the precise claim is "float front, TF32
trailing with cheap Ozaki recovery, is enough for NR," which is *stronger* than plain FP32.
Validation 🟢: NR-outer-loop ablation — float-front vs FP64 factor, compare iteration count and final
mismatch across the case suite; report the size/conditioning boundary. **No other solver needed.**
**Verdict: DEFENSIBLE and self-validating.** The strongest *positive* claim: it **contradicts a
published, peer-reviewed assumption** and is settled by an in-house experiment.

**B2. "On power-grid Jacobians, the tensor-core/low-precision factorization yields only a small,
bounded win — quantified, with the mechanism — and the obvious mixed-precision tricks regress."**
The trailing GEMM has contraction depth `K = nc ≤ 16–20`; TF32 mma at that K wins **only ~1.1×
(median, best-vs-best)**, and below ~10K nodes (`nc≈1–2`) it wins **structurally nothing**. The
honest decomposition (experiment #3) shows the tensor-core mma itself is **+6–9 %** of the float
speedup; the larger ~62 % is precision-agnostic kernel engineering. Two "obvious" upgrades were
built and **deleted as regressions**: an **FP64-master mixed path** (cast/writeback/2× memory on
latency-bound work; `deprecated/precision_mixed/`) and **FP16 trailing** (diverges at 25k/70k;
`deprecated/fp16/`).
Prior art: the survey found **no source quantifying tensor-core efficiency in this tiny-K
sparse-direct regime.** It is folklore that big fronts go to cuBLAS/TC and tiny fronts get hand
kernels [1][7] — but folklore is *qualitative*; nobody publishes the tiny-K crossover number or the
mixed-path-loses result on this matrix class.
Critical view: a "we measured ~1.1×" headline is unglamorous, and the value is precisely its honesty
— the old 1.2–1.3× headline was a baseline-cap-inflation artifact, now corrected.
Validation 🟢: data already in-repo (`factorize/{small,big,large}.cuh`, sweep over front width); package as a
characterization (TC mma vs scalar vs accuracy, plus the two deleted mixed paths as measured
negatives). **No other solver needed.**
**Verdict: DEFENSIBLE and self-validating — the most robust contribution.** A measured, reproducible
*bounded-and-negative* result on the correct matrix class, filling a documented gap. The Ozaki recovery
(relres 3.97e-2 → 4.77e-5 at near-zero cost) is the methodological core: *use low-precision TC but keep
FP32 accuracy*.

### Angle C — GPU-resident

**C1. "A GPU-resident sparse direct solver for power systems."**
Prior art: **Swirydowicz [8] uses exactly this term and concept** for ACOPF. The label is taken.
**Verdict: REJECT as novelty.** Don't claim "GPU-resident" as a new idea.

**C2. "The entire *batched* NR iteration as a single replayed CUDA graph with zero per-iteration
host↔device transfer (only a scalar convergence read)."** Assembly → solve → update → mismatch → norm
are one captured graph; the data-dependent convergence test is the sole host touch. The solver's
external/capturable mode (`CLS_INTERNAL_GRAPH` OFF + `set_stream`) is what makes this possible —
factor/solve issue onto cuPF's stream with no internal sync.
Prior art: [8] is GPU-resident but **single-system ACOPF KKT**, not a whole-iteration graph, not
batched; Wang [5] and MDPI-2024 *hide* (not remove) the H↔D transaction; no located work captures the
**whole batched NR iteration** as one graph. Critical view: this is a **systems/integration**
contribution that lives largely in **cuPF**, not the linear solver proper; CUDA graphs are a standard
tool, so the novelty is the *application*, not the mechanism.
Validation 🟢: per-iteration host time and H↔D bytes vs a non-graph baseline; cite [5]'s "hide the H↔D
transaction" as the cost we remove structurally. **No other solver needed.**
**Verdict: DEFENSIBLE but narrow**, and credit-shared with cuPF. State it as "removes the H↔D overhead
prior batched solvers engineer around," not as inventing GPU residency.

### Angle D — Multifrontal optimization (kernel + structural engineering)

**D1. "Four front-size tiers, each with a dedicated kernel, routed by a boundary derived from a
physical limit — not a swept constant or an occupancy-fill gate."** Front size (and, for the big|large
split, precision) picks one of four kernels: **tiny** (≤32, warp-packed sub-group), **small** (33–64,
whole-front shared-resident), **big** (65–shared-capacity, panel-resident: only the L/U panels in
shared, CB in global), **large** (global-resident, panels staged). The three boundaries are warp width
(32), the whole-front shared occupancy crossover (64), and the whole-front shared capacity (159 float /
111 double, derived from the 99 KiB opt-in budget). The one non-size condition is a clean **batch-regime
split on the big tier's shared strategy**: at B=1 it uses the whole-front kernel (per-front critical
path, D3), at B>1 the panel-resident kernel (batch bandwidth, D2). **Precision is orthogonal to the
tier** — in TF32 mode the trailing GEMM takes the TF32 tensor-core path (Ozaki-corrected → FP32-band) in
every float tier (small, big, large), so the selected precision never silently degrades to another in
some tier. (The big-tier panel kernel recovers occupancy, so its TF32 mma has no idle headroom and is
measurably slower than FP32 scalar there — an accepted cost for a consistent TF32 path; `--precision
fp32` gives the fast scalar big tier.) A level's heterogeneous fronts are split per-tier (not promoted
to the largest), and independent subtrees run on separate streams.
Prior art: STRUMPACK uses MAGMA vbatched + a naive small-front kernel [1]; Spatula [7] names the
bottleneck (0.004 % peak, "crude batching," load imbalance) but its fix is a hardware accelerator. The
*claim* here is not just "we wrote fast kernels" but that the **tier boundaries are first-principles**
(each marks where one hardware resource — warp packing, shared occupancy, shared capacity — gives out),
so the routing is reproducible and explainable rather than swept.
Critical view: this is **engineering**, not new theory; the win is over a naive baseline, and MAGMA
vbatched is a non-trivial baseline we have **not** measured against. The deterministic routing gives up
one piece of the old occupancy-adaptive tuning — promoting tiny fronts to the block kernel at B=1 — a
trade we make for explainability; its B=1 cost is the thing to measure (the big-tier TF32 lever, by
contrast, is preserved by the batch-regime split above).
Validation 🟢 (in-house per-kernel ablations: `tier_split` on/off, `--no-multistream`; correctness
verified across all four tiers × FP64/FP32/TF32) and 🔴 (vs MAGMA vbatched / cuDSS for a fair "library
kernel" baseline).
**Verdict: DEFENSIBLE as a characterized method with first-principles tier boundaries**; its
**competitive** standing needs the head-to-head.

**D2. "Breaking the whole-front shared-memory cap with a panel-resident big kernel recovers batched
DRAM bandwidth — the largest measured batch win, and a structural (not micro-) insight."** Big-tier
fronts staged whole (`fsz²`) into shared run at 1–2 blocks/SM → 8–16 % occupancy → ~35–40 % of peak
DRAM. Staging only the L/U panels (`nc(fsz+uc) ≈ fsz²/3`) and leaving the contribution block in global
gives 3–4× more blocks/SM, lifts DRAM to 55–65 %, and fuses the extend-add into the parent so global
CB traffic stays one pass.
Prior art: shared-residency vs global-residency is standard; what is **not** located is the diagnosis
that on thin-K power-grid fronts the binding constraint is *shared footprint per block*, not the GEMM,
plus the fused-drain that keeps the global-CB pass single. Critical view: it is a mapping change, not
a new algorithm; the gate (`fsz≥112` large / `fsz≥64` heavy-occupancy big) is empirically tuned and the
big tier can regress without the stronger gate.
Validation 🟢: `CLS_MID_PANEL=0/1` ablation + ncu DRAM% (measured: DRAM 2–32 %→55–65 %, **USA B=64
−9.3 %**, default-on). **No other solver needed.**
**Verdict: DEFENSIBLE as a specific, measured structural optimization** — the single largest batched
factor lever. Code: `factorize/big.cuh` (`factor_big`);
[`03-optimization-notes/07-batch-factorize-structural-2026-06-13.md`](03-optimization-notes/07-batch-factorize-structural-2026-06-13.md).

**D3. "Single-system (B=1) is a distinct regime whose lever is tensor cores, shortening the critical
path."** B=1 factor is critical-path bound on the under-filled deep levels (~1 block/SM), where
scheduling/tiling/sync/amalgamation are all inert; TF32+Ozaki cuts the trailing on the critical path
(USA B=1 −17 %, *opposite-signed* to the B=64 lever D2 — the same plan wants opposite kernels at B=1
vs B=64).
Prior art: ND ordering is standard; what is located here is the *regime split* itself. Critical view:
this is a characterization, not a new algorithm.
Validation 🟢: the B=1 TF32 ablation (`--precision tf32` vs fp32, B=1).
**Verdict: DEFENSIBLE as a characterized regime split (B=1 lever = tensor cores)**; modest,
self-validating. Code: `solver.cpp`;
[`03-optimization-notes/06`](03-optimization-notes/06-b1-factorize-regime-2026-06-13.md).

### 3.5 Retired claims (the code that backed them was deleted)

The previous version of this doc listed three contributions that **no longer exist in the code**.
Keeping them would be a documentation–code mismatch:

- **"Partitioned-inverse per-front pivot block → batched multifrontal solve becomes a GEMV."**
  RETIRED. The selected-inversion path was built and **deleted** (`deprecated/selinv/`): power grids
  run *1 factor + 1 solve* per NR step, so pre-inverting the pivot block is a net loss, and the FP64
  inverse is throughput-bound on the 3090. The current solve is plain warp-packed triangular
  substitution. The solve speedups that *do* exist (~1.5×, sub-group packing / fixed-nc / spine /
  inverse-scatter / full-solve graph) are engineering, comparable to D1 — not a distinct algorithmic
  claim. See [`03-optimization-notes/04-solve-optimization-2026-06-10.md`](03-optimization-notes/04-solve-optimization-2026-06-10.md).
- **"TC-routable front coarsening — fuse panels to grow nc so tensor cores fire."** RETIRED. Built and
  **deleted** (`deprecated/amalgamation/`): best-vs-best net ≈ 0 (cap-inflation artifact), fragile
  multi-flag bundle, and a regression on case3012wp. The thin-K ceiling (work-weighted nc ≈ 4.6) is
  structural; no fusion crosses the arithmetic-intensity wall.
- **"Best-of-k ND ordering selection."** RETIRED (2026-06-15, `deprecated/best_of_k/`). Both the
  measured and the static-proxy selectors were removed; the analyze pipeline uses a single parallel-ND
  ordering. Not a contribution.

### Single-system (B=1) note

Within a fixed precision the B=1 path is at its critical-path floor (see
[`history/b1-single-system-optimization.md`](history/b1-single-system-optimization.md)); the B=1 story
is the regime of D3 (tensor-core lever), not a separate contribution.

---

## 4. What needs an external solver, and what does not

| Claim | Contribution type | In-house validation | Needs head-to-head? |
|---|---|---|---|
| B2 bounded-TC / mixed-path negative | characterization (negative) | sweep TC mma vs scalar + accuracy + the two deleted mixed paths | **No** |
| B1 float-front suffices for NR loop | characterization (accuracy) | NR iteration/residual ablation vs FP64 | **No** (head-to-head only strengthens) |
| C2 whole-iteration zero-transfer graph | systems/integration | per-iter host time + H↔D bytes | **No** |
| D2 panel-resident occupancy recovery | method (structural) | `CLS_MID_PANEL` ablation + ncu DRAM% (done) | **No** |
| D3 B=1 regime (TC lever) | characterization (regime) | B=1 `--precision tf32` ablation | **No** |
| D1 tiny-front kernels / tier split | method (engineering) | per-kernel ablation (done) | **for competitive claim only** |
| "Faster than cuDSS / KLU / Wang" | competitive | — | **Yes (the only true gate)** |

Six of seven rows do not need an external solver.

---

## 5. Honest verdict

- **A contribution exists now**, independent of any head-to-head: **B2** (a quantified, *bounded and
  negative* tensor-core result on power-grid thin-K fronts — ~1.1× median, structural zero below 10K,
  with the FP64-master and FP16 mixed paths measured as regressions — filling a documented gap) and
  **B1** (an all-float front, TF32+Ozaki, suffices for the NR loop — contradicting the closest
  competitor's explicit FP64-for-stability choice). Both are settled by experiments we run ourselves.
  **D2** (panel-resident occupancy recovery) is the largest measured *batch* lever and a structural
  insight; **D3** (B=1 regime — tensor cores are the lever) and **D1** (tiny-front kernels) are
  characterized methods; **C2** is a narrow systems result shared with cuPF.
- **Reject the strong claim** ("a new / faster GPU sparse solver"): batched shared-symbolic is prior
  art (now a cuDSS primitive) and the speed claim genuinely needs the head-to-head.
- **Retire the deleted claims**: the GEMV/partitioned-inverse solve, TC-routable front coarsening, and
  measured best-of-k ordering selection are no longer in the code — do not list them as contributions.
- **The correct sentence about comparison:** the head-to-head decides the **competitive standing** of
  D1 (and any "faster" wording) — it does **not** decide whether the work contributes. "No contribution
  because we didn't compare" is a category error: B1, B2, C2, D2, D3 are contributions *because they
  are novel and in-house-testable*.

**One line:** 기여는 있다 — "더 빠른 솔버"가 아니라 **(B2) 전력망 thin-K 에서 텐서코어/혼합정밀이 ~1.1×
상한이고 FP64-master·FP16 은 회귀라는 정량적 음의 결과**, **(B1) all-float front(TF32+Ozaki)가 NR 루프에
충분하다는, 선행연구의 FP64 가정 반증**, 그리고 **(D2) whole-front shared 한계를 깬 panel-resident 커널의
구조적 occupancy 회복(최대 배치 이득)** 이 핵심이며 모두 외부 솔버 없이 자체 검증된다. 폐기된 GEMV solve·
front coarsening 은 기여 목록에서 내린다. head-to-head 는 D1 의 "경쟁력"과 "더 빠르다" 문장에만 필요한
게이트일 뿐, 기여의 존재 여부를 가르는 관문이 아니다.

## 6. The experiments that turn these claims into validated results

1. **B1 (no external solver):** float-front vs FP64 factor across the case suite; report NR iteration
   count, final |ΔV| / mismatch, and the size/conditioning boundary where float fails (70k known to
   diverge). Deliverable: an accuracy-vs-size frontier; show TF32+Ozaki sits in the FP32 band.
2. **B2 (no external solver):** sweep front width; per width, TF32-mma vs FP32-scalar trailing time +
   accuracy; include the deleted FP64-master and FP16 paths as measured negatives; locate the K where
   TC would break even and show K=nc never reaches it on these grids.
3. **C2 (no external solver):** per-iteration host wall-time and H↔D bytes, graph vs non-graph
   (external/capturable mode).
4. **D2 (no external solver):** `CLS_MID_PANEL=0/1` × case × B sweep with ncu DRAM-throughput, across
   FP64/FP32/TF32.
5. **D3 (no external solver):** B=1 `--precision tf32` vs fp32 (the B=1 lever, USA −17 %); show the
   same plan wants opposite kernels at B=1 vs B=64.
6. **D1/competitive (external, the only gate):** add **MAGMA vbatched** and **cuDSS ≥ v0.6 (FP32 + IR)
   uniform-batch** as fair library baselines; KLU/NICSLU and ideally Wang [5] on the *same* Jacobians,
   same accuracy target.

Sources: [`01-orientation/02-related-work-and-novelty.md`](01-orientation/02-related-work-and-novelty.md)
(full citations [1]–[10]); primary texts for [5][7] and STRUMPACK read in full.
