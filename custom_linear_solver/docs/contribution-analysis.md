# Contribution analysis — claim-based, falsifiable, critically argued

This re-does the contribution question with a different method than
[`related-work-and-contribution.md`](related-work-and-contribution.md) (the literature survey,
which this builds on). It exists because the previous write-up concluded "no defensible
contribution until a head-to-head exists" — which conflates two different things.

## 0. Reframing (why the old verdict was wrong-shaped)

"We have not measured against cuDSS / KLU / STRUMPACK" disqualifies exactly one kind of claim —
a **competitive** claim ("we are faster than X"). It says nothing about whether the work makes
**characterization** or **method** claims that are novel and testable on their own. The right
question is therefore not *"did we win?"* but:

> **What falsifiable claims does this solver make, and which can be validated WITHOUT any other
> solver?**

A statement is a contribution here if it is (a) **not located in prior art** and (b) **testable**.
Most of our claims are testable with experiments we run entirely in-house (accuracy ablations,
occupancy/ROI profiling, per-kernel ablations). The external head-to-head upgrades a contribution
from "characterized method" to "characterized *and* competitive method" — it is a **strengthener,
not a gate** for everything except the literal speed claim. The previous doc let the missing
head-to-head sink claims that never depended on it. That is the error corrected below.

Stance: **critical, not promotional.** Each candidate claim is stated, then attacked, then judged.
Several are rejected.

---

## 1. What the solver actually is (code-grounded facts)

From this repo (`src/`), not aspirations:

- **Multifrontal LU** (no pivoting; diagonal-shift retry), nested-dissection ordering, relaxed-panel
  amalgamation, partitioned-inverse solve. Single-system and uniform-batched paths share one
  symbolic/analyze.
- **GPU-resident execution:** `analyze()` runs once and captures factor and solve into **CUDA
  graphs**; `factorize()`/`solve()` replay them on device pointers. cuPF can build the solver in
  *capturable* mode so the **entire Newton iteration** (Jacobian assembly → solve → voltage update →
  mismatch → norm) is one replayed graph; the only per-iteration host action is reading a scalar
  convergence norm.
- **Single shared symbolic across B systems** (batched): one etree/amalgamation/arena; front-major
  arena `B * front_total`; one kernel launch per etree level with `blockIdx.y = batch`.
- **FP32-native batched fronts** (`BatchPrecision::FP32`): the front arena is `float`, with **no
  FP64 master** — assembly, LU, trailing, extend-add and solve all FP32.
- **Tiny-front-specialized kernels:** warp-per-front (one warp/front, shared-staged) for the
  tens-of-thousands tiny bottom levels; shared-resident block kernel for mid fronts; high-thread
  kernel for the few big separators; warp-packed forward/backward solve.
- **Partitioned-inverse solve:** each front's `nc×nc` pivot block is inverted at the end of factor
  so the per-front solve is a **GEMV**, not a sequential triangular solve.
- **Tensor-core path exists and is measured** (`src/tc/`, batched): FP32-native WMMA trailing and an
  FP64-master WMMA trailing.

---

## 2. The literature, only as it bounds each claim (cited)

Primary-text facts used below (full text read locally where quoted):

- **Wang/Fraunhofer batched power flow [5]** (arXiv:2101.02270): **KLU column-LU refactorization**
  (Gilbert–Peierls, CCS) — *not* multifrontal; batched, one shared symbolic; **FP64, explicitly**:
  *"On both test platforms, double precision float number is used, due to the high numerical
  stability requirement during the LU refactorization."* CPU–GPU hybrid: it has a host↔device "DH
  Memory" transaction per batch that it **hides with streams**, not eliminates.
- **Zhou [6]:** batched whole-system LU for power grids; ~76× vs 1-thread KLU. Column/whole-system LU.
- **STRUMPACK [1]:** multifrontal; fronts done with **MAGMA vbatched** `dgetrf/dtrsm/dgemm` +
  a hand kernel for the smallest fronts; **FP64-centric**; general SuiteSparse, not power-grid.
- **Spatula [7]** (MICRO'23): on near-planar (circuit/grid-like) matrices general multifrontal hits
  **0.004 % of V100 peak**; *"batching is a crude [workaround] … causes load imbalance … level-by-
  level traversal … incurs additional DRAM traffic."*
- **Swirydowicz [8]:** title is literally *"**GPU-Resident** Sparse Direct Linear Solvers for AC
  Optimal Power Flow."* Single-system KKT; KLU + refactorization (cuSolverRf/Ginkgo); FP64; general
  GPU solvers often **lose to 1-core MA57**, problem-specialized refactorization wins 1.6–1.9×.
- **cuDSS [9b]:** real uniform-batch-of-many-systems API (v0.6+), FP32/FP64/double-double + IR,
  **no FP16/tensor-core factor mode**; vendor black box; **no independent power-grid Jacobian
  benchmark published**.
- **PanguLU [10]:** general distributed 2D-block; "batching" = multiple RHS on one factorization.
- **MDPI 2024 (Energies 17/6269):** batched power flow, **fast-decoupled**, **CPU-GPU hybrid**,
  binary interchange to fight H↔D throughput — not multifrontal, not GPU-resident.

Two facts recur and matter: (i) the nearest power-grid neighbors are **FP64 column-LU**, and
(ii) every GPU one **fights host↔device transfer** rather than removing it.

---

## 3. Candidate claims, by the angles requested — each attacked, then judged

Legend for **Validation**: 🟢 = testable in-house, no other solver needed; 🔴 = needs an external
baseline (head-to-head).

### Angle A — Power-flow (batched, shared-symbolic)

**A1. "Batched, single-shared-symbolic solving of many power-grid NR Jacobians."**
Prior art: Wang [5], Zhou [6]; and uniform-batch-of-systems is now a **cuDSS primitive** [9b].
Critical view: this is the headline idea and it is **fully prior art**.
**Verdict: REJECT.** Not a contribution. Do not claim it.

**A2. "Doing the batched power-grid solve with a *multifrontal / dense-front* kernel rather than
column-LU refactorization."**
Prior art: the power-grid batched solvers [5][6] use column/whole-system LU; multifrontal-batched
for this target is not located. Critical view: this is an **implementation choice**, not a new
algorithm — no new ordering/pivoting/complexity result. 🟢 (ablation: multifrontal vs a column-LU
batched baseline).
**Verdict: WEAK.** Frame as "an alternative realization," never as novelty on its own.

### Angle B — Precision (the strongest angles)

**B1. "FP32-native fronts (no FP64 master) suffice for the Newton outer loop on power-grid
Jacobians."** i.e. an all-FP32 batched factor+solve keeps NR convergence (iteration count, final
residual) at parity with FP64, with no iterative refinement.
Prior art: the closest neighbor **explicitly chose FP64 "due to the high numerical stability
requirement during the LU refactorization"** [5]; SuperLU_DIST batched is FP64 [3]; the ACOPF
winners are FP64 [8]; cuDSS power-grid is FP64/double-double [9b]. GLU3.0 is FP32 but circuit,
single-system, and only as a Maxwell-hardware workaround [4]. **No FP32-native batched power-grid
solver is located, and the field's stated assumption is the opposite.**
Critical view: "FP32 sparse LU" is old; absence-of-evidence is not proof; FP32 *does* fail on our
largest cases (70k diverges), so the claim must be **scoped** to where it holds.
Validation 🟢: NR-outer-loop ablation — FP32-native vs FP64 factor, compare iteration count and
final mismatch across the case suite; report the size/conditioning boundary where FP32 stops being
enough. **This needs no other solver.**
**Verdict: DEFENSIBLE and self-validating.** This is the single strongest *positive* claim because
it **contradicts a published, peer-reviewed assumption** and is settled by an in-house experiment.
It is a contribution the moment the ablation is run — independent of any head-to-head.

**B2. "On power-grid Jacobians, FP16/tensor cores and an FP64-master mixed path do NOT help the
factorization — quantified, with the mechanism."** The trailing GEMM has contraction depth
`K = nc ≤ 16–20`; WMMA loses to the FP32 FMA loop at that K, and an FP64-master path is *slower than
pure FP32* (cast/writeback/double-atomic/2× memory on latency-bound work), with FP16 trailing
diverging at 25k/70k.
Prior art: the survey found **no source quantifying tensor-core efficiency in this tiny-K
sparse-direct regime.** Critical view: it is "folklore" that big fronts go to cuBLAS/TC and tiny
fronts get hand kernels [1][7] — but folklore is **qualitative**; nobody publishes the tiny-K
crossover number or the FP64-master-loses-to-FP32 result on this matrix class.
Validation 🟢: the data already exists in this repo (`src/tc/`, batched benchmarks); package as a
characterization (TC vs FP32 vs FP64-master vs accuracy, swept over front width). **No other solver
needed.**
**Verdict: DEFENSIBLE and self-validating — the most robust contribution.** A measured, reproducible
negative result on the correct matrix class, filling a documented gap.

### Angle C — GPU-resident

**C1. "A GPU-resident sparse direct solver for power systems."**
Prior art: **Swirydowicz [8] uses exactly this term and concept** for ACOPF. Critical view: the
label is taken.
**Verdict: REJECT as novelty.** Don't claim "GPU-resident" as a new idea.

**C2. "The entire *batched* NR iteration as a single replayed CUDA graph with zero per-iteration
host↔device transfer (only a scalar convergence read)."** Assembly → solve → update → mismatch →
norm are one captured graph; the data-dependent convergence test is the sole host touch.
Prior art: [8] is GPU-resident but **single-system ACOPF KKT**, not a whole-iteration graph, not
batched; Wang [5] and MDPI-2024 are CPU-GPU hybrids that **hide** (not remove) the H↔D transaction;
no located work captures the **whole batched NR iteration** as one graph. Critical view: this is a
**systems/integration** contribution and it lives largely in **cuPF**, not the linear solver proper;
CUDA graphs are a standard tool, so the novelty is the *application*, not the mechanism.
Validation 🟢: measure per-iteration host time and H↔D bytes vs a non-graph baseline; cite [5]'s
"hide the H↔D transaction" as the cost we remove structurally. **No other solver needed.**
**Verdict: DEFENSIBLE but narrow**, and credit-shared with cuPF. State it as "removes the H↔D
overhead prior batched solvers engineer around," not as inventing GPU residency.

### Angle D — Multifrontal optimization (kernel engineering)

**D1. "Tiny-front-specialized kernels (warp-per-front, shared-resident mid, warp-packed solve) that
beat the library 'uniform batch' / `<32×32` fallback in the latency-bound regime."**
Prior art: STRUMPACK uses MAGMA vbatched + a naive small-front kernel [1]; Spatula [7] names the
bottleneck (0.004 % peak, "crude batching," load imbalance) but its fix is a hardware accelerator,
not better GPU kernels. Critical view: this is **engineering**, not new theory; the win is over a
naive baseline, and MAGMA vbatched is a non-trivial baseline we have **not** measured against.
Validation 🟢 (in-house ablation per kernel — already −25…−46 % vs our own FP64/Mixed/FP32) and 🔴
(vs MAGMA vbatched / cuDSS for a fair "library kernel" baseline).
**Verdict: DEFENSIBLE as a characterized method** (each kernel tied 1:1 to the bottleneck it
removes); its **competitive** standing is the part that needs the head-to-head.

**D2. "Partitioned-inverse per-front pivot block → the batched multifrontal SOLVE becomes a GEMV."**
Prior art: selected inversion exists (e.g. SelInv) but as a different goal; applying per-front
pivot-block inversion to convert the *batched power-grid multifrontal solve* into a GEMV is not
located. Critical view: it is a known trick repackaged; it trades a little factor time for solve
time. Validation 🟢: ablation (measured −22…−28 % backward, −13…−18 % forward).
**Verdict: DEFENSIBLE as a specific, measured optimization;** modest, self-validating.

### Single-system (B=1) note

The B=1 path is **at its critical-path floor** (see
[`history/b1-single-system-optimization.md`](history/b1-single-system-optimization.md)); within a
fixed precision it is not a contribution. The B=1 story belongs under B1 (precision lever) only.

---

## 4. What needs an external solver, and what does not

| Claim | Contribution type | In-house validation | Needs head-to-head? |
|---|---|---|---|
| B2 tiny-K TC / FP64-master negative result | characterization (negative) | sweep TC/FP32/FP64-master + accuracy | **No** |
| B1 FP32-native suffices for NR loop | characterization (accuracy) | NR iteration/residual ablation vs FP64 | **No** (head-to-head only strengthens) |
| C2 whole-iteration zero-transfer graph | systems/integration | per-iter host time + H↔D bytes | **No** |
| D2 partitioned-inverse GEMV solve | method (optimization) | solve ablation (done) | **No** |
| D1 tiny-front kernels | method (engineering) | per-kernel ablation (done) | **for competitive claim only** |
| "Faster than cuDSS / KLU / Wang" | competitive | — | **Yes (the only true gate)** |

The previous doc collapsed everything to the last row. Five of six rows do not need it.

---

## 5. Honest verdict

- **A contribution exists now**, independent of any head-to-head: **B2** (quantified tiny-K
  tensor-core / FP64-master negative result on power-grid multifrontal — a documented gap) and
  **B1** (FP32-native fronts suffice for the NR loop — contradicts the closest competitor's explicit
  FP64-for-stability choice). Both are settled by experiments we run ourselves. **D1/D2** are
  characterized methods; **C2** is a narrow systems result shared with cuPF.
- **Reject the strong claim** ("a new / faster GPU sparse solver"): batched shared-symbolic is prior
  art (now a cuDSS primitive) and the speed claim is the one thing that genuinely needs the
  head-to-head.
- **The correct sentence about comparison:** the head-to-head decides the **competitive standing**
  of D1 (and any "faster" wording) — it does **not** decide whether the work contributes. Saying
  "no contribution because we didn't compare" is a category error: B1, B2, C2, D2 are contributions
  *because they are novel and in-house-testable*, and the comparison would only add a competitive
  claim on top.

**One line:** 기여는 있다 — "더 빠른 솔버"가 아니라 **(B2) 정량화된 tiny-K 텐서코어/FP64-master
음의 결과**와 **(B1) FP32-native가 NR 루프에 충분하다는, 선행연구의 FP64 가정을 반증하는 결과**가
핵심이며, 둘 다 **외부 솔버 없이 자체 실험으로 검증 가능**하다. head-to-head는 D1의 "경쟁력"과
"더 빠르다" 문장에만 필요한 게이트일 뿐, 기여의 존재 여부를 가르는 관문이 아니다.

## 6. The experiments that turn these claims into validated results

1. **B1 (no external solver):** FP32-native vs FP64 factor across the case suite; report NR
   iteration count, final |ΔV| / mismatch, and the size/conditioning boundary where FP32 fails
   (70k known to diverge). Deliverable: an accuracy-vs-size frontier.
2. **B2 (no external solver):** sweep front width; for each, FP32-FMA vs FP16-WMMA vs FP64-master-TC
   trailing time + resulting accuracy; locate the K where TC would break even (and show K=nc never
   reaches it on these grids).
3. **C2 (no external solver):** per-iteration host wall-time and H↔D bytes, graph vs non-graph.
4. **D1/D2 (in-house done; competitive pending):** keep the per-kernel ablations; add **MAGMA
   vbatched** and **cuDSS ≥ v0.6 (FP32 + IR)** as the fair library baselines for the competitive row.
5. **Competitive gate (external):** cuDSS ≥ v0.6 uniform-batch (FP32+IR), KLU/NICSLU, and ideally
   Wang [5]; on the *same* Jacobians, same accuracy target.

Sources: see [`related-work-and-contribution.md`](related-work-and-contribution.md) §6 (full
citations [1]–[10]); primary texts for [5][7] and STRUMPACK read in full for the quotes above.
