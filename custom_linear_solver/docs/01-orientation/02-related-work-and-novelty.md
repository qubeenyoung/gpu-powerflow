# GPU sparse direct solvers for non-SPD matrices — landscape, why the famous solvers are *not* fast on power-grid Jacobians, and where our work contributes

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: non-SPD GPU 직접 솔버 landscape(인용)·power-grid Jacobian에서 유명 솔버가 느린 이유·우리 기여의 솔직한 novelty 평가.

This document answers four things the team asked for:
1. A surveyed comparison of existing open-source / published GPU sparse **direct** solvers for
   **non-SPD (unsymmetric)** matrices (STRUMPACK, SuperLU_DIST, GLU 3.0, PanGuLu, Basker/Kokkos,
   cuDSS, and the power-systems batched-LU literature).
2. The real reason a **famous, well-understood algorithm (multifrontal)** does not run fast on our
   matrices in those libraries — and whether the gap is genuinely *problem specialization*.
3. An honest statement of **our contribution** (including what is *not* novel).
4. A **logical structure (논리 구조)** for writing this up as research.

Method note: the external claims below come from a fan-out, adversarially-verified literature
search (22 sources fetched, 108 claims extracted, 25 verified by 3-vote majority, 23 confirmed).
Every external claim carries a citation. Claims about **our** solver come from this repo's measured
runs (RTX 3090, sm_86, B=64–128; see `docs/03-optimization-notes/01-fp32-batched-kernel-optimization.md` and
`docs/03-optimization-notes/02-factor-solve-analyze-optimization.md`). Where the literature is silent I say so explicitly.

---

## 1. The landscape (cited)

| Solver | Factorization | Target matrices | Batched (B systems, 1 symbolic)? | FP16 / mixed / tensor-core? | Documented GPU bottleneck |
|---|---|---|---|---|---|
| **STRUMPACK** | Multifrontal LU (unsymmetric) | General SuiteSparse; large dense fronts | No (single system; *level-batches fronts internally*) | No (FP64-centric) | **Latency/launch-bound on small fronts**; custom <32×32 kernel + MAGMA vbatched GETRF to cut launch overhead [1][2] |
| **SuperLU_DIST** | Supernodal right-looking LU | General; distributed | **Yes** — uniform-batched, one symbolic, fixed pivot, pivoting off [3] | **No — FP64 only**; demoed on **fusion plasma**, not power grids [3] | Launch/alloc/transfer latency dominates "many small problems"; needed a *new flattened L/U* struct because supernodal layout is unsuitable for batching [3] |
| **GLU 3.0** | Right-looking column LU (level-scheduled) | **Circuit** simulation matrices | No | No (separate FP32/FP64 runs, not mixed) | Double-U dependency-detection cost + fixed thread allocation [4] |
| **cuDSS** (NVIDIA) | Supernodal LU/LDL/Cholesky | General; NVIDIA markets it **for the US power grid** | **Yes — now has a real uniform-batch-of-many-systems API** (`CUDSS_CONFIG_UBATCH_SIZE/INDEX`, v0.6.0+; non-uniform batch v0.4.0) [9b] | FP32 / FP64 / **double-double** value types (v0.8.0) + **iterative refinement** (IR_TOL, IR_N_STEPS); **no FP16, no tensor-core factor mode** [9b] | Vendor; only NVIDIA-internal OPF numbers, no independent power-grid Jacobian benchmark [9][9b] |
| **Power-flow batched LU** (Wang/Fraunhofer 2021) | Gilbert–Peierls **column LU + refactorization** (not multifrontal) | **Power-grid NR power-flow Jacobians** sharing one Ybus pattern | **Yes** — many NR-PFs, one symbolic/fill structure, batched refactorization | **FP64 (double) — CONFIRMED from full text**, justified by "high numerical stability requirement during the LU refactorization"; no low/mixed precision [5] | Targets GPU under-saturation at small batch; fine-grained (4× threads) for earlier saturation [5] |
| **Batch LU** (Zhou et al. 2017) | Whole-system batched LU (packaged into one big problem) | **Power-grid** N-x security / Monte-Carlo PF, shared topology | **Yes** | Not stated | Up to **76× vs KLU** (1-thread); ~4× vs multicore KLU [6] |
| **PanguLU** (Fu et al., SC'23) | **Regular 2D block-cyclic** sparse direct (blocks stored *sparse*; per-block sparse-BLAS) — **not multifrontal, not large-dense-front** | General SuiteSparse; **distributed multi-GPU** (MPI+OpenMP+CUDA, up to 128 A100) | **No** — only **batched-RHS over one factorization** (`pangulu_gstrs` called repeatedly); **not B-systems-one-symbolic** | FP32 **and** FP64, real/complex (compile-time); **no FP16/mixed/tensor-core** [10] | Distributed-scaling solver (11.7×/18× vs SuperLU_DIST at scale); not a small-front / power-grid target |

**PanguLU resolved (was unverified):** it is a *general-purpose, distributed, 2D-block* sparse LU — **not** power-grid-specialized, **not** multifrontal, and its "batching" is multiple RHS on one factorization, **not** a uniform batch of many distinct systems. It does **not** scoop our batched/FP32/multifrontal/power-grid combination [10]. (Basker/KokkosKernels still produced no confirmed power-grid-relevant claims; treat as out of scope.)

---

## 2. Why the famous algorithm is *not* fast on our matrices — the literature confirms the diagnosis

The team's intuition ("multifrontal is famous; why aren't they as fast as us — is it
specialization?") is **correct, and independently documented**:

**(a) On power-grid / circuit-like graphs, general GPU multifrontal is latency/occupancy-bound, not
compute-bound — by an enormous margin.** An independent third-party profiling of STRUMPACK
(Spatula, MICRO'23) measured **0.3 GFLOP/s = 0.004 % of V100 peak on FullChip** (a near-planar
circuit matrix, largest supernode only 3047) versus **26 % of peak on `atmosmoddd`** (92 % of FLOPs
in supernodes > 4000) [7]. The cause is exactly the structural property our power-grid Jacobians
share: near-planar graphs ⇒ small nested-dissection separators ⇒ **the work scatters across tens of
thousands of small fronts/supernodes** where dense-GPU-LU throughput is near zero (the dense-LU
throughput curve "drops linearly below 10 000 and flattens around 20 000") [7].

> This is *quantitatively the same picture we measured* on our Jacobians: at ACTIVSg25k, **95 % of
> fronts are fsz ≤ 16** (only 5 % of the flops), and the factor is `compute < 60 %` **and**
> `DRAM < 60 %` on every level — i.e. latency-bound. The literature gives the general-solver number
> (0.004 % of peak); our profiling gives the same diagnosis on the same matrix class.

**(b) The standard mitigation — "uniform batching" of same-level fronts — is itself documented as a
crude, load-imbalanced workaround.** Grouping different-sized supernodes into one kernel "causes
load imbalance … poor SM utilization," and level-by-level traversal "incurs additional DRAM
traffic" [7]. STRUMPACK still needs a hand-written `< 32×32` kernel because vendor/MAGMA batched
routines "are not sufficiently optimized" for small fronts [1][2] — the very regime our warp-per-front
kernel targets.

**(c) On power-grid KKT/Jacobian systems specifically, general GPU direct solvers lose to a
single CPU core, and *problem-specialized* solvers win.** Swirydowicz et al. (ACOPF, IJEPES 2023):
across SuperLU, STRUMPACK, SSIDS, PaStiX, cuSolver, **"none … was substantially better than
[1-thread CPU] MA57, and turning on GPU acceleration often resulted in performance losses"** (e.g.
SSIDS 29 s CPU → 198 s GPU; MA57 6 s) [8]. The paper's *own* problem-specialized path — **KLU +
refactorization (cuSolverRf / Ginkgo) reusing a single fixed symbolic pattern + pivot sequence** —
beats MA57 by **1.6–1.9×** [8]. The winning lever is precisely *specialization*: one shared symbolic
factorization + refactorization, not a general dense-front solver.

**Answer to the question:** yes — it is specialization. General multifrontal/supernodal libraries
are engineered for matrices with **large dense fronts** (their FLOPs live there); power-grid
Jacobians have **no large fronts**, so those libraries run at a small fraction of peak and even the
batched fallback is crude. A solver built around the actual structure (many small fronts, one shared
symbolic, accuracy relaxable) is a different machine.

---

## 3. Where our solver sits, and what is / isn't novel

### What we built (measured, this repo)
A **uniform-batched, FP32-native, GPU multifrontal** factor+solve for power-grid NR Jacobians:
one shared symbolic/etree/amalgamation for all B systems; front arena FP32 (no FP64 master);
small-front levels run a **warp-per-front shared-cached kernel**, mid fronts a **shared-resident
block kernel** (per-level-sized shared, L/U-only write-back), the few big separators a
**1024-thread** kernel; solve uses **warp-packed** small levels. Result vs our own FP32 batched
baseline: factor+solve **−25…−28 %** on the large cases (ACTIVSg25k/70k), **−42…−46 % vs FP64**,
solve alone **−30…−41 %**, at **FP32 accuracy** (no iterative refinement needed).

### Honest novelty assessment (the team said "없으면 없다고 보고해도 돼")

| Component | Prior art? | Verdict |
|---|---|---|
| Batched, **single-shared-symbolic** power-grid Jacobian solving | **Yes** — Wang/Fraunhofer [5], Zhou [6], and the refactorization solvers [8] all reuse one symbolic pattern across many power-flow systems | **NOT novel.** This is established in the power-systems GPU literature. |
| Uniform-batched sparse direct factorization (B systems, 1 symbolic, no pivoting) | **Yes** — SuperLU_DIST [3] (FP64, fusion plasma); **and cuDSS now ships a uniform-batch-of-systems API** (v0.6.0+) [9b] | **NOT novel** as a concept — and as of cuDSS v0.6 it is a *vendor primitive*. Do not claim "first batched." |
| Doing it with a **multifrontal / dense-front** GPU kernel (vs column-LU refactorization) | The power-grid batched solvers [5][6] use **column / whole-system LU**, *not* multifrontal fronts | **Partially novel framing** — batched-multifrontal for this target is not documented, but it's an implementation choice, not a new algorithm. |
| **FP32-native fronts** (drop the FP64 master entirely, accept FP32 accuracy) for batched power-grid factorization | **Every nearest neighbor is FP64**: Wang/Fraunhofer is **explicitly FP64** ("high numerical stability requirement during the LU refactorization") [5], SuperLU_DIST batched is FP64 [3], the ACOPF winners are FP64 [8], cuDSS power-grid demo is FP64/double-double [9b]. GLU3.0 runs FP32 but only as a *Maxwell hardware workaround* on **circuit** matrices, single-system [4] | **No prior art found, and the field's stated assumption is the opposite** (FP64 "required"). Still absence-of-evidence, but now the closest competitor's own words frame FP32-native as a genuine departure. |
| **Small-front-specialized GPU kernels** (warp-per-front, shared-resident, adaptive threads) beating the library "uniform batch" / `<32×32` fallback | STRUMPACK's `<32×32` naive kernel [1][2] and "crude batching" [7] are the *baseline we improve on* | **Engineering contribution**, well-motivated by [7]; the specific kernel design is ours. |
| **FP16 / tensor cores** help this factorization | The survey found **no source** quantifying tensor-core efficiency at small supernode width K (≤16–32) | **Our empirical negative result is genuinely new data** (see below). |

**Bottom line on novelty:** the *headline idea* (batched, shared-symbolic, power-grid-specialized
GPU solve) is **not novel** — it has clear prior art [5][6]. What has **no exact prior art** in the
surveyed literature is the **specific combination**: *batched + FP32-native + multifrontal-dense-
front + small-front-specialized kernels* for power-grid Jacobians. That is a real but **narrow**
gap, and it is an absence-of-evidence claim, not a proof.

### The most defensible *scientific* contribution
Not "a faster solver" (we have only beaten **our own** FP32 baseline; we have **not** run
head-to-head vs cuDSS / KLU / STRUMPACK / the batched power-flow solvers — that is required before
any comparative speed claim). The defensible contributions are two **characterization** results:

1. **A negative result the literature is missing (RQ4 gap):** on power-grid Jacobians the rank-`nc`
   trailing GEMM has contraction depth `K = nc ≤ 16–20`, which is **too thin for FP16/WMMA tensor
   cores to beat the FP32 FMA loop**, and an FP64-master TC path is *slower than pure FP32* because
   the master's cast/writeback/double-atomic/2× memory overhead dominates the latency-bound work
   (FP16 trailing even diverges at 25k/70k). The survey confirmed **no published source quantifies
   tensor-core efficiency in this small-K sparse-direct regime** — so "tensor cores do not help
   power-grid multifrontal factorization, and why" is a publishable, currently-undocumented finding.

2. **The lever is FP32-native + small-front kernel engineering, not precision tricks.** We show the
   factor/solve is latency/occupancy-bound (matching [7]), and that the wins come from removing the
   FP64 master and from warp-per-front / shared-resident / warp-packed kernels — i.e. attacking the
   exact bottleneck [7] names but does not solve well (its `<32×32` naive kernel + crude batching).

---

## 4. Proposed logical structure (논리 구조) for the writeup

A defensible paper/report argues a **chain**, each link backed by a measurement or a citation:

1. **Motivation.** Power-system applications (contingency screening, time-series / Monte-Carlo PF,
   stochastic OPF) solve **many** Newton Jacobians that **share one sparsity pattern** → a
   uniform-batched solve is the natural primitive. [5][6]
2. **Gap.** General GPU sparse direct solvers are built for large dense fronts and run at ~0.004 %
   of peak on power-grid-class graphs [7]; on power-grid KKT/Jacobians they don't beat a single CPU
   core and GPU often *hurts* [8]. → A specialized solver is needed (this is the accepted finding we
   build on, not our claim).
3. **Diagnosis (our measurement).** Profile the batched multifrontal: 95 % of fronts are small, every
   level is `compute<60 % ∧ DRAM<60 %` = latency/occupancy-bound; the trailing GEMM (where tensor
   cores would apply) is a minority of time and has `K=nc≤20`. → Precision/tensor-core levers are
   inert here; the lever is launch/occupancy/traffic.
4. **Negative result (contribution 1).** Tensor cores and an FP64-master mixed path **lose to pure
   FP32** on this workload; quantify why (thin K, master overhead, divergence at scale). Literature
   has no counter-evidence (RQ4 gap). [this repo]
5. **Method (contribution 2).** FP32-native fronts + per-regime kernels: warp-per-front small
   levels, shared-resident mid fronts (L/U-only write-back), high-thread big separators, warp-packed
   solve; one shared symbolic/amalgamation across B. Tie each kernel to the bottleneck it removes.
6. **Evaluation.** (a) vs our FP64/Mixed/FP32 baselines — done (−25…−46 %). (b) **vs cuDSS, KLU,
   and the batched power-flow solvers [5][6] — REQUIRED, not yet done.** (c) accuracy: FP32 residual
   suffices for the NR outer loop (no IR) — needs the IR-vs-FP32 ablation [RQ6 gap].
7. **Positioning / novelty.** State plainly: batched shared-symbolic power-grid solving is prior art
   [5][6]; our delta is the FP32-native multifrontal + small-front kernels + the tensor-core negative
   result. Do **not** claim the batched concept as new.

---

## 5. What must still be done before any comparative claim (honesty gate)

- **Head-to-head benchmarks** vs cuDSS (current version), KLU/NICSLU (CPU sparse-LU king for
  circuits/power), and ideally the Wang/Fraunhofer batched solver. We currently only beat our own
  baselines.
- ~~**Resolve the precision question in prior art [5]**~~ **RESOLVED (full text):** the Fraunhofer
  batched power-flow solver is **FP64** — "On both test platforms, double precision float number is
  used, due to the high numerical stability requirement during the LU refactorization" [5]. This
  *strengthens* the FP32-native angle: the closest competitor explicitly chose FP64 for stability,
  so demonstrating FP32 suffices for the NR outer loop is a substantive (not cosmetic) departure.
- **IR-vs-FP32 accuracy ablation** on the NR outer loop (does FP32 factor change NR iteration count
  / convergence vs FP64?) — currently unaddressed in the literature (RQ6 gap).
- **Generality:** results are RTX 3090 / MATPOWER+ACTIVSg cases; confirm the front-size distribution
  and gains hold on other grids and GPUs.

---

## 5b. 두 경우의 논리 완성 (both-cases logic) — "기여 있음" vs "기여 없음"

The team asked to complete the argument **for both outcomes**, then judge. Here are the two
self-consistent cases, each as strong as the evidence honestly allows.

### CASE A — "기여 없음" (no defensible contribution)
*Argued as strongly as the evidence permits.*

1. **The headline idea is fully prior art.** Batched, single-shared-symbolic power-grid Jacobian
   solving is Wang/Fraunhofer [5] and Zhou [6]; uniform-batched sparse direct factorization is
   SuperLU_DIST [3] and, **as of cuDSS v0.6.0, a shipped NVIDIA vendor primitive** [9b]. None of the
   batched concept is ours.
2. **"Multifrontal instead of column-LU" is an implementation choice, not a new algorithm.** No new
   ordering, no new pivoting theory, no new complexity result.
3. **We have not beaten anyone but ourselves.** Every speedup number (−25…−46 %) is vs *our own*
   FP64/Mixed/FP32 baselines. No head-to-head vs cuDSS / KLU/NICSLU / STRUMPACK / Wang's solver
   exists. A "faster solver" claim is therefore unsupported.
4. **The FP32-native novelty is absence-of-evidence.** We found no FP32-native batched power-grid
   multifrontal solver — but a literature *survey* not finding one is not proof none exists, and FP32
   sparse LU itself is old (GLU3.0 [4]).
5. **The tensor-core negative result is "folklore."** The multifrontal-GPU community already routes
   only large fronts to cuBLAS/tensor-core GEMM and hand-writes small-front kernels [1][7]; that small
   fronts don't use tensor cores well is implicitly known.
→ **Conclusion of Case A:** the work is solid *engineering* on a known problem with known primitives;
   it is not a research contribution until a head-to-head win exists.

### CASE B — "기여 있음" (a defensible, narrow contribution)
*Argued as strongly as the evidence permits.*

1. **A combination with no located prior art.** *batched + FP32-native (no FP64 master) + multifrontal
   dense-front + small-front-specialized kernels + power-grid Jacobians.* Each nearest neighbor misses
   ≥1 axis: Wang [5]=FP64+column-LU; SuperLU_DIST [3]=FP64+fusion-plasma; cuDSS [9b]=FP64/dd vendor
   black-box, no small-front kernels published; GLU3.0 [4]=FP32-but-circuit-single-system; PanguLU
   [10]=FP64/FP32-but-general-distributed-2D-block, batched-RHS-only. The scoop check (2021–2026)
   found nothing closer.
2. **The FP32-native result contradicts the field's stated assumption.** The closest competitor [5]
   explicitly chose FP64 "due to the high numerical stability requirement during the LU
   refactorization." Demonstrating FP32-native fronts suffice for the NR outer loop (relres ≤ 1e-3,
   no IR) is a *substantive* departure, not a cosmetic one — it falsifies an assumption a peer-
   reviewed solver baked in.
3. **Two genuine characterization contributions (not "a faster solver"):**
   - **C1 — negative result, quantified, on the right matrix class.** On power-grid Jacobians the
     trailing GEMM has contraction depth `K = nc ≤ 16–20`; FP16/WMMA loses to the FP32 FMA loop, and
     an FP64-master mixed path is *slower than pure FP32* (cast/writeback/double-atomic/2× memory),
     with FP16 trailing diverging at 25k/70k. The "folklore" (Case A pt 5) is *qualitative*; **no
     source quantifies tensor-core inefficiency in this small-K sparse-direct power-grid regime** —
     turning folklore into a measured, reproducible result *is* the contribution.
   - **C2 — the lever is kernel engineering against a documented bottleneck.** Spatula [7] measured
     the bottleneck (0.004 % of peak on near-planar FullChip) but its mitigation is "crude batching";
     STRUMPACK still falls back to a naive `<32×32` kernel [1]. Our warp-per-front / shared-resident /
     warp-packed kernels attack exactly that regime and are tied 1:1 to the bottleneck they remove.
4. **Reproducible, structurally-grounded measurements** (front-size distribution, per-level
   compute/DRAM occupancy) back every claim — this is publishable *characterization* science even
   before a head-to-head.
→ **Conclusion of Case B:** the contribution is real but **narrow and of the "characterization +
   specialized-engineering" type**, not a "new fastest solver" claim.

### VERDICT (정직한 판정)
**Case B is the correct framing — but only at Case B's stated altitude, and only after one gate.**

- **Reject the strong claim** ("a novel/faster GPU sparse solver"). Case A is decisive against it:
  batched-shared-symbolic is prior art (now even a cuDSS primitive [9b]), and we have **no head-to-
  head win**. Do not write that sentence.
- **Accept the narrow claim.** The *combination* is unscooped, and the **two characterization
  results survive every check**: (C1) the quantified small-K tensor-core/FP64-master negative result —
  which the survey confirms is **undocumented** — and (C2) FP32-native + small-front kernel
  engineering that beats the FP64/Mixed paths and is motivated by the exact bottleneck the literature
  names but does not solve. The newly-confirmed fact that the closest competitor [5] **chose FP64 for
  stability** converts our FP32 result from "absence of evidence" into "contradicts a published
  assumption" — the single strongest sentence available to us.
- **The one gate that decides A-vs-B in print:** a **head-to-head** vs cuDSS (≥v0.6 uniform-batch,
  FP32 + IR — now the fair, strongest baseline since it does *exactly* batched-many-systems) and vs
  KLU/NICSLU. If we win or even match at FP32 accuracy, Case B stands as written. If cuDSS's batch
  API already matches us, the contribution collapses to **C1 alone** (still publishable as a negative
  result, but not a method paper). **C1 does not depend on the gate; C2's value does.**

**One-line judgment:** 기여는 **있다 — 단 "더 빠른 솔버"가 아니라 "정량화된 음의 결과(C1) + 병목-
조준 FP32-native 커널 엔지니어링(C2)"의 좁은 기여로서**, 그리고 **cuDSS≥v0.6 / KLU와의 head-to-head
한 판**을 통과해야 C2가 산다. C1은 그 관문과 무관하게 이미 방어 가능.

---

## 6. Sources
[1] Ghysels & Synk, *High-performance sparse multifrontal solvers on GPUs* (STRUMPACK), Parallel
Computing 2022 — https://www.sciencedirect.com/science/article/am/pii/S0167819122000059
[2] STRUMPACK GPU multifrontal w/ MAGMA vbatched, IJHPCA 2024/25 —
https://journals.sagepub.com/doi/full/10.1177/10943420241288567
[3] Boukaram, Hong, Liu, Shi, Li, *Batched Sparse Direct Solver … in SuperLU_DIST*, IJHPCA 2024 —
https://www.researchgate.net/publication/382142370
[4] Peng & Tan, *GLU3.0*, arXiv:1908.00204 (IEEE D&T 2020) — https://arxiv.org/abs/1908.00204
[5] Wang, Wende-von Berg, Braun, *Batched … power flow on GPU*, SEGN 2021, arXiv:2101.02270 —
https://www.sciencedirect.com/science/article/abs/pii/S2352467721000540
[6] Zhou et al., *GPU-Based Batch LU-Factorization Solver for … Massive Power Flows*, IEEE Trans.
Power Systems ~2017 — https://www.researchgate.net/publication/313235620
[7] *Spatula* (independent STRUMPACK profiling), MICRO'23 —
https://dl.acm.org/doi/fullHtml/10.1145/3613424.3623783
[8] Swirydowicz et al., *Linear solvers for power grid (ACOPF)*, IJEPES 2023, arXiv:2306.14337 —
https://arxiv.org/pdf/2306.14337
[9] NVIDIA cuDSS for the US power grid (vendor blog) —
https://developer.nvidia.com/blog/nvidia-cudss-library-removes-barriers-to-optimizing-the-us-power-grid/
[9b] cuDSS docs + release notes (uniform-batch API `CUDSS_CONFIG_UBATCH_SIZE` v0.6.0; FP32/FP64/
double-double value types + iterative refinement v0.8.0) — https://docs.nvidia.com/cuda/cudss/ ,
https://docs.nvidia.com/cuda/cudss/release_notes.html
[10] Fu, Zhang, Wang, …, Jin, W. Liu, *PanguLU: A Scalable Regular 2-D Block-Cyclic Sparse Direct
Solver on Distributed Heterogeneous Systems*, SC'23, DOI 10.1145/3581784.3607050 —
https://dl.acm.org/doi/10.1145/3581784.3607050 ; repo
https://github.com/SuperScientificSoftwareLaboratory/PanguLU

*Caveats:* STRUMPACK/SuperLU_DIST numbers are developer self-reports on vendor-selected matrices;
the STRUMPACK-vs-cuDSS 1.9× used an early cuDSS v0.1.0 and is stale; the "no FP32-native prior art"
finding is absence-of-evidence over the surveyed sample, not proof; two MDPI contingency-symbolic
claims were refuted (votes 1-2, 0-3) and excluded.
