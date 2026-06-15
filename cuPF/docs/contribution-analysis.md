# cuPF — contribution analysis (power-flow scope, claim-based)

**Scope.** The research question is **AC power flow**, not machine learning. Differentiability
(adjoint / implicit function theorem) and the PyTorch zero-copy bridge are **task requirements** cuPF
implements — they are *capabilities*, not the research contribution, and are out of scope here. The
GPU-kernel "tensor cores don't help" finding is a linear-algebra micro-result, **not** a power-flow
contribution, and is dropped from this framing.

What cuPF *is*, as a power-flow artifact: a **GPU-resident, uniform-batched AC Newton–Raphson
power-flow solver** whose linear step is a **direct multifrontal GPU factorization** specialized to
power-grid Jacobians, runnable **FP32-native / Mixed / FP64**. The contribution question is therefore:

> **What does this work contribute to GPU AC power flow?**

Stance: critical. Claims are stated, attacked, judged; broad ones are rejected. A claim counts as a
contribution iff it is **(a) not in the power-flow prior art** and **(b) testable** — and most are
testable **in-house** (no other solver needed); the head-to-head only gates the *competitive* wording.

---

## 1. Power-flow prior art (the comparison set)

| Work | Many-scenario batched? | Linear step | Precision | Host↔device |
|---|---|---|---|---|
| **Wang/Fraunhofer** (SEGN'21) [5] | **Yes** (1 Ybus pattern, batched NR-PF) | **KLU column-LU refactorization** (Gilbert–Peierls) | **FP64, explicit:** *"double precision … due to the high numerical stability requirement during the LU refactorization"* | CPU-GPU hybrid; **hides** the "DH Memory" transfer with streams |
| **Zhou** (TPS'17) [6] | Yes | whole-system batched LU | not stated | hybrid |
| **MDPI 2024** (Energies 17/6269) | Yes | **fast-decoupled** (not full NR) | — | CPU-GPU hybrid (binary interchange for throughput) |
| **ExaPF.jl** (Argonne) [E1] | single system | **iterative Krylov + Schwarz preconditioner** | FP64 | **GPU-resident, no H↔D in loop** |
| **Swirydowicz** (IJEPES'23) [8] | single (ACOPF KKT) | KLU + refactorization (cuSolverRf/Ginkgo) | FP64 | "GPU-resident" |
| **cuDSS** [9b] | uniform-batch API | supernodal LU | FP32/FP64/dd + IR | device |

Two facts dominate: **(i)** batched power flow is established, but every batched power-flow solver
uses **column-LU / whole-system / fast-decoupled** linear steps — **none is a direct multifrontal GPU
factorization**; **(ii)** every power-flow neighbor is **FP64** (the closest one *explicitly chose*
FP64 for stability), and the GPU-resident ones are single-system iterative.

KLU (column LU) is the power-flow direct-solver standard; the literature even notes GPU direct
solvers "rely on multifrontal/supernodal … **dense blocks** essential for GPU performance" — but
power-grid Jacobians have **no dense blocks**, which is exactly why multifrontal is an unusual (and
hard) choice here.

---

## 2. Candidate power-flow claims — attacked, then judged

Legend: 🟢 in-house testable; 🔴 needs an external solver.

### PF-1 (headline). "FP32-native (and Mixed) batched power-flow Newton is sufficient — within a characterized boundary."
An all-FP32 factorization (no FP64 master) keeps NR convergence (iteration count, final mismatch)
and solution accuracy at FP64 parity on power-grid Jacobians, up to a size/conditioning boundary we
measure; Mixed (FP64 state + FP32 linear step) extends that boundary.
- **Prior art:** the closest batched power-flow solver **explicitly avoided FP32** "due to the high
  numerical stability requirement during the LU refactorization" [5]; the other power-flow neighbors
  are FP64 [6][8][E1]. The literature has **no FP32 power-flow precision study**.
- **Critical view:** FP32 *does* fail on the largest / ill-conditioned grids (must be scoped, not
  claimed universally); "FP32 power flow" has surely been tried casually, but never *characterized*
  as a precision-vs-grid-size frontier with the NR convergence consequence.
- **Validation 🟢:** per case/precision, compare NR iterations, final ‖F‖, and bus-voltage error vs
  FP64; report the boundary where FP32 breaks (e.g. ACTIVSg70k diverges) and where Mixed rescues it.
  **No other solver required.**
- **Verdict: DEFENSIBLE, self-validating — the strongest power-flow contribution.** It is a
  power-flow *precision* result that **contradicts a published, peer-reviewed assumption**.

### PF-2. "A direct *multifrontal* GPU factorization for batched power flow, specialized to the power-grid Jacobian's tiny-front structure."
vs the field's column-LU refactorization [5][6], fast-decoupled [MDPI], and iterative [E1].
- **Prior art:** no batched power-flow solver uses a direct multifrontal GPU factorization; the
  general multifrontal libraries are built for dense fronts and collapse on power-grid graphs
  (latency-bound, ~0.004 % of peak [7]). So making multifrontal work here *requires* the tiny-front
  specialization (warp-per-front, shared-resident, amalgamation, partitioned-inverse solve).
- **Critical view:** "use multifrontal instead of column-LU" is an **implementation choice**, not a
  new power-flow algorithm; and the batched-shared-symbolic *idea* is prior art [5][6]. Its value is
  realized only together with PF-1 (FP32) and a head-to-head.
- **Validation 🟢** (ablate the specializations; characterize the power-grid front-size distribution
  that motivates them) **and 🔴** (vs KLU-refactorization batched [5] / cuDSS-batch for the
  competitive claim).
- **Verdict: DEFENSIBLE as a specialized realization**, weak on its own; it is the *engine* that
  makes PF-1 a working power-flow solver, not a standalone power-flow result.

### PF-3. "Fully GPU-resident *batched* NR power flow — the whole iteration on device, no per-iteration host↔device transfer."
- **Prior art:** the batched power-flow solvers are **CPU-GPU hybrids that *hide* the H↔D
  transaction** [5][MDPI], not remove it; ExaPF is GPU-resident but **single-system iterative** [E1].
  A fully device-resident *batched* NR-PF (whole iteration as one replayed graph; only a scalar
  convergence read on the host) is not located.
- **Critical view:** a **systems** result, and CUDA graphs are a standard tool; the novelty is the
  application to batched NR-PF, not the mechanism.
- **Validation 🟢:** per-iteration host time and H↔D bytes vs a non-graph baseline; cite [5]'s
  "hide the DH Memory transaction" as the cost removed structurally.
- **Verdict: DEFENSIBLE but narrow** (engineering/systems).

### Rejected (stated for completeness)
- **"Batched / single-shared-symbolic power flow"** → prior art [5][6] (now even a cuDSS primitive).
- **"GPU-resident power flow"** as a label → ExaPF [E1], Swirydowicz [8].
- **"Differentiable power flow" / PyTorch op** → out of research scope (task requirement); and prior
  art exists anyway (ExaPF AD; DPF).
- **"Tensor cores don't help the factorization"** → out of scope (GPU linear-algebra micro-result,
  not power flow).

---

## 3. Self-validating vs external-gated

| Power-flow claim | Type | In-house validation | Needs head-to-head? |
|---|---|---|---|
| PF-1 FP32/Mixed suffices for batched NR-PF | precision characterization | NR iters + ‖F‖ + V-error vs FP64, with boundary | **No** |
| PF-2 multifrontal-GPU realization | method/engineering | specialization ablations + front-size profile | **for competitive claim only** |
| PF-3 fully GPU-resident batched NR-PF | systems | per-iter host time / H↔D bytes | **No** |
| "faster than Wang / ExaPF / cuDSS-batch / pandapower" | competitive | — | **Yes (the only true gate)** |

---

## 4. Honest verdict

- **The power-flow contribution is PF-1:** *FP32-native (and Mixed) is sufficient for batched AC
  Newton–Raphson power flow within a characterized grid-size/conditioning boundary* — a precision
  finding that **overturns the FP64-for-stability assumption** the closest batched power-flow solver
  [5] explicitly relied on, and which the power-flow literature has **not** characterized. It is
  validated by an **in-house** experiment (no other solver needed).
- **PF-2 (direct multifrontal GPU engine) and PF-3 (full GPU residency of the batched iteration)**
  are the **enabling realization** — defensible as specialized engineering, novel as a *combination*
  in the power-flow setting, but not standalone power-flow research; PF-2's competitive value is the
  part that needs the head-to-head.
- **Reject** the broad claims (batched power flow, GPU-resident power flow, differentiable power
  flow) — all prior art or out of scope.
- **The comparison's role:** a head-to-head vs Wang's batched solver [5], ExaPF [E1], cuDSS-batch and
  pandapower decides the **competitive standing** of PF-2 and any "faster" wording. It does **not**
  decide whether the work contributes: **PF-1 and PF-3 are contributions because they are novel and
  in-house-testable.**

**한 줄:** 전력조류 연구 기여는 **PF-1 — 배치 AC NR 전력조류에서 FP32-native(및 Mixed)가
(특성화된 계통 규모/조건수 경계 안에서) 충분하다는 정밀도 결과**이며, 이는 가장 가까운 배치
전력조류 솔버 [5]가 명시적으로 채택한 "FP64 for stability" 가정을 **반증**하고 문헌이 다루지 않은
질문이다. 이를 가능케 하는 **직접 멀티프론탈 GPU 엔진(PF-2)** 과 **배치 NR의 완전 GPU 상주(PF-3)**
는 방어 가능한 특화 엔지니어링이고, 모두 **외부 솔버 없이 자체 실험으로 검증 가능**하다(head-to-head는
"더 빠르다"에만 필요).

## 5. Experiments (turn claims into validated results)
1. **PF-1 (in-house):** sweep cases × {FP64, Mixed, FP32}; record NR iterations, final ‖F‖,
   max bus-voltage error vs FP64; locate the size/conditioning boundary (FP32 break, Mixed rescue).
2. **PF-2 (in-house + external):** ablate warp-per-front / shared-resident / amalgamation /
   partitioned-inverse; report the power-grid front-size distribution that motivates each; then
   compare to KLU-refactorization batched [5] and cuDSS-batch on the same Jacobians.
3. **PF-3 (in-house):** per-iteration host wall-time and H↔D bytes, whole-iteration graph vs
   non-graph.
4. **Competitive gate (external):** vs Wang/Fraunhofer batched [5], ExaPF [E1], cuDSS-batch, and
   pandapower, on the same grids at matched accuracy.

## Sources
[5] Wang, Wende-von Berg, Braun, *Fast Parallel NR Power Flow Solver … CPU and GPU*, SEGN'21,
arXiv:2101.02270. [6] Zhou et al., *GPU Batch LU for Massive Power Flows*, IEEE TPS ~2017.
[7] *Spatula*, MICRO'23 (near-planar multifrontal at 0.004 % peak). [8] Swirydowicz et al.,
*GPU-Resident Sparse Direct Linear Solvers for ACOPF*, IJEPES'23, arXiv:2306.14337.
[9b] cuDSS docs/release notes (uniform-batch, FP32/FP64/dd + IR). [E1] ExaPF.jl —
https://github.com/exanauts/ExaPF.jl . MDPI 2024 — Energies 17/24/6269. Full citations:
`../custom_linear_solver/docs/related-work-and-contribution.md` §6.
