# Mixed precision for power flow — rationale & extension (research plan)

Goal: replace "we tried Mixed and it converged" with a **principled rationale** for *why* solving the
NR linear system in FP32 while keeping the rest in FP64 is correct, and then a plan to test **how far
into FP32 we can push** (Q2: full FP32 floors at ~1e-3 — can we beat that?).

**No build/run in this document — plan only.** The literature grounding below is the 근거; the
experiments are specified but deferred.

---

## 0. Current per-operation precision map (confirmed from code)

`CudaMixedStorage : CudaBatchedStorage<double, float>` — comment: *"상태(Ybus/V/Va/Vm/Sbus/Ibus/F/
normF)는 FP64로 유지하고, Jacobian 값과 solve 결과(d_J_values/d_dx)만 FP32."*

| operation | FP64 profile | **Mixed** | FP32 profile |
|---|---|---|---|
| state V, Va, Vm, Sbus | double | **double** | float |
| `ibus` (Ibus = Ybus·V) | double accumulate | **double accumulate** | float |
| `mismatch` F = S(V) − Sbus | double | **double** | float |
| `mismatch_norm` ‖F‖ | double | **double** | float |
| `jacobian` values | double | **float** | float |
| `factorize` / `solve` (dx) | double | **float** | float |
| `voltage_update` (V ← V − dx) | double | **double state, float dx → cast** | float |
| `prepare_rhs` | — | **down-cast FP64 F → FP32 RHS** | — |

So **Mixed = FP64 residual chain + FP64 state, FP32 Jacobian + FP32 linear solve.** This is exactly
the split whose rationale we must establish.

---

## Part A — Rationale for Mixed (Q1): why FP32 linear solve + FP64 rest is correct

The NR loop with this split is **not** an ad-hoc trick — it is the composition of two well-established
numerical-analysis results. The rationale has two independent legs.

### A1. The linear step may be inexact — *inexact Newton theory*
Newton's equation `J·dx = F` need only be solved **approximately**: with residual
`‖F − J·dx‖ ≤ η_k‖F‖` (η_k = "forcing term"), if `η_k < 1` uniformly the iteration is **locally
convergent**, and the convergence order is governed by how η_k decays (Dembo–Eisenstat–Steihaug 1982
[N1]; Eisenstat–Walker 1996 [N2]).
- An **FP32 factorization + triangular solve** produces a step with relative residual
  `η ≈ O(κ(J)·u_FP32)` — empirically the FP32 linear relres we already see (~1e-3…1e-6). That is a
  **valid forcing term η < 1**, so convergence is *guaranteed by theory*, not luck. The cost is a
  drop from quadratic to linear/superlinear convergence (a few extra NR iterations), which matches
  the doc's note *"Mixed는 Jacobian이 FP32라 ill-conditioned 계통에서 반복이 늘 수 있다."*

### A2. The final accuracy is set by the *residual* precision — *mixed-precision iterative refinement*
cuPF's NR loop is structurally **nonlinear iterative refinement / defect correction**: each iteration
(i) computes the residual `F` in FP64, (ii) solves `J·dx = F` with an FP32 factorization, (iii)
updates the FP64 state `V ← V − dx`. This is the three-precision IR pattern of Carson–Higham [N3][N4]
(Higham–Mary survey [N5]): *factorization precision* `u_f = FP32`, *working/update precision*
`u_w = FP64`, *residual precision* `u_r = FP64`.
- Their result: the attainable accuracy is governed by the **residual precision `u_r`**, not the
  factorization precision `u_f`, **provided `κ(J)·u_f ≲ 1`**. Hence Mixed reaches the **FP64**
  mismatch tolerance even though the factor/solve are FP32 — *because the residual `F` that defines
  convergence is FP64*. The FP32 factor only slows the rate; it cannot move the converged solution.

### A3. Falsifiable predictions (what the rationale commits us to)
1. Mixed final ‖F‖ reaches the **FP64** tolerance (not an FP32 floor). ✔ already observed; the
   rationale explains *why*.
2. Mixed NR iteration count ≈ FP64 + (0…few), rising with κ(J).
3. Mixed **breaks** (loses accuracy / stalls) when `κ(J)·u_FP32 ≳ 1`, i.e. roughly `κ(J) ≳ 10⁷`
   (u_FP32 ≈ 6×10⁻⁸). This predicts the size/conditioning boundary — a *derived* boundary, not an
   empirical surprise.

### A4. Evidence to collect (deferred experiments)
- Per-iteration **forcing term** η = FP32 linear relative residual; show η < 1 every iteration.
- NR iteration count Mixed vs FP64 across the case suite.
- Final ‖F‖ Mixed vs FP64 (expect both hit tol until the κ boundary).
- **κ(J)** (or a cheap estimate, e.g. 1-norm condition estimator) per case near convergence; test
  prediction A3.3: Mixed degrades exactly where `κ(J)·u_FP32 → 1`.
- Map each measurement to [N1]–[N5] so the writeup reads "Mixed works *because* inexact-Newton η<1
  and IR accuracy = u_residual, and breaks *because* κ·u_f→1," not "it worked."

---

## Part B — How far into FP32? Breaking the ~1e-3 full-FP32 floor (Q2)

### B1. Diagnose the floor (hypothesis + analysis)
Full FP32 makes the **residual chain** (Ibus, F, normF) FP32. Near convergence
`S(V) → Sbus`, so `F = S(V) − Sbus` is a **difference of nearly equal O(1) quantities →
catastrophic cancellation** [G1]. The relative error of F is `~ u·‖S‖/‖F‖`, which **diverges as
‖F‖→0**; in FP32 (u ≈ 6×10⁻⁸) this floors the *achievable* ‖F‖ at `~ u·‖S‖ ~ 10⁻³…10⁻⁴` — the
observed wall. Also the FP32 state update accumulates O(u) error per iteration.
**Claim to test:** the 1e-3 floor is set by the **precision of the residual evaluation (Ibus + F +
norm)**, *not* by the Jacobian/factorization. (Mixed already proves the factor can be FP32 without a
floor — so the floor must live in the FP32 residual chain.)

### B2. Per-operation precision "dial" + ablation matrix
Treat precision as a per-op knob over {`ibus`, `mismatch`(F), `mismatch_norm`, `jacobian`,
`factorize/solve`, `voltage_update`}. Ablation: start from full FP32 and promote ops to FP64 one/few
at a time; find the **minimal subset that must exceed FP32** to reach 1e-6.
- **Hypothesis:** the minimal set is `{mismatch F, mismatch_norm}` plus the **accumulator** of
  `ibus` (the SpMV sum) and the `voltage_update` accumulation — i.e. the cancellation-prone sums.
  Jacobian + factor + solve can stay FP32 (this is what Mixed already shows, minus the Jacobian).
- Output: a table (config → attainable ‖F‖, NR iters), naming the cheapest config that breaks 1e-3.

### B3. Make the critical ops cheap *and* accurate (so "mostly FP32" still breaks the floor)
Rather than full FP64 on the critical ops, keep FP32 storage/throughput but recover accuracy in the
cancellation-prone sums:
- **Compensated summation** (Kahan / Neumaier / 2Sum, error-free transforms) in the `ibus` SpMV and
  the `mismatch` reduction → FP32 data movement, ~FP64 accuracy of the sum [G2][N5].
- **FP64 accumulate, FP32 store** for Ibus/F (mixed-precision accumulation — common GPU pattern).
- **Double-single (two-FP32)** residual as a lighter-than-FP64 option.
- **Defect-correction view:** if only `F` (+norm) is computed accurately and everything else is FP32,
  the loop is again IR with `u_r` better than `u_f` — Part A's guarantee applies, so accuracy should
  track `u_r`.
- Each technique's *cost* must be measured: compensated sums add 3–6× the flops of the reduction
  (cheap relative to factor), but the point is they preserve the FP32 *memory/throughput* advantage
  while removing the floor.

### B4. Validation plan (deferred)
- Run the B2 ablation; for each config record attainable ‖F‖ and NR iterations across cases.
- For the compensated-FP32 residual: show it reaches the same ‖F‖ as FP64-residual at FP32 memory
  cost; confirm the floor is gone.
- Tie back to B1: the config that breaks the floor is exactly the one that fixes the residual
  cancellation — confirming the diagnosis.

---

## Risks / things that move the boundary
- **κ(J) of power-flow Jacobians**: depends on grid size, R/X ratios, loading, flat vs warm start.
  The A3.3 boundary is κ-driven, so the precision boundary is *grid-dependent* — characterize it,
  don't claim a universal "FP32 is fine."
- **Per-unit scaling / PV-PQ block scaling** changes the cancellation magnitude in F — note it.
- **Compensated FP32 erodes the throughput win** if overused; the goal is the *minimal* high-accuracy
  set (B2), not "promote everything."
- This is power-flow-scoped (NR forward). The same rationale extends to the adjoint solve, but that
  is a separate (out-of-research-scope) capability.

## Deliverable
A **precision-vs-accuracy frontier with a theoretical prediction it matches**: Mixed is justified by
inexact-Newton (η<1) + IR (accuracy = residual precision) and bounded by κ·u_f; the full-FP32 floor
is the residual-cancellation limit, removable by accurate (FP64 or compensated) residual ops while the
bulk stays FP32. That is the 근거-backed mixed-precision result for batched power flow.

## Sources (rationale)
[N1] Dembo, Eisenstat, Steihaug, *Inexact Newton Methods*, SIAM J. Numer. Anal. 19(2), 1982 —
https://epubs.siam.org/doi/10.1137/0719025
[N2] Eisenstat, Walker, *Choosing the Forcing Terms in an Inexact Newton Method*, SIAM J. Sci.
Comput. 17(1), 1996 — https://epubs.siam.org/doi/10.1137/0917003
[N3] Carson, Higham, *Accelerating the Solution of Linear Systems by Iterative Refinement in Three
Precisions*, SIAM J. Sci. Comput. 40(2), 2018.
[N4] Carson, Higham, *A New Analysis of Iterative Refinement and its Application to Accurate Solution
of Ill-Conditioned Sparse Linear Systems*, SIAM J. Sci. Comput. 39(6), 2017.
[N5] Higham, Mary, *Mixed precision algorithms in numerical linear algebra*, Acta Numerica, 2022 —
https://eprints.maths.manchester.ac.uk/2849/1/paper_eprint.pdf
[G1] Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic* (catastrophic
cancellation) — https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
[G2] Higham, *The accuracy of floating point summation*, SIAM J. Sci. Comput. 14(4), 1993 (Kahan /
compensated summation).
Power-flow precision context: Wang/Fraunhofer [5] chose FP64 "due to the high numerical stability
requirement during the LU refactorization" — the assumption Part A's rationale and Part B's
experiments are designed to delimit. (See `../custom_linear_solver/docs/related-work-and-contribution.md`.)
