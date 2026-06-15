# Measuring precision error scientifically — proof of the FP32 asymmetry

**Question.** Why can the **Jacobian assembly + linear factorization/solve** run in FP32, while the
**state V + power-mismatch** cannot (full FP32 floors at ~1e-3)? This document gives the **metrics**
(standard numerical-analysis quantities, not "we tried it"), the **proof** built from them, and the
**input/output data distribution** that is the root cause. Measured with
`tests/precision_study/measure.py` (RTX-independent — accuracy is precision-determined).

The whole asymmetry reduces to one principle:

> **An operation can be FP32 iff the precision metric that *governs the converged answer* stays
> O(u₃₂) for it.** For the linear solve that metric is the **backward error** (small, κ-independent,
> and self-corrected by the outer loop). For the mismatch it is the **forward error of an
> ill-conditioned cancellation** (large, irreducible, and it *defines* the answer).

`u₃₂ = 2⁻²⁴ ≈ 5.96e-8` (FP32 unit roundoff).

---

## 1. The scientific metrics (and why each is the right one)

| metric | definition | what it bounds | the right tool for |
|---|---|---|---|
| **normwise backward error** ω (Rigal–Gaches) | `ω = ‖b−Jx̂‖ / (‖J‖‖x̂‖ + ‖b‖)` | smallest relative perturbation of (J,b) for which x̂ is exact | **the linear solve** |
| **condition number** κ(J) | `‖J‖‖J⁻¹‖` (1-norm est., Hager) | forward error ≤ κ·ω | amplification of solve error |
| **summation / dot-product condition** κ_sum | `Σ_j|Y_ij V_j| / |Σ_j Y_ij V_j|` per bus | rel. error of computing Ibus_i | **the power mismatch** |
| **running-error bound** (Higham, ASNA ch.3–4) | `|fl(Σxⱼ)−Σxⱼ| ≤ γₙ Σ|xⱼ|`, γₙ≈nu | absolute error of a sum from input rounding | the mismatch floor |
| **self-correction test** | does the next iteration recompute the quantity? | whether a one-step error persists | step (no) vs residual (yes, it's the answer) |

The pairing is the whole point: the linear solve is judged by **backward** error (the inverse problem
"what problem did we actually solve"), the mismatch by **forward** error (the direct problem "how
wrong is the number we output"), because of how each feeds the Newton fixed point.

---

## 2. Proof A — FP32 Jacobian/linear-solve is safe (backward stability + self-correction)

Measured at the converged operating point (random RHS so the residual ≠ 0; this probes the *solver*,
not the mismatch):

| case | κ(J) | κ(J)·u₃₂ | **fp32 ω (backward)** | fp32 forward err | forward / (κ·u₃₂) |
|---|---|---|---|---|---|
| case118 | 5.7e3 | 3.4e-4 | **1.8e-8** | 3.0e-6 | 0.01 |
| case300 | 2.4e5 | 1.5e-2 | **2.0e-8** | 3.2e-4 | 0.02 |
| case1354pegase | 6.3e5 | 3.8e-2 | **1.7e-8** | 1.9e-5 | — |
| case3120sp | 1.4e6 | 8.4e-2 | **3.5e-8** | 8.7e-5 | — |
| case9241pegase | 1.3e7 | 7.9e-1 | **2.5e-8** | 1.9e-4 | — |

**Reading:** the FP32 LU **backward error ω = (1.8–3.5)e-8 ≈ u₃₂ for every case, independent of
κ(J)** (κ spans 5.7e3 → 1.3e7). That is the definition of a **backward-stable** solver (Higham,
*Accuracy and Stability of Numerical Algorithms*, ch. 9): the FP32 step is the *exact* Newton step of
a coefficient matrix perturbed by only ~u₃₂.

Two facts make this sufficient:
1. **Inexact Newton (Dembo–Eisenstat–Steihaug).** The NR step only needs `‖F−J·dx‖ ≤ η‖F‖` with
   η<1. Here η ≈ ω ≈ u₃₂ ≪ 1 — a textbook-good forcing term, so convergence is guaranteed and the
   iteration count is unchanged (measured earlier: Mixed iters = FP64 iters).
2. **Self-correction.** The forward error in the step (≤ κ·u₃₂) does **not** persist: the next
   iteration recomputes F in FP64 and re-solves, so the step error is discarded, not accumulated. It
   only changes the contraction rate, never the fixed point.

⇒ **FP32 in the Jacobian/factorization/solve cannot move the converged solution.** Proven by ω≈u₃₂
(κ-independent) + the self-correction property.

---

## 3. Proof B — FP32 state/mismatch is NOT safe (ill-conditioned cancellation, no correction)

The mismatch `F = S(V) − Sbus`, `S = V·conj(Ybus·V)`, is the quantity we need accurately — it is the
convergence test *and* the RHS. Its evaluation is **ill-conditioned**: each bus current
`Ibus_i = Σ_j Y_ij V_j` sums terms far larger than the net result.

Running-error prediction vs measurement (round only the *inputs* V, Ybus to FP32 at the converged V;
all arithmetic exact FP64):

| case | u₃₂·median(Σ|Y_ij V_j|) | u₃₂·max(Σ|Y_ij V_j|) | **measured fp32 floor** |
|---|---|---|---|
| case118 | 4.8e-6 | 4.6e-5 | **1.6e-5** |
| case300 | 6.0e-6 | 2.9e-4 | **1.3e-4** |
| case1354pegase | 4.9e-5 | 2.1e-3 | **7.6e-4** |
| case3120sp | 1.5e-4 | 3.0e-3 | **2.3e-3** |
| case9241pegase | 3.3e-5 | 3.1e-3 | **1.6e-3** |

**Reading:** the measured FP32 mismatch floor sits between `u₃₂·median` and `u₃₂·max` of the row sum
`Σ_j|Y_ij V_j|`, i.e. it is exactly the **running-error bound** `|ΔF_i| ≈ u₃₂·Σ_j|Y_ij V_j|`. The
relative error is `κ_sum·u₃₂` with κ_sum = Σ|Y_ij V_j|/|Ibus_i| (median κ_sum ≈ 2e2 for case118 up
to ≈3e4 for case3120sp).

Two facts make this fatal:
1. **No backward-error escape.** Unlike the solve, there is no "we solved a nearby problem" — F is the
   output. The relevant metric is the **forward** error, and it is `κ_sum·u₃₂ ~ 1e-3`.
2. **No self-correction.** F *defines* the fixed point / convergence test, so its error floors the
   attainable ‖F‖ directly; later iterations cannot fix it (they trust the wrong F).
3. **Summation tricks don't help.** The error is **input rounding** of V and Ybus *before* the sum;
   Kahan / FP64-accumulate only fix summation *order*. (Confirmed: "FP64-accumulate + FP32-store"
   ablation still floored — see `mixed-precision-findings.md` §B.)

⇒ **FP32 in V/Ybus/mismatch floors the answer at ~u₃₂·κ_sum ~ 1e-3.** Proven by the running-error
bound matching the measured floor + the no-correction property.

---

## 4. The root cause is in the data distribution

Percentiles (50/90/99/max) at the converged operating point — this is *why* κ_sum is large:

| case | \|V\| (unknown) | \|Y_ij·V_j\| terms (summed) | \|Ibus\|=\|S\| (net result) | κ_sum |
|---|---|---|---|---|
| case118 | 0.99 / 1.02 / 1.05 | 15 / 67 / 187 / **389** | 0.37 / 1.5 / 4.9 / 6 | med **217** |
| case1354pegase | 1.04 / 1.07 / 1.08 / 1.11 | 154 / 1.6e3 / 5e3 / **1.7e4** | 0.45 / 2.6 / 12.7 / 32 | med **3.0e3** |
| case3120sp | 1.06 / 1.08 / 1.12 / 1.26 | 85 / 1.4e3 / 1.5e4 / **2.5e4** | 0.047 / 0.19 / 3.1 / 18 | med **3.4e4** |
| case9241pegase | 1.03 / 1.06 / 1.10 / 1.18 | 74 / 1.5e3 / 5.3e3 / **2.6e4** | 0.21 / 2.1 / 9.5 / 34 | med **2.7e3** |

The distribution tells the whole story:
- **\|V\| is tight (≈0.9–1.26)** — the *unknown* is benign and perfectly FP32-representable on its own.
- **The summed terms \|Y_ij·V_j\| are large and wide (up to 10⁴)** — line flows into a bus are big.
- **The net \|Ibus\|=\|S\| is small (median 0.05–0.5)** — those big flows nearly cancel at each bus
  (Kirchhoff: in ≈ out).
- **κ_sum = (big terms)/(small net) ≈ 10²–10⁴** — this cancellation is the amplifier that turns the
  u₃₂ input rounding into a ~1e-3 mismatch error.

So FP32 is not "generally too coarse": **the unknown V is fine; it is the power equation `S=V·conj(YV)`
that is ill-conditioned by physical cancellation, and the linear solve's relevant (backward) error
never sees that cancellation.** Same u₃₂ input rounding, opposite consequence — because the *metric
that governs the answer* differs between the two operations.

---

## 5. Conclusion (the asymmetry, proven)

| operation | governing metric | value under FP32 | corrected later? | FP32 verdict |
|---|---|---|---|---|
| Jacobian + factor + solve | backward error ω | ω ≈ u₃₂ (κ-independent) | yes (re-solved each iter) | **safe** |
| state V + Ibus + mismatch F | forward err of κ_sum-conditioned sum | κ_sum·u₃₂ ≈ 1e-3 | no (it *is* the residual) | **unsafe** |

This is the rigorous version of the Mixed rationale: FP32 belongs exactly on the **backward-stable,
self-correcting** operation (the expensive factorization/solve), and FP64 is mandatory on the
**ill-conditioned, answer-defining** operation (the cheap O(n) mismatch/state). Methods to push the
floor down must reduce κ_sum or use ≥FP64 representation of V/Ybus there — not better summation.

**Scientific measurement recipe (reusable):** for any op, ask *(i)* is the relevant metric backward
or forward error?  *(ii)* what is its condition number (κ(J) for the solve; κ_sum=Σ|terms|/|sum| for
reductions)?  *(iii)* is the error self-corrected downstream? FP32 is admissible only when the
governing metric is O(u₃₂) — which (i)+(ii)+(iii) decide *a priori*, before any timing run.

Harness: `tests/precision_study/measure.py`. Theory: Higham, *Accuracy and Stability of Numerical
Algorithms* (backward error, running-error, summation condition); Dembo–Eisenstat–Steihaug 1982
(inexact Newton); see also [`mixed-precision-findings.md`](mixed-precision-findings.md).
