# Mixed precision for power flow — findings (rationale established with evidence)

Companion to [`mixed-precision-plan.md`](mixed-precision-plan.md). This reports the results of the
A/B experiments. **Method:** a controlled NR AC power-flow harness in NumPy/SciPy with **exact
per-operation precision control** (true FP32 sparse LU via SuiteSparse `splu`, FP32 storage emulated
by round-trip casting). State/admittance/Jacobian from PYPOWER (case9–300) and pandapower
(pegase/rte/sp/GB); κ also cross-checked on the **actual cuPF NR Jacobians** in `cls_linsys/*.mtx`.
Accuracy is always reported as the **TRUE** mismatch ‖F‖∞ recomputed in FP64 from the current state
(so a low-precision solver that *thinks* it converged is still scored honestly). Harness:
`/tmp/pf_prec/` (`harness.py`, `driver_A.py`, `driver_B.py`).

Profiles: **FP64** (all double); **Mixed** = cuPF's profile (state V, Ibus, mismatch F, ‖F‖,
update in FP64; **Jacobian values + factorization + triangular solve in FP32**); **FP32** (all single).

---

## Part A — Why "FP32 linear solve + FP64 rest" is correct (rationale + evidence)

| case | n | κ₁(J) | FP64 iters/‖F‖ | **Mixed iters/‖F‖** | FP32 iters/‖F‖ | Mixed η̄ (forcing term) |
|---|---|---|---|---|---|---|
| case30 | 30 | 1.3e3 | 3 / 9.6e-10 | **3 / 9.7e-10** | 49 / 4.7e-6 | 1.4e-6 |
| case118 | 118 | 5.7e3 | 4 / 1.1e-10 | **4 / 1.2e-10** | 49 / 1.3e-5 | 9.1e-7 |
| case300 | 300 | 2.4e5 | 5 / 1.4e-12 | **5 / 8.1e-11** | 49 / 3.0e-4 | 3.0e-5 |
| case1354pegase | 1354 | 6.8e5 | 5 / 3.2e-12 | **5 / 1.6e-10** | 49 / 2.0e-3 | 2.5e-5 |
| case3120sp | 3120 | 1.4e6 | 6 / 1.0e-11 | **6 / 1.5e-11** | 49 / 4.1e-3 | 5.4e-5 |
| case9241pegase | 9241 | 1.3e7 | 6 / 4.6e-12 | **6 / 7.5e-10** | 49 / 2.2e-3 | 1.8e-4 |
| GBnetwork | 2224 | 1.9e7 | 5 / 6.7e-11 | **5 / 1.3e-9** | 49 / 2.3e-2 | 3.5e-4 |

**The split is justified by two classical results, both now confirmed on power-flow Jacobians:**

1. **Inexact Newton (Dembo–Eisenstat–Steihaug 1982).** `J·dx=F` need only be solved to
   `‖F−J·dx‖ ≤ η‖F‖` with a forcing term η<1. The FP32 factorization's measured relative residual is
   **η̄ = 1e-6 … 4e-4 — always < 1**, growing with κ but staying far below 1. So Mixed convergence is
   *guaranteed by theory*, and indeed **Mixed takes the same iteration count as FP64** on every case.
2. **Mixed-precision iterative refinement (Carson–Higham).** The NR loop *is* defect correction:
   FP64 residual, FP32 factor/step, FP64 update. The attainable accuracy is set by the **residual
   precision (FP64)**, not the factor precision (FP32). Measured: **Mixed reaches the same FP64-grade
   ‖F‖ (1e-9…1e-13) as the all-FP64 solver**, while FP32 floors (next section).

**On the conditioning boundary (honest correction).** A worst-case bound says the FP32 *forward*
error is ≲ κ·u₃₂, which would warn of trouble at κ ≳ 10⁷ (κ·u₃₂ ≈ 1). But on the **actual cuPF
Jacobians** the FP32 single-solve forward error is far below that bound:

| (cls_linsys J) | 1354 | 3120sp | 6470rte | 9241 | ACTIVSg25k | ACTIVSg70k |
|---|---|---|---|---|---|---|
| κ·u₃₂ (worst-case) | 0.04 | 0.10 | 0.5 | 0.8 | 1.6 | **42** |
| FP32 solve fwd-err | 1e-5 | 1.3e-4 | 7.7e-5 | 2.9e-4 | 1.7e-4 | **2.2e-3** |
| FP32 solve η | 4e-6 | 1.1e-4 | 7.8e-5 | 3.3e-6 | 2.0e-4 | **1.1e-3** |

Even at ACTIVSg70k (κ·u₃₂=42) the FP32 solve has **η=1.1e-3 < 1** — a valid forcing term. So **the
FP32 *linear solve* (Mixed) is robust across the whole suite, including 70k.** The κ·u₃₂ bound is
conservative; power-flow Jacobians are far more benign than the worst case.

---

## Part B — Can other ops be FP32? Why full FP32 floors at ~1e-3, and what fixes it

Per-operation ablation, final **true** ‖F‖∞ (60-iter cap):

| config | case300 | 1354pegase | 3120sp | 9241pegase |
|---|---|---|---|---|
| FP32 (all) | 4.4e-4 | 1.7e-3 | 4.8e-3 | 2.2e-3 |
| FP32, **state→FP64** only | 4.2e-4 | 1.5e-3 | 3.4e-3 | 3.1e-3 |
| FP32, **residual→FP64** only | 1.0e-4 | 1.2e-3 | 1.9e-3 | 1.8e-3 |
| FP32, **residual = FP32-store / FP64-accumulate** | 1.6e-4 | 2.3e-3 | 3.0e-3 | 4.3e-3 |
| FP32, state→FP64 **and** residual(acc64) | 1.8e-4 | 6.6e-4 | 2.0e-3 | 1.5e-3 |
| **Mixed** (state **and** residual FP64) | **8.1e-11** | **1.6e-10** | **1.5e-11** | **7.5e-10** |

**Findings (the hypothesis was partly wrong — this is sharper):**
- Fixing **state alone** or **residual alone** to FP64 does **not** break the floor; both stay ~1e-3.
- **FP64 accumulation with FP32 storage (the "cheap compensated" fix) does NOT help.** This proves the
  floor is **not a summation/accumulation error** — it is the **FP32 *representation* of the state V
  and the admittances Ybus feeding the power mismatch** `S = V·conj(Ybus·V)`. Near convergence
  `F = S − Sbus` is a tiny difference of O(1) quantities; an FP32 V/Ybus carries ~6e-8 relative error
  that, through that cancellation, floors ‖F‖ at ~1e-3 for transmission-scale grids. Compensated
  summation (Kahan/2Sum/FP64-accumulate) cannot recover information already lost to FP32 rounding of
  the *inputs*.
- Only when **both** the state and the residual are FP64 (= Mixed) does ‖F‖ reach 1e-10.

**So which ops can be FP32, and which cannot:**

| operation | precision-critical? | cost (per iteration) |
|---|---|---|
| state V, voltage update | **must be FP64** | O(n) — cheap |
| Ibus=Ybus·V, mismatch F, ‖F‖ | **must be FP64** | O(nnz)≈O(n) — cheap |
| Jacobian assembly | **FP32 OK** | O(nnz) |
| **factorization + triangular solve** | **FP32 OK** (Part A) | **O(expensive) — the bottleneck** |

The precision-critical operations are exactly the **cheap O(n)/O(nnz)** ones; the operation that
dominates runtime — the sparse **factorization/solve** — is precisely the one that tolerates FP32.

**Answer to "can we overcome the 1e-3 floor while pushing more ops to FP32?":** No — not by
accumulation tricks, because the floor is an input-representation limit, not a summation limit. The
only way below 1e-3 is FP64 (or an equally-expensive double-single) storage of V and Ybus — which is
what Mixed already does. And since those ops are cheap, pushing them to FP32 would save almost nothing
while flooring accuracy. **Mixed is therefore near-optimal: it spends FP64 only where it is both
necessary and cheap, and FP32 exactly on the expensive factorization where Part A proves it is safe.**

---

## Conclusion (the 근거, not "it worked")

- **cuPF Mixed (FP32 factor/solve + FP64 state/residual) is correct by construction**, justified by
  inexact-Newton (measured η < 1 on all cases, incl. ACTIVSg70k) and mixed-precision iterative
  refinement (final accuracy = residual precision = FP64). Evidence: Mixed matches FP64 in both
  iteration count and final ‖F‖ across the suite.
- **Full FP32 floors at ~1e-3** because the FP32 representation of V/Ybus loses the small power
  mismatch to catastrophic cancellation; this is **not** fixable by compensated summation (proven by
  ablation), only by FP64 state+residual.
- **The optimal precision split is Mixed**: FP64 on the cheap state+mismatch (irreducible), FP32 on
  the expensive factorization/solve (safe). Going "more FP32" is both unhelpful (the FP64 ops are
  cheap) and harmful (floors accuracy).
- **For the largest / most ill-conditioned grids (ACTIVSg70k, κ·u₃₂≈42)** the FP32 *solve* is still a
  valid inexact-Newton step (η≈1e-3<1), so Mixed remains the recommended profile; full FP32 fails
  there because of the representation floor, not the solve.

**Caveats / not yet done:** full-NR Mixed was run up to case9241/GBnetwork (κ≈1.3–1.9e7) and the FP32
*solve* validated on the real 25k/70k Jacobians; an end-to-end Mixed NR on ACTIVSg25k/70k needs their
Ybus/Sbus (only the linear systems are in `cls_linsys`). The emulation rounds FP32 by casting (the
standard mixed-precision study method); a GPU run would confirm timings (out of scope here — accuracy
is precision-determined and reproduced faithfully). κ is a 1-norm estimate (Hager/`onenormest`).

Theory sources: Dembo–Eisenstat–Steihaug 1982 (inexact Newton); Eisenstat–Walker 1996 (forcing
terms); Carson–Higham 2017/2018 and Higham–Mary 2022 (mixed-precision IR); Higham 1993 (compensated
summation); Wang/Fraunhofer 2021 (the FP64-for-stability assumption this overturns). Full citations in
[`mixed-precision-plan.md`](mixed-precision-plan.md).
