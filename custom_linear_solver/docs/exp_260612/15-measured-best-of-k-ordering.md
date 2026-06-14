# 15 — Measured best-of-k ordering selection (the doc-14 actionable finding, implemented)

**Date:** 2026-06-13. **Regime:** B=1, fp32 + fp64. RTX 3090.
**Result:** Doc 14 closed the *tailored-ND* hypotheses (all refuted) and named the one real ordering lever: **the production best-of-k selector (`CLS_ORDER_K`, tail-cube proxy) mis-ranks — it picks a seed up to ~8% slower than the trivial default.** This doc *implements and measures the fix*: a **measured** best-of-k that times a real factorize per candidate seed and keeps the fastest. It captures **6–13% at B=1** (the largest ordering win in this experiment) and is correct by construction (no proxy can mis-rank a direct measurement).

## The proxy is anti-informative (decisive, case8387 B=1 fp32, 16 seeds)
Per-seed measured factor time vs the `ordering_cost_model` tail-cube proxy the production `CLS_ORDER_K` selects on (run-to-run noise ±0.2%, so the ranks are real):

| seed | proxy cost | measured ms | note |
|---|---|---|---|
| **14** | 3.28e6 (13th) | **0.3007** | TRUE best |
| 12 | 2.82e6 | 0.3049 | |
| 0 (default) | 2.96e6 | 0.3219 | |
| **11** | **2.38e6 (1st)** | 0.3263 | **proxy PICKS this** |
| 9 | 2.56e6 (2nd) | 0.3503 | proxy's 2nd pick = near-worst |

The proxy's #1 pick (seed 11) is **8.5% slower than the true best** and slower even than the trivial default seed 0; its #2 pick is nearly the worst. Fill does not discriminate either (`front present slots` = 110572 identical across all 16 seeds — METIS holds total fill constant, only the *tree shape* varies, which is exactly what the proxy fails to rank). **The only reliable selector is direct measurement.**

## Available win (B=1 fp32, brute-force 16-seed sweep, ±0.2% noise)
| case | default (seed 0) | best seed | win |
|---|---|---|---|
| case8387 (13K) | 0.3219 | 0.3007 (s14) | **6.6%** |
| ACTIVSg25k (25K) | 0.7689 | 0.6704 (s2) | **12.8%** |
| SyntheticUSA (70K) | 2.1995 | 1.9753 (s13) | **10.2%** |

The best seed is **case-dependent and unpredictable** (s14 / s2 / s13) — no fixed seed or cheap proxy finds it. Note this is *larger* than the doc-04 ceiling estimate (25K 1.05×): the panel-resident kernel (doc 10) reshaped the front-cost landscape since doc 04, widening the ordering spread again.

## Implementation — `CLS_ORDER_MEASURE_K` (`src/solver.cpp` `Solver::analyze`)
Opt-in env, default off; production unchanged. When `CLS_ORDER_MEASURE_K=K>1` and numeric values are present:
- For each seed `[metis_seed .. metis_seed+K)`: build one serial-ND plan (new `PlanBuildOptions::single_seed_only` bypasses the env best-of-k / GPU-ND branches in `build_plan_from_csr`), then **time a real factorize** via a private RAII `State` (`measure_candidate_factor_ms`: `warmup` untimed + median of `reps`, defaults 3/5, `CLS_ORDER_MEASURE_WARMUP`/`_REPS`). The private state mirrors `Solver::factorize`'s value-type dispatch exactly and frees its arenas on scope exit, leaving the solver state untouched.
- Keep the fastest plan; install it into `impl_`. Prints `[analyze] measure-select K=.. -> seed=.. factor_ms=..`.

## Measured selection picks the true best, on every case
| case | selects | factor ms | vs production default (parallel ND, median of 3) |
|---|---|---|---|
| case8387 | seed 14 | 0.303 | 0.329 → **−8%** |
| ACTIVSg25k | seed 2 | 0.670 | 0.753 → **−11%** |
| SyntheticUSA | seed 13 | 1.976 | 2.157 → **−8.4%** |

Exactly the brute-force optimum each time. **Second benefit — determinism:** the production parallel-ND default is nondeterministic (USA default swung 2.06–2.43 across runs, an 18% spread); measured serial selection removes that.

**Precision-adaptive (a feature):** at fp64 the same case8387 selects a *different* seed (s5, not s14) — fp64 fronts are 2× the footprint with different occupancy, so the optimal ordering genuinely differs, and a measured selector tunes to whatever you actually run. fp64 correctness exact (relres 3.1e-14).

## Cost — honest, and why it's near-free *if you were already paying for best-of-k*
Total analyze wall (ACTIVSg25k): default ~2.8s, `MEASURE_K=8` ~18s, `K=16` ~33s. The overhead is dominated by serial-ND **plan building** (~1.9s/candidate), **not** the factorize timing (~6ms/candidate). The existing broken `CLS_ORDER_K=16` proxy path *already pays that same ND-build cost* — measured selection adds only the negligible factorize timing on top, and actually finds the optimum. So:
- **It is the correct drop-in replacement for the broken `CLS_ORDER_K` proxy, at ~the same cost.** If you set best-of-k at all, set `CLS_ORDER_MEASURE_K`, not `CLS_ORDER_K`.
- **For a one-shot solve it is NOT worth it** (tens of seconds of ND building to save microseconds/factorize). It pays off only under heavy same-pattern factorize reuse — time-series power flow (8760 snapshots), N-1 contingency (thousands), Monte-Carlo, or repeated Newton in optimization — which is exactly the regime where any best-of-k is justified.

## Status / reproduce
Env-gated (`CLS_ORDER_MEASURE_K`, default off; production default = parallel-ND K=1, unchanged). Supersedes the broken `CLS_ORDER_K` proxy (doc 14). `CLS_ORDER_MEASURE_K=16 build/custom_linear_solver_run <case> --precision {fp32,fp64} --single-precision fp64 --batch 1 --repeat 20 --warmup 6 --serial-nd --metis-seed 0`; add `--analyze-info` for per-candidate `measure-cand` lines.
