# Milestone M1 — Reordering (METIS ND, min-fill selection)

**Completed:** 2026-05-26 (cycles 2–4)
**Tag:** `m1`

## Goal
mysolver generates its own fill-reducing ordering (cuDSS Analysis step 1). Numeric
factorization stays delegated to KLU (gradual replacement).

## What landed
- `src/mysolver/reordering/metis_nd.{hpp,cpp}` — symmetric pattern (A+Aᵀ) → METIS
  `METIS_NodeND` ordering; `klu_metis_user_order` callback (KLU `ordering=3`).
- mysolver keeps two candidate orderings — KLU AMD and METIS ND — and
  `factorize()` picks the one with smaller actual fill (`lnz+unz`).
- `--matrices NAME[,..]` explicit verification set in the benchmark; run_cycle
  accepts a comma-separated `active_matrix_set`.
- Fill-in (`lnz+unz`) exposed in both mysolver and klu CSV messages (§204).

## Key finding
METIS ND is **not uniformly better than AMD**. It wins on large-fill matrices
(wang3 fill −31%, rajat15 −11%) but is worse on the onetone2 circuit matrix
(2.08× fill, berr 4.43e-7 > KLU 1.31e-7 → would fail gate A). Min-fill selection
resolves this: mysolver fill ≤ KLU fill on every matrix.

## Exit criteria (PLAN §M1)
| Criterion | Result |
|---|---|
| Accuracy maintained (gate A: berr ≤ KLU) | ✅ 12/12 |
| mysolver uses its own ordering | ✅ (ND: wang3, rajat15; AMD: other 10) |
| Full set passes (5 SuiteSparse + matpower case30..case8387pegase) | ✅ all success |
| Performance: no regression (host-bound) | ⚠️ see below |

## Known interim cost (accepted by user, option A)
The min-fill selection factors **both** candidate orderings (~2× factor time,
+46–239% vs m0). PLAN §M1 sets no absolute perf target and §3.2 sanctions
intended regressions; the change is a net **structural** improvement (fill ≤ KLU,
ND wins kept, accuracy preserved/improved). The cost is to be removed in **M2**,
where the symbolic stage (elimination tree, CXSparse `cs_etree`/`cs_counts`)
provides a reliable, BTF-aware cheap fill prediction so only the winning ordering
is factored. Merged to master as a documented exception to strict
optimal-on-master, with the cost tracked in `STATE.regression_alerts`.

## Numbers (M1 full set, mysolver berr vs KLU; ordering chosen)
```
wang3     ND  berr 4.18e-15  fill -31%      onetone2  AMD berr 1.31e-7 (=KLU)
rajat15   ND  berr 8.56e-11  fill -11%      memplus   AMD berr 8.0e-16 (=KLU)
rajat27   AMD berr 1.53e-5 (=KLU)           case30/118/1197/ACTIVSg2000/3012wp/
                                            6468rte/8387pegase  AMD (=KLU)
```
Artifacts: `report/benchmark/cycle_0004.csv`.
