# Milestone M2 — Symbolic Factorize (etree + supernode + dependency)

**Completed:** 2026-05-26 (cycles 5–10)
**Tag:** `m2`

## Goal
Build the symbolic data structures a supernodal factorization needs, verified
against a reference (CXSparse). Numeric work stays on KLU (gate A) until M3.

## What landed (`src/mysolver/symbolic/`)
- `elimination_tree.{hpp,cpp}` — symmetric pattern (A+Aᵀ), elimination tree
  (cs_etree-equivalent), postorder, column counts (cs_counts-equivalent),
  predicted fill + `predicted_fill_perm`.
- `supernode.{hpp,cpp}` — fundamental supernode partition (Liu-Ng-Peyton).
- `schedule.{hpp,cpp}` — supernode etree + level grouping (independent supernodes
  per level = GPU launch unit; forward = levels 0..L, backward reversed).
- Also (cycle 7): `predicted_fill` wired into `analyze` to remove the M1
  double-factor — AMD-favoured matrices now factor once (incl. KPI case6468rte).
- `src/tools/symbolic_stats.cpp` → `report/benchmark/symbolic_stats.csv`.

## Verification
- Unit tests `tests/mysolver` 12/12: etree (tridiagonal path + cs_etree parity),
  column counts (tridiagonal + cs_counts parity), supernodes (tridiagonal +
  dense), schedule (chain + single level).
- **Real-data parity (§243):** etree matches CXSparse `cs_etree` on **all 12**
  benchmark matrices (0 failures).

## Exit criteria (PLAN §M2)
| Criterion | Result |
|---|---|
| Accuracy maintained (gate A: berr ≤ KLU) | ✅ 12/12 |
| smoke + case_ACTIVSg2000 + case3012wp pass | ✅ |
| etree vs cs_etree (rajat27 etc.) | ✅ all 12 match |
| symbolic_stats.csv produced | ✅ |
| Performance (numeric still fallback) | no target; double-factor removed |

## Supernode statistics (AMD ordering)
```
matrix            n      snodes  avg   max    levels
case6468rte(KPI) 12643   6310   2.00   34     64
wang3            26064  17520   1.49  1446    37   <- big dense block (high fill)
onetone2         36057  23306   1.55   114    141
rajat15          37261  31158   1.20   159    119
case8387pegase   14908   7961   1.87    56     70
```
Full table: `report/benchmark/symbolic_stats.csv`.

## Bridge to M3
The level schedule + supernode partition are the inputs to the M3 GPU numeric
factorization (dense supernode blocks via cuBLAS, level-by-level launches). The
KPI is case6468rte factor: 3.2ms (CPU/KLU) → match cuDSS 0.6ms.
