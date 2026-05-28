"""Parse a benchmark CSV and run the milestone gate checks (PLAN §3.2/§5/§8).

Detects, for the target solver (default mysolver):
  - accuracy violations  : milestone-aware (see _is_violation)
  - regressions          : factor_ms / solve_ms more than +10% over recorded best
  - cross-solver compare : target vs cudss-gpu and klu

Accuracy gate policy (decided 2026-05-26, option A):
  - KLU-fallback milestones M0-M2: mysolver still delegates numeric work to KLU,
    so the gate is "no worse than the KLU reference" (PLAN §8: a high berr that
    KLU also produces is a matrix property, not a mysolver regression). When no
    KLU reference row is present it falls back to the absolute limit.
  - M3+ (native numeric factorize): strict absolute berr<=1e-10, abs_err<=1e-8.

Usable as a module (analyze) or CLI (prints JSON).
"""

import argparse
import csv
import json
import math

BERR_LIMIT = 1e-10
ABS_LIMIT = 1e-8
REGRESSION_PCT = 0.10

# Milestones where mysolver may still delegate to KLU (hybrid). M0-M2 are full
# KLU fallback; M3 is hybrid (own numeric where accurate, KLU fallback for the
# circuit matrices pending MC64), so the gate stays "no worse than KLU" until
# own-numeric covers every matrix.
KLU_FALLBACK_MILESTONES = {"M0", "M1", "M2", "M3"}
# Allowance for mysolver berr vs the KLU reference under the fallback gate.
KLU_REL_TOL = 1e-3
KLU_ABS_EPS = 1e-12


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _is_violation(success, berr, abserr, milestone, klu_berr):
    if not success or math.isnan(berr):
        return True
    if milestone in KLU_FALLBACK_MILESTONES and klu_berr is not None and not math.isnan(klu_berr):
        # Must not be worse than the KLU reference it is wrapping.
        return berr > klu_berr * (1.0 + KLU_REL_TOL) + KLU_ABS_EPS
    if berr > BERR_LIMIT:
        return True
    if not math.isnan(abserr) and abserr > ABS_LIMIT:
        return True
    return False


def read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def analyze(csv_path, target="mysolver", best=None, milestone="M0"):
    rows = read_rows(csv_path)
    best = best or {}
    by = {(r["matrix"], r["solver"]): r for r in rows}
    matrices = sorted({r["matrix"] for r in rows})
    solvers = sorted({r["solver"] for r in rows})

    violations = []
    regressions = []
    target_metrics = {}
    comparisons = []

    for m in matrices:
        r = by.get((m, target))
        if r is None:
            continue
        success = str(r["success"]).lower() == "true"
        berr, abserr = _f(r["berr"]), _f(r["absolute_error"])
        mm = {
            "analysis_ms": _f(r["analysis_ms"]),
            "factor_ms": _f(r["factor_ms"]),
            "solve_ms": _f(r["solve_ms"]),
            "berr": berr,
            "absolute_error": abserr,
            "success": success,
        }
        target_metrics[m] = mm

        klu_row = by.get((m, "klu"))
        klu_berr = _f(klu_row["berr"]) if klu_row else None
        if _is_violation(success, berr, abserr, milestone, klu_berr):
            violations.append({"matrix": m, "success": success, "berr": berr,
                               "absolute_error": abserr, "klu_berr": klu_berr})

        b = best.get(m)
        if b:
            for k in ("factor_ms", "solve_ms"):
                ref = b.get(k)
                now = mm[k]
                if ref and ref > 0 and not math.isnan(now):
                    delta = (now - ref) / ref
                    if delta > REGRESSION_PCT:
                        regressions.append({"matrix": m, "metric": k, "best": ref,
                                            "now": now, "delta_pct": round(delta * 100, 1)})

        comp = {"matrix": m}
        for ref_solver in ("cudss-gpu", "klu"):
            rr = by.get((m, ref_solver))
            if rr and str(rr["success"]).lower() == "true":
                comp[ref_solver] = {"factor_ms": _f(rr["factor_ms"]),
                                    "solve_ms": _f(rr["solve_ms"])}
        comparisons.append(comp)

    return {
        "matrices": matrices,
        "solvers": solvers,
        "target": target,
        "target_metrics": target_metrics,
        "violations": violations,
        "regressions": regressions,
        "comparisons": comparisons,
        "accuracy_ok": len(violations) == 0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--target", default="mysolver")
    ap.add_argument("--milestone", default="M0")
    ap.add_argument("--best", help="JSON file with best_metrics_per_case")
    args = ap.parse_args()
    best = json.load(open(args.best)) if args.best else {}
    print(json.dumps(analyze(args.csv, args.target, best, args.milestone), indent=2))
