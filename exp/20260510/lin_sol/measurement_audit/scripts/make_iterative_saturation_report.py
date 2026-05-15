#!/usr/bin/env python3
"""Summarize AMGx/Ginkgo saturation sweep and profile logs."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "measurement_audit"
SWEEP_CSV = AUDIT / "results" / "iterative_saturation_sweep.csv"
SUMMARY_CSV = AUDIT / "results" / "iterative_saturation_summary.csv"
PROFILE_CSV = AUDIT / "results" / "iterative_bottleneck_profile.csv"
REPORT = ROOT / "report" / "iterative_solver_saturation_diagnosis.md"

PROFILE_FILES = {
    "AMGx setup/profile start": AUDIT / "logs" / "iterative_saturation" / "ncu_amgx_gmres_case9241_200.csv",
    "AMGx solve-window": AUDIT / "logs" / "iterative_saturation" / "ncu_amgx_gmres_case9241_200_solve.csv",
    "Ginkgo GMRES setup/profile start": AUDIT / "logs" / "iterative_saturation" / "ncu_ginkgo_gmres_case9241_200.csv",
    "Ginkgo GMRES solve-window": AUDIT / "logs" / "iterative_saturation" / "ncu_ginkgo_gmres_case9241_200_solve.csv",
    "Ginkgo BiCGSTAB mixed window": AUDIT / "logs" / "iterative_saturation" / "ncu_ginkgo_bicgstab_case9241_100_solve.csv",
}


def as_float(value: str):
    if value == "" or value is None:
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def load_sweep() -> list[dict]:
    rows = []
    with SWEEP_CSV.open() as f:
        for row in csv.DictReader(f):
            converted = {k: as_float(v) for k, v in row.items()}
            rows.append(converted)
    return rows


def group_key(row: dict) -> tuple[str, str, str]:
    config = str(row["config"])
    return str(row["library"]), config.rsplit("_", 1)[0], str(row["case"])


def classify(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["requested_max_iters"])
    ok_rows = [r for r in rows if r["status"] == "ok" and isinstance(r["relative_residual_2"], float)]
    if not ok_rows:
        return {
            "limit_type": "no_successful_runs",
            "selected_row": rows[-1],
            "best_row": rows[-1],
            "post_limit_note": "No successful data points.",
        }

    best = min(ok_rows, key=lambda r: r["relative_residual_2"])
    last = ok_rows[-1]
    if best is not last and last["relative_residual_2"] > best["relative_residual_2"] * 1.5:
        return {
            "limit_type": "unstable_after_best_residual",
            "selected_row": best,
            "best_row": best,
            "post_limit_note": "Residual worsens substantially after the best cap; extra iterations are numerically unhelpful.",
        }

    for row in ok_rows:
        if row is last:
            break
        improvement_to_last = row["relative_residual_2"] / last["relative_residual_2"]
        if improvement_to_last < 1.2:
            return {
                "limit_type": "practical_plateau",
                "selected_row": row,
                "best_row": best,
                "post_limit_note": "Remaining sweep improves residual by less than 20%.",
            }

    return {
        "limit_type": "not_saturated_by_2000",
        "selected_row": last,
        "best_row": best,
        "post_limit_note": "Residual is still improving at the largest tested cap, but has not reached tolerance.",
    }


def make_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    out = []
    for (library, config_family, case), group in sorted(groups.items()):
        c = classify(group)
        selected = c["selected_row"]
        best = c["best_row"]
        out.append({
            "library": library,
            "config_family": config_family,
            "case": case,
            "limit_type": c["limit_type"],
            "saturation_or_limit_iters": int(selected["requested_max_iters"]),
            "actual_iterations": int(selected["actual_iterations"]) if isinstance(selected["actual_iterations"], float) else selected["actual_iterations"],
            "time_to_limit_solve_ms": selected["solve_ms"],
            "time_to_limit_total_solver_ms": selected["total_solver_ms"],
            "residual_at_limit": selected["relative_residual_2"],
            "error_at_limit": selected["relative_error_to_x_ref_2"],
            "best_residual_iters": int(best["requested_max_iters"]) if isinstance(best["requested_max_iters"], float) else best["requested_max_iters"],
            "best_residual": best["relative_residual_2"],
            "best_solve_ms": best["solve_ms"],
            "per_iter_solve_ms_at_limit": selected["per_iter_solve_ms"],
            "converged_at_limit": selected["converged"],
            "note": c["post_limit_note"],
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_profile(path: Path, label: str) -> list[dict]:
    ids: dict[str, str] = {}
    if not path.exists():
        return []
    with path.open(newline="", errors="replace") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("==") or row[0] == "ID":
                continue
            if len(row) > 4 and row[0].isdigit():
                ids.setdefault(row[0], row[4])
    counts = Counter(ids.values())
    return [
        {
            "profile": label,
            "captured_launches": len(ids),
            "kernel": kernel,
            "launch_count": count,
        }
        for kernel, count in counts.most_common(20)
    ]


def profile_rows() -> list[dict]:
    rows: list[dict] = []
    for label, path in PROFILE_FILES.items():
        rows.extend(parse_profile(path, label))
    return rows


def fmt(x, digits=3):
    if isinstance(x, bool):
        return "yes" if x else "no"
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if abs(x) >= 1000:
            return f"{x:.1f}"
        if abs(x) < 1e-3 and x != 0:
            return f"{x:.3e}"
        return f"{x:.{digits}f}"
    return str(x)


def report_text(summary: list[dict]) -> str:
    by_case = defaultdict(list)
    for row in summary:
        by_case[row["case"]].append(row)

    lines = [
        "# AMGx and Ginkgo Saturation Diagnosis",
        "",
        "This note extends the linear solver measurement audit by checking where the iterative GPU solvers stop making useful residual progress on the large power-flow Jacobian systems. It reuses the existing dumped systems and wrappers, and it does not touch production cuPF source code.",
        "",
        "## Method",
        "",
        "- Cases: `case2869pegase` iteration 0 and `case9241pegase` iteration 0.",
        "- dtype: FP64.",
        "- AMGx configs: GMRES/FGMRES with AMG aggregation and BlockJacobi smoother.",
        "- Ginkgo configs: GMRES/BiCGSTAB with Jacobi, matching the current wrapper capability.",
        "- Iteration caps: 50, 100, 200, 400, 800, 1000, 1500, 2000.",
        "- Saturation is reported as a practical plateau when all remaining tested caps improve residual by less than 20%; otherwise the largest tested cap is reported as not saturated. If residual worsens substantially after the best point, the limit is classified as instability after best residual.",
        "",
        "## Saturation Summary",
        "",
        "| case | solver/config | limit type | iter cap | solve ms | residual at limit | best residual | note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in sorted(by_case):
        for row in sorted(by_case[case], key=lambda r: (r["library"], r["config_family"])):
            lines.append(
                f"| {case} | {row['config_family']} | {row['limit_type']} | "
                f"{row['saturation_or_limit_iters']} | {fmt(row['time_to_limit_solve_ms'])} | "
                f"{fmt(row['residual_at_limit'])} | {fmt(row['best_residual'])} | {row['note']} |"
            )

    lines.extend([
        "",
        "## What Saturates",
        "",
        "For `case9241pegase`, AMGx reaches a practical plateau very early, around 100 to 200 iterations. The residual is about `1.08e-1` at 100 iterations and about `1.02e-1` at 200 iterations, but only reaches about `9.92e-2` at 2000 iterations while solve time grows from roughly 67-141 ms to more than 1.2 s. This is a convergence-quality limit, not a raw throughput limit.",
        "",
        "For `case2869pegase`, AMGx continues to improve through 2000 iterations, reaching about `5.2e-5`, but it still does not meet the FP64 tolerance and takes about 1.0 s of solve time. It is not saturated by 2000 iterations, but the time-to-accuracy tradeoff is poor for Newton linear solves.",
        "",
        "Ginkgo GMRES with Jacobi improves more smoothly. On `case9241pegase` it becomes a practical plateau around 1500 iterations: residual improves from about `6.58e-4` to `6.06e-4` by 2000, while solve time rises from about 757 ms to about 1009 ms. On `case2869pegase`, it is still improving at 2000 iterations but remains above the requested tolerance.",
        "",
        "Ginkgo BiCGSTAB with Jacobi is unstable on these Jacobians. On `case2869pegase`, the best observed residual occurs near 1000 iterations and then worsens. On `case9241pegase`, it improves at 100 iterations but then diverges badly as the cap increases.",
        "",
        "## Bottleneck Evidence",
        "",
        "Nsight Compute launch sampling was used only for operation identification, not timing, because profiler overhead changes the measured solve time.",
        "",
        "AMGx setup/profile-start captures aggregation hierarchy construction: `size2_selector::*`, CUB scans/sorts, and device fills. The solve-window captures repeated `amgx::csrmv`, BlockJacobi presmoothing, `aggregation::restrictResidualKernel`, AXPBY vector updates, and CUB reductions. The bottleneck is the repeated AMG V-cycle/Krylov iteration work, dominated by sparse matrix-vector products, BlockJacobi smoothing, residual restriction, and vector reductions. On `case9241pegase`, those iterations keep consuming time after the residual has essentially plateaued.",
        "",
        "Ginkgo setup/profile-start captures CSR/COO conversion and cuSPARSE scan/sort kernels plus Jacobi setup. The solve-window captures `csr::abstract_classical_spmv`, Jacobi `kernel::apply`, residual-norm kernels, cuBLAS `dot`/`nrm2` reductions, dense vector updates, and GMRES Hessenberg QR kernels. The bottleneck is the Krylov iteration loop: sparse matvec plus Jacobi application plus global reduction/orthogonalization work. For GMRES, the reduction/orthogonalization component is visible in the dot/nrm2 and Hessenberg QR kernels.",
        "",
        "## Limitation",
        "",
        "The current Ginkgo wrapper only wires Jacobi preconditioning even though the config parser accepts a preconditioner field. Therefore this diagnosis is a fair saturation audit for the implemented Ginkgo path, not a full Ginkgo best-preconditioner study. AMGx residual history was not available from the wrapper because `AMGX_solver_get_iteration_residual` reported that residual history was not recorded, so saturation was inferred by cap sweep.",
        "",
        "## Files",
        "",
        f"- Sweep CSV: `{SWEEP_CSV.relative_to(ROOT)}`",
        f"- Summary CSV: `{SUMMARY_CSV.relative_to(ROOT)}`",
        f"- Profile CSV: `{PROFILE_CSV.relative_to(ROOT)}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    sweep = load_sweep()
    summary = make_summary(sweep)
    profiles = profile_rows()
    write_csv(SUMMARY_CSV, summary)
    write_csv(PROFILE_CSV, profiles)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
