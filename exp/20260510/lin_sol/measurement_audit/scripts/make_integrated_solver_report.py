#!/usr/bin/env python3
"""Create an integrated measurement-validity report across all solvers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "measurement_audit" / "results" / "raw_json"
RESULTS = ROOT / "measurement_audit" / "results"
REPORT = ROOT / "report" / "linear_solver_integrated_measurement_validity_v5.md"
SUMMARY_CSV = RESULTS / "integrated_solver_optimality_summary.csv"

CASES = ["case2869pegase", "case9241pegase"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def rhs_norm(case: str) -> float:
    meta = load_json(ROOT / "datasets" / "dumped_systems" / case / "iter_000" / "meta.json")
    return float(meta.get("rhs_norm_2") or 0.0)


def enrich(path: Path, label: str, config: str, interpretation: str) -> dict[str, Any]:
    d = load_json(path)
    case = d.get("case_name")
    if d.get("absolute_residual_2") is None and d.get("relative_residual_2") is not None and case:
        abs_res = float(d["relative_residual_2"]) * rhs_norm(str(case))
        d["absolute_residual_2"] = abs_res
        d["scaled_residual_2"] = abs_res / max(1.0, rhs_norm(str(case)))
    return {
        "case": d.get("case_name"),
        "solver": label,
        "config": config,
        "build_status": d.get("build_status"),
        "converged": d.get("converged"),
        "analysis_ms": d.get("analysis_ms"),
        "factorization_ms": d.get("factorization_ms"),
        "solve_ms": d.get("solve_ms"),
        "total_solver_ms": d.get("total_solver_ms"),
        "repeated_newton_ms": d.get("numeric_factor_plus_solve_ms"),
        "end_to_end_ms": d.get("total_end_to_end_ms"),
        "num_iterations": d.get("num_iterations"),
        "relative_residual_2": d.get("relative_residual_2"),
        "absolute_residual_2": d.get("absolute_residual_2"),
        "scaled_residual_2": d.get("scaled_residual_2"),
        "relative_error_to_x_ref_2": d.get("relative_error_to_x_ref_2"),
        "gpu_residency": d.get("gpu_resident_after_initial_load"),
        "phase_visibility": d.get("phase_visibility") or infer_phase(label),
        "validity_interpretation": interpretation,
        "raw_json": str(path),
    }


def infer_phase(label: str) -> str:
    if label == "cuDSS":
        return "analysis_factor_solve_reusable"
    if label == "cuSolverSP":
        return "monolithic_qr"
    if label in {"AMGx", "Ginkgo"}:
        return "setup_solve_iterative"
    if label == "STRUMPACK":
        return "analysis_factor_solve_host_mpi"
    if label == "SuperLU_DIST":
        return "ABglobal_one_shot_stats"
    return "unknown"


def best_amgx(case: str) -> dict[str, Any]:
    paths = sorted(RAW.glob(f"integrated_amgx_*_{case}_iter000_fp64.json"))
    rows = []
    for p in paths:
        d = load_json(p)
        rows.append((float(d.get("scaled_residual_2") or d.get("relative_residual_2") or 1e300), float(d.get("total_solver_ms") or 1e300), p, d.get("integrated_config_name", p.stem)))
    rows.sort()
    _, _, path, config = rows[0]
    return enrich(path, "AMGx", config, "large-grid-tested; did not converge on representative large cases")


def best_ginkgo(case: str) -> dict[str, Any]:
    paths = [
        RAW / f"v4_ginkgo_gmres_jacobi_{case}_iter000_fp64.json",
        RAW / f"v4_ginkgo_bicgstab_jacobi_{case}_iter000_fp64.json",
    ]
    rows = []
    for p in paths:
        d = load_json(p)
        rows.append((float(d.get("scaled_residual_2") or d.get("relative_residual_2") or 1e300), float(d.get("total_solver_ms") or 1e300), p, d.get("solver_configuration", p.stem)))
    rows.sort()
    _, _, path, config = rows[0]
    return enrich(path, "Ginkgo", str(config), "CUDA executor but Jacobi-only wrapper; did not converge on representative large cases")


def superlu_best(case: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for p in sorted(RAW.glob(f"superlu_opt_*_{case}_*.json")):
        d = load_json(p)
        if d.get("build_status") == "ok" and d.get("converged") is True:
            rows.append((float(d.get("total_solver_ms") or 1e300), p, d))
    rows.sort()
    best_path = rows[0][1]
    best = enrich(best_path, "SuperLU_DIST", "MMD_AT_PLUS_A best observed; acc_offload=0", "best observed is CPU-dominant ABglobal, not GPU evidence")

    gpu_rows = []
    for _, p, d in rows:
        if float(d.get("superlu_acc_offload", d.get("acc_offload", -1)) or -1) == 1:
            gpu_rows.append((float(d.get("total_solver_ms") or 1e300), p))
    gpu_path = gpu_rows[0][1]
    gpu = enrich(gpu_path, "SuperLU_DIST", "MMD_AT_PLUS_A best with acc_offload=1", "CUDA-enabled hybrid ABglobal; slower than offload-off")
    return best, gpu


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        rows.extend([
            enrich(RAW / f"v4_cudss_{case}_iter000_fp64.json", "cuDSS", "general CSR LU, analysis reused", "valid GPU direct baseline and cuPF integration baseline"),
            enrich(RAW / f"v4_cusolver_{case}_iter000_fp64.json", "cuSolverSP", "csrlsvqr monolithic QR", "valid NVIDIA monolithic QR comparison, not reusable factorization"),
            best_amgx(case),
            best_ginkgo(case),
            enrich(RAW / f"v4_strumpack_np1_{case}_iter000_fp64.json", "STRUMPACK", "MPIDist np=1 default no-compression", "valid external MPI/hybrid direct baseline; not full-GPU evidence"),
        ])
        best, gpu = superlu_best(case)
        rows.append(best)
        rows.append(gpu)
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: Any) -> str:
    if v is None or v == "":
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    try:
        x = float(v)
        if not math.isfinite(x):
            return "n/a"
        if abs(x) >= 1000:
            return f"{x:,.1f}"
        if abs(x) >= 1:
            return f"{x:.3f}"
        return f"{x:.3e}"
    except Exception:
        return str(v)


def md_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(t for t, _ in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for _, k in cols) + " |")
    return "\n".join(lines)


def solver_validity_rows() -> list[dict[str, Any]]:
    return [
        {
            "solver": "cuDSS",
            "measurement_status": "valid best GPU baseline",
            "best_effort_status": "strong",
            "gpu_evidence": "yes, GPU-resident cuDSS phases timed with CUDA events",
            "cuPF_relevance": "high",
            "remaining_issue": "Need production integration tuning, but benchmark measurement is sound.",
        },
        {
            "solver": "cuSolverSP/RF",
            "measurement_status": "valid monolithic QR",
            "best_effort_status": "reasonable for available raw CSR API",
            "gpu_evidence": "yes for cuSolverSP QR path",
            "cuPF_relevance": "medium/low",
            "remaining_issue": "QR is monolithic; RF needs supplied LU factors and is not a drop-in Jacobian solver here.",
        },
        {
            "solver": "AMGx",
            "measurement_status": "valid iterative evidence",
            "best_effort_status": "limited finite grid",
            "gpu_evidence": "yes, GPU iterative library",
            "cuPF_relevance": "low as standalone Newton solve",
            "remaining_issue": "GMRES/FGMRES AMG BlockJacobi did not converge on >=5K cases even at 1000 iterations.",
        },
        {
            "solver": "Ginkgo",
            "measurement_status": "valid as Jacobi-only wrapper",
            "best_effort_status": "incomplete",
            "gpu_evidence": "yes, CUDA executor used",
            "cuPF_relevance": "low until advanced preconditioners wired",
            "remaining_issue": "Wrapper only tests GMRES/BiCGSTAB with Jacobi; no ILU/ParILU/ISAI path.",
        },
        {
            "solver": "STRUMPACK",
            "measurement_status": "valid external MPI/hybrid np=1",
            "best_effort_status": "reasonable external baseline",
            "gpu_evidence": "weak/qualified; CUDA build but SLATE warning means not full GPU",
            "cuPF_relevance": "medium as external baseline, low as default",
            "remaining_issue": "np>1 instability in prior audit; no SLATE full-GPU path.",
        },
        {
            "solver": "SuperLU_DIST",
            "measurement_status": "fixed and reclassified",
            "best_effort_status": "reasonable for ABglobal after MMD/offload sweep",
            "gpu_evidence": "no for best result; offload-off is fastest, offload-on is hybrid and slower",
            "cuPF_relevance": "medium as CPU/MPI external baseline, low as GPU default",
            "remaining_issue": "High-level ABglobal one-shot; ParMETIS/COLAMD absent; reusable lower-level path not validated.",
        },
    ]


def write_report(rows: list[dict[str, Any]]) -> None:
    large_cols = [
        ("case", "case"),
        ("solver", "solver"),
        ("config", "config"),
        ("conv", "converged"),
        ("analysis", "analysis_ms"),
        ("factor", "factorization_ms"),
        ("solve", "solve_ms"),
        ("solver ms", "total_solver_ms"),
        ("reuse ms", "repeated_newton_ms"),
        ("scaled resid", "scaled_residual_2"),
        ("GPU/residency", "gpu_residency"),
    ]
    direct = [r for r in rows if r["solver"] in {"cuDSS", "cuSolverSP", "STRUMPACK", "SuperLU_DIST"}]
    iterative = [r for r in rows if r["solver"] in {"AMGx", "Ginkgo"}]
    REPORT.write_text(f"""# Integrated Linear Solver Measurement Validity Report v5

## Purpose

This report integrates the v3 measurement audit, v4 large-case rerun, SuperLU_DIST ordering diagnosis, and the latest solver optimality checks. The goal is not to crown the fastest number in isolation, but to decide which measurements are valid evidence for choosing a sparse linear solver library for cuPF power-flow Jacobian systems.

Representative large cases are `case2869pegase` and `case9241pegase`, iteration 0, FP64.

## Main Large-Case Summary

{md_table(rows, large_cols)}

`reuse ms` is only populated when the wrapper exposes a meaningful repeated Newton-style factorization-plus-solve number. For cuDSS this is the strongest cuPF-relevant timing because symbolic analysis is reusable for a fixed sparsity pattern. For monolithic or one-shot MPI wrappers, blank reuse timing means the current wrapper does not validate that use case.

## Solver Validity Classification

{md_table(solver_validity_rows(), [
    ("solver", "solver"),
    ("measurement status", "measurement_status"),
    ("best-effort status", "best_effort_status"),
    ("GPU evidence", "gpu_evidence"),
    ("cuPF relevance", "cuPF_relevance"),
    ("remaining issue", "remaining_issue"),
])}

## Direct Solver Findings

{md_table(direct, [
    ("case", "case"),
    ("solver", "solver"),
    ("config", "config"),
    ("solver ms", "total_solver_ms"),
    ("reuse ms", "repeated_newton_ms"),
    ("scaled resid", "scaled_residual_2"),
    ("phase visibility", "phase_visibility"),
    ("interpretation", "validity_interpretation"),
])}

Key points:

- `cuDSS` is the cleanest GPU direct-solver evidence: general CSR LU, explicit analysis/factor/solve phases, CUDA-event timing, and validated analysis reuse. Its one-shot time is competitive, and its repeated Newton factor+solve time is sub-millisecond to about 1 ms on the two large cases.
- `cuSolverSP` is valid as a monolithic QR comparison, but it is not a reusable factorization comparison. `cuSolverRF` remains outside this wrapper because it requires externally supplied LU factors.
- `STRUMPACK` solves accurately and is useful as an external MPI/hybrid direct baseline. Prior logs warn that SLATE is required for full GPU support, so it should not be treated as fully GPU-resident evidence.
- `SuperLU_DIST` needed reclassification. NATURAL ordering was not best effort. MMD ordering fixed the catastrophic factorization time. The fastest observed SuperLU_DIST result disables SuperLU GPU offload, so it is a CPU-dominant MPI/ABglobal baseline, not GPU sparse direct evidence.

## Iterative Solver Findings

{md_table(iterative, [
    ("case", "case"),
    ("solver", "solver"),
    ("config", "config"),
    ("conv", "converged"),
    ("iters", "num_iterations"),
    ("solver ms", "total_solver_ms"),
    ("scaled resid", "scaled_residual_2"),
    ("rel err", "relative_error_to_x_ref_2"),
    ("interpretation", "validity_interpretation"),
])}

AMGx was rerun on the large cases with GMRES/FGMRES AMG BlockJacobi at 200 and 1000 iterations. The 1000-iteration runs reduce residuals but still do not converge. Ginkgo large-case evidence remains limited because the existing wrapper only wires Jacobi-preconditioned GMRES/BiCGSTAB, and those did not converge on the large representative systems.

This does not prove AMGx or Ginkgo are poor libraries. It shows that the tested standalone iterative configurations are not robust drop-in Newton linear solvers for these Jacobians.

## SuperLU_DIST Correction

The SuperLU_DIST story changed materially:

1. `METIS_AT_PLUS_A` caused the earlier `Invalid ISPEC` because ParMETIS is not enabled.
2. `NATURAL` ordering runs were valid but not best effort; they caused huge numeric LU factorization time.
3. `MMD_AT_PLUS_A` / `MMD_ATA` made the solve accurate and much faster.
4. `np=1` was best on these single-node matrix sizes; larger MPI rank counts and alternate process-grid shapes were slower.
5. `superlu_acc_offload=0` was fastest:
   - `case2869pegase`: about 15.3 ms solver time.
   - `case9241pegase`: about 52.1 ms solver time.

Therefore SuperLU_DIST should be reported as a valid external CPU/MPI direct-solver baseline after configuration repair, not as a GPU-resident sparse direct solver baseline.

## Updated Annual-Report Interpretation

The evidence for selecting cuDSS should be phrased carefully:

cuDSS remains the most suitable default sparse linear solver for cuPF because it is a direct solver for general nonsymmetric sparse Jacobians, exposes and reuses analysis/factorization structure in a way that matches repeated Newton power-flow solves, runs through a GPU-resident NVIDIA library interface, and has low integration complexity compared with MPI/hybrid external solvers.

The conclusion should not claim that every alternative is simply slower. The better evidence is:

- cuSolverSP is monolithic QR and not the reusable LU/refactorization path cuPF wants.
- AMGx and Ginkgo did not show robust standalone convergence under the finite tested configurations.
- STRUMPACK is accurate but external MPI/hybrid and not full-GPU in this build.
- SuperLU_DIST can be fast after MMD/offload tuning, but the fastest path is CPU-dominant and ABglobal one-shot, not cuDSS-equivalent GPU evidence.

CSV summary: `{SUMMARY_CSV.relative_to(ROOT)}`
""")


def main() -> None:
    rows = collect_rows()
    write_csv(rows)
    write_report(rows)
    print(REPORT)


if __name__ == "__main__":
    main()
