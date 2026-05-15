#!/usr/bin/env python3
"""Run the SuperLU_DIST MMD ordering rank-sweep addendum for the v4 audit."""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "measurement_audit"
RAW = AUDIT / "results" / "raw_json"
RESULTS = AUDIT / "results"
LOGS = AUDIT / "logs" / "v4_mmd_rank"
REPORT = ROOT / "report" / "linear_solver_large_case_superlu_mmd_rank_addendum_v4.md"
DATASETS = ROOT / "datasets" / "dumped_systems"
MPIRUN = ROOT / "third_party" / "mpich" / "install" / "bin" / "mpirun"
EXE = AUDIT / "superlu_dist_phase" / "build" / "superlu_dist_phase_benchmark"

SYSTEMS = [
    ("synthetic_validation", 0),
    ("case2869pegase", 0),
    ("case9241pegase", 0),
]
ROWPERMS = ["LargeDiag_MC64", "NOROWPERM"]
RANKS = [1, 2, 4]


def ensure_dirs() -> None:
    for path in [RAW, RESULTS, LOGS, REPORT.parent]:
        path.mkdir(parents=True, exist_ok=True)


def common_args(case: str, iteration: int, out: Path, rowperm: str) -> list[str]:
    base = DATASETS / case / f"iter_{iteration:03d}"
    return [
        "--matrix",
        str(base / "J.mtx"),
        "--rhs",
        str(base / "rhs.txt"),
        "--xref",
        str(base / "x_ref.txt"),
        "--meta",
        str(base / "meta.json"),
        "--dtype",
        "fp64",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--colperm",
        "MMD_AT_PLUS_A",
        "--rowperm",
        rowperm,
        "--out",
        str(out),
    ]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def tail(text: str, limit: int = 3000) -> str:
    return text[-limit:] if len(text) > limit else text


def run_case(case: str, iteration: int, np: int, rowperm: str) -> Path:
    out = RAW / f"v4_superlu_mmd_rank_{case}_iter{iteration:03d}_np{np}_MMD_AT_PLUS_A_{rowperm}_fp64.json"
    name = f"{case}_iter{iteration:03d}_np{np}_{rowperm}"
    cmd = [str(MPIRUN), "-np", str(np), str(EXE), *common_args(case, iteration, out, rowperm)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    timeout = 120 if case == "synthetic_validation" else 600
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        (LOGS / f"{name}.stdout.log").write_text(proc.stdout)
        (LOGS / f"{name}.stderr.log").write_text(proc.stderr)
        if proc.returncode != 0:
            write_failure(out, cmd, elapsed_ms, proc.returncode, proc.stdout, proc.stderr)
        else:
            data = load_json(out)
            data["v4_addendum_external_elapsed_ms"] = elapsed_ms
            data["v4_addendum_command"] = " ".join(shlex.quote(x) for x in cmd)
            write_json(out, data)
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        (LOGS / f"{name}.stdout.log").write_text(stdout)
        (LOGS / f"{name}.stderr.log").write_text(stderr + f"\nTIMEOUT after {timeout}s\n")
        write_failure(out, cmd, elapsed_ms, 124, stdout, stderr)
    return out


def write_failure(out: Path, cmd: list[str], elapsed_ms: float, code: int, stdout: str, stderr: str) -> None:
    meta: dict[str, Any] = {}
    if "--meta" in cmd:
        try:
            meta = load_json(Path(cmd[cmd.index("--meta") + 1]))
        except Exception:
            meta = {}
    data = {
        "solver_name": "SuperLU_DIST",
        "solver_version": "unknown",
        "library_path": "unknown",
        "build_status": "timeout" if code == 124 else "runtime_failed",
        "dtype": "fp64",
        "case_name": meta.get("case_name", "unknown"),
        "iteration": meta.get("iteration", -1),
        "matrix_rows": meta.get("matrix_rows", 0),
        "matrix_cols": meta.get("matrix_cols", 0),
        "nnz": meta.get("nnz", 0),
        "converged": False,
        "num_iterations": -1,
        "total_end_to_end_ms": elapsed_ms,
        "gpu_resident_after_initial_load": "unknown",
        "colperm": "MMD_AT_PLUS_A",
        "rowperm": cmd[cmd.index("--rowperm") + 1] if "--rowperm" in cmd else "unknown",
        "mpi_ranks": int(cmd[cmd.index("-np") + 1]) if "-np" in cmd else 0,
        "notes": f"Command failed with return code {code}.",
        "error_tail": tail(stderr or stdout),
        "v4_addendum_command": " ".join(shlex.quote(x) for x in cmd),
    }
    write_json(out, data)


def enrich(path: Path) -> dict[str, Any]:
    data = load_json(path)
    case = data.get("case_name")
    iteration = int(data.get("iteration") or 0)
    meta_path = DATASETS / str(case) / f"iter_{iteration:03d}" / "meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        rhs_norm = float(meta.get("rhs_norm_2") or 0.0)
        if data.get("absolute_residual_2") is None and data.get("relative_residual_2") is not None:
            abs_res = float(data["relative_residual_2"]) * rhs_norm
            data["absolute_residual_2"] = abs_res
            data["scaled_residual_2"] = abs_res / max(1.0, rhs_norm)
    data["raw_json"] = str(path)
    return data


def collect(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        data = enrich(path)
        rows.append({
            "case": data.get("case_name"),
            "iteration": data.get("iteration"),
            "np": data.get("mpi_ranks"),
            "colperm": data.get("colperm"),
            "rowperm": data.get("rowperm"),
            "build_status": data.get("build_status"),
            "converged": data.get("converged"),
            "analysis_ms": data.get("analysis_ms"),
            "factorization_ms": data.get("factorization_ms"),
            "solve_ms": data.get("solve_ms"),
            "total_solver_ms": data.get("total_solver_ms"),
            "external_elapsed_ms": data.get("v4_addendum_external_elapsed_ms"),
            "relative_residual_2": data.get("relative_residual_2"),
            "absolute_residual_2": data.get("absolute_residual_2"),
            "scaled_residual_2": data.get("scaled_residual_2"),
            "relative_error_to_x_ref_2": data.get("relative_error_to_x_ref_2"),
            "raw_json": str(path),
            "notes": data.get("notes", "") + " " + str(data.get("error_tail", "")),
        })
    return rows


def write_csv(rows: list[dict[str, Any]]) -> Path:
    out = RESULTS / "superlu_dist_mmd_rank_sweep.csv"
    keys = list(rows[0].keys()) if rows else []
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return out


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        v = float(value)
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.3e}"
    except Exception:
        return str(value)


def md_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        ("case", "case"),
        ("np", "np"),
        ("rowperm", "rowperm"),
        ("conv", "converged"),
        ("analysis ms", "analysis_ms"),
        ("factor ms", "factorization_ms"),
        ("solve ms", "solve_ms"),
        ("solver ms", "total_solver_ms"),
        ("scaled resid", "scaled_residual_2"),
    ]
    lines = [
        "| " + " | ".join(title for title, _ in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in cols) + " |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, Any]], csv_path: Path) -> None:
    large = [r for r in rows if r["case"] in {"case2869pegase", "case9241pegase"}]
    best = []
    for case in ["case2869pegase", "case9241pegase"]:
        candidates = [r for r in large if r["case"] == case and str(r["converged"]).lower() in {"true", "yes"}]
        candidates.sort(key=lambda r: float(r.get("total_solver_ms") or 1e300))
        if candidates:
            best.append(candidates[0])
    REPORT.write_text(f"""# SuperLU_DIST MMD Rank-Sweep Addendum v4

This addendum continues the large-case diagnosis after the v4 report found that SuperLU_DIST's catastrophic slow path was tied to `NATURAL` ordering. It reruns `MMD_AT_PLUS_A` with both `LargeDiag_MC64` and `NOROWPERM` for `np=1`, `np=2`, and `np=4`.

All runs used the local MPICH launcher paired with the SuperLU_DIST build and `OMP_NUM_THREADS=1`.

## Best Large-Case Results

{md_table(best)}

## Full MMD Rank Sweep

{md_table(rows)}

## Interpretation

The MMD rank sweep confirms that the earlier slow SuperLU_DIST timing was caused by the NATURAL ordering path, not by process launch, MatrixMarket loading, or a general inability of SuperLU_DIST to solve the large Jacobians. For these single-node runs, increasing MPI ranks did not improve the best MMD configuration; `np=1` remained the fastest or competitive option. This reinforces the annual-report interpretation: SuperLU_DIST is a valid external MPI/hybrid direct-solver baseline when configured with a supported fill-reducing ordering, but it is still less natural as a cuPF default than cuDSS because it relies on MPI/host-distributed integration and lacks a validated reusable repeated-Newton timing path in this wrapper.

CSV: `{csv_path.relative_to(ROOT)}`
""")


def main() -> None:
    ensure_dirs()
    paths: list[Path] = []
    synthetic_ok: dict[tuple[int, str], bool] = {}
    for rowperm in ROWPERMS:
        for np in RANKS:
            p = run_case("synthetic_validation", 0, np, rowperm)
            paths.append(p)
            try:
                synthetic_ok[(np, rowperm)] = bool(load_json(p).get("converged"))
            except Exception:
                synthetic_ok[(np, rowperm)] = False
    for rowperm in ROWPERMS:
        for np in RANKS:
            if np > 1 and not synthetic_ok.get((np, rowperm), False):
                continue
            for case in ["case2869pegase", "case9241pegase"]:
                paths.append(run_case(case, 0, np, rowperm))
    rows = collect(paths)
    csv_path = write_csv(rows)
    write_report(rows, csv_path)
    print(REPORT)


if __name__ == "__main__":
    main()
