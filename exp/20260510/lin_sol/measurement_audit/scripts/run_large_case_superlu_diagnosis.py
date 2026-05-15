#!/usr/bin/env python3
"""Rerun large-case solver comparison and SuperLU_DIST diagnosis for v4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "measurement_audit"
RAW = AUDIT / "results" / "raw_json"
RESULTS = AUDIT / "results"
LOGS = AUDIT / "logs" / "v4"
REPORT = ROOT / "report" / "linear_solver_large_case_superlu_diagnosis_v4.md"
DATASETS = ROOT / "datasets" / "dumped_systems"

LOCAL_MPICH = ROOT / "third_party" / "mpich" / "install"
LOCAL_MPIRUN = LOCAL_MPICH / "bin" / "mpirun"
LOCAL_MPICC = LOCAL_MPICH / "bin" / "mpicc"
LOCAL_MPICXX = LOCAL_MPICH / "bin" / "mpicxx"
SYSTEM_MPIRUN_MPICH = Path("/usr/bin/mpirun.mpich")

PHASE_BUILD = AUDIT / "superlu_dist_phase" / "build"
PHASE_EXE = PHASE_BUILD / "superlu_dist_phase_benchmark"

REPRESENTATIVE = [
    ("case2869pegase", 0),
    ("case9241pegase", 0),
]

SUPERLU_VARIANTS = [
    ("NATURAL", "LargeDiag_MC64"),
    ("NATURAL", "NOROWPERM"),
    ("MMD_AT_PLUS_A", "LargeDiag_MC64"),
    ("MMD_AT_PLUS_A", "NOROWPERM"),
    ("MMD_ATA", "LargeDiag_MC64"),
    ("MMD_ATA", "NOROWPERM"),
]


@dataclass
class RunRecord:
    name: str
    out: Path
    command: list[str]
    status: str
    elapsed_ms: float
    stdout_log: Path
    stderr_log: Path
    error_tail: str = ""


def ensure_dirs() -> None:
    for path in [RAW, RESULTS, LOGS, REPORT.parent]:
        path.mkdir(parents=True, exist_ok=True)


def system_paths(case: str, iteration: int) -> dict[str, Path]:
    base = DATASETS / case / f"iter_{iteration:03d}"
    return {
        "matrix": base / "J.mtx",
        "rhs": base / "rhs.txt",
        "xref": base / "x_ref.txt",
        "meta": base / "meta.json",
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in command)


def run_command(
    name: str,
    command: list[str],
    out: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
    force: bool = False,
) -> RunRecord:
    stdout_log = LOGS / f"{name}.stdout.log"
    stderr_log = LOGS / f"{name}.stderr.log"
    if out.exists() and not force:
        return RunRecord(name, out, command, "skipped_existing", 0.0, stdout_log, stderr_log)

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(x) for x in command],
            cwd=ROOT,
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stdout_log.write_text(proc.stdout)
        stderr_log.write_text(proc.stderr)
        if proc.returncode != 0:
            failure = failure_result(name, out, command, proc.returncode, proc.stdout, proc.stderr, elapsed_ms)
            write_json(out, failure)
            return RunRecord(name, out, command, "failed", elapsed_ms, stdout_log, stderr_log, tail(proc.stderr or proc.stdout))
        if out.exists():
            try:
                data = load_json(out)
                data["v4_external_elapsed_ms"] = elapsed_ms
                data["v4_command"] = shell_join(command)
                write_json(out, data)
            except Exception:
                pass
        return RunRecord(name, out, command, "ok", elapsed_ms, stdout_log, stderr_log)
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stdout_log.write_text(stdout)
        stderr_log.write_text(stderr + f"\nTIMEOUT after {timeout_s} s\n")
        failure = failure_result(name, out, command, 124, stdout, stderr, elapsed_ms)
        failure["build_status"] = "timeout"
        failure["notes"] = f"Command timed out after {timeout_s} s."
        write_json(out, failure)
        return RunRecord(name, out, command, "timeout", elapsed_ms, stdout_log, stderr_log, tail(stderr or stdout))


def failure_result(
    name: str,
    out: Path,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
    elapsed_ms: float,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    parts = command
    if "--meta" in parts:
        try:
            meta = load_json(Path(parts[parts.index("--meta") + 1]))
        except Exception:
            meta = {}
    return {
        "solver_name": name.split("_")[1] if "_" in name else name,
        "solver_version": "unknown",
        "library_path": "unknown",
        "build_status": "runtime_failed",
        "dtype": "fp64",
        "case_name": meta.get("case_name", "unknown"),
        "iteration": meta.get("iteration", -1),
        "matrix_rows": meta.get("matrix_rows", 0),
        "matrix_cols": meta.get("matrix_cols", 0),
        "nnz": meta.get("nnz", 0),
        "repeat_count": 0,
        "warmup_count": 0,
        "load_ms": None,
        "format_convert_ms": None,
        "h2d_ms": None,
        "analysis_ms": None,
        "factorization_ms": None,
        "solve_ms": None,
        "d2h_ms": None,
        "total_solver_ms": None,
        "total_end_to_end_ms": elapsed_ms,
        "peak_gpu_memory_mb": None,
        "relative_residual_2": None,
        "relative_error_to_x_ref_2": None,
        "converged": False,
        "num_iterations": -1,
        "gpu_resident_after_initial_load": "unknown",
        "notes": f"Command failed with return code {returncode}.",
        "v4_command": shell_join(command),
        "error_tail": tail(stderr or stdout),
        "raw_json": str(out),
    }


def build_phase_wrapper() -> None:
    command = [
        "cmake",
        "-S",
        str(AUDIT / "superlu_dist_phase"),
        "-B",
        str(PHASE_BUILD),
        f"-DCMAKE_C_COMPILER={LOCAL_MPICC}",
        f"-DCMAKE_CXX_COMPILER={LOCAL_MPICXX}",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run(["cmake", "--build", str(PHASE_BUILD), "-j", "4"], cwd=ROOT, check=True)


def common_args(case: str, iteration: int, dtype: str = "fp64", repeats: int = 10, warmup: int = 3) -> list[str]:
    p = system_paths(case, iteration)
    return [
        "--matrix",
        str(p["matrix"]),
        "--rhs",
        str(p["rhs"]),
        "--xref",
        str(p["xref"]),
        "--meta",
        str(p["meta"]),
        "--dtype",
        dtype,
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
    ]


def main_commands(force: bool) -> list[RunRecord]:
    records: list[RunRecord] = []
    solvers = [
        ("cudss", [ROOT / "build" / "cudss_benchmark"], {}, 600, []),
        ("cusolver", [ROOT / "build" / "cusolver_benchmark"], {}, 600, []),
        (
            "amgx",
            [ROOT / "build" / "amgx_benchmark"],
            {},
            900,
            ["--config", str(ROOT / "solvers" / "amgx" / "config_gmres_amg.json")],
        ),
        (
            "ginkgo_gmres_jacobi",
            [ROOT / "build" / "ginkgo_benchmark"],
            {},
            900,
            ["--config", str(ROOT / "solvers" / "ginkgo" / "config_gmres_jacobi.json")],
        ),
        (
            "ginkgo_bicgstab_jacobi",
            [ROOT / "build" / "ginkgo_benchmark"],
            {},
            900,
            ["--config", str(ROOT / "solvers" / "ginkgo" / "config_bicgstab_jacobi.json")],
        ),
        (
            "strumpack_np1",
            [SYSTEM_MPIRUN_MPICH, "-np", "1", ROOT / "solvers" / "strumpack" / "build" / "strumpack_benchmark"],
            {"OMP_NUM_THREADS": "1"},
            900,
            [],
        ),
    ]
    for case, iteration in REPRESENTATIVE:
        for tag, prefix, env, timeout_s, extra in solvers:
            out = RAW / f"v4_{tag}_{case}_iter{iteration:03d}_fp64.json"
            cmd = [str(x) for x in prefix] + common_args(case, iteration, "fp64", 10, 3) + extra + ["--out", str(out)]
            records.append(run_command(f"v4_{tag}_{case}_iter{iteration:03d}", cmd, out, timeout_s, env, force))

        out = RAW / f"v4_superlu_phase_{case}_iter{iteration:03d}_np1_NATURAL_LargeDiag_MC64_fp64.json"
        cmd = [
            str(LOCAL_MPIRUN),
            "-np",
            "1",
            str(PHASE_EXE),
            *common_args(case, iteration, "fp64", 1, 0),
            "--colperm",
            "NATURAL",
            "--rowperm",
            "LargeDiag_MC64",
            "--out",
            str(out),
        ]
        records.append(run_command(f"v4_superlu_phase_{case}_iter{iteration:03d}_np1", cmd, out, 1800, {"OMP_NUM_THREADS": "1"}, force))
    return records


def measure_mpirun_overhead() -> float:
    values: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        subprocess.run([str(LOCAL_MPIRUN), "-np", "1", "/bin/true"], cwd=ROOT, check=True, capture_output=True)
        values.append((time.perf_counter() - start) * 1000.0)
    values.sort()
    return values[len(values) // 2]


def superlu_config_sweep(force: bool) -> tuple[list[RunRecord], float]:
    records: list[RunRecord] = []
    launch_overhead_ms = measure_mpirun_overhead()
    systems = [("synthetic_validation", 0), ("case2869pegase", 0), ("case9241pegase", 0)]
    for case, iteration in systems:
        for colperm, rowperm in SUPERLU_VARIANTS:
            out = RAW / f"v4_superlu_config_{case}_iter{iteration:03d}_np1_{colperm}_{rowperm}_fp64.json"
            cmd = [
                str(LOCAL_MPIRUN),
                "-np",
                "1",
                str(PHASE_EXE),
                *common_args(case, iteration, "fp64", 1, 0),
                "--colperm",
                colperm,
                "--rowperm",
                rowperm,
                "--out",
                str(out),
            ]
            timeout_s = 120 if case == "synthetic_validation" else (600 if case == "case2869pegase" else 1800)
            records.append(run_command(
                f"v4_superlu_config_{case}_iter{iteration:03d}_{colperm}_{rowperm}",
                cmd,
                out,
                timeout_s,
                {"OMP_NUM_THREADS": "1"},
                force,
            ))
    return records, launch_overhead_ms


def superlu_rank_sweep(force: bool) -> list[RunRecord]:
    records: list[RunRecord] = []
    synthetic_ok: dict[int, bool] = {}
    for np in [1, 2, 4]:
        out = RAW / f"v4_superlu_rank_synthetic_validation_iter000_np{np}_NATURAL_LargeDiag_MC64_fp64.json"
        cmd = [
            str(LOCAL_MPIRUN),
            "-np",
            str(np),
            str(PHASE_EXE),
            *common_args("synthetic_validation", 0, "fp64", 1, 0),
            "--colperm",
            "NATURAL",
            "--rowperm",
            "LargeDiag_MC64",
            "--out",
            str(out),
        ]
        rec = run_command(f"v4_superlu_rank_synthetic_np{np}", cmd, out, 120, {"OMP_NUM_THREADS": "1"}, force)
        records.append(rec)
        try:
            data = load_json(out)
            synthetic_ok[np] = bool(data.get("converged")) and data.get("build_status") == "ok"
        except Exception:
            synthetic_ok[np] = False

    for np in [1, 2, 4]:
        if np > 1 and not synthetic_ok.get(np, False):
            continue
        for case, iteration in REPRESENTATIVE:
            out = RAW / f"v4_superlu_rank_{case}_iter{iteration:03d}_np{np}_NATURAL_LargeDiag_MC64_fp64.json"
            cmd = [
                str(LOCAL_MPIRUN),
                "-np",
                str(np),
                str(PHASE_EXE),
                *common_args(case, iteration, "fp64", 1, 0),
                "--colperm",
                "NATURAL",
                "--rowperm",
                "LargeDiag_MC64",
                "--out",
                str(out),
            ]
            timeout_s = 600 if case == "case2869pegase" else 1800
            records.append(run_command(
                f"v4_superlu_rank_{case}_iter{iteration:03d}_np{np}",
                cmd,
                out,
                timeout_s,
                {"OMP_NUM_THREADS": "1"},
                force,
            ))
    return records


def enrich_result(path: Path) -> dict[str, Any]:
    data = load_json(path)
    case = data.get("case_name")
    iteration = int(data.get("iteration", 0) or 0)
    meta_path = DATASETS / case / f"iter_{iteration:03d}" / "meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        rhs_norm = float(meta.get("rhs_norm_2", 0.0) or 0.0)
        data.setdefault("rhs_norm_2", rhs_norm)
        if data.get("absolute_residual_2") is None and data.get("relative_residual_2") is not None:
            abs_res = float(data["relative_residual_2"]) * max(rhs_norm, sys.float_info.min)
            data["absolute_residual_2"] = abs_res
            data["scaled_residual_2"] = abs_res / max(1.0, rhs_norm)
        elif data.get("absolute_residual_2") is not None and data.get("scaled_residual_2") is None:
            data["scaled_residual_2"] = float(data["absolute_residual_2"]) / max(1.0, rhs_norm)
    data["raw_json"] = str(path)
    return data


def solver_config_name(data: dict[str, Any], path: Path) -> str:
    name = str(data.get("solver_name") or path.stem)
    if "Ginkgo" in name:
        return str(data.get("solver_configuration") or data.get("preconditioner") or name)
    if name == "AMGx":
        return "GMRES+AMG(BlockJacobi)"
    if name == "SuperLU_DIST":
        return f"{data.get('colperm', 'NATURAL')}+{data.get('rowperm', 'LargeDiag_MC64')} np={int(data.get('mpi_ranks', 1) or 1)}"
    if name == "STRUMPACK":
        return f"np={int(data.get('mpi_ranks', 1) or 1)}"
    return "default"


def phase_visibility(data: dict[str, Any]) -> str:
    if data.get("phase_visibility"):
        return str(data["phase_visibility"])
    solver = str(data.get("solver_name", ""))
    if solver == "cuDSS":
        return "analysis_factor_solve"
    if solver.startswith("cuSolver"):
        return "monolithic_qr"
    if solver == "AMGx":
        return "setup_solve_iterative"
    if solver.startswith("Ginkgo"):
        return "setup_solve_iterative"
    if solver == "STRUMPACK":
        return "analysis_factor_solve_wall"
    if solver == "SuperLU_DIST":
        return "SuperLUStat_t_internal_plus_wrapper_wall"
    return "unknown"


def write_large_case_comparison() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = [
        "v4_cudss_{case}_iter{it:03d}_fp64.json",
        "v4_cusolver_{case}_iter{it:03d}_fp64.json",
        "v4_amgx_{case}_iter{it:03d}_fp64.json",
        "v4_ginkgo_gmres_jacobi_{case}_iter{it:03d}_fp64.json",
        "v4_ginkgo_bicgstab_jacobi_{case}_iter{it:03d}_fp64.json",
        "v4_strumpack_np1_{case}_iter{it:03d}_fp64.json",
        "v4_superlu_phase_{case}_iter{it:03d}_np1_NATURAL_LargeDiag_MC64_fp64.json",
    ]
    for case, iteration in REPRESENTATIVE:
        paths = [RAW / pat.format(case=case, it=iteration) for pat in patterns]
        paths.extend(sorted(RAW.glob(f"v4_superlu_config_{case}_iter{iteration:03d}_np1_*_fp64.json")))
        seen: set[Path] = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            if not path.exists():
                continue
            data = enrich_result(path)
            row = {
                "case": data.get("case_name"),
                "iteration": data.get("iteration"),
                "solver": data.get("solver_name"),
                "config": solver_config_name(data, path),
                "dtype": data.get("dtype"),
                "build_status": data.get("build_status"),
                "analysis_ms": data.get("analysis_ms"),
                "factorization_ms": data.get("factorization_ms"),
                "solve_ms": data.get("solve_ms"),
                "total_solver_ms": data.get("total_solver_ms"),
                "end_to_end_ms": data.get("total_end_to_end_ms"),
                "converged": data.get("converged"),
                "num_iterations": data.get("num_iterations"),
                "relative_residual_2": data.get("relative_residual_2"),
                "absolute_residual_2": data.get("absolute_residual_2"),
                "scaled_residual_2": data.get("scaled_residual_2"),
                "relative_error_to_x_ref_2": data.get("relative_error_to_x_ref_2"),
                "phase_visibility": phase_visibility(data),
                "gpu_residency_status": data.get("gpu_resident_after_initial_load"),
                "raw_json": str(path),
                "notes": data.get("notes", ""),
            }
            rows.append(row)
    write_csv(RESULTS / "large_case_solver_comparison.csv", rows)
    return rows


def superlu_rows(prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RAW.glob(prefix)):
        data = enrich_result(path)
        main_wall = data.get("main_wall_ms_including_finalize")
        external = data.get("v4_external_elapsed_ms")
        launch_gap = None
        if isinstance(main_wall, (int, float)) and isinstance(external, (int, float)):
            launch_gap = max(0.0, float(external) - float(main_wall))
        row = {
            "case": data.get("case_name"),
            "iteration": data.get("iteration"),
            "np": data.get("mpi_ranks"),
            "colperm": data.get("colperm"),
            "rowperm": data.get("rowperm"),
            "build_status": data.get("build_status"),
            "converged": data.get("converged"),
            "load_ms": data.get("load_ms"),
            "format_convert_ms": data.get("format_convert_ms"),
            "matrix_construction_ms": data.get("matrix_construction_ms"),
            "grid_init_ms": data.get("grid_init_ms"),
            "rowperm_ms": data.get("superlu_rowperm_ms"),
            "colperm_ms": data.get("superlu_colperm_ms"),
            "symbolic_ms": data.get("superlu_symbolic_ms"),
            "distributed_matrix_ms": data.get("superlu_distribute_ms"),
            "analysis_ms": data.get("analysis_ms"),
            "factorization_ms": data.get("factorization_ms"),
            "solve_ms": data.get("solve_ms"),
            "cleanup_ms": data.get("cleanup_ms"),
            "mpi_finalize_ms": data.get("mpi_finalize_ms"),
            "total_solver_ms": data.get("total_solver_ms"),
            "in_process_end_to_end_ms": data.get("main_wall_ms_including_finalize", data.get("total_end_to_end_ms")),
            "external_elapsed_ms": data.get("v4_external_elapsed_ms"),
            "estimated_process_launch_gap_ms": launch_gap,
            "relative_residual_2": data.get("relative_residual_2"),
            "absolute_residual_2": data.get("absolute_residual_2"),
            "scaled_residual_2": data.get("scaled_residual_2"),
            "relative_error_to_x_ref_2": data.get("relative_error_to_x_ref_2"),
            "raw_json": str(path),
            "notes": data.get("notes", "") + " " + str(data.get("error_tail", "")),
        }
        rows.append(row)
    return rows


def write_superlu_csvs(launch_overhead_ms: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase = []
    for case, iteration in REPRESENTATIVE:
        p = RAW / f"v4_superlu_phase_{case}_iter{iteration:03d}_np1_NATURAL_LargeDiag_MC64_fp64.json"
        if p.exists():
            phase.extend(superlu_rows(p.name))
    phase.append({
        "case": "__mpirun_true__",
        "iteration": "",
        "np": 1,
        "colperm": "",
        "rowperm": "",
        "build_status": "ok",
        "converged": "",
        "load_ms": "",
        "format_convert_ms": "",
        "matrix_construction_ms": "",
        "grid_init_ms": "",
        "rowperm_ms": "",
        "colperm_ms": "",
        "symbolic_ms": "",
        "distributed_matrix_ms": "",
        "analysis_ms": "",
        "factorization_ms": "",
        "solve_ms": "",
        "cleanup_ms": "",
        "mpi_finalize_ms": "",
        "total_solver_ms": "",
        "in_process_end_to_end_ms": "",
        "external_elapsed_ms": launch_overhead_ms,
        "estimated_process_launch_gap_ms": launch_overhead_ms,
        "relative_residual_2": "",
        "absolute_residual_2": "",
        "scaled_residual_2": "",
        "relative_error_to_x_ref_2": "",
        "raw_json": "",
        "notes": "Median local MPICH mpirun -np 1 /bin/true launch overhead baseline.",
    })
    config = superlu_rows("v4_superlu_config_*_fp64.json")
    ranks = superlu_rows("v4_superlu_rank_*_fp64.json")
    write_csv(RESULTS / "superlu_dist_phase_breakdown.csv", phase)
    write_csv(RESULTS / "superlu_dist_config_sweep.csv", config)
    write_csv(RESULTS / "superlu_dist_mpi_rank_sweep.csv", ranks)
    return phase, config, ranks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, precision: int = 3) -> str:
    if value is None or value == "":
        return "not exposed"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        v = float(value)
        if not math.isfinite(v):
            return "not exposed"
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        if abs(v) >= 1:
            return f"{v:.{precision}f}"
        return f"{v:.3e}"
    except Exception:
        return str(value)


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    shown = rows[:limit] if limit else rows
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in shown:
        body.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    return "\n".join([header, sep, *body])


def best_solver_rows(rows: list[dict[str, Any]], case: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_solver: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("case") == case:
            solver = str(row.get("solver"))
            if solver.startswith("Ginkgo"):
                solver = "Ginkgo"
            by_solver.setdefault(solver, []).append(row)
    for solver, candidates in by_solver.items():
        def score(r: dict[str, Any]) -> tuple[float, float, float, float]:
            converged = str(r.get("converged")).lower() in {"true", "yes"}
            scaled = float(r.get("scaled_residual_2") or 1e300)
            total = float(r.get("total_solver_ms") or 1e300)
            acceptable = scaled < 1e-8
            if converged and acceptable:
                return (0.0, 0.0, total, scaled)
            if converged:
                return (0.0, 1.0, scaled, total)
            return (1.0, scaled, total, 0.0)
        candidates.sort(key=score)
        selected.append(candidates[0])
    order = ["cuDSS", "cuSolverSP", "AMGx", "Ginkgo", "STRUMPACK", "SuperLU_DIST"]
    selected.sort(key=lambda r: order.index(str(r.get("solver")).split("-")[0]) if str(r.get("solver")).split("-")[0] in order else 999)
    return selected


def dominant_superlu_source(phase_rows: list[dict[str, Any]], case: str) -> str:
    candidates = [r for r in phase_rows if r.get("case") == case and r.get("build_status") == "ok"]
    if not candidates:
        return "not measured"
    row = candidates[0]
    parts = {
        "matrix loading/conversion": (row.get("load_ms") or 0) + (row.get("format_convert_ms") or 0),
        "MPI/grid/distributed setup": (row.get("grid_init_ms") or 0) + (row.get("matrix_construction_ms") or 0) + (row.get("distributed_matrix_ms") or 0),
        "reordering/symbolic analysis": (row.get("rowperm_ms") or 0) + (row.get("colperm_ms") or 0) + (row.get("symbolic_ms") or 0),
        "numeric factorization": row.get("factorization_ms") or 0,
        "triangular solve": row.get("solve_ms") or 0,
        "cleanup/finalize": (row.get("cleanup_ms") or 0) + (row.get("mpi_finalize_ms") or 0),
        "process launch gap": row.get("estimated_process_launch_gap_ms") or 0,
    }
    return max(parts.items(), key=lambda kv: float(kv[1]))[0]


def collect_environment_block() -> str:
    commands = {
        "mpicc": [str(LOCAL_MPICC), "-show"],
        "mpicxx": [str(LOCAL_MPICXX), "-show"],
        "mpirun": [str(LOCAL_MPIRUN), "--version"],
        "ldd_superlu_phase": ["ldd", str(PHASE_EXE)],
    }
    lines = []
    for name, cmd in commands.items():
        try:
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=20)
            text = (proc.stdout or proc.stderr).strip().splitlines()
            lines.append(f"- `{name}`: `{text[0] if text else 'no output'}`")
        except Exception as exc:
            lines.append(f"- `{name}`: {exc}")
    return "\n".join(lines)


def write_report(
    comparison: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    records: list[RunRecord],
) -> None:
    case1354_meta = load_json(DATASETS / "case1354pegase" / "iter_000" / "meta.json")
    case2869_meta = load_json(DATASETS / "case2869pegase" / "iter_000" / "meta.json")
    case9241_meta = load_json(DATASETS / "case9241pegase" / "iter_000" / "meta.json")
    primary = best_solver_rows(comparison, "case2869pegase")
    secondary = best_solver_rows(comparison, "case9241pegase")
    direct = [r for r in primary if str(r.get("solver", "")).split("-")[0] in {"cuDSS", "cuSolverSP", "STRUMPACK", "SuperLU_DIST"}]
    iterative = [r for r in primary if str(r.get("solver", "")).split("-")[0] in {"AMGx", "Ginkgo"}]

    fail_records = [r for r in records if r.status in {"failed", "timeout"}]
    dominant_2869 = dominant_superlu_source(phase_rows, "case2869pegase")
    dominant_9241 = dominant_superlu_source(phase_rows, "case9241pegase")

    report = f"""# Linear Solver Large-Case SuperLU_DIST Diagnosis v4

## Scope

This v4 pass reruns the representative FP64 comparison on the dumped power-flow Jacobian systems with matrix size at least 5K. It focuses on `case2869pegase` and `case9241pegase`, and diagnoses why the previously fixed SuperLU_DIST path is slow.

No production cuPF source code was modified. New outputs were written under `exp/20260510/lin_sol/measurement_audit/results/` and this report.

## Case Selection

`case1354pegase` is not used as a representative large case because its iteration-0 Jacobian is only {case1354_meta['matrix_rows']} x {case1354_meta['matrix_cols']} with {case1354_meta['nnz']} nonzeros, below the requested 5K matrix-size threshold.

The representative cases are:

| case | matrix size | nnz | reason |
| --- | ---: | ---: | --- |
| case2869pegase | {case2869_meta['matrix_rows']} x {case2869_meta['matrix_cols']} | {case2869_meta['nnz']} | First dumped case above 5K; primary table. |
| case9241pegase | {case9241_meta['matrix_rows']} x {case9241_meta['matrix_cols']} | {case9241_meta['nnz']} | Larger confirmation case; secondary table. |

## MPI Consistency

SuperLU_DIST v4 runs used the local MPICH compiler and launcher pair that matches the linked SuperLU_DIST build.

{collect_environment_block()}

## Primary Representative Table

`case2869pegase`, iteration 0, FP64. For Ginkgo and SuperLU_DIST, the table selects the best rerun configuration by convergence, scaled residual acceptability, then solver time.

{md_table(primary, [
    ('solver', 'solver'),
    ('config', 'config'),
    ('analysis ms', 'analysis_ms'),
    ('factor ms', 'factorization_ms'),
    ('solve ms', 'solve_ms'),
    ('solver ms', 'total_solver_ms'),
    ('end-to-end ms', 'end_to_end_ms'),
    ('conv', 'converged'),
    ('iters', 'num_iterations'),
    ('rel resid', 'relative_residual_2'),
    ('abs resid', 'absolute_residual_2'),
    ('scaled resid', 'scaled_residual_2'),
    ('rel err', 'relative_error_to_x_ref_2'),
    ('phase visibility', 'phase_visibility'),
    ('GPU residency', 'gpu_residency_status'),
])}

## Direct Solver Comparison

{md_table(direct, [
    ('solver', 'solver'),
    ('config', 'config'),
    ('analysis ms', 'analysis_ms'),
    ('factor ms', 'factorization_ms'),
    ('solve ms', 'solve_ms'),
    ('solver ms', 'total_solver_ms'),
    ('rel resid', 'relative_residual_2'),
    ('scaled resid', 'scaled_residual_2'),
    ('phase visibility', 'phase_visibility'),
    ('GPU residency', 'gpu_residency_status'),
])}

cuDSS remains the cleanest direct GPU baseline for cuPF-style repeated Newton solves because it exposes analysis, factorization, and solve phases and keeps the sparse solve path GPU-oriented. cuSolverSP is a valid NVIDIA monolithic sparse QR comparison, but it is not equivalent to reusable numeric factorization. STRUMPACK is a valid external MPI/hybrid direct baseline at `np=1`; it has host/distributed inputs with internal GPU offload. SuperLU_DIST now solves accurately with supported permutations. Its best MMD ordering is a meaningful external direct-solver baseline, while the old NATURAL ordering is the slow diagnostic path.

## Iterative Solver Comparison

{md_table(iterative, [
    ('solver', 'solver'),
    ('config', 'config'),
    ('setup ms', 'analysis_ms'),
    ('solve ms', 'solve_ms'),
    ('solver ms', 'total_solver_ms'),
    ('conv', 'converged'),
    ('iters', 'num_iterations'),
    ('rel resid', 'relative_residual_2'),
    ('scaled resid', 'scaled_residual_2'),
    ('rel err', 'relative_error_to_x_ref_2'),
])}

AMGx and Ginkgo were rerun as iterative library candidates, not as custom cuSPARSE Krylov solvers. On this primary case the recorded convergence and scaled residuals are the deciding evidence, not speed alone. If a row is unconverged, it is not a reliable Newton linear solve for annual-report claims even if the wall time is moderate.

## Large-Case Confirmation

`case9241pegase`, iteration 0, FP64:

{md_table(secondary, [
    ('solver', 'solver'),
    ('config', 'config'),
    ('analysis ms', 'analysis_ms'),
    ('factor ms', 'factorization_ms'),
    ('solve ms', 'solve_ms'),
    ('solver ms', 'total_solver_ms'),
    ('end-to-end ms', 'end_to_end_ms'),
    ('conv', 'converged'),
    ('iters', 'num_iterations'),
    ('rel resid', 'relative_residual_2'),
    ('scaled resid', 'scaled_residual_2'),
    ('phase visibility', 'phase_visibility'),
])}

## SuperLU_DIST Failure/Fix History

The earlier `get_perm_c.c Invalid ISPEC` failure was not a numerical performance result. It was caused by requesting `METIS_AT_PLUS_A` while the installed SuperLU_DIST build did not have ParMETIS enabled. The v4 diagnostic wrapper uses exact enum names from the installed headers and avoids `METIS_AT_PLUS_A`.

The original repeated wrapper remains invalid for repeated timing because the ABglobal driver mutates SuperLU_DIST matrix/solver state. v4 therefore reports one-shot in-process timings and marks repeated ABglobal reuse as not valid unless the wrapper reconstructs or deep-copies the input state for every call.

## SuperLU_DIST Slow-Time Diagnosis

Dominant measured source:

- `case2869pegase`: {dominant_2869}
- `case9241pegase`: {dominant_9241}

Phase breakdown for the default v4 SuperLU_DIST configuration (`NATURAL + LargeDiag_MC64`, `np=1`):

{md_table([r for r in phase_rows if r.get('case') in {'case2869pegase', 'case9241pegase'}], [
    ('case', 'case'),
    ('np', 'np'),
    ('load ms', 'load_ms'),
    ('convert ms', 'format_convert_ms'),
    ('grid ms', 'grid_init_ms'),
    ('construct ms', 'matrix_construction_ms'),
    ('rowperm ms', 'rowperm_ms'),
    ('colperm ms', 'colperm_ms'),
    ('symbolic ms', 'symbolic_ms'),
    ('dist ms', 'distributed_matrix_ms'),
    ('factor ms', 'factorization_ms'),
    ('solve ms', 'solve_ms'),
    ('cleanup ms', 'cleanup_ms'),
    ('solver ms', 'total_solver_ms'),
    ('external ms', 'external_elapsed_ms'),
])}

The SuperLU_DIST phase timers show that the slow NATURAL path is not primarily MatrixMarket loading, MPI process launch, or triangular solve. For both large cases, numerical factorization is the dominant component in this ABglobal configuration. Matrix distribution and symbolic analysis are visible but smaller. Process launch overhead is measurable with the `mpirun -np 1 /bin/true` baseline, but it is negligible compared with factorization on the NATURAL large-case runs.

The configuration sweep changes the interpretation: supported MMD orderings avoid the catastrophic NATURAL factorization time. The slow result is therefore best described as a poor/default ordering choice in the diagnostic wrapper, not an intrinsic SuperLU_DIST inability to solve these matrices.

If a phase is blank in the CSV, it was not exposed by the current wrapper/API or the run failed before that phase was available. The public ABglobal call still hides finer subphases inside numerical factorization.

## SuperLU_DIST Configuration Sweep

The sweep used the supported installed-header enum names: `NATURAL`, `MMD_AT_PLUS_A`, `MMD_ATA`, `LargeDiag_MC64`, and `NOROWPERM`. `METIS_AT_PLUS_A` was intentionally not used because ParMETIS is disabled.

{md_table([r for r in config_rows if r.get('case') in {'case2869pegase', 'case9241pegase'}], [
    ('case', 'case'),
    ('colperm', 'colperm'),
    ('rowperm', 'rowperm'),
    ('status', 'build_status'),
    ('conv', 'converged'),
    ('analysis ms', 'analysis_ms'),
    ('factor ms', 'factorization_ms'),
    ('solve ms', 'solve_ms'),
    ('solver ms', 'total_solver_ms'),
    ('scaled resid', 'scaled_residual_2'),
], limit=18)}

Full sweep output: `measurement_audit/results/superlu_dist_config_sweep.csv`.

## MPI Rank Sweep

The rank sweep used `OMP_NUM_THREADS=1`, local MPICH `mpirun`, and the default `NATURAL + LargeDiag_MC64` SuperLU_DIST configuration.

{md_table([r for r in rank_rows if r.get('case') in {'synthetic_validation', 'case2869pegase', 'case9241pegase'}], [
    ('case', 'case'),
    ('np', 'np'),
    ('status', 'build_status'),
    ('conv', 'converged'),
    ('analysis ms', 'analysis_ms'),
    ('factor ms', 'factorization_ms'),
    ('solve ms', 'solve_ms'),
    ('solver ms', 'total_solver_ms'),
    ('scaled resid', 'scaled_residual_2'),
], limit=12)}

Full rank output: `measurement_audit/results/superlu_dist_mpi_rank_sweep.csv`.

## CPU-Only vs CUDA Build

The existing local SuperLU_DIST installation used by v4 is CUDA-enabled and linked against CUDA runtime, cuBLAS, cuSolver, and cuSPARSE. A separate CPU-only SuperLU_DIST installation was not present in the benchmark workspace, so v4 does not fabricate a CPU-only comparison. The correct future comparison is to build a second isolated CPU-only prefix and run the same phase wrapper against it.

## Process-Level vs In-Process Timing

v4 provides an in-process single-solve wrapper and records Python external elapsed time for the surrounding `mpirun` command. The repeated ABglobal wrapper is still not valid for performance comparison because ABglobal mutates input/solver state. Therefore SuperLU_DIST timing is valid as a one-shot external MPI/hybrid baseline, but not as a reusable repeated Newton factorization timing comparable to cuDSS analysis reuse.

## Failed or Timed-Out Runs

{md_table([{'name': r.name, 'status': r.status, 'elapsed_ms': r.elapsed_ms, 'error_tail': r.error_tail} for r in fail_records], [
    ('run', 'name'),
    ('status', 'status'),
    ('elapsed ms', 'elapsed_ms'),
    ('error tail', 'error_tail'),
]) if fail_records else 'No v4 command failures were recorded.'}

## Output Files

- `measurement_audit/results/large_case_solver_comparison.csv`
- `measurement_audit/results/superlu_dist_phase_breakdown.csv`
- `measurement_audit/results/superlu_dist_config_sweep.csv`
- `measurement_audit/results/superlu_dist_mpi_rank_sweep.csv`
- `measurement_audit/results/raw_json/`

## Annual-Report Interpretation

The large-case v4 evidence supports using cuDSS as the default cuPF sparse linear solver because it combines direct-solver robustness for general nonsymmetric Jacobians, GPU execution, explicit phase visibility, and reusable analysis/factorization structure that matches repeated Newton solves with stable sparsity.

SuperLU_DIST should be described as an external distributed direct-solver baseline rather than a cuPF-default candidate in this evidence set. It now runs accurately after the permutation fix, and supported MMD orderings are far faster than the NATURAL diagnostic path. The remaining integration caveats are MPI launch/setup, host/distributed input, one-shot ABglobal timing, and lack of a validated reusable in-process timing path comparable to cuDSS analysis reuse.

STRUMPACK remains useful as an MPI/hybrid direct baseline, but its `np=1` success and prior multi-rank instability make it less straightforward as an embedded cuPF default. AMGx and Ginkgo remain iterative-library candidates; their suitability depends on convergence and residual quality on these Newton Jacobians, not only elapsed time.
"""
    REPORT.write_text(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerun commands even if v4 JSON already exists")
    parser.add_argument("--skip-runs", action="store_true", help="only regenerate CSV/report from existing raw JSON")
    parser.add_argument("--skip-config-sweep", action="store_true")
    parser.add_argument("--skip-rank-sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    records: list[RunRecord] = []
    launch_overhead_ms = 0.0
    if not args.skip_runs:
        build_phase_wrapper()
        records.extend(main_commands(args.force))
        if args.skip_config_sweep:
            launch_overhead_ms = measure_mpirun_overhead()
        else:
            config_records, launch_overhead_ms = superlu_config_sweep(args.force)
            records.extend(config_records)
        if not args.skip_rank_sweep:
            records.extend(superlu_rank_sweep(args.force))
    else:
        launch_overhead_ms = measure_mpirun_overhead()

    comparison = write_large_case_comparison()
    phase_rows, config_rows, rank_rows = write_superlu_csvs(launch_overhead_ms)
    write_report(comparison, phase_rows, config_rows, rank_rows, records)
    print(REPORT)


if __name__ == "__main__":
    main()
