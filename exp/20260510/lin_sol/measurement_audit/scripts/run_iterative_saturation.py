#!/usr/bin/env python3
"""Run max-iteration saturation sweeps for AMGx and Ginkgo.

This script intentionally stays inside exp/20260510/lin_sol and reuses the
existing benchmark wrappers and dumped Jacobian systems. It varies only the
iteration cap so that residual-vs-time plateaus can be identified without
changing production cuPF code.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "measurement_audit"
CONFIG_DIR = AUDIT / "configs" / "iterative_saturation"
RAW_DIR = AUDIT / "results" / "raw_json" / "iterative_saturation"
LOG_DIR = AUDIT / "logs" / "iterative_saturation"
OUT_CSV = AUDIT / "results" / "iterative_saturation_sweep.csv"

CASES = [
    ("case2869pegase", "iter_000"),
    ("case9241pegase", "iter_000"),
]

MAX_ITERS = [50, 100, 200, 400, 800, 1000, 1500, 2000]

AMGX_SOLVERS = ["GMRES", "FGMRES"]
GINKGO_SOLVERS = ["gmres", "bicgstab"]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def amgx_config(solver: str, max_iters: int) -> dict:
    return {
        "config_version": 2,
        "solver": {
            "scope": "main",
            "solver": solver,
            "max_iters": max_iters,
            "tolerance": 1e-8,
            "norm": "L2",
            "convergence": "RELATIVE_INI",
            "monitor_residual": 1,
            "obtain_timings": 1,
            "gmres_n_restart": 30,
            "preconditioner": {
                "scope": "amg",
                "solver": "AMG",
                "algorithm": "AGGREGATION",
                "selector": "SIZE_2",
                "smoother": "BLOCK_JACOBI",
                "coarse_solver": "DENSE_LU_SOLVER",
                "cycle": "V",
                "max_iters": 1,
                "max_levels": 50,
                "min_coarse_rows": 32,
                "presweeps": 1,
                "postsweeps": 1,
            },
        },
    }


def ginkgo_config(solver: str, max_iters: int) -> dict:
    return {
        "solver": solver,
        "preconditioner": "jacobi",
        "max_iterations": max_iters,
        "tolerance": 1e-8,
    }


def run_command(cmd: list[str], stdout_path: Path, stderr_path: Path, timeout_s: int = 300) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=out,
                stderr=err,
                timeout=timeout_s,
                check=False,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            err.write(f"\nTIMEOUT after {timeout_s}s\n")
            return 124


def load_result(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def tail(path: Path, n: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def result_row(
    library: str,
    config_name: str,
    case_name: str,
    iteration_dir: str,
    max_iters: int,
    out_json: Path,
    command: list[str],
    rc: int,
    stdout_path: Path,
    stderr_path: Path,
) -> dict:
    result = load_result(out_json)
    meta_path = ROOT / "datasets" / "dumped_systems" / case_name / iteration_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    rhs_norm = float(meta.get("rhs_norm_2", math.nan))
    rel_res = result.get("relative_residual_2")
    abs_res = None if rel_res is None else float(rel_res) * rhs_norm
    scaled_res = None if abs_res is None else abs_res / max(1.0, rhs_norm)
    iterations = result.get("num_iterations")
    solve_ms = result.get("solve_ms")
    per_iter_ms = None
    if isinstance(iterations, (int, float)) and iterations and iterations > 0 and isinstance(solve_ms, (int, float)):
        per_iter_ms = float(solve_ms) / float(iterations)
    return {
        "library": library,
        "config": config_name,
        "case": case_name,
        "iteration": int(meta.get("iteration", 0)),
        "matrix_rows": meta.get("matrix_rows"),
        "nnz": meta.get("nnz"),
        "dtype": "fp64",
        "requested_max_iters": max_iters,
        "actual_iterations": iterations,
        "converged": result.get("converged"),
        "relative_residual_2": rel_res,
        "absolute_residual_2": abs_res,
        "scaled_residual_2": scaled_res,
        "relative_error_to_x_ref_2": result.get("relative_error_to_x_ref_2"),
        "analysis_ms": result.get("analysis_ms"),
        "solve_ms": solve_ms,
        "total_solver_ms": result.get("total_solver_ms"),
        "total_end_to_end_ms": result.get("total_end_to_end_ms"),
        "per_iter_solve_ms": per_iter_ms,
        "peak_gpu_memory_mb": result.get("peak_gpu_memory_mb"),
        "status": "ok" if rc == 0 and result else f"failed_rc_{rc}",
        "raw_json": str(out_json.relative_to(ROOT)),
        "stdout_log": str(stdout_path.relative_to(ROOT)),
        "stderr_log": str(stderr_path.relative_to(ROOT)),
        "attempted_command": " ".join(command),
        "stderr_tail": tail(stderr_path, 8),
    }


def write_rows(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "library",
        "config",
        "case",
        "iteration",
        "matrix_rows",
        "nnz",
        "dtype",
        "requested_max_iters",
        "actual_iterations",
        "converged",
        "relative_residual_2",
        "absolute_residual_2",
        "scaled_residual_2",
        "relative_error_to_x_ref_2",
        "analysis_ms",
        "solve_ms",
        "total_solver_ms",
        "total_end_to_end_ms",
        "per_iter_solve_ms",
        "peak_gpu_memory_mb",
        "status",
        "raw_json",
        "stdout_log",
        "stderr_log",
        "attempted_command",
        "stderr_tail",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    amgx_exe = ROOT / "build" / "amgx_benchmark"
    ginkgo_exe = ROOT / "build" / "ginkgo_benchmark"
    if not amgx_exe.exists() or not ginkgo_exe.exists():
        raise SystemExit("AMGx or Ginkgo benchmark executable is missing")

    for case_name, iteration_dir in CASES:
        system_dir = ROOT / "datasets" / "dumped_systems" / case_name / iteration_dir
        matrix = system_dir / "J.mtx"
        rhs = system_dir / "rhs.txt"
        xref = system_dir / "x_ref.txt"
        meta = system_dir / "meta.json"

        for solver in AMGX_SOLVERS:
            for max_iters in MAX_ITERS:
                config_name = f"amgx_{solver.lower()}_amg_block_jacobi_{max_iters}"
                config_path = CONFIG_DIR / f"{config_name}.json"
                write_json(config_path, amgx_config(solver, max_iters))
                out_json = RAW_DIR / f"{config_name}_{case_name}_iter000_fp64.json"
                stdout_path = LOG_DIR / f"{config_name}_{case_name}.stdout.log"
                stderr_path = LOG_DIR / f"{config_name}_{case_name}.stderr.log"
                cmd = [
                    str(amgx_exe),
                    "--matrix", str(matrix),
                    "--rhs", str(rhs),
                    "--xref", str(xref),
                    "--meta", str(meta),
                    "--dtype", "fp64",
                    "--repeats", "1",
                    "--warmup", "0",
                    "--out", str(out_json),
                    "--config", str(config_path),
                ]
                print("RUN", " ".join(cmd), flush=True)
                rc = run_command(cmd, stdout_path, stderr_path)
                rows.append(result_row("AMGx", config_name, case_name, iteration_dir, max_iters, out_json, cmd, rc, stdout_path, stderr_path))
                write_rows(rows)

        for solver in GINKGO_SOLVERS:
            for max_iters in MAX_ITERS:
                config_name = f"ginkgo_{solver}_jacobi_{max_iters}"
                config_path = CONFIG_DIR / f"{config_name}.json"
                write_json(config_path, ginkgo_config(solver, max_iters))
                out_json = RAW_DIR / f"{config_name}_{case_name}_iter000_fp64.json"
                stdout_path = LOG_DIR / f"{config_name}_{case_name}.stdout.log"
                stderr_path = LOG_DIR / f"{config_name}_{case_name}.stderr.log"
                cmd = [
                    str(ginkgo_exe),
                    "--matrix", str(matrix),
                    "--rhs", str(rhs),
                    "--xref", str(xref),
                    "--meta", str(meta),
                    "--dtype", "fp64",
                    "--repeats", "1",
                    "--warmup", "0",
                    "--out", str(out_json),
                    "--config", str(config_path),
                ]
                print("RUN", " ".join(cmd), flush=True)
                rc = run_command(cmd, stdout_path, stderr_path)
                rows.append(result_row("Ginkgo", config_name, case_name, iteration_dir, max_iters, out_json, cmd, rc, stdout_path, stderr_path))
                write_rows(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
