#!/usr/bin/env python3
"""Run the second-pass Ginkgo, SuperLU_DIST, and STRUMPACK benchmarks."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw_json"
SUMMARY = ROOT / "results" / "summary_csv" / "summary_second_pass.csv"

SYSTEM_ORDER = [
    "synthetic_validation",
    "case14",
    "case118",
    "case300",
    "case1354pegase",
    "case2869pegase",
    "case9241pegase",
]

RESULT_FIELDS = [
    "solver_name",
    "solver_version",
    "library_path",
    "build_status",
    "dtype",
    "case_name",
    "iteration",
    "matrix_rows",
    "matrix_cols",
    "nnz",
    "repeat_count",
    "warmup_count",
    "load_ms",
    "format_convert_ms",
    "h2d_ms",
    "analysis_ms",
    "factorization_ms",
    "solve_ms",
    "d2h_ms",
    "total_solver_ms",
    "total_end_to_end_ms",
    "peak_gpu_memory_mb",
    "relative_residual_2",
    "relative_error_to_x_ref_2",
    "converged",
    "num_iterations",
    "gpu_resident_after_initial_load",
    "notes",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def systems() -> list[Path]:
    metas: list[Path] = []
    for case in SYSTEM_ORDER:
        case_dir = ROOT / "datasets" / "dumped_systems" / case
        if case_dir.exists():
            metas.extend(sorted(case_dir.glob("iter_*/meta.json")))
    return metas


def iter_tag(meta: dict[str, Any]) -> str:
    return f"iter{int(meta.get('iteration', -1)):03d}"


def result_path(meta: dict[str, Any], label: str, dtype: str) -> Path:
    return RAW_DIR / f"{meta['case_name']}_{iter_tag(meta)}_{label}_{dtype}_second_pass.json"


def base_result(meta: dict[str, Any], solver: str, dtype: str, status: str, notes: str) -> dict[str, Any]:
    return {
        "solver_name": solver,
        "solver_version": "unavailable",
        "library_path": "unavailable",
        "build_status": status,
        "dtype": dtype,
        "case_name": meta.get("case_name"),
        "iteration": meta.get("iteration"),
        "matrix_rows": meta.get("matrix_rows"),
        "matrix_cols": meta.get("matrix_cols"),
        "nnz": meta.get("nnz"),
        "repeat_count": int(os.environ.get("LIN_SOL_REPEATS", "10")),
        "warmup_count": int(os.environ.get("LIN_SOL_WARMUP", "3")),
        "load_ms": None,
        "format_convert_ms": None,
        "h2d_ms": None,
        "analysis_ms": None,
        "factorization_ms": None,
        "solve_ms": None,
        "d2h_ms": None,
        "total_solver_ms": None,
        "total_end_to_end_ms": None,
        "peak_gpu_memory_mb": None,
        "relative_residual_2": None,
        "relative_error_to_x_ref_2": None,
        "converged": False,
        "num_iterations": -1,
        "gpu_resident_after_initial_load": "unknown",
        "notes": notes,
        "timing_stats": {},
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def command_text(cmd: list[str]) -> str:
    return " ".join(cmd)


def run_command(
    cmd: list[str],
    out_path: Path,
    meta: dict[str, Any],
    solver: str,
    dtype: str,
    env: dict[str, str],
    timeout_s: int,
) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            old = load_json(out_path)
            if old.get("build_status") not in {"unavailable", "build_failed"}:
                return
        except Exception:
            pass

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT.parent.parent.parent,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        if completed.returncode != 0:
            data = base_result(
                meta,
                solver,
                dtype,
                "runtime_failed",
                "command failed: "
                + command_text(cmd)
                + "; stderr tail: "
                + completed.stderr[-2000:].replace("\n", " | "),
            )
            data["attempted_command"] = command_text(cmd)
            data["stdout_tail"] = completed.stdout[-2000:]
            data["stderr_tail"] = completed.stderr[-2000:]
            data["total_end_to_end_ms"] = (time.perf_counter() - start) * 1000.0
            write_json(out_path, data)
        elif not out_path.exists():
            data = base_result(meta, solver, dtype, "runtime_failed", "command returned zero but did not create output JSON")
            data["attempted_command"] = command_text(cmd)
            data["stdout_tail"] = completed.stdout[-2000:]
            data["stderr_tail"] = completed.stderr[-2000:]
            data["total_end_to_end_ms"] = (time.perf_counter() - start) * 1000.0
            write_json(out_path, data)
    except subprocess.TimeoutExpired as exc:
        data = base_result(
            meta,
            solver,
            dtype,
            "timeout",
            f"command timed out after {timeout_s}s: {command_text(cmd)}; stderr tail: {(exc.stderr or '')[-2000:]}",
        )
        data["attempted_command"] = command_text(cmd)
        data["stdout_tail"] = (exc.stdout or "")[-2000:]
        data["stderr_tail"] = (exc.stderr or "")[-2000:]
        data["total_end_to_end_ms"] = (time.perf_counter() - start) * 1000.0
        write_json(out_path, data)


def executable(path: Path) -> bool:
    return path.exists() and os.access(path, os.X_OK)


def make_summary(paths: list[Path]) -> None:
    rows: list[dict[str, Any]] = []
    extra_keys: set[str] = set()
    for path in sorted(paths):
        data = load_json(path)
        row = {k: data.get(k) for k in RESULT_FIELDS}
        for k, v in data.items():
            if k not in RESULT_FIELDS and k != "timing_stats":
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v, sort_keys=True)
                else:
                    row[k] = v
                extra_keys.add(k)
        rows.append(row)

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = RESULT_FIELDS + sorted(extra_keys)
    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    repeats = int(os.environ.get("LIN_SOL_REPEATS", "10"))
    warmup = int(os.environ.get("LIN_SOL_WARMUP", "3"))
    timeout_s = int(os.environ.get("LIN_SOL_TIMEOUT", "240"))

    ginkgo_exe = ROOT / "build" / "ginkgo_benchmark"
    superlu_exe = ROOT / "build" / "superlu_dist_benchmark"
    strumpack_exe = ROOT / "solvers" / "strumpack" / "build" / "strumpack_benchmark"
    local_mpirun = ROOT / "third_party" / "mpich" / "install" / "bin" / "mpirun"

    ginkgo_env = os.environ.copy()
    ginkgo_env["LD_LIBRARY_PATH"] = (
        str(ROOT / "third_party" / "ginkgo" / "install" / "lib")
        + ":"
        + ginkgo_env.get("LD_LIBRARY_PATH", "")
    )
    superlu_env = os.environ.copy()
    superlu_env["LD_LIBRARY_PATH"] = (
        str(ROOT / "third_party" / "mpich" / "install" / "lib")
        + ":"
        + superlu_env.get("LD_LIBRARY_PATH", "")
    )
    strumpack_env = os.environ.copy()

    generated: list[Path] = []
    for meta_path in systems():
        meta = load_json(meta_path)
        system_dir = meta_path.parent
        common = [
            "--matrix",
            str(system_dir / "J.mtx"),
            "--rhs",
            str(system_dir / "rhs.txt"),
            "--xref",
            str(system_dir / "x_ref.txt"),
            "--meta",
            str(meta_path),
            "--repeats",
            str(repeats),
            "--warmup",
            str(warmup),
        ]

        if executable(ginkgo_exe):
            configs = [
                ("ginkgo_gmres_jacobi", ROOT / "solvers" / "ginkgo" / "config_gmres_jacobi.json"),
                ("ginkgo_bicgstab_jacobi", ROOT / "solvers" / "ginkgo" / "config_bicgstab_jacobi.json"),
            ]
            for label, config in configs:
                for dtype in ("fp64", "fp32"):
                    out = result_path(meta, label, dtype)
                    generated.append(out)
                    cmd = [
                        str(ginkgo_exe),
                        *common,
                        "--dtype",
                        dtype,
                        "--config",
                        str(config),
                        "--out",
                        str(out),
                    ]
                    run_command(cmd, out, meta, "Ginkgo", dtype, ginkgo_env, timeout_s)

        if executable(superlu_exe) and executable(local_mpirun):
            for dtype in ("fp64", "fp32"):
                out = result_path(meta, "superlu_dist_np1", dtype)
                generated.append(out)
                cmd = [
                    str(local_mpirun),
                    "-np",
                    "1",
                    str(superlu_exe),
                    *common,
                    "--dtype",
                    dtype,
                    "--out",
                    str(out),
                ]
                run_command(cmd, out, meta, "SuperLU_DIST", dtype, superlu_env, timeout_s)

        if executable(strumpack_exe):
            for dtype in ("fp64", "fp32"):
                out = result_path(meta, "strumpack_np1", dtype)
                generated.append(out)
                cmd = [
                    "/usr/bin/mpirun.mpich",
                    "-np",
                    "1",
                    str(strumpack_exe),
                    *common,
                    "--dtype",
                    dtype,
                    "--out",
                    str(out),
                ]
                run_command(cmd, out, meta, "STRUMPACK", dtype, strumpack_env, timeout_s)

    make_summary(sorted(RAW_DIR.glob("*_second_pass.json")))
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
