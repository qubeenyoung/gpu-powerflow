#!/usr/bin/env python3
"""Run all available solver wrappers on dumped systems and summarize results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DUMP_ROOT = EXP_ROOT / "datasets" / "dumped_systems"
RAW_DIR = EXP_ROOT / "results" / "raw_json"
SUMMARY_DIR = EXP_ROOT / "results" / "summary_csv"
LOG_DIR = EXP_ROOT / "logs"
BUILD_DIR = EXP_ROOT / "build"


SUMMARY_FIELDS = [
    "case",
    "iteration",
    "solver",
    "dtype",
    "build_status",
    "matrix_rows",
    "matrix_cols",
    "nnz",
    "analysis_ms",
    "factorization_ms",
    "solve_ms",
    "total_solver_ms",
    "total_end_to_end_ms",
    "relative_residual_2",
    "relative_error_to_x_ref_2",
    "converged",
    "num_iterations",
    "gpu_resident_after_initial_load",
    "notes",
    "raw_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-root", default=str(DUMP_ROOT))
    parser.add_argument("--build-dir", default=str(BUILD_DIR))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--systems", default="", help="Comma-separated case_name filter")
    parser.add_argument("--solvers", default="", help="Comma-separated solver filter")
    parser.add_argument("--dtypes", default="fp64,fp32")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_systems(dump_root: Path, systems_filter: str) -> List[Dict[str, Any]]:
    filters = {s.strip() for s in systems_filter.split(",") if s.strip()}
    systems: List[Dict[str, Any]] = []
    for meta_path in sorted(dump_root.glob("*/iter_*/meta.json")):
        meta = load_json(meta_path)
        if filters and meta.get("case_name") not in filters:
            continue
        system_dir = meta_path.parent
        systems.append(
            {
                "case_name": meta.get("case_name", system_dir.parent.name),
                "iteration": int(meta.get("iteration", 0)),
                "matrix": system_dir / "J.mtx",
                "rhs": system_dir / "rhs.txt",
                "xref": system_dir / "x_ref.txt",
                "meta": meta_path,
                "meta_data": meta,
            }
        )
    systems.sort(key=lambda s: (s["case_name"] != "synthetic_validation", s["case_name"], s["iteration"]))
    return systems


def executable(path: Path) -> Optional[str]:
    return str(path) if path.exists() and os.access(path, os.X_OK) else None


def solver_specs(build_dir: Path) -> List[Dict[str, Any]]:
    python = sys.executable
    return [
        {
            "name": "cudss",
            "display": "cuDSS",
            "cmd": executable(build_dir / "cudss_benchmark"),
            "dtypes": ["fp64", "fp32"],
            "config": None,
        },
        {
            "name": "cusolver",
            "display": "cuSolverSP",
            "cmd": executable(build_dir / "cusolver_benchmark"),
            "dtypes": ["fp64", "fp32"],
            "config": None,
        },
        {
            "name": "amgx",
            "display": "AMGx",
            "cmd": executable(build_dir / "amgx_benchmark"),
            "dtypes": ["fp64", "fp32"],
            "config": EXP_ROOT / "solvers" / "amgx" / "config_gmres_amg.json",
        },
        {
            "name": "ginkgo",
            "display": "Ginkgo",
            "cmd": f"{python} {EXP_ROOT / 'solvers' / 'ginkgo' / 'ginkgo_wrapper.py'}",
            "dtypes": ["fp64", "fp32"],
            "config": None,
        },
        {
            "name": "superlu_dist",
            "display": "SuperLU_DIST",
            "cmd": f"{python} {EXP_ROOT / 'solvers' / 'superlu_dist' / 'superlu_dist_wrapper.py'}",
            "dtypes": ["fp64", "fp32"],
            "config": None,
        },
        {
            "name": "strumpack",
            "display": "STRUMPACK",
            "cmd": f"{python} {EXP_ROOT / 'solvers' / 'strumpack' / 'strumpack_wrapper.py'}",
            "dtypes": ["fp64", "fp32"],
            "config": None,
        },
    ]


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def synthesize_failure(
    *,
    out_path: Path,
    spec: Dict[str, Any],
    system: Dict[str, Any],
    dtype: str,
    warmup: int,
    repeats: int,
    status: str,
    notes: str,
) -> Dict[str, Any]:
    meta = system["meta_data"]
    result = {
        "solver_name": spec["display"],
        "solver_version": "unavailable",
        "library_path": "",
        "build_status": status,
        "dtype": dtype,
        "case_name": meta.get("case_name", ""),
        "iteration": meta.get("iteration", -1),
        "matrix_rows": meta.get("matrix_rows", 0),
        "matrix_cols": meta.get("matrix_cols", 0),
        "nnz": meta.get("nnz", 0),
        "repeat_count": repeats,
        "warmup_count": warmup,
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
        "gpu_resident_after_initial_load": "unavailable",
        "notes": notes,
        "timing_stats": {},
        "raw_json": str(out_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_solver(spec: Dict[str, Any], system: Dict[str, Any], dtype: str, warmup: int, repeats: int, timeout: int) -> Dict[str, Any]:
    out_name = f"{safe_name(system['case_name'])}_iter{system['iteration']:03d}_{spec['name']}_{dtype}.json"
    out_path = RAW_DIR / out_name
    if spec["cmd"] is None:
        return synthesize_failure(
            out_path=out_path,
            spec=spec,
            system=system,
            dtype=dtype,
            warmup=warmup,
            repeats=repeats,
            status="build_failed",
            notes=f"{spec['display']} executable was not found in {BUILD_DIR}. Build or install logs should be checked under {LOG_DIR}.",
        )

    base_cmd = spec["cmd"].split()
    cmd = [
        *base_cmd,
        "--matrix",
        str(system["matrix"]),
        "--rhs",
        str(system["rhs"]),
        "--xref",
        str(system["xref"]),
        "--meta",
        str(system["meta"]),
        "--dtype",
        dtype,
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--out",
        str(out_path),
    ]
    if spec.get("config"):
        cmd.extend(["--config", str(spec["config"])])
    log_base = LOG_DIR / f"{out_path.stem}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        (log_base.with_suffix(".stdout.log")).write_text(proc.stdout, encoding="utf-8")
        (log_base.with_suffix(".stderr.log")).write_text(proc.stderr, encoding="utf-8")
        if out_path.exists():
            result = load_json(out_path)
            result.setdefault("command", " ".join(cmd))
            result["raw_json"] = str(out_path)
            out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return result
        notes = (
            f"Command failed before writing output JSON; returncode={proc.returncode}; "
            f"stderr_tail={proc.stderr[-1000:]}"
        )
        return synthesize_failure(
            out_path=out_path,
            spec=spec,
            system=system,
            dtype=dtype,
            warmup=warmup,
            repeats=repeats,
            status="runtime_failed",
            notes=notes,
        )
    except subprocess.TimeoutExpired as exc:
        notes = f"Command timed out after {timeout}s; elapsed={time.time() - started:.1f}s; command={' '.join(cmd)}"
        return synthesize_failure(
            out_path=out_path,
            spec=spec,
            system=system,
            dtype=dtype,
            warmup=warmup,
            repeats=repeats,
            status="runtime_failed",
            notes=notes,
        )


def write_summary(results: List[Dict[str, Any]]) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out = SUMMARY_DIR / "summary.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case": result.get("case_name", ""),
                    "iteration": result.get("iteration", ""),
                    "solver": result.get("solver_name", ""),
                    "dtype": result.get("dtype", ""),
                    "build_status": result.get("build_status", ""),
                    "matrix_rows": result.get("matrix_rows", ""),
                    "matrix_cols": result.get("matrix_cols", ""),
                    "nnz": result.get("nnz", ""),
                    "analysis_ms": result.get("analysis_ms", ""),
                    "factorization_ms": result.get("factorization_ms", ""),
                    "solve_ms": result.get("solve_ms", ""),
                    "total_solver_ms": result.get("total_solver_ms", ""),
                    "total_end_to_end_ms": result.get("total_end_to_end_ms", ""),
                    "relative_residual_2": result.get("relative_residual_2", ""),
                    "relative_error_to_x_ref_2": result.get("relative_error_to_x_ref_2", ""),
                    "converged": result.get("converged", ""),
                    "num_iterations": result.get("num_iterations", ""),
                    "gpu_resident_after_initial_load": result.get("gpu_resident_after_initial_load", ""),
                    "notes": result.get("notes", ""),
                    "raw_json": result.get("raw_json", ""),
                }
            )
    return out


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    systems = discover_systems(Path(args.dump_root), args.systems)
    if not systems:
        raise SystemExit(f"No dumped systems found under {args.dump_root}")
    solver_filter = {s.strip() for s in args.solvers.split(",") if s.strip()}
    dtype_filter = {d.strip() for d in args.dtypes.split(",") if d.strip()}
    specs = [s for s in solver_specs(Path(args.build_dir)) if not solver_filter or s["name"] in solver_filter or s["display"] in solver_filter]
    results: List[Dict[str, Any]] = []
    for system in systems:
        for spec in specs:
            for dtype in spec["dtypes"]:
                if dtype not in dtype_filter:
                    continue
                print(f"[run_all] {system['case_name']} iter {system['iteration']} {spec['display']} {dtype}", flush=True)
                result = run_solver(spec, system, dtype, args.warmup, args.repeats, args.timeout)
                results.append(result)
    summary = write_summary(results)
    print(summary)


if __name__ == "__main__":
    main()
