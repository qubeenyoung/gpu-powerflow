#!/usr/bin/env python3
"""Shared JSON helpers for unavailable solver wrappers."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict


REQUIRED_RESULT_FIELDS = [
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


def parse_common_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--rhs", required=True)
    parser.add_argument("--xref", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--dtype", choices=["fp64", "fp32"], required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="")
    return parser.parse_args()


def load_meta(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nan() -> float:
    return float("nan")


def unavailable_result(
    *,
    solver_name: str,
    build_status: str,
    args: argparse.Namespace,
    notes: str,
    solver_version: str = "unavailable",
    library_path: str = "",
    gpu_resident_after_initial_load: str = "unavailable",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    meta = load_meta(args.meta)
    result: Dict[str, Any] = {
        "solver_name": solver_name,
        "solver_version": solver_version,
        "library_path": library_path,
        "build_status": build_status,
        "dtype": args.dtype,
        "case_name": meta.get("case_name", ""),
        "iteration": meta.get("iteration", -1),
        "matrix_rows": meta.get("matrix_rows", 0),
        "matrix_cols": meta.get("matrix_cols", 0),
        "nnz": meta.get("nnz", 0),
        "repeat_count": args.repeats,
        "warmup_count": args.warmup,
        "load_ms": nan(),
        "format_convert_ms": nan(),
        "h2d_ms": nan(),
        "analysis_ms": nan(),
        "factorization_ms": nan(),
        "solve_ms": nan(),
        "d2h_ms": nan(),
        "total_solver_ms": nan(),
        "total_end_to_end_ms": nan(),
        "peak_gpu_memory_mb": nan(),
        "relative_residual_2": nan(),
        "relative_error_to_x_ref_2": nan(),
        "converged": False,
        "num_iterations": -1,
        "gpu_resident_after_initial_load": gpu_resident_after_initial_load,
        "notes": notes,
        "timestamp_unix": time.time(),
    }
    if extra:
        result.update(extra)
    return result


def write_result(path: str | Path, result: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True, sort_keys=True)
        f.write("\n")
