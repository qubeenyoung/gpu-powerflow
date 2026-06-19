"""Shared benchmark scaffolding for all user-facing benchmark runners.

All benchmark variants write the same ``runs.csv`` contract under
``benchmark/results/<run-name>/<variant>/``.  Dataset preparation helpers live
in :mod:`benchmark.common.matpower_data`; this module only owns run manifests,
case selection, CSV writing, and common CLI defaults.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .matpower_data import (
    DEFAULT_DATASET_ROOT,
    PreprocessedCase,
    ReferenceResult,
    load_case,
    resolve_case_paths,
    solve_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark" / "results"

REQUIRED_RUN_FIELDS = [
    "mode",
    "variant",
    "case_name",
    "case_path",
    "backend",
    "compute",
    "linear_solver",
    "jacobian",
    "entrypoint",
    "repeat_idx",
    "warmup",
    "success",
    "converged",
    "iterations",
    "error_message",
    "n_bus",
    "ybus_nnz",
    "n_ref",
    "n_pv",
    "n_pq",
    "initialize_ms",
    "solve_ms",
    "total_ms",
    "output_mismatch",
]

EXTRA_RUN_FIELDS = [
    "reported_solve_ms",
    "device_solve_ms",
    "reference_converged",
    "reference_iterations",
    "reference_final_mismatch",
    "max_abs_v_error",
    "rms_abs_v_error",
    "max_abs_vm_error",
    "max_abs_va_error",
]

RUN_FIELDNAMES = REQUIRED_RUN_FIELDS + EXTRA_RUN_FIELDS


def now_run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def bool_for_csv(value: object) -> str:
    return "1" if bool(value) else "0"


class CsvSink:
    def __init__(self, path: Path, fieldnames: list[str] = RUN_FIELDNAMES):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "CsvSink":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cases", nargs="*", help="Case names or .m paths. Defaults to all case*.m files.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases after resolution.")
    parser.add_argument("--run-name", default=now_run_name())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--reference-tolerance", type=float, default=1e-10)
    parser.add_argument("--reference-max-iter", type=int, default=80)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--no-aggregate", action="store_true", help="Skip writing summary.csv / summary.md.")


def selected_case_paths(args: argparse.Namespace) -> list[Path]:
    paths = resolve_case_paths(args.dataset_root, args.cases)
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
    return paths


def run_root(args: argparse.Namespace) -> Path:
    return args.output_root / args.run_name


def variant_dir(args: argparse.Namespace, variant: str) -> Path:
    return run_root(args) / variant


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_skip(out: Path, reason: str, manifest: dict[str, Any] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "SKIPPED.txt").write_text(reason.rstrip() + "\n", encoding="utf-8")
    if manifest is not None:
        write_json(out / "run.json", {**manifest, "skipped": True, "skip_reason": reason})


def manifest(args: argparse.Namespace, mode: str, variant: str, cases: list[Path], **extra: Any) -> dict[str, Any]:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "variant": variant,
        "dataset_root": str(args.dataset_root),
        "cases": [str(path) for path in cases],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "reference_tolerance": args.reference_tolerance,
        "reference_max_iter": args.reference_max_iter,
    }
    payload.update(extra)
    return payload


def dimensions(case: PreprocessedCase) -> dict[str, int]:
    return {
        "n_bus": int(case.ybus.shape[0]),
        "ybus_nnz": int(case.ybus.nnz),
        "n_ref": int(case.ref.size),
        "n_pv": int(case.pv.size),
        "n_pq": int(case.pq.size),
    }


def reference_fields(reference: ReferenceResult) -> dict[str, Any]:
    return {
        "reference_converged": bool_for_csv(reference.converged),
        "reference_iterations": int(reference.iterations),
        "reference_final_mismatch": float(reference.final_mismatch),
    }


def load_case_and_reference(path: Path, args: argparse.Namespace) -> tuple[PreprocessedCase, ReferenceResult]:
    case = load_case(path)
    reference = solve_reference(case, args.reference_tolerance, args.reference_max_iter)
    return case, reference
