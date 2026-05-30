#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from ctypes.util import find_library
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = Path("/tmp/cupf_custom_py_build")
DEFAULT_MAT_ROOT = Path("/datasets/matpower_mat")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
BACKENDS = ("cudss", "custom")
RAW_FIELDS = [
    "case",
    "backend",
    "repeat",
    "status",
    "initialize_ms",
    "solve_ms",
    "total_ms",
    "iterations",
    "final_mismatch",
    "converged",
    "n_bus",
    "ybus_nnz",
    "n_pv",
    "n_pq",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cuPF FP64 cuDSS and custom linear solver on MATPOWER cases."
    )
    parser.add_argument("--mat-root", type=Path, default=DEFAULT_MAT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cases", nargs="*", help="Optional case stems. Defaults to all .mat cases.")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing raw CSV.")
    return parser.parse_args()


def prepare_imports(build_dir: Path) -> None:
    for path in (build_dir, REPO_ROOT / "cuPF" / "python", REPO_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_cuda_synchronizer():
    candidates = []
    found = find_library("cudart")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            "libcudart.so",
            "libcudart.so.12",
            "/usr/local/cuda/lib64/libcudart.so",
        ]
    )
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
        except OSError:
            continue
        lib.cudaDeviceSynchronize.restype = ctypes.c_int

        def sync() -> None:
            err = int(lib.cudaDeviceSynchronize())
            if err != 0:
                raise RuntimeError(f"cudaDeviceSynchronize failed: {err}")

        return sync

    def no_sync() -> None:
        return None

    return no_sync


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def case_names(mat_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return [Path(name).stem for name in requested]
    return sorted(path.stem for path in mat_root.glob("*.mat"))


def make_options(cupf: Any, backend: str):
    opts = cupf.NewtonOptions()
    opts.backend = cupf.BackendKind.CUDA
    opts.compute = cupf.ComputePolicy.FP64
    if backend == "custom":
        opts.cuda_linear_solver = cupf.CudaLinearSolverKind.Custom
    else:
        opts.cuda_linear_solver = cupf.CudaLinearSolverKind.CuDSS
    return opts


def row_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["case"], row["backend"], int(row["repeat"])


def load_completed(raw_csv: Path, resume: bool) -> set[tuple[str, str, int]]:
    if not resume or not raw_csv.exists():
        return set()
    with raw_csv.open("r", newline="", encoding="utf-8") as fh:
        return {row_key(row) for row in csv.DictReader(fh)}


def append_row(raw_csv: Path, row: dict[str, Any]) -> None:
    exists = raw_csv.exists()
    with raw_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RAW_FIELDS})
        fh.flush()
        os.fsync(fh.fileno())


def benchmark_once(cupf: Any, sync_cuda, case_data: Any, backend: str, repeat: int, args: argparse.Namespace) -> dict[str, Any]:
    ybus = case_data.Ybus
    cfg = cupf.NRConfig()
    cfg.tolerance = args.tolerance
    cfg.max_iter = args.max_iter

    row: dict[str, Any] = {
        "case": case_data.case_stem,
        "backend": backend,
        "repeat": repeat,
        "n_bus": int(ybus.shape[0]),
        "ybus_nnz": int(ybus.nnz),
        "n_pv": int(case_data.pv.size),
        "n_pq": int(case_data.pq.size),
    }

    solver = None
    try:
        solver = cupf.NewtonSolver(make_options(cupf, backend))
        sync_cuda()
        t0 = time.perf_counter()
        solver.initialize(
            ybus.indptr,
            ybus.indices,
            ybus.data,
            int(ybus.shape[0]),
            int(ybus.shape[1]),
            case_data.pv,
            case_data.pq,
        )
        sync_cuda()
        t1 = time.perf_counter()

        result = solver.solve(
            ybus.indptr,
            ybus.indices,
            ybus.data,
            int(ybus.shape[0]),
            int(ybus.shape[1]),
            case_data.Sbus,
            case_data.V0,
            case_data.pv,
            case_data.pq,
            cfg,
        )
        sync_cuda()
        t2 = time.perf_counter()

        row.update(
            {
                "status": "ok",
                "initialize_ms": (t1 - t0) * 1000.0,
                "solve_ms": (t2 - t1) * 1000.0,
                "total_ms": (t2 - t0) * 1000.0,
                "iterations": int(result.iterations),
                "final_mismatch": float(result.final_mismatch),
                "converged": bool(result.converged),
                "error": "",
            }
        )
    except Exception as exc:
        sync_cuda()
        row.update(
            {
                "status": "error",
                "initialize_ms": "",
                "solve_ms": "",
                "total_ms": "",
                "iterations": "",
                "final_mismatch": "",
                "converged": "",
                "error": repr(exc),
            }
        )
    finally:
        del solver
        gc.collect()
        sync_cuda()
    return row


def numeric(row: dict[str, str], field: str) -> float:
    return float(row[field])


def summarize(raw_csv: Path, summary_csv: Path, comparison_csv: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(raw_csv.open("r", newline="", encoding="utf-8")))
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["backend"])].append(row)

    summary_fields = [
        "case",
        "backend",
        "n_bus",
        "ybus_nnz",
        "n_pv",
        "n_pq",
        "runs",
        "ok_runs",
        "converged_runs",
        "initialize_mean_ms",
        "initialize_median_ms",
        "initialize_min_ms",
        "initialize_max_ms",
        "initialize_std_ms",
        "solve_mean_ms",
        "solve_median_ms",
        "solve_min_ms",
        "solve_max_ms",
        "solve_std_ms",
        "total_mean_ms",
        "total_median_ms",
        "final_mismatch_median",
        "iterations_median",
        "first_error",
    ]

    summaries: list[dict[str, Any]] = []
    for (case, backend), group_rows in sorted(groups.items()):
        ok = [row for row in group_rows if row["status"] == "ok"]

        def stats(field: str, reducer: str) -> float | str:
            if not ok:
                return ""
            values = [numeric(row, field) for row in ok]
            if reducer == "mean":
                return statistics.fmean(values)
            if reducer == "median":
                return statistics.median(values)
            if reducer == "min":
                return min(values)
            if reducer == "max":
                return max(values)
            if reducer == "std":
                return statistics.stdev(values) if len(values) > 1 else 0.0
            raise ValueError(reducer)

        first = group_rows[0]
        first_error = next((row["error"] for row in group_rows if row["status"] != "ok"), "")
        summaries.append(
            {
                "case": case,
                "backend": backend,
                "n_bus": first["n_bus"],
                "ybus_nnz": first["ybus_nnz"],
                "n_pv": first["n_pv"],
                "n_pq": first["n_pq"],
                "runs": len(group_rows),
                "ok_runs": len(ok),
                "converged_runs": sum(1 for row in ok if row["converged"] == "True"),
                "initialize_mean_ms": stats("initialize_ms", "mean"),
                "initialize_median_ms": stats("initialize_ms", "median"),
                "initialize_min_ms": stats("initialize_ms", "min"),
                "initialize_max_ms": stats("initialize_ms", "max"),
                "initialize_std_ms": stats("initialize_ms", "std"),
                "solve_mean_ms": stats("solve_ms", "mean"),
                "solve_median_ms": stats("solve_ms", "median"),
                "solve_min_ms": stats("solve_ms", "min"),
                "solve_max_ms": stats("solve_ms", "max"),
                "solve_std_ms": stats("solve_ms", "std"),
                "total_mean_ms": stats("total_ms", "mean"),
                "total_median_ms": stats("total_ms", "median"),
                "final_mismatch_median": statistics.median([numeric(row, "final_mismatch") for row in ok]) if ok else "",
                "iterations_median": statistics.median([numeric(row, "iterations") for row in ok]) if ok else "",
                "first_error": first_error,
            }
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_case[row["case"]][row["backend"]] = row

    comparison_fields = [
        "case",
        "n_bus",
        "ybus_nnz",
        "cudss_ok_runs",
        "custom_ok_runs",
        "cudss_converged_runs",
        "custom_converged_runs",
        "cudss_initialize_mean_ms",
        "custom_initialize_mean_ms",
        "initialize_custom_over_cudss",
        "cudss_solve_mean_ms",
        "custom_solve_mean_ms",
        "solve_custom_over_cudss",
        "cudss_total_mean_ms",
        "custom_total_mean_ms",
        "total_custom_over_cudss",
        "status",
    ]
    comparisons: list[dict[str, Any]] = []
    for case in sorted(by_case):
        cudss = by_case[case].get("cudss")
        custom = by_case[case].get("custom")
        n_bus = (cudss or custom or {}).get("n_bus", "")
        ybus_nnz = (cudss or custom or {}).get("ybus_nnz", "")

        def ratio(num: Any, den: Any) -> float | str:
            try:
                den_f = float(den)
                if den_f == 0.0:
                    return ""
                return float(num) / den_f
            except Exception:
                return ""

        status = "ok"
        if not cudss or not custom:
            status = "missing_backend"
        elif int(cudss["ok_runs"]) == 0 or int(custom["ok_runs"]) == 0:
            status = "backend_error"
        comparisons.append(
            {
                "case": case,
                "n_bus": n_bus,
                "ybus_nnz": ybus_nnz,
                "cudss_ok_runs": cudss["ok_runs"] if cudss else 0,
                "custom_ok_runs": custom["ok_runs"] if custom else 0,
                "cudss_converged_runs": cudss["converged_runs"] if cudss else 0,
                "custom_converged_runs": custom["converged_runs"] if custom else 0,
                "cudss_initialize_mean_ms": cudss["initialize_mean_ms"] if cudss else "",
                "custom_initialize_mean_ms": custom["initialize_mean_ms"] if custom else "",
                "initialize_custom_over_cudss": ratio(
                    custom["initialize_mean_ms"] if custom else "",
                    cudss["initialize_mean_ms"] if cudss else "",
                ),
                "cudss_solve_mean_ms": cudss["solve_mean_ms"] if cudss else "",
                "custom_solve_mean_ms": custom["solve_mean_ms"] if custom else "",
                "solve_custom_over_cudss": ratio(
                    custom["solve_mean_ms"] if custom else "",
                    cudss["solve_mean_ms"] if cudss else "",
                ),
                "cudss_total_mean_ms": cudss["total_mean_ms"] if cudss else "",
                "custom_total_mean_ms": custom["total_mean_ms"] if custom else "",
                "total_custom_over_cudss": ratio(
                    custom["total_mean_ms"] if custom else "",
                    cudss["total_mean_ms"] if cudss else "",
                ),
                "status": status,
            }
        )

    with comparison_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparisons)

    return {
        "raw_rows": len(rows),
        "summary_rows": len(summaries),
        "comparison_rows": len(comparisons),
        "cases_with_both_backends_ok": sum(1 for row in comparisons if row["status"] == "ok"),
        "comparisons": comparisons,
        "summaries": summaries,
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value == "" or value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(report_path: Path, metadata: dict[str, Any], summary: dict[str, Any]) -> None:
    comparisons = summary["comparisons"]
    ok = [row for row in comparisons if row["status"] == "ok"]
    ratio_rows = [row for row in ok if row["total_custom_over_cudss"] != ""]
    faster_custom = [row for row in ratio_rows if float(row["total_custom_over_cudss"]) < 1.0]

    lines = [
        "# cuPF FP64 Linear Solver Benchmark - 2026-05-29",
        "",
        "## Setup",
        "",
        f"- Cases: {metadata['case_count']} MATPOWER cases from `{metadata['mat_root']}`",
        f"- Repeats: {metadata['repeats']} measured runs per case/backend",
        "- Backends: cuDSS FP64 and custom FP64",
        f"- NR config: tolerance={metadata['tolerance']}, max_iter={metadata['max_iter']}",
        "- Timing split: `initialize()` and `solve()` wall time, with `cudaDeviceSynchronize()` after each call.",
        "- One unrecorded warmup is run on the first case for both backends to avoid CUDA context creation skew.",
        "",
        "## Artifacts",
        "",
        "- `raw_runs.csv`: every measured run",
        "- `summary.csv`: per-case/backend aggregate",
        "- `comparison.csv`: cuDSS vs custom ratios per case",
        "- `metadata.json`: environment and command metadata",
        "",
        "## Result Summary",
        "",
        f"- Raw rows: {summary['raw_rows']}",
        f"- Case comparison rows: {summary['comparison_rows']}",
        f"- Both backends produced at least one successful run: {summary['cases_with_both_backends_ok']} / {metadata['case_count']}",
        f"- Custom total mean faster than cuDSS: {len(faster_custom)} / {len(ratio_rows)} comparable cases",
        "",
    ]
    if ratio_rows:
        init_ratios = [float(row["initialize_custom_over_cudss"]) for row in ratio_rows if row["initialize_custom_over_cudss"] != ""]
        solve_ratios = [float(row["solve_custom_over_cudss"]) for row in ratio_rows if row["solve_custom_over_cudss"] != ""]
        total_ratios = [float(row["total_custom_over_cudss"]) for row in ratio_rows if row["total_custom_over_cudss"] != ""]
        lines.extend(
            [
                f"- Median initialize ratio custom/cuDSS: {statistics.median(init_ratios):.3f}x",
                f"- Median solve ratio custom/cuDSS: {statistics.median(solve_ratios):.3f}x",
                f"- Median total ratio custom/cuDSS: {statistics.median(total_ratios):.3f}x",
                "",
            ]
        )

    failed = [row for row in comparisons if row["status"] != "ok"]
    if failed:
        lines.extend(["## Failed Or Incomplete Cases", ""])
        lines.append("| case | status | cudss ok | custom ok |")
        lines.append("|---|---:|---:|---:|")
        for row in failed:
            lines.append(
                f"| {row['case']} | {row['status']} | {row['cudss_ok_runs']} | {row['custom_ok_runs']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Per-Case Mean Times",
            "",
            "| case | buses | cuDSS init ms | custom init ms | init ratio | cuDSS solve ms | custom solve ms | solve ratio | total ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["n_bus"]),
                    fmt(row["cudss_initialize_mean_ms"]),
                    fmt(row["custom_initialize_mean_ms"]),
                    fmt(row["initialize_custom_over_cudss"], 3),
                    fmt(row["cudss_solve_mean_ms"]),
                    fmt(row["custom_solve_mean_ms"]),
                    fmt(row["solve_custom_over_cudss"], 3),
                    fmt(row["total_custom_over_cudss"], 3),
                ]
            )
            + " |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_imports(args.build_dir)

    import cupf
    from python.converters.common import preprocess_case

    sync_cuda = load_cuda_synchronizer()
    cases = case_names(args.mat_root, args.cases)
    if not cases:
        raise RuntimeError(f"No .mat cases found under {args.mat_root}")

    raw_csv = args.output_dir / "raw_runs.csv"
    summary_csv = args.output_dir / "summary.csv"
    comparison_csv = args.output_dir / "comparison.csv"
    metadata_path = args.output_dir / "metadata.json"
    report_path = args.output_dir / "README.md"
    completed = load_completed(raw_csv, resume=not args.no_resume)

    metadata = {
        "date": "2026-05-29",
        "case_count": len(cases),
        "cases": cases,
        "backends": list(BACKENDS),
        "repeats": args.repeats,
        "mat_root": str(args.mat_root),
        "output_dir": str(args.output_dir),
        "build_dir": str(args.build_dir),
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "python": sys.version,
        "platform": platform.platform(),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "nvidia_smi": command_output(["nvidia-smi", "-L"]),
        "command": " ".join(sys.argv),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    warmup_done = False
    for case_idx, name in enumerate(cases, start=1):
        print(f"[case {case_idx}/{len(cases)}] loading {name}", flush=True)
        try:
            case_data = preprocess_case(name, mat_root=args.mat_root)
        except Exception as exc:
            for repeat in range(1, args.repeats + 1):
                for backend in BACKENDS:
                    key = (name, backend, repeat)
                    if key in completed:
                        continue
                    append_row(
                        raw_csv,
                        {
                            "case": name,
                            "backend": backend,
                            "repeat": repeat,
                            "status": "error",
                            "error": f"preprocess failed: {exc!r}",
                        },
                    )
                    completed.add(key)
            continue

        if not warmup_done:
            for backend in BACKENDS:
                print(f"[warmup] {backend} on {case_data.case_stem}", flush=True)
                _ = benchmark_once(cupf, sync_cuda, case_data, backend, 0, args)
            warmup_done = True

        for repeat in range(1, args.repeats + 1):
            for backend in BACKENDS:
                key = (case_data.case_stem, backend, repeat)
                if key in completed:
                    continue
                print(f"[run] {case_data.case_stem} {backend} repeat={repeat}", flush=True)
                row = benchmark_once(cupf, sync_cuda, case_data, backend, repeat, args)
                append_row(raw_csv, row)
                completed.add(key)
                status = row["status"]
                if status == "ok":
                    print(
                        "[ok] "
                        f"{case_data.case_stem} {backend} repeat={repeat} "
                        f"init={row['initialize_ms']:.2f}ms solve={row['solve_ms']:.2f}ms "
                        f"conv={row['converged']} iter={row['iterations']}",
                        flush=True,
                    )
                else:
                    print(f"[error] {case_data.case_stem} {backend}: {row['error']}", flush=True)

    summary = summarize(raw_csv, summary_csv, comparison_csv)
    write_report(report_path, metadata, summary)
    print(f"[done] wrote {raw_csv}", flush=True)
    print(f"[done] wrote {summary_csv}", flush=True)
    print(f"[done] wrote {comparison_csv}", flush=True)
    print(f"[done] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
