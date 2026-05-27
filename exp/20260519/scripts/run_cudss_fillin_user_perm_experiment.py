#!/usr/bin/env python3
"""Compare METIS symbolic fill estimates with cuDSS analyze and user permutations."""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
DEFAULT_BENCH = EXP_ROOT / "build" / "cudss_pf_benchmark"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_REPORT_DIR = EXP_ROOT / "report"
DEFAULT_FIG_DIR = EXP_ROOT / "figs" / "cudss_fillin_user_perm"
DEFAULT_CUDSS_THREADING_LIB = Path(
    "/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so"
)
DEFAULT_CASES = [
    "case118",
    "case1197",
    "case3012wp",
    "case6468rte",
    "case8387pegase",
]
PERM_MODES = ["default", "metis"]

SUMMARY_FIELDS = [
    "status",
    "case_name",
    "cudss_perm_mode",
    "precision",
    "rhs_mode",
    "cudss_version",
    "n_bus",
    "n_pv",
    "n_pq",
    "linear_dim",
    "linear_nnz",
    "warmup",
    "repeats",
    "metis_ordering_ms",
    "metis_symbolic_ms",
    "metis_num_fronts",
    "metis_num_levels",
    "metis_max_front_size",
    "metis_total_dense_entries",
    "metis_total_dense_bytes",
    "metis_dense_entries_per_nnz",
    "cudss_analysis_lu_nnz",
    "cudss_factor_lu_nnz",
    "cudss_factor_lu_nnz_per_nnz",
    "analysis_ms_mean",
    "factor_ms_mean",
    "solve_ms_mean",
    "total_ms_mean",
    "relative_residual",
    "relative_error",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--precision", choices=["fp64", "fp32"], default="fp64")
    parser.add_argument("--rhs-mode", choices=["synthetic", "mismatch"], default="synthetic")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--local-metis-symbolic", action="store_true")
    parser.add_argument("--enable-mt", action="store_true", default=True)
    parser.add_argument("--disable-mt", action="store_false", dest="enable_mt")
    parser.add_argument("--cudss-threading-lib", type=Path, default=DEFAULT_CUDSS_THREADING_LIB)
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if args.enable_mt and args.cudss_threading_lib.exists():
        lib = str(args.cudss_threading_lib)
        env["CUDSS_THREADING_LIB"] = lib
        preload_items = [item for item in env.get("LD_PRELOAD", "").split() if item]
        if lib not in preload_items:
            preload_items.insert(0, lib)
        env["LD_PRELOAD"] = " ".join(preload_items)
    return env


def parse_csv_row(stdout: str) -> Dict[str, str]:
    rows = list(csv.DictReader(io.StringIO(stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"expected one CSV row, got {len(rows)} rows: {stdout[-1000:]}")
    return rows[0]


def failure_row(case_name: str, mode: str, args: argparse.Namespace, notes: str) -> Dict[str, str]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "status": "failed",
            "case_name": case_name,
            "cudss_perm_mode": mode,
            "precision": args.precision,
            "rhs_mode": args.rhs_mode,
            "warmup": str(args.warmup),
            "repeats": str(args.repeats),
            "notes": notes,
        }
    )
    return row


def run_one(args: argparse.Namespace, case_name: str, mode: str, env: Dict[str, str]) -> Dict[str, str]:
    case_dir = args.dataset_root / case_name
    if not case_dir.exists():
        return failure_row(case_name, mode, args, f"missing case dir: {case_dir}")

    cmd = [
        str(args.bench),
        "--case-dir",
        str(case_dir),
        "--case",
        case_name,
        "--precision",
        args.precision,
        "--rhs-mode",
        args.rhs_mode,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--cudss-perm",
        mode,
        "--csv",
    ]
    if args.local_metis_symbolic:
        cmd.append("--metis-symbolic")
    if args.enable_mt:
        cmd.append("--enable-mt")
        if args.cudss_threading_lib.exists():
            cmd.extend(["--threading-lib", str(args.cudss_threading_lib)])

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=args.timeout,
    )
    if completed.returncode != 0:
        notes = f"returncode={completed.returncode}; stderr_tail={completed.stderr[-1000:]}"
        return failure_row(case_name, mode, args, notes)

    try:
        row = parse_csv_row(completed.stdout)
    except Exception as exc:  # noqa: BLE001
        return failure_row(case_name, mode, args, str(exc))
    row["status"] = "ok"
    row["notes"] = ""
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: Dict[str, str], field: str) -> float:
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return float("nan")


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def fmt_int(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return ""


def pct_delta(new: float, base: float) -> float:
    if not np.isfinite(new) or not np.isfinite(base) or base == 0.0:
        return float("nan")
    return 100.0 * (new / base - 1.0)


def rows_by_case_mode(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    return {(row.get("case_name", ""), row.get("cudss_perm_mode", "")): row for row in rows}


def write_markdown(path: Path, rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key = rows_by_case_mode(rows)
    has_local_symbolic = any(row.get("metis_total_dense_entries") for row in rows)
    lines: List[str] = []
    lines.append("# cuDSS Fill-In and METIS User Permutation Experiment")
    lines.append("")
    lines.append(f"- precision: `{args.precision}`")
    lines.append(f"- rhs_mode: `{args.rhs_mode}`")
    lines.append(f"- warmup/repeats: `{args.warmup}/{args.repeats}`")
    lines.append(f"- cuDSS threading layer: `{'enabled' if args.enable_mt else 'disabled'}`")
    lines.append("")
    if has_local_symbolic:
        lines.append(
            "METIS symbolic entries are dense frontal-storage entries from the local symbolic path; "
            "cuDSS LU_NNZ is `CUDSS_DATA_LU_NNZ` after `CUDSS_PHASE_ANALYSIS`."
        )
    else:
        lines.append(
            "Fill-in is measured as cuDSS `CUDSS_DATA_LU_NNZ` immediately after "
            "`CUDSS_PHASE_ANALYSIS`; `metis` means METIS NodeND was supplied through "
            "`CUDSS_DATA_USER_PERM` before analysis."
        )
    lines.append("")
    lines.append("## Fill Metrics")
    lines.append("")
    if has_local_symbolic:
        lines.append(
            "| case | n | J nnz | METIS dense entries | METIS/J nnz | cuDSS LU default | cuDSS LU METIS user | LU METIS/default |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append(
            "| case | n | J nnz | cuDSS LU default | default/J nnz | cuDSS LU METIS user | METIS/J nnz | LU METIS/default |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for case_name in args.cases:
        default = by_key.get((case_name, "default"), {})
        metis = by_key.get((case_name, "metis"), {})
        default_lu = as_float(default, "cudss_analysis_lu_nnz")
        metis_lu = as_float(metis, "cudss_analysis_lu_nnz")
        matrix_nnz = as_float(default, "linear_nnz")
        if not np.isfinite(matrix_nnz):
            matrix_nnz = as_float(metis, "linear_nnz")
        if has_local_symbolic:
            lines.append(
                "| {case} | {n} | {nnz} | {dense} | {dense_ratio} | {lu_default} | {lu_metis} | {lu_ratio} |".format(
                    case=case_name,
                    n=fmt_int(default.get("linear_dim", "") or metis.get("linear_dim", "")),
                    nnz=fmt_int(default.get("linear_nnz", "") or metis.get("linear_nnz", "")),
                    dense=fmt_int(default.get("metis_total_dense_entries", "") or metis.get("metis_total_dense_entries", "")),
                    dense_ratio=fmt_float(default.get("metis_dense_entries_per_nnz", "") or metis.get("metis_dense_entries_per_nnz", ""), 2),
                    lu_default=fmt_int(default_lu),
                    lu_metis=fmt_int(metis_lu),
                    lu_ratio=fmt_float(metis_lu / default_lu if default_lu else float("nan"), 3),
                )
            )
        else:
            lines.append(
                "| {case} | {n} | {nnz} | {lu_default} | {default_ratio} | {lu_metis} | {metis_ratio} | {lu_ratio} |".format(
                    case=case_name,
                    n=fmt_int(default.get("linear_dim", "") or metis.get("linear_dim", "")),
                    nnz=fmt_int(default.get("linear_nnz", "") or metis.get("linear_nnz", "")),
                    lu_default=fmt_int(default_lu),
                    default_ratio=fmt_float(default_lu / matrix_nnz if matrix_nnz else float("nan"), 3),
                    lu_metis=fmt_int(metis_lu),
                    metis_ratio=fmt_float(metis_lu / matrix_nnz if matrix_nnz else float("nan"), 3),
                    lu_ratio=fmt_float(metis_lu / default_lu if default_lu else float("nan"), 3),
                )
            )
    lines.append("")
    lines.append("## Timing Impact")
    lines.append("")
    lines.append(
        "| case | analysis default ms | analysis METIS ms | factor default ms | factor METIS ms | factor delta | solve default ms | solve METIS ms | solve delta |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case_name in args.cases:
        default = by_key.get((case_name, "default"), {})
        metis = by_key.get((case_name, "metis"), {})
        factor_default = as_float(default, "factor_ms_mean")
        factor_metis = as_float(metis, "factor_ms_mean")
        solve_default = as_float(default, "solve_ms_mean")
        solve_metis = as_float(metis, "solve_ms_mean")
        lines.append(
            "| {case} | {analysis_default} | {analysis_metis} | {factor_default} | {factor_metis} | {factor_delta}% | {solve_default} | {solve_metis} | {solve_delta}% |".format(
                case=case_name,
                analysis_default=fmt_float(default.get("analysis_ms_mean", "")),
                analysis_metis=fmt_float(metis.get("analysis_ms_mean", "")),
                factor_default=fmt_float(factor_default),
                factor_metis=fmt_float(factor_metis),
                factor_delta=fmt_float(pct_delta(factor_metis, factor_default), 1),
                solve_default=fmt_float(solve_default),
                solve_metis=fmt_float(solve_metis),
                solve_delta=fmt_float(pct_delta(solve_metis, solve_default), 1),
            )
        )
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for row in failures:
            lines.append(f"- `{row.get('case_name', '')}` `{row.get('cudss_perm_mode', '')}`: {row.get('notes', '')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_timings(path: Path, rows: List[Dict[str, str]], cases: List[str]) -> None:
    by_key = rows_by_case_mode(rows)
    x = np.arange(len(cases))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    for ax, field, title in [
        (axes[0], "factor_ms_mean", "Factorization time"),
        (axes[1], "solve_ms_mean", "Solve time"),
    ]:
        default_values = [as_float(by_key.get((case, "default"), {}), field) for case in cases]
        metis_values = [as_float(by_key.get((case, "metis"), {}), field) for case in cases]
        ax.bar(x - width / 2, default_values, width, label="cuDSS default")
        ax.bar(x + width / 2, metis_values, width, label="METIS user perm")
        ax.set_ylabel("ms")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(cases, rotation=20, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fill(path: Path, rows: List[Dict[str, str]], cases: List[str]) -> None:
    by_key = rows_by_case_mode(rows)
    x = np.arange(len(cases))
    width = 0.25
    default_lu = [as_float(by_key.get((case, "default"), {}), "cudss_analysis_lu_nnz") for case in cases]
    metis_lu = [as_float(by_key.get((case, "metis"), {}), "cudss_analysis_lu_nnz") for case in cases]
    dense_entries = [as_float(by_key.get((case, "default"), {}), "metis_total_dense_entries") for case in cases]
    has_dense_entries = any(np.isfinite(value) for value in dense_entries)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    if has_dense_entries:
        ax.bar(x - width, default_lu, width, label="cuDSS LU_NNZ default")
        ax.bar(x, metis_lu, width, label="cuDSS LU_NNZ METIS user")
        ax.bar(x + width, dense_entries, width, label="METIS symbolic dense entries")
    else:
        width = 0.35
        ax.bar(x - width / 2, default_lu, width, label="cuDSS LU_NNZ default")
        ax.bar(x + width / 2, metis_lu, width, label="cuDSS LU_NNZ METIS user")
    ax.set_yscale("log")
    ax.set_ylabel("entry count (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.bench.exists():
        raise FileNotFoundError(f"benchmark binary not found: {args.bench}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

    env = build_env(args)
    rows: List[Dict[str, str]] = []
    for case_name in args.cases:
        for mode in PERM_MODES:
            rows.append(run_one(args, case_name, mode, env))

    csv_path = args.report_dir / "cudss_metis_user_perm_fillin.csv"
    md_path = args.report_dir / "cudss_metis_user_perm_fillin.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, args)
    plot_timings(args.fig_dir / "cudss_metis_user_perm_timings.png", rows, list(args.cases))
    plot_fill(args.fig_dir / "cudss_metis_fill_metrics.png", rows, list(args.cases))
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {args.fig_dir}")


if __name__ == "__main__":
    main()
