#!/usr/bin/env python3
"""Run cuDSS phase timings for representative MATPOWER case dumps."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
DEFAULT_BENCH = EXP_ROOT / "build" / "cudss_pf_benchmark"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_REPORT_DIR = EXP_ROOT / "report"
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

SUMMARY_FIELDS = [
    "status",
    "case_name",
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
    "analysis_ms_mean",
    "analysis_ms_median",
    "analysis_ms_min",
    "analysis_ms_max",
    "analysis_ms_stddev",
    "factor_ms_mean",
    "factor_ms_median",
    "factor_ms_min",
    "factor_ms_max",
    "factor_ms_stddev",
    "solve_ms_mean",
    "solve_ms_median",
    "solve_ms_min",
    "solve_ms_max",
    "solve_ms_stddev",
    "total_ms_mean",
    "total_ms_median",
    "total_ms_min",
    "total_ms_max",
    "total_ms_stddev",
    "residual_norm",
    "relative_residual",
    "relative_error",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cuDSS analysis/factorization/solve timings and write exp/20260519/report outputs."
    )
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--precision", default="fp64", choices=["fp64", "fp32"])
    parser.add_argument("--rhs-mode", default="synthetic", choices=["synthetic", "mismatch"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=1800)
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


def metadata_for_case(dataset_root: Path, case_name: str) -> Dict[str, object]:
    path = dataset_root / case_name / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_row(stdout: str) -> Dict[str, str]:
    rows = list(csv.DictReader(io.StringIO(stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"expected one CSV row, got {len(rows)} rows: {stdout[-1000:]}")
    return rows[0]


def failure_row(
    *,
    case_name: str,
    args: argparse.Namespace,
    notes: str,
    metadata: Dict[str, object],
) -> Dict[str, str]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "status": "failed",
            "case_name": case_name,
            "precision": args.precision,
            "rhs_mode": args.rhs_mode,
            "n_bus": str(metadata.get("n_bus", "")),
            "warmup": str(args.warmup),
            "repeats": str(args.repeats),
            "notes": notes,
        }
    )
    return row


def run_case(args: argparse.Namespace, case_name: str, env: Dict[str, str]) -> Dict[str, str]:
    case_dir = args.dataset_root / case_name
    metadata = metadata_for_case(args.dataset_root, case_name)
    if not case_dir.exists():
        return failure_row(case_name=case_name, args=args, metadata=metadata, notes=f"missing case dir: {case_dir}")

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
        "--csv",
    ]
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
        return failure_row(case_name=case_name, args=args, metadata=metadata, notes=notes)

    try:
        row = parse_csv_row(completed.stdout)
    except Exception as exc:  # noqa: BLE001
        return failure_row(case_name=case_name, args=args, metadata=metadata, notes=str(exc))
    row["status"] = "ok"
    row["notes"] = ""
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: str, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def fmt_sci(value: str) -> str:
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return ""


def write_markdown(path: Path, rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# cuDSS Power Flow Jacobian Benchmark")
    lines.append("")
    lines.append(f"- precision: `{args.precision}`")
    lines.append(f"- rhs_mode: `{args.rhs_mode}`")
    lines.append(f"- warmup/repeats: `{args.warmup}/{args.repeats}`")
    lines.append(f"- cuDSS threading layer: `{'enabled' if args.enable_mt else 'disabled'}`")
    lines.append("")
    lines.append(
        "| status | case | n_bus | linear_dim | nnz | analysis ms | factorize ms | solve ms | total ms | rel residual | rel error |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {status} | {case} | {n_bus} | {dim} | {nnz} | {analysis} | {factor} | {solve} | {total} | {relres} | {relerr} |".format(
                status=row.get("status", ""),
                case=row.get("case_name", ""),
                n_bus=row.get("n_bus", ""),
                dim=row.get("linear_dim", ""),
                nnz=row.get("linear_nnz", ""),
                analysis=fmt_float(row.get("analysis_ms_mean", "")),
                factor=fmt_float(row.get("factor_ms_mean", "")),
                solve=fmt_float(row.get("solve_ms_mean", "")),
                total=fmt_float(row.get("total_ms_mean", "")),
                relres=fmt_sci(row.get("relative_residual", "")),
                relerr=fmt_sci(row.get("relative_error", "")),
            )
        )
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for row in failures:
            lines.append(f"- `{row.get('case_name', '')}`: {row.get('notes', '')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.bench.exists():
        raise FileNotFoundError(f"benchmark binary not found: {args.bench}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

    env = build_env(args)
    rows: List[Dict[str, str]] = []
    for index, case_name in enumerate(args.cases, start=1):
        row = run_case(args, case_name, env)
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"[OK] {index}/{len(args.cases)} {case_name} "
                f"analysis={fmt_float(row['analysis_ms_mean'])}ms "
                f"factor={fmt_float(row['factor_ms_mean'])}ms "
                f"solve={fmt_float(row['solve_ms_mean'])}ms"
            )
        else:
            print(f"[FAIL] {index}/{len(args.cases)} {case_name}: {row['notes']}")

    csv_path = args.report_dir / "cudss_pf_case_timings.csv"
    md_path = args.report_dir / "cudss_pf_case_timings.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, args)
    print(f"[DONE] csv={csv_path}")
    print(f"[DONE] md={md_path}")


if __name__ == "__main__":
    main()
