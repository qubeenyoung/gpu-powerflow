#!/usr/bin/env python3
"""Compare GLU CUDA numeric factorization against the 20260519 cuDSS report."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "exp" / "20260520"
DEFAULT_BENCH = EXP_ROOT / "build" / "glu_pf_benchmark"
DEFAULT_CUDSS_CSV = REPO_ROOT / "exp" / "20260519" / "report" / "cudss_pf_case_timings.csv"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_OUT_CSV = EXP_ROOT / "report" / "glu_vs_cudss_factorize.csv"
DEFAULT_OUT_MD = EXP_ROOT / "report" / "glu_vs_cudss_factorize.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GLU on the cuDSS timing cases and compare factorization time."
    )
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--cudss-csv", type=Path, default=DEFAULT_CUDSS_CSV)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--perturb", action="store_true")
    return parser.parse_args()


def read_cudss_rows(path: Path, requested_cases: Iterable[str] | None) -> List[Dict[str, str]]:
    requested = set(requested_cases or [])
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if row.get("status", "ok") == "ok"]
    if requested:
        rows = [row for row in rows if row["case_name"] in requested]
    return rows


def run_glu_once(bench: Path, case_dir: Path, perturb: bool) -> Dict[str, str]:
    cmd = [
        str(bench),
        "--case-dir",
        str(case_dir),
        "--rhs-mode",
        "synthetic",
        "--csv",
    ]
    if perturb:
        cmd.append("--perturb")
    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"GLU benchmark did not emit CSV rows for {case_dir}")
    return next(csv.DictReader(lines))


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        nan = float("nan")
        return {"mean": nan, "median": nan, "min": nan, "max": nan, "stddev": nan}
    mean = statistics.fmean(values)
    stddev = math.sqrt(statistics.fmean([(value - mean) ** 2 for value in values]))
    return {
        "mean": mean,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stddev": stddev,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_name",
        "n_bus",
        "linear_dim",
        "linear_nnz",
        "cudss_factor_ms_mean",
        "glu_numeric_gpu_event_ms_mean",
        "glu_numeric_gpu_event_ms_median",
        "glu_numeric_gpu_event_ms_min",
        "glu_numeric_gpu_event_ms_max",
        "glu_numeric_host_ms_mean",
        "glu_vs_cudss_factor_ratio",
        "cudss_vs_glu_factor_speedup",
        "glu_relative_residual",
        "glu_relative_error",
        "warmup",
        "repeats",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) >= 1.0e4 or (value != 0.0 and abs(value) < 1.0e-3):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ratios = [float(row["glu_vs_cudss_factor_ratio"]) for row in rows]
    arithmetic_ratio = statistics.fmean(ratios) if ratios else float("nan")
    geometric_ratio = (
        math.exp(statistics.fmean([math.log(value) for value in ratios]))
        if ratios
        else float("nan")
    )
    with path.open("w") as f:
        f.write("# GLU CUDA Numeric Factorization vs cuDSS\n\n")
        f.write("- cuDSS source: `exp/20260519/report/cudss_pf_case_timings.csv`\n")
        f.write("- mode: `fp64`, `synthetic` RHS, cuDSS threading enabled\n")
        if rows:
            f.write(f"- GLU runs: warmup/repeats `{rows[0]['warmup']}/{rows[0]['repeats']}`\n")
        f.write("- GLU factor time uses `numeric_gpu_event_ms`, parsed from GLU `Total GPU time` inside `LUonDevice`.\n")
        f.write("- GLU uses single-precision numeric kernels because upstream GLU defines `REAL` as `float`.\n\n")
        f.write("## Summary\n\n")
        f.write(f"- GLU CUDA factorization is slower than cuDSS on all {len(rows)} cases.\n")
        f.write(f"- Arithmetic mean GLU/cuDSS factor ratio: `{arithmetic_ratio:.2f}x`.\n")
        f.write(f"- Geometric mean GLU/cuDSS factor ratio: `{geometric_ratio:.2f}x`.\n")
        if ratios:
            f.write(
                f"- Best observed ratio: `{min(ratios):.2f}x`; "
                f"worst observed ratio: `{max(ratios):.2f}x`.\n"
            )
        f.write("\n")
        f.write(
            "| case | n_bus | dim | nnz | cuDSS factor ms | GLU CUDA factor ms | "
            "GLU/cuDSS | cuDSS/GLU speedup | GLU rel residual | GLU rel error |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['case_name']} | {row['n_bus']} | {row['linear_dim']} | "
                f"{row['linear_nnz']} | {fmt(row['cudss_factor_ms_mean'])} | "
                f"{fmt(row['glu_numeric_gpu_event_ms_mean'])} | "
                f"{fmt(row['glu_vs_cudss_factor_ratio'], 1)} | "
                f"{fmt(row['cudss_vs_glu_factor_speedup'], 3)} | "
                f"{fmt(row['glu_relative_residual'])} | {fmt(row['glu_relative_error'])} |\n"
            )


def main() -> int:
    args = parse_args()
    cudss_rows = read_cudss_rows(args.cudss_csv, args.cases)
    output_rows: List[Dict[str, object]] = []

    for cudss in cudss_rows:
        case_name = cudss["case_name"]
        case_dir = args.dataset_root / case_name
        if not case_dir.exists():
            raise FileNotFoundError(case_dir)

        for _ in range(args.warmup):
            run_glu_once(args.bench, case_dir, args.perturb)

        gpu_event_ms: List[float] = []
        host_ms: List[float] = []
        rel_residual = float("nan")
        rel_error = float("nan")
        for _ in range(args.repeats):
            row = run_glu_once(args.bench, case_dir, args.perturb)
            gpu_event_ms.append(float(row["numeric_gpu_event_ms"]))
            host_ms.append(float(row["numeric_ms"]))
            rel_residual = float(row["relative_residual"])
            rel_error = float(row["relative_error"])

        gpu = stats(gpu_event_ms)
        host = stats(host_ms)
        cudss_factor = float(cudss["factor_ms_mean"])
        ratio = gpu["mean"] / cudss_factor
        speedup = cudss_factor / gpu["mean"]
        output_rows.append(
            {
                "case_name": case_name,
                "n_bus": int(cudss["n_bus"]),
                "linear_dim": int(cudss["linear_dim"]),
                "linear_nnz": int(cudss["linear_nnz"]),
                "cudss_factor_ms_mean": cudss_factor,
                "glu_numeric_gpu_event_ms_mean": gpu["mean"],
                "glu_numeric_gpu_event_ms_median": gpu["median"],
                "glu_numeric_gpu_event_ms_min": gpu["min"],
                "glu_numeric_gpu_event_ms_max": gpu["max"],
                "glu_numeric_host_ms_mean": host["mean"],
                "glu_vs_cudss_factor_ratio": ratio,
                "cudss_vs_glu_factor_speedup": speedup,
                "glu_relative_residual": rel_residual,
                "glu_relative_error": rel_error,
                "warmup": args.warmup,
                "repeats": args.repeats,
            }
        )
        print(f"{case_name}: GLU/CUDSS factor ratio {ratio:.1f}x")

    write_csv(args.out_csv, output_rows)
    write_markdown(args.out_md, output_rows)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
