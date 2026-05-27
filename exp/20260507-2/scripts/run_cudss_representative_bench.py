#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
import subprocess


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_BENCH = EXP_ROOT / "build" / "cudss_jf_bench"
DEFAULT_DUMP_ROOT = EXP_ROOT / "raw" / "cupf_jf_dumps"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "cudss_representative_j1.csv"
DEFAULT_CUDSS_THREADING_LIB = Path(
    "/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so"
)
DEFAULT_CASES = [
    "case1197",
    "case2736sp",
    "case3375wp",
    "case6468rte",
    "case_ACTIVSg10k",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cuDSS analyze/factorize/solve timings for representative J1/F1 dumps."
    )
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--precision", default="fp64", choices=["fp64", "fp32"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--enable-mt", action="store_true", default=True)
    parser.add_argument("--disable-mt", action="store_false", dest="enable_mt")
    parser.add_argument("--cudss-threading-lib", type=Path, default=DEFAULT_CUDSS_THREADING_LIB)
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.enable_mt and args.cudss_threading_lib.exists():
        lib = str(args.cudss_threading_lib)
        env["CUDSS_THREADING_LIB"] = lib
        preload_items = [item for item in env.get("LD_PRELOAD", "").split() if item]
        if lib not in preload_items:
            preload_items.insert(0, lib)
        env["LD_PRELOAD"] = " ".join(preload_items)
    return env


def run_case(args: argparse.Namespace, case_name: str, env: dict[str, str]) -> dict[str, str]:
    case_dir = args.dump_root / case_name
    matrix_path = case_dir / f"J{args.iteration}.txt"
    rhs_path = case_dir / f"F{args.iteration}.txt"
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    if not rhs_path.exists():
        raise FileNotFoundError(rhs_path)

    cmd = [
        str(args.bench),
        "--case",
        case_name,
        "--matrix",
        str(matrix_path),
        "--rhs",
        str(rhs_path),
        "--precision",
        args.precision,
        "--repeats",
        str(args.repeats),
        "--csv",
    ]
    if args.enable_mt:
        cmd.append("--enable-mt")
        if args.cudss_threading_lib.exists():
            cmd.extend(["--threading-lib", str(args.cudss_threading_lib)])

    completed = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case_name} failed with returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"unexpected CSV output for {case_name}: {completed.stdout}")
    row = rows[0]
    row["iteration"] = str(args.iteration)
    return row


def write_markdown(csv_path: Path, rows: list[dict[str, str]]) -> None:
    md_path = csv_path.with_suffix(".md")
    fields = [
        "case_name",
        "n",
        "nnz",
        "analyze_sec",
        "factorize_sec",
        "solve_sec",
        "total_sec",
        "relative_residual",
    ]
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(fields) + " |\n")
        fh.write("|" + "|".join("---" for _ in fields) + "|\n")
        for row in rows:
            values = []
            for field in fields:
                value = row[field]
                if field.endswith("_sec") or field == "relative_residual":
                    value = f"{float(value):.6e}"
                values.append(value)
            fh.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    if not args.bench.exists():
        raise FileNotFoundError(args.bench)

    env = build_env(args)
    rows: list[dict[str, str]] = []
    for case_name in args.cases:
        row = run_case(args, case_name, env)
        rows.append(row)
        print(
            f"[OK] {case_name} n={row['n']} analyze={float(row['analyze_sec']):.6e} "
            f"factorize={float(row['factorize_sec']):.6e} solve={float(row['solve_sec']):.6e}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_name",
        "precision",
        "iteration",
        "n",
        "nnz",
        "repeats",
        "analyze_sec",
        "factorize_sec",
        "solve_sec",
        "total_sec",
        "residual_norm",
        "relative_residual",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output, rows)
    print(f"[DONE] csv={args.output}")
    print(f"[DONE] md={args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
