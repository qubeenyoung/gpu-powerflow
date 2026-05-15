#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import subprocess


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_BENCH = EXP_ROOT / "build" / "gmres_block_jacobi_bench"
DEFAULT_DUMP_ROOT = EXP_ROOT / "raw" / "cupf_jf_dumps"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "gmres_representative_j1_sweep.csv"
DEFAULT_CASES = [
    "case1197",
    "case2736sp",
    "case3375wp",
    "case6468rte",
    "case_ACTIVSg10k",
]


def parse_int_list(values: list[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                result.append(int(part))
    return result


def parse_str_list(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GMRES + METIS block-Jacobi timings for representative J1/F1 dumps."
    )
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--block-sizes", nargs="*", default=["32", "64"])
    parser.add_argument("--restarts", nargs="*", default=["8", "16"])
    parser.add_argument("--max-iters", nargs="*", default=["16", "32"])
    parser.add_argument("--apply-modes", nargs="*", default=["inverse_gemv"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp64"])
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument("--preconditioner", default="metis_block_jacobi")
    return parser.parse_args()


def run_one(
    args: argparse.Namespace,
    case_name: str,
    block_size: int,
    restart: int,
    max_iters: int,
    apply_mode: str,
) -> dict[str, str]:
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
        "--solver",
        "gmres",
        "--gmres-restart",
        str(restart),
        "--gmres-max-iters",
        str(max_iters),
        "--gmres-rtol",
        str(args.rtol),
        "--preconditioner",
        args.preconditioner,
        "--block-size",
        str(block_size),
        "--block-jacobi-precision",
        args.precision,
        "--block-jacobi-apply",
        apply_mode,
        "--csv",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case_name} b{block_size} r{restart} m{max_iters} failed "
            f"with returncode={completed.returncode}\nstdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"unexpected CSV output for {case_name}: {completed.stdout}")
    row = rows[0]
    row["iteration"] = str(args.iteration)
    return row


def choose_best(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_name"], []).append(row)

    best_rows: list[dict[str, str]] = []
    for case_name in sorted(by_case):
        case_rows = by_case[case_name]
        converged = [row for row in case_rows if row["converged"] == "1"]
        if converged:
            best = min(converged, key=lambda row: float(row["solve_total_sec"]))
        else:
            best = min(case_rows, key=lambda row: float(row["relative_residual"]))
        best_rows.append(best)
    return best_rows


def write_markdown(path: Path, rows: list[dict[str, str]], title: str) -> None:
    fields = [
        "case_name",
        "n",
        "block_size",
        "restart",
        "max_iters",
        "converged",
        "iterations",
        "relative_residual",
        "setup_including_analyze_sec",
        "solve_total_sec",
        "gmres_loop_sec",
        "preconditioner_apply_sec",
        "spmv_sec",
        "dot_reduction_sec",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
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

    block_sizes = parse_int_list(args.block_sizes)
    restarts = parse_int_list(args.restarts)
    max_iters_values = parse_int_list(args.max_iters)
    apply_modes = parse_str_list(args.apply_modes)

    rows: list[dict[str, str]] = []
    total = len(args.cases) * len(block_sizes) * len(restarts) * len(max_iters_values) * len(apply_modes)
    count = 0
    for case_name in args.cases:
        for block_size in block_sizes:
            for restart in restarts:
                for max_iters in max_iters_values:
                    for apply_mode in apply_modes:
                        count += 1
                        row = run_one(args, case_name, block_size, restart, max_iters, apply_mode)
                        rows.append(row)
                        print(
                            f"[OK] {count}/{total} {case_name} b={block_size} "
                            f"r={restart} max={max_iters} conv={row['converged']} "
                            f"iters={row['iterations']} rel={float(row['relative_residual']):.3e} "
                            f"solve={float(row['solve_total_sec']):.3e}"
                        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(
        args.output.with_suffix(".md"),
        rows,
        "GMRES Representative J1 Sweep",
    )
    write_markdown(
        args.output.with_name(args.output.stem + "_best.md"),
        choose_best(rows),
        "GMRES Representative J1 Best Per Case",
    )
    print(f"[DONE] csv={args.output}")
    print(f"[DONE] md={args.output.with_suffix('.md')}")
    print(f"[DONE] best={args.output.with_name(args.output.stem + '_best.md')}")


if __name__ == "__main__":
    main()
