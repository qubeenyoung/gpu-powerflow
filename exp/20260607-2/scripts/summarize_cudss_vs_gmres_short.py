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
DEFAULT_SWEEP = EXP_ROOT / "results" / "gmres_representative_j1_sweep.csv"
DEFAULT_CUDSS = EXP_ROOT / "results" / "cudss_representative_j1.csv"
DEFAULT_OUT = EXP_ROOT / "results" / "cudss_vs_gmres_short_summary.csv"
DEFAULT_HISTORY_DIR = EXP_ROOT / "results" / "gmres_short_residual_history"
SAMPLE_ITERS = [1, 2, 4, 8, 16, 24, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cuDSS factorize+solve and short GMRES residual trends."
    )
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--cudss-csv", type=Path, default=DEFAULT_CUDSS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--iteration", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


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


def run_history(args: argparse.Namespace, row: dict[str, str]) -> tuple[dict[str, str], Path]:
    case_name = row["case_name"]
    case_dir = args.dump_root / case_name
    history_path = args.history_dir / f"{case_name}.csv"
    args.history_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.bench),
        "--case",
        case_name,
        "--matrix",
        str(case_dir / f"J{args.iteration}.txt"),
        "--rhs",
        str(case_dir / f"F{args.iteration}.txt"),
        "--solver",
        "gmres",
        "--gmres-restart",
        row["restart"],
        "--gmres-max-iters",
        row["max_iters"],
        "--gmres-rtol",
        row["rtol"],
        "--preconditioner",
        row["preconditioner"],
        "--block-size",
        row["block_size"],
        "--block-jacobi-precision",
        row["block_jacobi_precision"],
        "--block-jacobi-apply",
        row["block_jacobi_apply"],
        "--residual-history",
        str(history_path),
        "--csv",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case_name} history run failed with returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    output_rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(output_rows) != 1:
        raise RuntimeError(f"unexpected GMRES CSV output for {case_name}: {completed.stdout}")
    return output_rows[0], history_path


def sampled_history(history_path: Path) -> tuple[dict[int, float], dict[int, float]]:
    rows = read_csv(history_path)
    by_iter_rel = {
        int(row["iteration"]): float(row["estimated_relative_residual"])
        for row in rows
    }
    by_iter_abs: dict[int, float] = {}
    for row in rows:
        iteration = int(row["iteration"])
        if "estimated_residual_norm" in row and row["estimated_residual_norm"]:
            by_iter_abs[iteration] = float(row["estimated_residual_norm"])
        elif "rhs_norm" in row and row["rhs_norm"]:
            by_iter_abs[iteration] = (
                float(row["estimated_relative_residual"]) * float(row["rhs_norm"])
            )
        else:
            by_iter_abs[iteration] = float("nan")
    return (
        {iteration: by_iter_rel.get(iteration, float("nan")) for iteration in SAMPLE_ITERS},
        {iteration: by_iter_abs.get(iteration, float("nan")) for iteration in SAMPLE_ITERS},
    )


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "case_name",
        "n",
        "cudss_factorize_solve_sec",
        "gmres_short_solve_sec",
        "gmres_final_true_absres",
        "gmres_final_true_relres",
        "est_absres_iter1",
        "est_absres_iter8",
        "est_absres_iter16",
        "est_absres_iter32",
        "est_relres_iter1",
        "est_relres_iter8",
        "est_relres_iter16",
        "est_relres_iter32",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# cuDSS vs Short GMRES\n\n")
        fh.write("| " + " | ".join(fields) + " |\n")
        fh.write("|" + "|".join("---" for _ in fields) + "|\n")
        for row in rows:
            values = []
            for field in fields:
                value = row[field]
                if field != "case_name" and field != "n":
                    value = f"{float(value):.6e}"
                values.append(value)
            fh.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    best_rows = choose_best(read_csv(args.sweep_csv))
    cudss_by_case = {row["case_name"]: row for row in read_csv(args.cudss_csv)}

    summary_rows: list[dict[str, str]] = []
    for best in best_rows:
        gmres_row, history_path = run_history(args, best)
        history_rel, history_abs = sampled_history(history_path)
        cudss = cudss_by_case[best["case_name"]]
        cudss_factorize_solve = float(cudss["factorize_sec"]) + float(cudss["solve_sec"])
        summary = {
            "case_name": best["case_name"],
            "n": best["n"],
            "nnz": best["nnz"],
            "cudss_analyze_sec": cudss["analyze_sec"],
            "cudss_factorize_sec": cudss["factorize_sec"],
            "cudss_solve_sec": cudss["solve_sec"],
            "cudss_factorize_solve_sec": f"{cudss_factorize_solve:.12g}",
            "gmres_block_size": gmres_row["block_size"],
            "gmres_restart": gmres_row["restart"],
            "gmres_max_iters": gmres_row["max_iters"],
            "gmres_short_solve_sec": gmres_row["solve_total_sec"],
            "gmres_setup_including_analyze_sec": gmres_row["setup_including_analyze_sec"],
            "gmres_final_true_absres": gmres_row["residual_norm"],
            "gmres_final_true_relres": gmres_row["relative_residual"],
        }
        for iteration in SAMPLE_ITERS:
            summary[f"est_absres_iter{iteration}"] = f"{history_abs[iteration]:.12g}"
            summary[f"est_relres_iter{iteration}"] = f"{history_rel[iteration]:.12g}"
        summary_rows.append(summary)
        print(
            f"[OK] {best['case_name']} cudss_f+s={cudss_factorize_solve:.6e} "
            f"gmres={float(gmres_row['solve_total_sec']):.6e} "
            f"abs32={float(gmres_row['residual_norm']):.3e} "
            f"rel32={float(gmres_row['relative_residual']):.3e}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_markdown(args.output.with_suffix(".md"), summary_rows)
    print(f"[DONE] csv={args.output}")
    print(f"[DONE] md={args.output.with_suffix('.md')}")
    print(f"[DONE] histories={args.history_dir}")


if __name__ == "__main__":
    main()
