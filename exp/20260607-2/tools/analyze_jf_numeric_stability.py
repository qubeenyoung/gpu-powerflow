#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, onenormest, splu


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_JF_ROOT = EXP_ROOT / "raw" / "cupf_jf_dumps"
DEFAULT_SUMMARY = DEFAULT_JF_ROOT / "linear_system_dump_summary.csv"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "jf_numeric_stability_all_iterations.csv"
DEFAULT_REPORT = EXP_ROOT / "results" / "jf_numeric_stability_all_iterations.md"


@dataclass(frozen=True)
class DumpCase:
    name: str
    n_bus: int
    linear_dim: int
    linear_nnz: int
    iterations: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate numerical-stability metrics for dumped cuPF J/F systems."
    )
    parser.add_argument("--jf-root", type=Path, default=DEFAULT_JF_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case filter. Empty means all cases in the dump summary.",
    )
    parser.add_argument(
        "--iterations",
        default="",
        help="Comma-separated iteration filter. Empty means all iterations available per case.",
    )
    parser.add_argument(
        "--condest-t",
        type=int,
        default=2,
        help="Block count used by scipy.sparse.linalg.onenormest.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_cases(path: Path) -> list[DumpCase]:
    rows: list[DumpCase] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                DumpCase(
                    name=row["case_name"],
                    n_bus=int(row["n_bus"]),
                    linear_dim=int(row["linear_dim"]),
                    linear_nnz=int(row["linear_nnz"]),
                    iterations=tuple(int(item) for item in row["available_iterations"].split()),
                )
            )
    return rows


def load_csr_dump(path: Path) -> csr_matrix:
    tokens = path.read_text(encoding="utf-8").split()
    pos = 0

    def expect(token: str) -> None:
        nonlocal pos
        got = tokens[pos] if pos < len(tokens) else "<eof>"
        if got != token:
            raise ValueError(f"{path}: expected {token}, got {got}")
        pos += 1

    expect("type")
    matrix_type = tokens[pos]
    pos += 1
    if matrix_type != "csr_matrix":
        raise ValueError(f"{path}: expected csr_matrix, got {matrix_type}")
    expect("rows")
    rows = int(tokens[pos])
    pos += 1
    expect("cols")
    cols = int(tokens[pos])
    pos += 1
    expect("nnz")
    nnz = int(tokens[pos])
    pos += 1
    expect("row_ptr")
    row_ptr = np.asarray(tokens[pos : pos + rows + 1], dtype=np.int64)
    pos += rows + 1
    expect("col_idx")
    col_idx = np.asarray(tokens[pos : pos + nnz], dtype=np.int64)
    pos += nnz
    expect("values")
    values = np.asarray(tokens[pos : pos + nnz], dtype=np.float64)
    if row_ptr[0] != 0 or row_ptr[-1] != nnz:
        raise ValueError(f"{path}: malformed row_ptr")
    return csr_matrix((values, col_idx, row_ptr), shape=(rows, cols))


def load_vector_dump(path: Path) -> np.ndarray:
    tokens = path.read_text(encoding="utf-8").split()
    pos = 0
    if tokens[pos] != "type":
        raise ValueError(f"{path}: missing type")
    pos += 1
    vector_type = tokens[pos]
    pos += 1
    if vector_type != "vector":
        raise ValueError(f"{path}: expected vector, got {vector_type}")
    if tokens[pos] != "size":
        raise ValueError(f"{path}: missing size")
    pos += 1
    size = int(tokens[pos])
    pos += 1
    if tokens[pos] != "values":
        raise ValueError(f"{path}: missing values")
    pos += 1
    values = np.zeros(size, dtype=np.float64)
    for _ in range(size):
        index = int(tokens[pos])
        value = float(tokens[pos + 1])
        values[index] = value
        pos += 2
    return values


def finite_min(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.min(values)) if values.size else math.nan


def finite_max(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else math.nan


def positive_min(values: np.ndarray) -> float:
    values = values[np.isfinite(values) & (values > 0.0)]
    return float(np.min(values)) if values.size else math.nan


def safe_ratio(num: float, den: float) -> float:
    if den == 0.0 or not math.isfinite(num) or not math.isfinite(den):
        return math.nan
    return num / den


def fmt(value: float, precision: int = 2) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{precision}e}"


def compute_metrics(case: DumpCase, iteration: int, repeat_dir: Path, condest_t: int) -> dict[str, object]:
    matrix_path = repeat_dir / f"jacobian_iter{iteration}.txt"
    vector_path = repeat_dir / f"residual_iter{iteration}.txt"
    if not vector_path.exists():
        vector_path = repeat_dir / f"residual_before_update_iter{iteration}.txt"

    start = time.perf_counter()
    matrix = load_csr_dump(matrix_path)
    rhs = load_vector_dump(vector_path)
    parse_seconds = time.perf_counter() - start

    if matrix.shape[0] != rhs.size:
        raise ValueError(f"{case.name} iter {iteration}: shape mismatch {matrix.shape} vs {rhs.size}")

    abs_values = np.abs(matrix.data)
    row_abs_sum = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    col_abs_sum = np.asarray(np.abs(matrix).sum(axis=0)).ravel()
    diag_abs = np.abs(matrix.diagonal())
    row_nonzero_min = positive_min(row_abs_sum)
    col_nonzero_min = positive_min(col_abs_sum)
    diag_nonzero_min = positive_min(diag_abs)
    rhs_l2 = float(np.linalg.norm(rhs))
    rhs_linf = float(np.linalg.norm(rhs, ord=np.inf))
    rhs_rms = float(rhs_l2 / math.sqrt(rhs.size)) if rhs.size else math.nan

    factor_start = time.perf_counter()
    lu = splu(matrix.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0)
    factor_seconds = time.perf_counter() - factor_start

    norm1 = finite_max(col_abs_sum)

    def inv_matvec(x: np.ndarray) -> np.ndarray:
        return lu.solve(x)

    def inv_rmatvec(x: np.ndarray) -> np.ndarray:
        return lu.solve(x, trans="T")

    inv_op = LinearOperator(matrix.shape, matvec=inv_matvec, rmatvec=inv_rmatvec, dtype=np.float64)
    cond_start = time.perf_counter()
    inv_norm1_est = float(onenormest(inv_op, t=condest_t))
    condest_seconds = time.perf_counter() - cond_start
    condest_1 = norm1 * inv_norm1_est

    solve_start = time.perf_counter()
    dx = lu.solve(rhs)
    solve_seconds = time.perf_counter() - solve_start
    residual = matrix @ dx - rhs
    solve_abs_resid_l2 = float(np.linalg.norm(residual))
    solve_rel_resid_l2 = safe_ratio(solve_abs_resid_l2, rhs_l2)
    dx_l2 = float(np.linalg.norm(dx))
    dx_linf = float(np.linalg.norm(dx, ord=np.inf))
    dx_rms = float(dx_l2 / math.sqrt(dx.size)) if dx.size else math.nan

    u_diag_abs = np.abs(lu.U.diagonal())
    u_diag_min = positive_min(u_diag_abs)
    u_diag_max = finite_max(u_diag_abs)
    u_pivot_spread = safe_ratio(u_diag_max, u_diag_min)
    fill_ratio = safe_ratio(float(lu.L.nnz + lu.U.nnz), float(matrix.nnz))

    return {
        "case": case.name,
        "n_bus": case.n_bus,
        "iteration": iteration,
        "n": matrix.shape[0],
        "nnz": matrix.nnz,
        "matrix_abs_min_nonzero": positive_min(abs_values),
        "matrix_abs_max": finite_max(abs_values),
        "matrix_value_spread": safe_ratio(finite_max(abs_values), positive_min(abs_values)),
        "row_abs_sum_min": row_nonzero_min,
        "row_abs_sum_max": finite_max(row_abs_sum),
        "row_abs_sum_spread": safe_ratio(finite_max(row_abs_sum), row_nonzero_min),
        "col_abs_sum_min": col_nonzero_min,
        "col_abs_sum_max": finite_max(col_abs_sum),
        "col_abs_sum_spread": safe_ratio(finite_max(col_abs_sum), col_nonzero_min),
        "diag_abs_min_nonzero": diag_nonzero_min,
        "diag_abs_max": finite_max(diag_abs),
        "diag_abs_spread": safe_ratio(finite_max(diag_abs), diag_nonzero_min),
        "zero_diag_count": int(np.count_nonzero(diag_abs == 0.0)),
        "norm1": norm1,
        "inv_norm1_est": inv_norm1_est,
        "condest_1": condest_1,
        "lu_fill_ratio": fill_ratio,
        "u_diag_abs_min_nonzero": u_diag_min,
        "u_diag_abs_max": u_diag_max,
        "u_pivot_spread": u_pivot_spread,
        "rhs_l2": rhs_l2,
        "rhs_linf": rhs_linf,
        "rhs_rms": rhs_rms,
        "dx_l2": dx_l2,
        "dx_linf": dx_linf,
        "dx_rms": dx_rms,
        "solve_abs_resid_l2": solve_abs_resid_l2,
        "solve_rel_resid_l2": solve_rel_resid_l2,
        "parse_seconds": parse_seconds,
        "factor_seconds": factor_seconds,
        "condest_seconds": condest_seconds,
        "solve_seconds": solve_seconds,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def write_report(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)
    for case_rows in by_case.values():
        case_rows.sort(key=lambda row: int(row["iteration"]))

    common_iterations = sorted(
        set.intersection(
            *(set(int(row["iteration"]) for row in case_rows) for case_rows in by_case.values())
        )
    )

    lines: list[str] = []
    lines.append("# J/F Numeric Stability Summary")
    lines.append("")
    lines.append("- `condest_1` is a sparse 1-norm condition estimate from SuperLU solves.")
    lines.append("- `pivot spread` is `max(abs(diag(U))) / min(nonzero abs(diag(U)))` from SuperLU.")
    lines.append("- `F_inf` and `dx_inf` show RHS/correction scale for the same linear system.")
    lines.append("")
    lines.append("## Aggregate By Iteration")
    lines.append("")
    lines.append("| iter | cases | median condest_1 | median pivot spread | median F_inf | median dx_inf | median solve relres |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    all_iterations = sorted({int(row["iteration"]) for row in rows})
    for iteration in all_iterations:
        iter_rows = [row for row in rows if int(row["iteration"]) == iteration]
        lines.append(
            f"| {iteration} | {len(iter_rows)} | "
            f"{fmt(median([float(row['condest_1']) for row in iter_rows]))} | "
            f"{fmt(median([float(row['u_pivot_spread']) for row in iter_rows]))} | "
            f"{fmt(median([float(row['rhs_linf']) for row in iter_rows]))} | "
            f"{fmt(median([float(row['dx_linf']) for row in iter_rows]))} | "
            f"{fmt(median([float(row['solve_rel_resid_l2']) for row in iter_rows]))} |"
        )

    if common_iterations:
        baseline_iter = common_iterations[0]
        baseline = {
            str(row["case"]): row
            for row in rows
            if int(row["iteration"]) == baseline_iter
        }
        lines.append("")
        lines.append("## Median Ratio To Iteration 0")
        lines.append("")
        lines.append("| iter | condest_1 | pivot spread | F_inf | dx_inf |")
        lines.append("|---:|---:|---:|---:|---:|")
        for iteration in common_iterations:
            iter_rows = [row for row in rows if int(row["iteration"]) == iteration]
            lines.append(
                f"| {iteration} | "
                f"{fmt(median([safe_ratio(float(row['condest_1']), float(baseline[str(row['case'])]['condest_1'])) for row in iter_rows]), 3)} | "
                f"{fmt(median([safe_ratio(float(row['u_pivot_spread']), float(baseline[str(row['case'])]['u_pivot_spread'])) for row in iter_rows]), 3)} | "
                f"{fmt(median([safe_ratio(float(row['rhs_linf']), float(baseline[str(row['case'])]['rhs_linf'])) for row in iter_rows]), 3)} | "
                f"{fmt(median([safe_ratio(float(row['dx_linf']), float(baseline[str(row['case'])]['dx_linf'])) for row in iter_rows]), 3)} |"
            )

    lines.append("")
    lines.append("## Case Detail")
    lines.append("")
    lines.append("| case | iter | n | condest_1 | pivot spread | row spread | col spread | F_inf | dx_inf | relres |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case in sorted(by_case):
        for row in by_case[case]:
            lines.append(
                f"| {case} | {row['iteration']} | {row['n']} | "
                f"{fmt(float(row['condest_1']))} | "
                f"{fmt(float(row['u_pivot_spread']))} | "
                f"{fmt(float(row['row_abs_sum_spread']))} | "
                f"{fmt(float(row['col_abs_sum_spread']))} | "
                f"{fmt(float(row['rhs_linf']))} | "
                f"{fmt(float(row['dx_linf']))} | "
                f"{fmt(float(row['solve_rel_resid_l2']))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    case_filter = set(split_csv(args.cases))
    iteration_filter = {int(item) for item in split_csv(args.iterations)}
    cases = read_cases(args.summary)
    if case_filter:
        cases = [case for case in cases if case.name in case_filter]

    rows: list[dict[str, object]] = []
    for case in cases:
        repeat_dir = args.jf_root / case.name / "repeat_00"
        iterations = case.iterations
        if iteration_filter:
            iterations = tuple(iteration for iteration in iterations if iteration in iteration_filter)
        for iteration in iterations:
            print(f"[RUN] case={case.name} iter={iteration}", flush=True)
            row = compute_metrics(case, iteration, repeat_dir, args.condest_t)
            rows.append(row)
            print(
                "[OK] "
                f"case={case.name} iter={iteration} "
                f"condest_1={row['condest_1']:.3e} "
                f"F_inf={row['rhs_linf']:.3e} "
                f"dx_inf={row['dx_linf']:.3e}",
                flush=True,
            )
    if not rows:
        raise SystemExit("no rows produced")
    write_csv(rows, args.output)
    write_report(rows, args.report)
    print(f"[DONE] output={args.output}")
    print(f"[DONE] report={args.report}")


if __name__ == "__main__":
    main()
