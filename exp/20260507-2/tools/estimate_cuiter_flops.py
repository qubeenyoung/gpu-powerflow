#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_SUMMARY = EXP_ROOT / "results" / "bicgstab_iter2_bs8_all78_summary.csv"
DEFAULT_ITERS = EXP_ROOT / "results" / "bicgstab_iter2_bs8_all78_iters.csv"
DEFAULT_OUT_PREFIX = EXP_ROOT / "results" / "bicgstab_iter2_bs8_all78_flops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate cuITER BiCGSTAB + METIS block-Jacobi FLOPs from "
            "hybrid_nr_bench summary/iteration CSV files."
        )
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--iters", type=Path, default=DEFAULT_ITERS)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--block-apply",
        choices=["inverse_gemv", "lu_solve"],
        default="inverse_gemv",
        help="Block-Jacobi apply kernel used by the run.",
    )
    parser.add_argument(
        "--count-bj-setup",
        action="store_true",
        help=(
            "Also estimate dense block LU/inverse setup FLOPs when setup timing "
            "columns indicate that block-Jacobi setup ran."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except ValueError:
        return default
    if math.isnan(result):
        return default
    return result


def safe_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def boolish(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def estimate_sum_block_size_sq(row: dict[str, str], n: int) -> float:
    num_blocks = safe_int(row, "num_bus_partitions")
    avg = safe_float(row, "avg_block_unknowns")
    std = safe_float(row, "std_block_unknowns")
    if num_blocks > 0 and avg > 0.0:
        return num_blocks * (avg * avg + std * std)

    max_block = safe_int(row, "max_block_unknowns")
    if max_block <= 0:
        return float(n)
    full_blocks, tail = divmod(n, max_block)
    return full_blocks * float(max_block * max_block) + float(tail * tail)


def estimate_bj_apply_flops_per_apply(row: dict[str, str], n: int, block_apply: str) -> float:
    sum_sq = estimate_sum_block_size_sq(row, n)
    if block_apply == "inverse_gemv":
        # Dense y = B^{-1} x per block: m rows with m multiplies and m - 1 adds.
        return max(0.0, 2.0 * sum_sq - float(n))
    # Dense forward + backward triangular solves. Division is counted as one flop.
    return max(0.0, 2.0 * sum_sq)


def setup_ran(row: dict[str, str]) -> bool:
    if boolish(row.get("bj_cache_reused")):
        return False
    return (
        safe_float(row, "bj_setup_total_seconds") > 0.0
        or safe_float(row, "bj_inverse_build_seconds") > 0.0
        or safe_float(row, "gmres_trial_setup_seconds") > 0.0
        or safe_float(row, "linear_setup_seconds") > 0.0
    )


def estimate_bj_setup_flops(row: dict[str, str], block_apply: str) -> float:
    num_blocks = safe_int(row, "num_bus_partitions")
    leading_dim = safe_int(row, "max_block_unknowns")
    if num_blocks <= 0 or leading_dim <= 0:
        return 0.0
    m3 = float(num_blocks) * float(leading_dim) ** 3
    getrf = (2.0 / 3.0) * m3
    if block_apply == "inverse_gemv":
        # cublas getriBatched is approximately another 4/3 m^3 per dense block.
        return getrf + (4.0 / 3.0) * m3
    return getrf


def cuiter_row(row: dict[str, str]) -> bool:
    text = " ".join(
        [
            row.get("solver_used", ""),
            row.get("stop_reason", ""),
            row.get("setting", ""),
        ]
    ).lower()
    return "bicgstab" in text or safe_float(row, "bicgstab_total_seconds") > 0.0


def linear_iters(row: dict[str, str]) -> int:
    for key in ("bicgstab_refinement_iters", "linear_iters"):
        value = safe_int(row, key)
        if value > 0:
            return value
    return 0


def estimate_row_flops(
    row: dict[str, str],
    dims: dict[str, int],
    block_apply: str,
    count_bj_setup: bool,
) -> dict[str, float | int | str]:
    n = dims["n"]
    nnz = dims["nnz"]
    iters = linear_iters(row) if cuiter_row(row) else 0

    spmv_flops_per_iter = 4.0 * float(nnz)
    bj_apply_per_apply = estimate_bj_apply_flops_per_apply(row, n, block_apply)
    bj_apply_flops_per_iter = 2.0 * bj_apply_per_apply
    dot_flops_per_iter = max(0.0, 8.0 * float(n) - 4.0)

    update_total = 0.0
    for iter_idx in range(iters):
        update_total += (8.0 if iter_idx == 0 else 12.0) * float(n)

    spmv_total = spmv_flops_per_iter * float(iters)
    bj_apply_total = bj_apply_flops_per_iter * float(iters)
    dot_total = dot_flops_per_iter * float(iters)
    norm_total = (4.0 * float(n)) if iters > 0 else 0.0
    solve_total = spmv_total + bj_apply_total + dot_total + update_total + norm_total

    bj_setup_total = 0.0
    if count_bj_setup and iters > 0 and setup_ran(row):
        bj_setup_total = estimate_bj_setup_flops(row, block_apply)

    total = solve_total + bj_setup_total
    per_linear_iter = solve_total / float(iters) if iters > 0 else 0.0
    return {
        "case_name": row.get("case_name", ""),
        "nr_iter": row.get("nr_iter", ""),
        "solver_used": row.get("solver_used", ""),
        "linear_iters": iters,
        "n": n,
        "nnz": nnz,
        "cuiter_spmv_flops_est": spmv_total,
        "cuiter_block_jacobi_apply_flops_est": bj_apply_total,
        "cuiter_dot_reduction_flops_est": dot_total,
        "cuiter_vector_update_flops_est": update_total,
        "cuiter_norm_flops_est": norm_total,
        "cuiter_solve_flops_est": solve_total,
        "cuiter_bj_setup_flops_est": bj_setup_total,
        "cuiter_total_flops_est": total,
        "cuiter_flops_per_bicgstab_iter_est": per_linear_iter,
        "step_accepted": row.get("step_accepted", row.get("accepted", "")),
        "fallback_used": row.get("fallback_used", row.get("fallback", "")),
        "stop_reason": row.get("stop_reason", ""),
    }


def build_dims(summary_rows: Iterable[dict[str, str]]) -> dict[str, dict[str, int]]:
    dims: dict[str, dict[str, int]] = {}
    for row in summary_rows:
        case_name = row.get("case_name", "")
        n = safe_int(row, "n")
        nnz = safe_int(row, "nnz")
        if case_name and n > 0 and nnz > 0:
            dims[case_name] = {
                "n": n,
                "nnz": nnz,
                "nr_iters": safe_int(row, "nr_iters"),
            }
    return dims


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    iter_rows: list[dict[str, object]],
    dims_by_case: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    by_case: dict[str, dict[str, object]] = {}
    for row in iter_rows:
        case_name = str(row["case_name"])
        entry = by_case.setdefault(
            case_name,
            {
                "case_name": case_name,
                "n": row["n"],
                "nnz": row["nnz"],
                "nr_iters": dims_by_case.get(case_name, {}).get("nr_iters", 0),
                "cuiter_nr_steps": 0,
                "bicgstab_linear_iters": 0,
                "cuiter_spmv_flops_est": 0.0,
                "cuiter_block_jacobi_apply_flops_est": 0.0,
                "cuiter_dot_reduction_flops_est": 0.0,
                "cuiter_vector_update_flops_est": 0.0,
                "cuiter_norm_flops_est": 0.0,
                "cuiter_solve_flops_est": 0.0,
                "cuiter_bj_setup_flops_est": 0.0,
                "cuiter_total_flops_est": 0.0,
            },
        )
        if int(row["linear_iters"]) > 0:
            entry["cuiter_nr_steps"] = int(entry["cuiter_nr_steps"]) + 1
            entry["bicgstab_linear_iters"] = int(entry["bicgstab_linear_iters"]) + int(
                row["linear_iters"]
            )
        for key in (
            "cuiter_spmv_flops_est",
            "cuiter_block_jacobi_apply_flops_est",
            "cuiter_dot_reduction_flops_est",
            "cuiter_vector_update_flops_est",
            "cuiter_norm_flops_est",
            "cuiter_solve_flops_est",
            "cuiter_bj_setup_flops_est",
            "cuiter_total_flops_est",
        ):
            entry[key] = float(entry[key]) + float(row[key])

    summary_rows: list[dict[str, object]] = []
    for entry in by_case.values():
        cuiter_steps = int(entry["cuiter_nr_steps"])
        nr_iters = int(entry["nr_iters"])
        linear_iter_count = int(entry["bicgstab_linear_iters"])
        total = float(entry["cuiter_total_flops_est"])
        solve = float(entry["cuiter_solve_flops_est"])
        entry["cuiter_total_gflops_est"] = total / 1.0e9
        entry["cuiter_solve_gflops_est"] = solve / 1.0e9
        entry["cuiter_flops_per_cuiter_nr_step_est"] = (
            total / float(cuiter_steps) if cuiter_steps else 0.0
        )
        entry["cuiter_flops_per_nr_iter_est"] = total / float(nr_iters) if nr_iters else 0.0
        entry["cuiter_flops_per_bicgstab_iter_est"] = (
            solve / float(linear_iter_count) if linear_iter_count else 0.0
        )
        summary_rows.append(entry)
    return sorted(summary_rows, key=lambda item: str(item["case_name"]))


def format_sci(value: object) -> str:
    try:
        return f"{float(value):.6e}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, rows: list[dict[str, object]], count_bj_setup: bool) -> None:
    fields = [
        "case_name",
        "n",
        "nnz",
        "nr_iters",
        "cuiter_nr_steps",
        "bicgstab_linear_iters",
        "cuiter_total_gflops_est",
        "cuiter_flops_per_cuiter_nr_step_est",
        "cuiter_flops_per_bicgstab_iter_est",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# cuITER FLOP Estimate\n\n")
        fh.write(
            "Scope: BiCGSTAB solve work plus block-Jacobi apply FLOPs. "
            "Block-Jacobi setup FLOPs are "
            + ("included" if count_bj_setup else "reported as 0")
            + ".\n\n"
        )
        fh.write("| " + " | ".join(fields) + " |\n")
        fh.write("|" + "|".join("---" for _ in fields) + "|\n")
        for row in rows:
            values = []
            for field in fields:
                value = row[field]
                if "flops" in field or "gflops" in field:
                    value = format_sci(value)
                values.append(str(value))
            fh.write("| " + " | ".join(values) + " |\n")
        fh.write("\n## Formula\n\n")
        fh.write("- CSR SpMV: `2 * nnz` FLOPs; BiCGSTAB uses two SpMVs per linear iteration.\n")
        fh.write("- Block-Jacobi inverse GEMV apply: `sum_blocks(2*m^2 - m)` FLOPs per apply; BiCGSTAB uses two applies per linear iteration.\n")
        fh.write("- Dot reductions: `8*n - 4` FLOPs per BiCGSTAB linear iteration.\n")
        fh.write("- Vector updates: `8*n` FLOPs for the first BiCGSTAB iteration and `12*n` afterwards.\n")
        fh.write("- Norm checks: `4*n` FLOPs per cuITER solve.\n")
        if count_bj_setup:
            fh.write(
                "- Block-Jacobi setup: dense batched LU plus optional inverse, estimated from "
                "`num_blocks * max_block_unknowns^3`.\n"
            )


def main() -> None:
    args = parse_args()
    summary_rows = read_csv(args.summary)
    raw_iter_rows = read_csv(args.iters)
    dims_by_case = build_dims(summary_rows)
    if not dims_by_case:
        raise RuntimeError(f"no n/nnz case dimensions found in {args.summary}")

    iter_rows: list[dict[str, object]] = []
    cumulative_by_case: dict[str, float] = {}
    for row in raw_iter_rows:
        case_name = row.get("case_name", "")
        if case_name not in dims_by_case:
            continue
        estimate = estimate_row_flops(
            row,
            dims_by_case[case_name],
            args.block_apply,
            args.count_bj_setup,
        )
        cumulative = cumulative_by_case.get(case_name, 0.0) + float(
            estimate["cuiter_total_flops_est"]
        )
        cumulative_by_case[case_name] = cumulative
        estimate["cuiter_cumulative_flops_est"] = cumulative
        estimate["cuiter_total_gflops_est"] = float(estimate["cuiter_total_flops_est"]) / 1.0e9
        iter_rows.append(estimate)

    summary_estimates = summarize(iter_rows, dims_by_case)

    iter_fields = [
        "case_name",
        "nr_iter",
        "solver_used",
        "linear_iters",
        "n",
        "nnz",
        "cuiter_spmv_flops_est",
        "cuiter_block_jacobi_apply_flops_est",
        "cuiter_dot_reduction_flops_est",
        "cuiter_vector_update_flops_est",
        "cuiter_norm_flops_est",
        "cuiter_solve_flops_est",
        "cuiter_bj_setup_flops_est",
        "cuiter_total_flops_est",
        "cuiter_total_gflops_est",
        "cuiter_flops_per_bicgstab_iter_est",
        "cuiter_cumulative_flops_est",
        "step_accepted",
        "fallback_used",
        "stop_reason",
    ]
    summary_fields = [
        "case_name",
        "n",
        "nnz",
        "nr_iters",
        "cuiter_nr_steps",
        "bicgstab_linear_iters",
        "cuiter_spmv_flops_est",
        "cuiter_block_jacobi_apply_flops_est",
        "cuiter_dot_reduction_flops_est",
        "cuiter_vector_update_flops_est",
        "cuiter_norm_flops_est",
        "cuiter_solve_flops_est",
        "cuiter_bj_setup_flops_est",
        "cuiter_total_flops_est",
        "cuiter_solve_gflops_est",
        "cuiter_total_gflops_est",
        "cuiter_flops_per_cuiter_nr_step_est",
        "cuiter_flops_per_nr_iter_est",
        "cuiter_flops_per_bicgstab_iter_est",
    ]

    iter_path = args.out_prefix.with_name(args.out_prefix.name + "_iters.csv")
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    md_path = args.out_prefix.with_suffix(".md")
    write_csv(iter_path, iter_rows, iter_fields)
    write_csv(summary_path, summary_estimates, summary_fields)
    write_markdown(md_path, summary_estimates, args.count_bj_setup)
    print(f"[DONE] iter_flops={iter_path}")
    print(f"[DONE] summary_flops={summary_path}")
    print(f"[DONE] report={md_path}")


if __name__ == "__main__":
    main()
