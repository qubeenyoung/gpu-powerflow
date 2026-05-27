#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_BINARY = EXP_ROOT / "build" / "hybrid_nr_bench"
DEFAULT_RESULTS = EXP_ROOT / "results"
DEFAULT_CASES = [
    "case2383wp",
    "case3120sp",
    "case9241pegase",
    "case13659pegase",
    "case6468rte",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NR iteration reduction hybrid sweep.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-nr-iters", type=int, default=50)
    parser.add_argument("--full-restart-cross", action="store_true")
    return parser.parse_args()


def bool_value(value: str) -> bool:
    return value.lower() == "true"


def safe_float(value: str) -> float:
    return float(value) if value else 0.0


def run_config(args: argparse.Namespace,
               block_size: int,
               gmres_iters: int,
               gmres_restart: int,
               accept_ratio: float,
               config_id: int) -> tuple[Path, Path]:
    tag = (
        f"nr_iter_reduce_bs{block_size}_r{gmres_restart}_i{gmres_iters}"
        f"_a{str(accept_ratio).replace('.', 'p')}"
    )
    summary_path = args.results_dir / f"{tag}.csv"
    iter_path = args.results_dir / f"{tag}_iters.csv"
    cmd = [
        str(args.binary),
        "--case", args.cases,
        "--solver", "hybrid",
        "--warmup", str(args.warmup),
        "--max-nr-iters", str(args.max_nr_iters),
        "--cudss-bootstrap-iters", "1",
        "--cudss-polish-threshold", "1e-4",
        "--force-gmres-min-steps", "0",
        "--block-size", str(block_size),
        "--gmres-restart", str(gmres_restart),
        "--gmres-max-iters", str(gmres_iters),
        "--gmres-fixed-iter-mode", "true",
        "--fallback-policy", "immediate",
        "--accept-iterative-by-mismatch", "true",
        "--accept-mismatch-ratio", str(accept_ratio),
        "--reject-mismatch-ratio", "1.05",
        "--block-jacobi-precision", "fp32",
        "--block-jacobi-apply", "inverse_gemv",
        "--output", str(summary_path),
        "--iter-output", str(iter_path),
    ]
    print(
        f"[{config_id:02d}] block={block_size} restart={gmres_restart} "
        f"iters={gmres_iters} accept={accept_ratio}"
    )
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise RuntimeError(f"hybrid_nr_bench failed for {tag}")
    return summary_path, iter_path


def load_iteration_metrics(iter_path: Path) -> dict[tuple[str, int], dict[str, float]]:
    metrics: dict[tuple[str, int], dict[str, float]] = {}
    with iter_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            case = row["case_name"]
            gmres_trial = (
                safe_float(row.get("gmres_trial_setup_seconds", "0"))
                + safe_float(row.get("gmres_trial_solve_seconds", "0"))
            )
            fallback_wasted = gmres_trial if bool_value(row.get("fallback_used", "false")) else 0.0
            key = (case, 0)
            item = metrics.setdefault(
                key,
                {
                    "accepted_ratio_sum": 0.0,
                    "accepted_ratio_max": 0.0,
                    "accepted_ratio_count": 0.0,
                    "gmres_trial_sum": 0.0,
                    "gmres_trial_count": 0.0,
                    "fallback_wasted_sum": 0.0,
                    "fallback_wasted_count": 0.0,
                },
            )
            if gmres_trial > 0.0:
                item["gmres_trial_sum"] += gmres_trial
                item["gmres_trial_count"] += 1.0
            if fallback_wasted > 0.0:
                item["fallback_wasted_sum"] += fallback_wasted
                item["fallback_wasted_count"] += 1.0
            if row["solver_used"] == "gmres_middle" and bool_value(row["step_accepted"]):
                before = safe_float(row["mismatch_inf_before"])
                after = safe_float(row["mismatch_inf_after"])
                if before > 0.0:
                    ratio = after / before
                    item["accepted_ratio_sum"] += ratio
                    item["accepted_ratio_max"] = max(item["accepted_ratio_max"], ratio)
                    item["accepted_ratio_count"] += 1.0
    return metrics


def append_rows(report_rows: list[dict[str, object]],
                summary_path: Path,
                iter_path: Path,
                block_size: int,
                gmres_iters: int,
                gmres_restart: int,
                accept_ratio: float) -> None:
    iter_metrics = load_iteration_metrics(iter_path)
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            metrics = iter_metrics.get((row["case_name"], 0), {})
            accepted_count = metrics.get("accepted_ratio_count", 0.0)
            trial_count = metrics.get("gmres_trial_count", 0.0)
            fallback_count = metrics.get("fallback_wasted_count", 0.0)
            report_rows.append(
                {
                    "case": row["case_name"],
                    "block_size": block_size,
                    "gmres_restart": gmres_restart,
                    "gmres_iters": gmres_iters,
                    "accept_mismatch_ratio": accept_ratio,
                    "converged": row["converged"],
                    "nr_iters": row["nr_iters"],
                    "cudss_calls": row["cudss_calls"],
                    "gmres_calls": row["gmres_calls"],
                    "accepted": row["accepted_gmres_steps"],
                    "rejected": row["rejected_gmres_steps"],
                    "fallback": row["fallback_calls"],
                    "polish": row["polish_calls"],
                    "hybrid_time": row["total_seconds"],
                    "pure_cudss_time": row["pure_cudss_total_seconds"],
                    "speedup": row["speedup_vs_pure_cudss"],
                    "avg_accepted_mismatch_ratio": (
                        metrics.get("accepted_ratio_sum", 0.0) / accepted_count
                        if accepted_count > 0.0
                        else 0.0
                    ),
                    "max_accepted_mismatch_ratio": metrics.get("accepted_ratio_max", 0.0),
                    "avg_gmres_trial_time": (
                        metrics.get("gmres_trial_sum", 0.0) / trial_count
                        if trial_count > 0.0
                        else 0.0
                    ),
                    "avg_fallback_wasted_time": (
                        metrics.get("fallback_wasted_sum", 0.0) / fallback_count
                        if fallback_count > 0.0
                        else 0.0
                    ),
                }
            )


def write_report(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "case",
        "block_size",
        "gmres_restart",
        "gmres_iters",
        "accept_mismatch_ratio",
        "converged",
        "nr_iters",
        "cudss_calls",
        "gmres_calls",
        "accepted",
        "rejected",
        "fallback",
        "polish",
        "hybrid_time",
        "pure_cudss_time",
        "speedup",
        "avg_accepted_mismatch_ratio",
        "max_accepted_mismatch_ratio",
        "avg_gmres_trial_time",
        "avg_fallback_wasted_time",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, object]] = []
    block_sizes = [32, 64]
    gmres_iters_values = [1, 2]
    accept_ratios = [0.7, 0.85, 0.9, 0.95]
    if args.full_restart_cross:
        restart_pairs = [(restart, iters) for iters in gmres_iters_values for restart in [1, 2]]
    else:
        restart_pairs = [(iters, iters) for iters in gmres_iters_values]

    config_id = 0
    for block_size in block_sizes:
        for gmres_restart, gmres_iters in restart_pairs:
            for accept_ratio in accept_ratios:
                config_id += 1
                summary_path, iter_path = run_config(
                    args, block_size, gmres_iters, gmres_restart, accept_ratio, config_id
                )
                append_rows(
                    report_rows,
                    summary_path,
                    iter_path,
                    block_size,
                    gmres_iters,
                    gmres_restart,
                    accept_ratio,
                )

    report_path = args.results_dir / "nr_iteration_reduction_sweep.csv"
    write_report(report_rows, report_path)
    print(f"[DONE] report={report_path}")


if __name__ == "__main__":
    main()
