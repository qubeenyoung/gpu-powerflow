#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path


METRICS = [
    "sm__sass_thread_inst_executed_op_dadd_pred_on",
    "sm__sass_thread_inst_executed_op_dfma_pred_on",
    "sm__sass_thread_inst_executed_op_dmul_pred_on",
    "sm__sass_thread_inst_executed_op_fadd_pred_on",
    "sm__sass_thread_inst_executed_op_ffma_pred_on",
    "sm__sass_thread_inst_executed_op_fmul_pred_on",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure dynamic GPU floating-point instruction FLOPs with Nsight Compute. "
            "Pass the profiled command after --."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ncu", type=Path, default=Path("/usr/local/cuda/bin/ncu"))
    parser.add_argument("--kernel-name-include", default="", help="Optional regex filter.")
    parser.add_argument("--launch-count", type=int, default=0)
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Path for raw Nsight Compute CSV. Defaults to OUTPUT with .ncu.csv suffix.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("profiled command is required after --")
    return args


def ncu_command(args: argparse.Namespace, raw_output: Path) -> list[str]:
    cmd = [
        str(args.ncu),
        "--target-processes",
        "all",
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--metrics",
        ",".join(METRICS),
        "--log-file",
        str(raw_output),
    ]
    if args.launch_count > 0:
        cmd.extend(["--launch-count", str(args.launch_count)])
    if args.launch_skip > 0:
        cmd.extend(["--launch-skip", str(args.launch_skip)])
    cmd.extend(args.command)
    return cmd


def read_ncu_rows(path: Path) -> list[dict[str, str]]:
    csv_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("=="):
                continue
            if line.startswith('"') or line.startswith(","):
                csv_lines.append(line)
    if len(csv_lines) < 3:
        return []
    reader = csv.DictReader(csv_lines)
    rows = list(reader)
    if rows:
        rows = rows[1:]  # Nsight Compute emits a units row after the header.
    return rows


def metric(row: dict[str, str], name: str) -> float:
    value = row.get(name + ".sum", "")
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return 0.0


def classify_kernel(kernel_name: str) -> str:
    lower = kernel_name.lower()
    if "cudss" in lower:
        return "cudss"
    if "cublas" in lower:
        return "cublas"
    if any(token in lower for token in ("bicgstab", "block_inverse", "block_lu", "csr_spmv")):
        return "cuiter"
    if any(token in lower for token in ("mr1", "gmres", "residual", "dot_reduction")):
        return "cuiter"
    return "other"


def convert_rows(rows: list[dict[str, str]], include_regex: str) -> list[dict[str, object]]:
    pattern = re.compile(include_regex) if include_regex else None
    converted: list[dict[str, object]] = []
    for row in rows:
        kernel_name = row.get("Kernel Name", "")
        if pattern is not None and not pattern.search(kernel_name):
            continue
        dadd = metric(row, "sm__sass_thread_inst_executed_op_dadd_pred_on")
        dfma = metric(row, "sm__sass_thread_inst_executed_op_dfma_pred_on")
        dmul = metric(row, "sm__sass_thread_inst_executed_op_dmul_pred_on")
        fadd = metric(row, "sm__sass_thread_inst_executed_op_fadd_pred_on")
        ffma = metric(row, "sm__sass_thread_inst_executed_op_ffma_pred_on")
        fmul = metric(row, "sm__sass_thread_inst_executed_op_fmul_pred_on")
        fp64_flops = dadd + dmul + 2.0 * dfma
        fp32_flops = fadd + fmul + 2.0 * ffma
        converted.append(
            {
                "id": row.get("ID", ""),
                "kernel_name": kernel_name,
                "kernel_class": classify_kernel(kernel_name),
                "fp64_add_inst": dadd,
                "fp64_mul_inst": dmul,
                "fp64_fma_inst": dfma,
                "fp32_add_inst": fadd,
                "fp32_mul_inst": fmul,
                "fp32_fma_inst": ffma,
                "fp64_flops": fp64_flops,
                "fp32_flops": fp32_flops,
                "total_flops": fp64_flops + fp32_flops,
            }
        )
    return converted


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    totals: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row["kernel_class"])
        entry = totals.setdefault(
            key,
            {
                "kernel_class": key,
                "kernel_count": 0,
                "fp64_flops": 0.0,
                "fp32_flops": 0.0,
                "total_flops": 0.0,
            },
        )
        entry["kernel_count"] = int(entry["kernel_count"]) + 1
        for metric_name in ("fp64_flops", "fp32_flops", "total_flops"):
            entry[metric_name] = float(entry[metric_name]) + float(row[metric_name])
    total = {
        "kernel_class": "all",
        "kernel_count": sum(int(item["kernel_count"]) for item in totals.values()),
        "fp64_flops": sum(float(item["fp64_flops"]) for item in totals.values()),
        "fp32_flops": sum(float(item["fp32_flops"]) for item in totals.values()),
        "total_flops": sum(float(item["total_flops"]) for item in totals.values()),
    }
    rows_out = [total]
    rows_out.extend(sorted(totals.values(), key=lambda item: str(item["kernel_class"])))
    for row in rows_out:
        row["total_gflops"] = float(row["total_flops"]) / 1.0e9
    return rows_out


def main() -> None:
    args = parse_args()
    raw_output = args.raw_output or args.output.with_suffix(".ncu.csv")
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ncu_command(args, raw_output)
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as stdout_file:
        completed = subprocess.run(
            cmd,
            text=True,
            stdout=stdout_file,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "ncu failed with returncode="
                + str(completed.returncode)
                + "\nstderr="
                + completed.stderr
            )

    raw_rows = read_ncu_rows(raw_output)
    rows = convert_rows(raw_rows, args.kernel_name_include)
    fieldnames = [
        "id",
        "kernel_name",
        "kernel_class",
        "fp64_add_inst",
        "fp64_mul_inst",
        "fp64_fma_inst",
        "fp32_add_inst",
        "fp32_mul_inst",
        "fp32_fma_inst",
        "fp64_flops",
        "fp32_flops",
        "total_flops",
    ]
    write_csv(args.output, rows, fieldnames)

    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary_fields = [
        "kernel_class",
        "kernel_count",
        "fp64_flops",
        "fp32_flops",
        "total_flops",
        "total_gflops",
    ]
    write_csv(summary_path, summarize(rows), summary_fields)
    print(f"[DONE] kernels={args.output}")
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] raw_ncu={raw_output}")


if __name__ == "__main__":
    main()
