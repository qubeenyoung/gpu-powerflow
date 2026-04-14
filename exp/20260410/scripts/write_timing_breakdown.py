#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write TIMING_BREAKDOWN.md from summary.csv.")
    parser.add_argument("run_root", type=Path, help="Run directory under exp/20260410/results/")
    return parser.parse_args()


def load_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["case_stem"], row["implementation"]), []).append(row)
    return groups


def parse_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def avg(groups: dict[tuple[str, str], list[dict[str, str]]],
        case_stem: str,
        implementation: str,
        key: str) -> float | None:
    values = [parse_float(row, key) for row in groups.get((case_stem, implementation), [])]
    filtered = [value for value in values if value is not None]
    return mean(filtered) if filtered else None


def fmt_ms(value_sec: float | None) -> str:
    if value_sec is None:
        return "n/a"
    value_ms = value_sec * 1000.0
    if value_ms >= 100.0:
        return f"{value_ms:.3f}"
    if value_ms >= 1.0:
        return f"{value_ms:.3f}"
    return f"{value_ms:.4f}"


def pct(part: float | None, total: float | None) -> str:
    if part is None or total in (None, 0.0):
        return "n/a"
    return f"{(part / total) * 100.0:.1f}%"


def infer_gpu_device(rows: list[dict[str, str]]) -> str:
    for row in rows:
        value = row.get("gpu_device_requested", "")
        if value:
            return value
    return "n/a"


def case_labels(rows: list[dict[str, str]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        labels.setdefault(row["case_stem"], row["case_name"])
    return labels


def write_breakdown(run_root: Path) -> Path:
    summary_path = run_root / "summary.csv"
    rows = load_rows(summary_path)
    groups = group_rows(rows)
    labels = case_labels(rows)
    gpu_device = infer_gpu_device(rows)
    case_stems = sorted({row["case_stem"] for row in rows})

    lines = [
        "# Timing Breakdown",
        "",
        f"- run: `{run_root.name}`",
        "- benchmark binary: `/workspace/cuPF/build/bench-cuda-timing/cupf_case_benchmark`",
        "- units: `ms`",
    ]

    manifest_path = run_root / "manifest.json"
    if manifest_path.exists():
        lines.extend([
            "- warmup: see `manifest.json`",
            "- measured repeats: see `manifest.json`",
        ])

    lines.append(f"- gpu: `CUDA_VISIBLE_DEVICES={gpu_device}`")
    lines.extend([
        "",
        "## CUDA Analyze",
        "",
        "| case | impl | analyze mean (ms) | cudssAnalysis (ms) | cudssAnalysis share | cudssCreate (ms) | cusparseSetup (ms) | initialFactorization (ms) | uploadJacobianMaps (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for case_stem in case_stems:
        label = labels.get(case_stem, case_stem)
        for implementation, impl_label in (
            ("cpp_cuda_edge", "cuda edge"),
            ("cpp_cuda_vertex", "cuda vertex"),
        ):
            analyze = avg(groups, case_stem, implementation, "analyze_sec")
            analysis = avg(groups, case_stem, implementation, "CUDA.analyze.cudssAnalysis.avg_sec")
            if analyze is None:
                continue
            lines.append(
                f"| {label} | {impl_label} | {fmt_ms(analyze)} | "
                f"{fmt_ms(analysis)} | {pct(analysis, analyze)} | "
                f"{fmt_ms(avg(groups, case_stem, implementation, 'CUDA.analyze.cudssCreate.avg_sec'))} | "
                f"{fmt_ms(avg(groups, case_stem, implementation, 'CUDA.analyze.cusparseSetup.avg_sec'))} | "
                f"{fmt_ms(avg(groups, case_stem, implementation, 'CUDA.analyze.cudssFactorization.avg_sec'))} | "
                f"{fmt_ms(avg(groups, case_stem, implementation, 'CUDA.analyze.uploadJacobianMaps.avg_sec'))} |"
            )

    lines.extend([
        "",
        "## Whole Solve",
        "",
        "| case | impl | solve mean (ms) |",
        "|---|---:|---:|",
    ])

    for case_stem in case_stems:
        label = labels.get(case_stem, case_stem)
        for implementation, impl_label in (
            ("cpp_optimized", "cpu optimized"),
            ("cpp_cuda_edge", "cuda edge"),
            ("cpp_cuda_vertex", "cuda vertex"),
        ):
            solve = avg(groups, case_stem, implementation, "solve_sec")
            if solve is None:
                continue
            lines.append(f"| {label} | {impl_label} | {fmt_ms(solve)} |")

    lines.extend([
        "",
        "## Linear Solve Per Iteration",
        "",
        "| case | impl | solveLinearSystem avg (ms) | rhsPrepare avg (ms) | refactorization avg (ms) | solve avg (ms) | refac share | solve share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for case_stem in case_stems:
        label = labels.get(case_stem, case_stem)

        cpu_total = avg(groups, case_stem, "cpp_optimized", "NR.solveLinearSystem.avg_sec")
        cpu_refac = avg(groups, case_stem, "cpp_optimized", "CPU.solve.factorize.avg_sec")
        cpu_solve = avg(groups, case_stem, "cpp_optimized", "CPU.solve.solve.avg_sec")
        if cpu_total is not None:
            lines.append(
                f"| {label} | cpu optimized | {fmt_ms(cpu_total)} | n/a | "
                f"{fmt_ms(cpu_refac)} | {fmt_ms(cpu_solve)} | {pct(cpu_refac, cpu_total)} | {pct(cpu_solve, cpu_total)} |"
            )

        for implementation, impl_label in (
            ("cpp_cuda_edge", "cuda edge"),
            ("cpp_cuda_vertex", "cuda vertex"),
        ):
            total = avg(groups, case_stem, implementation, "NR.solveLinearSystem.avg_sec")
            if total is None:
                continue
            rhs = avg(groups, case_stem, implementation, "CUDA.solve.rhsPrepare.avg_sec")
            refac = avg(groups, case_stem, implementation, "CUDA.solve.refactorization.avg_sec")
            solve = avg(groups, case_stem, implementation, "CUDA.solve.solve.avg_sec")
            lines.append(
                f"| {label} | {impl_label} | {fmt_ms(total)} | {fmt_ms(rhs)} | "
                f"{fmt_ms(refac)} | {fmt_ms(solve)} | {pct(refac, total)} | {pct(solve, total)} |"
            )

    output_path = run_root / "TIMING_BREAKDOWN.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output_path = write_breakdown(args.run_root)
    print(output_path)


if __name__ == "__main__":
    main()
