#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_DUMP_ROOT = EXP_ROOT / "raw" / "cupf_jfdx_dumps_gt2k_until_convergence"
DEFAULT_SUMMARY = DEFAULT_DUMP_ROOT / "linear_system_dump_summary.csv"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "f_trajectory_gt2k_until_convergence.csv"
DEFAULT_REPORT = EXP_ROOT / "results" / "f_trajectory_gt2k_until_convergence.md"


@dataclass(frozen=True)
class CaseInfo:
    name: str
    n_bus: int
    linear_dim: int
    linear_nnz: int
    dump_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report per-case F trajectory from cuPF vector dumps.")
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_cases(path: Path) -> list[CaseInfo]:
    cases: list[CaseInfo] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cases.append(
                CaseInfo(
                    name=row["case_name"],
                    n_bus=int(row["n_bus"]),
                    linear_dim=int(row["linear_dim"]),
                    linear_nnz=int(row["linear_nnz"]),
                    dump_dir=Path(row["dump_dir"]),
                )
            )
    return cases


def available_iterations(repeat_dir: Path, prefix: str) -> list[int]:
    pattern = re.compile(rf"{re.escape(prefix)}_iter(\d+)\.txt$")
    out: list[int] = []
    for path in sorted(repeat_dir.glob(f"{prefix}_iter*.txt")):
        match = pattern.search(path.name)
        if match:
            out.append(int(match.group(1)))
    return out


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


def safe_ratio(num: float, den: float) -> float:
    if den == 0.0 or not math.isfinite(num) or not math.isfinite(den):
        return math.nan
    return num / den


def fmt(value: float, precision: int = 3) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{precision}e}"


def vector_metrics(values: np.ndarray) -> dict[str, float | int]:
    abs_values = np.abs(values)
    max_index = int(np.argmax(abs_values)) if abs_values.size else -1
    l2 = float(np.linalg.norm(values))
    return {
        "F_linf": float(abs_values[max_index]) if max_index >= 0 else math.nan,
        "F_l2": l2,
        "F_rms": l2 / math.sqrt(values.size) if values.size else math.nan,
        "F_max_index": max_index,
    }


def build_rows(cases: list[CaseInfo]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        repeat_dir = case.dump_dir / "repeat_00"
        f_iterations = available_iterations(repeat_dir, "residual")
        j_iterations = set(available_iterations(repeat_dir, "jacobian"))
        dx_iterations = set(available_iterations(repeat_dir, "dx"))
        prev: dict[str, object] | None = None
        initial: dict[str, object] | None = None
        for iteration in f_iterations:
            vector = load_vector_dump(repeat_dir / f"residual_iter{iteration}.txt")
            metrics = vector_metrics(vector)
            row: dict[str, object] = {
                "case": case.name,
                "n_bus": case.n_bus,
                "linear_dim": case.linear_dim,
                "linear_nnz": case.linear_nnz,
                "iteration": iteration,
                "has_J": "__YES__" if iteration in j_iterations else "__NO__",
                "has_dx": "__YES__" if iteration in dx_iterations else "__NO__",
            }
            row.update(metrics)
            if prev is None:
                row["prev_F_linf_ratio"] = math.nan
                row["prev_F_l2_ratio"] = math.nan
            else:
                row["prev_F_linf_ratio"] = safe_ratio(float(row["F_linf"]), float(prev["F_linf"]))
                row["prev_F_l2_ratio"] = safe_ratio(float(row["F_l2"]), float(prev["F_l2"]))
            if initial is None:
                initial = row
            row["initial_F_linf_ratio"] = safe_ratio(float(row["F_linf"]), float(initial["F_linf"]))
            row["initial_F_l2_ratio"] = safe_ratio(float(row["F_l2"]), float(initial["F_l2"]))
            rows.append(row)
            prev = row
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "n_bus",
        "linear_dim",
        "linear_nnz",
        "iteration",
        "has_J",
        "has_dx",
        "F_linf",
        "F_l2",
        "F_rms",
        "F_max_index",
        "prev_F_linf_ratio",
        "prev_F_l2_ratio",
        "initial_F_linf_ratio",
        "initial_F_l2_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]], path: Path, dump_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_case: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)
    for case_rows in by_case.values():
        case_rows.sort(key=lambda item: int(item["iteration"]))

    lines: list[str] = []
    lines.append("# F Trajectory, >2K Cases")
    lines.append("")
    lines.append(f"- Dump root: `{dump_root}`")
    lines.append("- Rows are per case and per Newton mismatch evaluation; no cross-case averaging is used.")
    lines.append("- Final converged rows often have `has_J=__NO__` and `has_dx=__NO__` because cuPF checks convergence before building the next linear system.")
    lines.append("")
    lines.append("## Case Summary")
    lines.append("")
    lines.append("| case | buses | F evals | J/dx solves | F_inf start | F_inf end | end/start | largest one-step ratio | monotone F_inf |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for case in sorted(by_case, key=lambda name: (int(by_case[name][0]["n_bus"]), name)):
        case_rows = by_case[case]
        start = case_rows[0]
        end = case_rows[-1]
        solve_count = sum(1 for row in case_rows if row["has_J"] == "__YES__" and row["has_dx"] == "__YES__")
        ratios = [float(row["prev_F_linf_ratio"]) for row in case_rows[1:]]
        finite_ratios = [value for value in ratios if math.isfinite(value)]
        largest_ratio = max(finite_ratios) if finite_ratios else math.nan
        monotone = all(value <= 1.0 for value in finite_ratios)
        lines.append(
            f"| {case} | {start['n_bus']} | {len(case_rows)} | {solve_count} | "
            f"{fmt(float(start['F_linf']))} | {fmt(float(end['F_linf']))} | "
            f"{fmt(float(end['initial_F_linf_ratio']))} | {fmt(largest_ratio)} | "
            f"{'yes' if monotone else 'no'} |"
        )

    lines.append("")
    lines.append("## Per-Case Iteration Detail")
    for case in sorted(by_case, key=lambda name: (int(by_case[name][0]["n_bus"]), name)):
        lines.append("")
        lines.append(f"### {case}")
        lines.append("")
        lines.append("| iter | J | dx | F_inf | F_l2 | F_inf / prev | F_inf / iter0 | max abs index |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for row in by_case[case]:
            lines.append(
                f"| {row['iteration']} | {row['has_J']} | {row['has_dx']} | "
                f"{fmt(float(row['F_linf']))} | {fmt(float(row['F_l2']))} | "
                f"{fmt(float(row['prev_F_linf_ratio']))} | "
                f"{fmt(float(row['initial_F_linf_ratio']))} | "
                f"{row['F_max_index']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cases = read_cases(args.summary)
    rows = build_rows(cases)
    write_csv(rows, args.output)
    write_report(rows, args.report, args.dump_root)
    print(f"[DONE] output={args.output}")
    print(f"[DONE] report={args.report}")


if __name__ == "__main__":
    main()
