#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIN_INPUT = ROOT / "lin_solver" / "comparison.csv"
CUPF_INPUT = ROOT / "cupf" / "comparison.csv"
CUPF_RAW = ROOT / "cupf" / "raw_runs.csv"
OUT = ROOT / "figs"

PHASES = [
    ("analyze", "Analyze"),
    ("factorize", "Factorize"),
    ("solve", "Solve"),
]

CUPF_STAGES = [
    ("initialize", "Initialize"),
    ("solve", "Solve"),
    ("total", "Total"),
]

COLORS = {
    "cudss": "#3b6fb6",
    "custom": "#c94f44",
}


def read_lin_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with LIN_INPUT.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok":
                continue
            out: dict[str, object] = {"case": row["case"]}
            for key in [
                "n",
                "nnz",
                "cudss_analyze_mean_ms",
                "custom_analyze_mean_ms",
                "cudss_factorize_mean_ms",
                "custom_factorize_mean_ms",
                "cudss_solve_mean_ms",
                "custom_solve_mean_ms",
            ]:
                out[key] = float(row[key])
            out["cudss_total_ms"] = (
                out["cudss_analyze_mean_ms"]
                + out["cudss_factorize_mean_ms"]
                + out["cudss_solve_mean_ms"]
            )
            out["custom_total_ms"] = (
                out["custom_analyze_mean_ms"]
                + out["custom_factorize_mean_ms"]
                + out["custom_solve_mean_ms"]
            )
            rows.append(out)
    return sorted(rows, key=lambda item: item["n"])


def read_cupf_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with CUPF_INPUT.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok":
                continue
            out: dict[str, object] = {"case": row["case"]}
            for key in [
                "n_bus",
                "cudss_initialize_mean_ms",
                "custom_initialize_mean_ms",
                "cudss_solve_mean_ms",
                "custom_solve_mean_ms",
                "cudss_total_mean_ms",
                "custom_total_mean_ms",
            ]:
                out[key] = float(row[key])
            rows.append(out)
    return sorted(rows, key=lambda item: item["n_bus"])


def read_cupf_iteration_stats() -> tuple[int, int, dict[int, int]]:
    by_case: dict[str, dict[str, list[int]]] = {}
    with CUPF_RAW.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok":
                continue
            by_backend = by_case.setdefault(row["case"], {})
            by_backend.setdefault(row["backend"], []).append(int(row["iterations"]))

    same_cases = 0
    distribution: dict[int, int] = {}
    for backends in by_case.values():
        cudss = backends.get("cudss", [])
        custom = backends.get("custom", [])
        if cudss and custom and cudss == custom:
            same_cases += 1
            distribution[cudss[0]] = distribution.get(cudss[0], 0) + 1
    return len(by_case), same_cases, dict(sorted(distribution.items()))


def arrays(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def annotate_largest(ax: plt.Axes, rows: list[dict[str, object]], x_key: str, y_key: str) -> None:
    for row in sorted(rows, key=lambda item: float(item[x_key]), reverse=True)[:3]:
        ax.annotate(
            str(row["case"]).replace("case_", ""),
            xy=(float(row[x_key]), float(row[y_key])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#303030",
        )


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color("#707070")


def plot_phase_times(rows: list[dict[str, object]], x_key: str, xlabel: str, name: str) -> None:
    x = arrays(rows, x_key)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, (phase, title) in zip(axes, PHASES):
        cudss_key = f"cudss_{phase}_mean_ms"
        custom_key = f"custom_{phase}_mean_ms"
        ax.plot(x, arrays(rows, cudss_key), "o-", ms=4, lw=1.5, color=COLORS["cudss"], label="cuDSS")
        ax.plot(x, arrays(rows, custom_key), "s-", ms=4, lw=1.5, color=COLORS["custom"], label="Custom")
        style_axes(ax, xlabel, "time (ms)")
        ax.set_title(title)
        annotate_largest(ax, rows, x_key, custom_key)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Linear solver phase time by case size", fontsize=14)
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def plot_total_times(rows: list[dict[str, object]], x_key: str, xlabel: str, name: str) -> None:
    x = arrays(rows, x_key)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(x, arrays(rows, "cudss_total_ms"), "o-", ms=4, lw=1.6, color=COLORS["cudss"], label="cuDSS")
    ax.plot(x, arrays(rows, "custom_total_ms"), "s-", ms=4, lw=1.6, color=COLORS["custom"], label="Custom")
    style_axes(ax, xlabel, "analyze + factorize + solve (ms)")
    ax.set_title("Total direct-solver phase time")
    annotate_largest(ax, rows, x_key, "custom_total_ms")
    ax.legend(frameon=False, loc="best")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def plot_speedups(rows: list[dict[str, object]], x_key: str, xlabel: str, name: str) -> None:
    x = arrays(rows, x_key)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for phase, label in PHASES:
        speedup = arrays(rows, f"cudss_{phase}_mean_ms") / arrays(rows, f"custom_{phase}_mean_ms")
        ax.plot(x, speedup, "o-", ms=4, lw=1.5, label=label)
    total_speedup = arrays(rows, "cudss_total_ms") / arrays(rows, "custom_total_ms")
    ax.plot(x, total_speedup, "k^-", ms=5, lw=1.8, label="Total")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("speedup (cuDSS time / custom time)")
    ax.set_title("Custom solver speedup by case size")
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.8)
    ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    speedup_by_case = {str(row["case"]): value for row, value in zip(rows, total_speedup)}
    for row in sorted(rows, key=lambda item: float(item[x_key]), reverse=True)[:3]:
        ax.annotate(
            str(row["case"]).replace("case_", ""),
            xy=(float(row[x_key]), speedup_by_case[str(row["case"])]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#303030",
        )
    ax.legend(frameon=False, loc="best")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def plot_cupf_times(rows: list[dict[str, object]], x_key: str, xlabel: str, name: str) -> None:
    x = arrays(rows, x_key)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, (stage, title) in zip(axes, CUPF_STAGES):
        cudss_key = f"cudss_{stage}_mean_ms"
        custom_key = f"custom_{stage}_mean_ms"
        ax.plot(x, arrays(rows, cudss_key), "o-", ms=4, lw=1.5, color=COLORS["cudss"], label="cuDSS")
        ax.plot(x, arrays(rows, custom_key), "s-", ms=4, lw=1.5, color=COLORS["custom"], label="Custom")
        style_axes(ax, xlabel, "time (ms)")
        ax.set_title(title)
        annotate_largest(ax, rows, x_key, custom_key)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("cuPF runtime by CUDA linear solver and case size", fontsize=14)
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def write_notes(lin_rows: list[dict[str, object]], cupf_rows: list[dict[str, object]]) -> None:
    lin_total_speedup = arrays(lin_rows, "cudss_total_ms") / arrays(lin_rows, "custom_total_ms")
    analyze_speedup = arrays(lin_rows, "cudss_analyze_mean_ms") / arrays(lin_rows, "custom_analyze_mean_ms")
    factor_speedup = arrays(lin_rows, "cudss_factorize_mean_ms") / arrays(lin_rows, "custom_factorize_mean_ms")
    lin_solve_speedup = arrays(lin_rows, "cudss_solve_mean_ms") / arrays(lin_rows, "custom_solve_mean_ms")

    cupf_case_count, cupf_same_iteration_cases, iteration_distribution = read_cupf_iteration_stats()
    iteration_distribution_text = ", ".join(
        f"{iterations} iter: {count}" for iterations, count in iteration_distribution.items()
    )
    lines = [
        "# 2026-05-29 Figures",
        "",
        f"- Linear solver source: `{LIN_INPUT.relative_to(ROOT)}`",
        f"- cuPF source: `{CUPF_INPUT.relative_to(ROOT)}`",
        f"- Cases: {len(lin_rows)} linear-system rows, {len(cupf_rows)} cuPF rows",
        "- X-axis uses log scale because case sizes span several orders of magnitude.",
        "- Time figures use mean of 10 independent process runs per case/backend.",
        "",
        "## Linear Solver Files",
        "",
        "- `lin_solver_phase_time_vs_n.png/pdf`: analyze, factorize, solve vs Jacobian dimension",
        "- `lin_solver_total_time_vs_n.png/pdf`: analyze + factorize + solve vs Jacobian dimension",
        "- `lin_solver_speedup_vs_n.png/pdf`: cuDSS/custom speedup vs Jacobian dimension",
        "- `lin_solver_phase_time_vs_nnz.png/pdf`: same phase comparison vs nonzeros",
        "- `lin_solver_speedup_vs_nnz.png/pdf`: speedup vs nonzeros",
        "",
        "## cuPF Files",
        "",
        "- `cupf_time_vs_n_bus.png`: initialize, solve, total runtime vs bus count",
        "",
        "## Linear Solver Median Speedup",
        "",
        f"- Analyze: {np.median(analyze_speedup):.2f}x",
        f"- Factorize: {np.median(factor_speedup):.2f}x",
        f"- Solve: {np.median(lin_solve_speedup):.2f}x",
        f"- Total: {np.median(lin_total_speedup):.2f}x",
        "",
        "## cuPF Newton Iterations",
        "",
        f"- Source: `{CUPF_RAW.relative_to(ROOT)}` (`iterations` per measured run)",
        f"- Identical cuDSS/custom per-run iteration sequences: {cupf_same_iteration_cases} / {cupf_case_count} cases",
        f"- Case distribution: {iteration_distribution_text}",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lin_rows = read_lin_rows()
    plot_phase_times(lin_rows, "n", "Jacobian dimension n", "lin_solver_phase_time_vs_n")
    plot_total_times(lin_rows, "n", "Jacobian dimension n", "lin_solver_total_time_vs_n")
    plot_speedups(lin_rows, "n", "Jacobian dimension n", "lin_solver_speedup_vs_n")
    plot_phase_times(lin_rows, "nnz", "Jacobian nonzeros", "lin_solver_phase_time_vs_nnz")
    plot_speedups(lin_rows, "nnz", "Jacobian nonzeros", "lin_solver_speedup_vs_nnz")

    cupf_rows = read_cupf_rows()
    plot_cupf_times(cupf_rows, "n_bus", "bus count", "cupf_time_vs_n_bus")
    write_notes(lin_rows, cupf_rows)


if __name__ == "__main__":
    main()
