#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "exp"
    / "20260507"
    / "results"
    / "mismatch_trends"
    / "all_matpower_b1_cuda_mixed_dump_20260507"
)

ITERATION_COLORS = {
    2: "#9D755D",
    3: "#4C78A8",
    4: "#F58518",
    5: "#54A24B",
    6: "#B279A2",
    7: "#E45756",
    8: "#72B7B2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize mismatch trajectories grouped by solver iteration count.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def median_by_x(rows: list[dict[str, str]],
                x_key: str,
                y_func) -> tuple[list[int], list[float], list[float], list[float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        x = int(row[x_key])
        y = y_func(row)
        if math.isfinite(y):
            grouped.setdefault(x, []).append(y)
    xs = sorted(grouped)
    medians = [float(np.median(grouped[x])) for x in xs]
    lows = [float(np.percentile(grouped[x], 25)) for x in xs]
    highs = [float(np.percentile(grouped[x], 75)) for x in xs]
    return xs, medians, lows, highs


def clean_figure_dir(fig_dir: Path, keep: set[str]) -> None:
    for path in fig_dir.glob("*.png"):
        if path.name not in keep:
            path.unlink()


def save_iteration_count_medians(run_root: Path,
                                 summary_rows: list[dict[str, str]],
                                 vector_rows: list[dict[str, str]]) -> None:
    fig_dir = run_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    norm_figure = "iteration_count_median_norm.png"
    direction_figure = "iteration_count_median_direction.png"
    clean_figure_dir(fig_dir, {norm_figure, direction_figure})

    case_to_iterations = {
        row["case_name"]: int(row["iterations"])
        for row in summary_rows
    }
    counts = {
        iteration_count: sum(
            1 for value in case_to_iterations.values()
            if value == iteration_count
        )
        for iteration_count in sorted(set(case_to_iterations.values()))
    }

    fig_norm, ax_norm = plt.subplots(figsize=(8.6, 4.7), constrained_layout=True)
    fig_direction, ax_direction = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)

    for iteration_count in sorted(counts):
        cases = {
            case for case, value in case_to_iterations.items()
            if value == iteration_count
        }
        rows = [row for row in vector_rows if row["case_name"] in cases]
        if not rows:
            continue
        color = ITERATION_COLORS.get(iteration_count, "#333333")
        label = f"{iteration_count} iterations (n={counts[iteration_count]})"

        xs, ys, lows, highs = median_by_x(
            rows,
            "iteration",
            lambda row: math.log10(float(row["norm_inf"])),
        )
        ax_norm.plot(xs, ys, marker="o", color=color, label=label, linewidth=2)
        ax_norm.fill_between(xs, lows, highs, color=color, alpha=0.12, linewidth=0)

        transition_rows = [row for row in rows if row["cos_prev"] != ""]
        xs2, ys2, lows2, highs2 = median_by_x(
            transition_rows,
            "iteration",
            lambda row: float(row["cos_prev"]),
        )
        ax_direction.plot(xs2, ys2, marker="o", color=color, label=label, linewidth=2)
        ax_direction.fill_between(xs2, lows2, highs2, color=color, alpha=0.12, linewidth=0)
        if xs2:
            ax_direction.scatter(
                xs2[0],
                ys2[0],
                s=95,
                marker="o",
                facecolor="white",
                edgecolor=color,
                linewidth=2.2,
                zorder=5,
            )
            ax_direction.scatter(
                xs2[-1],
                ys2[-1],
                s=95,
                marker="D",
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=6,
            )

    ax_norm.set_xlabel("Newton iteration k")
    ax_norm.set_ylabel(r"median $\log_{10}(\|F_k\|_\infty)$")
    ax_norm.grid(True, alpha=0.25)
    ax_norm.legend(fontsize=8, ncols=2)

    ax_direction.axvspan(0.75, 1.25, color="#C7CCD1", alpha=0.16, linewidth=0)
    ax_direction.set_xlabel("Newton iteration k")
    ax_direction.set_ylabel(r"median $\cos(F_k, F_{k-1})$")
    ax_direction.axhline(0.0, color="black", linewidth=0.8)
    ax_direction.set_ylim(-1.05, 1.05)
    ax_direction.grid(True, alpha=0.25)
    group_legend = ax_direction.legend(fontsize=8, ncols=2, loc="upper right")
    ax_direction.add_artist(group_legend)
    marker_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markeredgewidth=2.0,
            markersize=7,
            label="first transition",
        ),
        Line2D(
            [0], [0],
            marker="D",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            label="last transition",
        ),
    ]
    ax_direction.legend(handles=marker_handles, fontsize=8, loc="lower right")

    fig_norm.savefig(fig_dir / norm_figure, dpi=220)
    fig_direction.savefig(fig_dir / direction_figure, dpi=220)
    plt.close(fig_norm)
    plt.close(fig_direction)


def main() -> int:
    args = parse_args()
    summary_rows = read_csv(args.run_root / "trajectory_summary.csv")
    vector_rows = read_csv(args.run_root / "vector_metrics.csv")
    save_iteration_count_medians(args.run_root, summary_rows, vector_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
