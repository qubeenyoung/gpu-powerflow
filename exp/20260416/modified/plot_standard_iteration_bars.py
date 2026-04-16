#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULT_ROOT = (
    Path(__file__).resolve().parent / "results" / "texas_gpu3_mt_auto_r10"
)

OPERATORS = [
    ("mismatch", "Mismatch", "#4C78A8"),
    ("jacobian", "Jacobian", "#72B7B2"),
    ("linear_factorize", "Factorize", "#F58518"),
    ("linear_solve", "Solve", "#54A24B"),
    ("voltage_update", "Voltage update", "#B279A2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot standard Newton iteration stacked operator breakdown."
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def build_bus_map(summary_rows: list[dict[str, str]]) -> dict[str, int]:
    bus_map: dict[str, int] = {}
    for row in summary_rows:
        if row.get("measurement_mode") != "operators":
            continue
        if row.get("profile") != "cuda_edge":
            continue
        bus_map[row["case_name"]] = int(row["buses"])
    return bus_map


def write_plot_data(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_name",
        "buses",
        "accounted_ms",
        *[f"{name}_ms" for name, _, _ in OPERATORS],
        *[f"{name}_pct" for name, _, _ in OPERATORS],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_stacked(
    *,
    rows: list[dict[str, Any]],
    mode: str,
    out_png: Path,
    out_svg: Path,
) -> None:
    x = list(range(len(rows)))
    xlabels = [f"{row['buses']:,}" for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    bottom = [0.0 for _ in rows]
    for name, label, color in OPERATORS:
        if mode == "ratio":
            values = [float(row[f"{name}_pct"]) for row in rows]
        else:
            values = [float(row[f"{name}_ms"]) for row in rows]
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.72)
        bottom = [a + b for a, b in zip(bottom, values)]

    if mode == "ratio":
        ax.set_title("Standard Newton Iteration Operator Share")
        ax.set_ylabel("Share of accounted iteration time (%)")
        ax.set_ylim(0, 100)
    else:
        ax.set_title("Standard Newton Iteration Operator Time")
        ax.set_ylabel("Time per full Newton iteration (ms)")
        for xpos, total in zip(x, bottom):
            ax.text(xpos, total, f"{total:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Buses")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.grid(axis="y", color="#D6D6D6", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)

    fig.savefig(out_png, dpi=180)
    fig.savefig(out_svg)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.result_root
    breakdown = read_csv(root / "standard_iteration_operator_breakdown.csv")
    summary = read_csv(root / "filtered_summary.csv")
    bus_map = build_bus_map(summary)

    rows: list[dict[str, Any]] = []
    for row in breakdown:
        case_name = row["case_name"]
        if case_name.startswith("__"):
            continue
        buses = bus_map.get(case_name)
        if buses is None:
            continue
        out: dict[str, Any] = {
            "case_name": case_name,
            "buses": buses,
            "accounted_ms": fnum(row["accounted_per_full_iteration_ms"]),
        }
        for name, _, _ in OPERATORS:
            out[f"{name}_ms"] = fnum(row[f"{name}_per_full_iteration_ms"])
            out[f"{name}_pct"] = fnum(row[f"{name}_share_pct"])
        rows.append(out)

    rows.sort(key=lambda item: (item["buses"], item["case_name"]))
    write_plot_data(root / "standard_iteration_stacked_bar_data.csv", rows)
    plot_stacked(
        rows=rows,
        mode="ratio",
        out_png=root / "standard_iteration_stacked_ratio.png",
        out_svg=root / "standard_iteration_stacked_ratio.svg",
    )
    plot_stacked(
        rows=rows,
        mode="value",
        out_png=root / "standard_iteration_stacked_values.png",
        out_svg=root / "standard_iteration_stacked_values.svg",
    )
    print(root / "standard_iteration_stacked_ratio.png")
    print(root / "standard_iteration_stacked_values.png")


if __name__ == "__main__":
    main()
