#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tutorial_cpp_reference_b1_ops_ms.csv"
OUT_DIR = ROOT / "figures" / "tutorial_cpp_operator_pies"

OPERATORS = [
    ("factorize", ["factorize_ms"], "#285C8E"),
    ("triangular\nsolve", ["triangular_solve_ms"], "#4DA3D9"),
    ("jacobian", ["jacobian_ms"], "#F2C94C"),
    ("ibus (SpMV)", ["ibus_ms"], "#8E44AD"),
    ("mismatch\n+norm", ["mismatch_ms", "mnorm_ms"], "#E74C3C"),
    ("voltage\nupdate", ["voltage_update_ms"], "#F39C12"),
]


def fmt_dur(value_ms: float) -> str:
    if value_ms >= 1000.0:
        return f"{value_ms / 1000.0:.3g} s"
    if value_ms >= 1.0:
        return f"{value_ms:.3g} ms"
    return f"{value_ms * 1000.0:.3g} us"


def fmt_pct(percent: float) -> str:
    if 0.0 < percent < 0.5:
        return "<1%"
    return f"{percent:.0f}%"


def draw(row: dict[str, str]) -> Path:
    case_name = row["case_name"]
    batch = row["B"]
    labels = [label for label, _, _ in OPERATORS]
    values = np.array([sum(float(row[col]) for col in cols) for _, cols, _ in OPERATORS],
                      dtype=float)
    colors = [item[2] for item in OPERATORS]
    pct = values / values.sum() * 100.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{case_name}_B{batch}_operator_pie.png"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 28,
        "font.weight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, (ax, label_ax) = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.6),
        dpi=160,
        gridspec_kw={"width_ratios": [1.02, 1.18], "wspace": 0.02},
    )
    ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        labels=None,
        radius=1.0,
        center=(0.0, 0.0),
        wedgeprops={"edgecolor": "#F7F7F7", "linewidth": 2.4},
    )

    ax.set(aspect="equal")
    ax.set_axis_off()
    label_ax.set_axis_off()
    label_ax.set_xlim(0.0, 1.18)
    label_ax.set_ylim(0.0, 1.0)

    fig.suptitle(
        f"{case_name} · matpower(cpp) · B={batch}  (iters {float(row['iterations']):.0f})  "
        f"[solve_total {fmt_dur(float(row['solve_total_ms']))}]",
        fontsize=21,
        fontweight="bold",
        y=0.965,
    )

    y_positions = np.linspace(0.82, 0.12, len(labels))
    for y, label, percent, value, color in zip(y_positions, labels, pct, values, colors):
        label_ax.text(0.05, y, "■", ha="left", va="center",
                      color=color, fontsize=32, fontweight="bold")
        label_ax.text(0.17, y, label.replace("\n", " "), ha="left", va="center",
                      color=color, fontsize=26, fontweight="bold")
        label_ax.text(1.12, y, f"{fmt_pct(percent)} ({fmt_dur(value)})",
                      ha="right", va="center", color=color, fontsize=22, fontweight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


if __name__ == "__main__":
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r["n_bus"]))
    for row in rows:
        print(draw(row))
