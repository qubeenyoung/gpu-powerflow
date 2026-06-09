#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cupf_mixed_cudss_fp32_mt_auto_scale1_maxiter10_ops_ms.csv"
OUT_DIR = ROOT / "figures" / "cupf_operator_pies"


OPERATORS = [
    ("factorize", "factorize_ms", "#285C8E"),
    ("triangular\nsolve", "triangular_solve_ms", "#4DA3D9"),
    ("mismatch\nnorm", "mnorm_ms", "#F39C12"),
    ("ibus", "ibus_ms", "#8E44AD"),
    ("jacobian", "jacobian_ms", "#F2C94C"),
    ("mismatch", "mismatch_ms", "#E74C3C"),
    ("voltage\nupdate", "voltage_update_ms", "#2ECC71"),
]


def load_rows() -> list[dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_row(case_name: str, batch: str) -> dict[str, str]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_name"] == case_name and row["B"] == batch:
                return row
    raise SystemExit(f"missing row: case={case_name} B={batch}")


def format_duration(value_ms: float) -> str:
    if value_ms >= 1000.0:
        return f"{value_ms / 1000.0:.3g} s"
    if value_ms >= 1.0:
        return f"{value_ms:.3g} ms"
    return f"{value_ms * 1000.0:.3g} us"


def format_percent(percent: float) -> str:
    if 0.0 < percent < 0.5:
        return "<1%"
    return f"{percent:.0f}%"


def draw(row: dict[str, str]) -> Path:
    case_name = row["case_name"]
    batch = row["B"]
    labels = [item[0] for item in OPERATORS]
    values = np.array([float(row[item[1]]) for item in OPERATORS], dtype=float)
    colors = [item[2] for item in OPERATORS]
    pct = values / values.sum() * 100.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{case_name}_B{batch}_operator_pie.png"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 28,
            "font.weight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, (ax, label_ax) = plt.subplots(
        1,
        2,
        figsize=(15.0, 8.2),
        dpi=180,
        gridspec_kw={"width_ratios": [1.02, 1.18], "wspace": 0.02},
    )
    wedges, _ = ax.pie(
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

    y_positions = np.linspace(0.84, 0.18, len(labels))
    for y, label, percent, value, color in zip(y_positions, labels, pct, values, colors):
        label_ax.text(
            0.05,
            y,
            "■",
            ha="left",
            va="center",
            color=color,
            fontsize=34,
            fontweight="bold",
        )
        label_ax.text(
            0.17,
            y,
            label.replace("\n", " "),
            ha="left",
            va="center",
            color=color,
            fontsize=28,
            fontweight="bold",
        )
        label_ax.text(
            1.12,
            y,
            f"{format_percent(percent)} ({format_duration(value)})",
            ha="right",
            va="center",
            color=color,
            fontsize=24,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


if __name__ == "__main__":
    rows = sorted(load_rows(), key=lambda r: (int(r["n_bus"]), int(r["B"])))
    for row in rows:
        print(draw(row))
