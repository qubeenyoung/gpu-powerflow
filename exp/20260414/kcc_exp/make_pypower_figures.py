#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/workspace/exp/20260414/kcc_exp")
INPUT = ROOT / "tables" / "pypower_operator_pie.csv"
FIGURES = ROOT / "figures"

CASES = [
    "case30_ieee",
    "case118_ieee",
    "case793_goc",
    "case1354_pegase",
    "case2746wop_k",
    "case4601_goc",
    "case8387_pegase",
    "case9241_pegase",
]

OPS = [
    ("mismatch", "Mismatch"),
    ("jacobian", "Jacobian"),
    ("solve", "Solve"),
    ("update_voltage", "Voltage update"),
]

COLORS = {
    "mismatch": "#4C78A8",
    "jacobian": "#F58518",
    "solve": "#54A24B",
    "update_voltage": "#E45756",
}


def read_newtonpf_rows() -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != "newtonpf":
                continue
            values[(row["case_name"], row["operator_short"])] = float(row["ms_mean"])
    return values


def case_percentages(values: dict[tuple[str, str], float], case_name: str) -> list[float]:
    raw = [values[(case_name, op)] for op, _ in OPS]
    total = sum(raw)
    return [100.0 * value / total for value in raw]


def make_case4601_pie(values: dict[tuple[str, str], float]) -> Path:
    case_name = "case4601_goc"
    raw = [values[(case_name, op)] for op, _ in OPS]
    total = sum(raw)
    labels = [f"{label}: {100.0 * value / total:.1f}%" for value, (_, label) in zip(raw, OPS)]
    colors = [COLORS[op] for op, _ in OPS]

    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)
    wedges, _, _ = ax.pie(
        raw,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.0 else "",
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    ax.axis("equal")
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    path = FIGURES / "pypower_newtonpf_case4601_pie.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_stack_percent(values: dict[tuple[str, str], float]) -> Path:
    x = list(range(len(CASES)))
    labels = [case.replace("_ieee", "").replace("_goc", "").replace("_pegase", "").replace("case", "") for case in CASES]

    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=300)
    bottoms = [0.0 for _ in CASES]
    for op, label in OPS:
        heights = []
        for case_name in CASES:
            pcts = dict(zip([key for key, _ in OPS], case_percentages(values, case_name)))
            heights.append(pcts[op])
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            width=0.72,
            label=label,
            color=COLORS[op],
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False)

    path = FIGURES / "pypower_newtonpf_stack_percent.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    values = read_newtonpf_rows()
    for path in (make_case4601_pie(values), make_stack_percent(values)):
        print(path)


if __name__ == "__main__":
    main()
