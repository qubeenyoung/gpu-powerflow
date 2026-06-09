#!/usr/bin/env python3
"""Plot front-size statistics for the 2606012 lab meeting."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, to_rgb

from summarize_fronts import CASES, FRONT_DIR, ROOT, average, quantile, read_rows


OUT_DIR = ROOT / "figures"
BY_CASE_DIR = OUT_DIR / "by_case"
DATA_DIR = ROOT / "data"

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
})

IN_BAR_FONT_SIZE = 14
N_LABEL_FONT_SIZE = 14

POW2_BUCKETS = [
    ("<=4", 0, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    (">256", 257, None),
]


def bucket_of(fsz: int) -> str:
    for label, lo, hi in POW2_BUCKETS:
        if fsz >= lo and (hi is None or fsz <= hi):
            return label
    raise ValueError(f"unbucketed fsz={fsz}")


def pct(count: int, total: int) -> float:
    return (100.0 * count / total) if total else 0.0


def case_short_name(case: str) -> str:
    return (
        case.replace("case_", "")
        .replace("case", "")
        .replace("ACTIVSg", "A")
        .replace("SyntheticUSA", "USA")
    )


def write_pow2_histogram(all_rows: dict[str, list[dict[str, int]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, case, _ in CASES:
        fronts = all_rows[case]
        counts = Counter(bucket_of(r["fsz"]) for r in fronts)
        for label, _, _ in POW2_BUCKETS:
            count = counts[label]
            rows.append({
                "case_group": group,
                "case": case,
                "bucket": label,
                "count": count,
                "pct": f"{pct(count, len(fronts)):.2f}",
            })

    with (DATA_DIR / "front_size_pow2_histogram.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_group", "case", "bucket", "count", "pct"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_pow2_stacked(hist_rows: list[dict[str, object]]) -> None:
    by_case: dict[str, dict[str, float]] = defaultdict(dict)
    for row in hist_rows:
        by_case[str(row["case"])][str(row["bucket"])] = float(row["pct"])

    cases = [case for _, case, _ in CASES]
    labels = [case_short_name(case) for case in cases]
    bucket_labels = [label for label, _, _ in POW2_BUCKETS]
    colors = plt.get_cmap("tab20").colors[: len(bucket_labels)]

    fig, ax = plt.subplots(figsize=(18, 10.5), constrained_layout=True)
    y = np.arange(len(cases))
    left = np.zeros(len(cases))
    for i, bucket in enumerate(bucket_labels):
        values = np.array([by_case[case].get(bucket, 0.0) for case in cases])
        ax.barh(y, values, left=left, label=bucket, color=colors[i], edgecolor="white", linewidth=0.6)
        left += values

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Front count share (%)")
    ax.set_title("Front-size distribution by case (pow2 buckets)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.savefig(OUT_DIR / "front_size_pow2_distribution_pct.png", dpi=180)
    plt.close(fig)


def plot_case_quantiles(all_rows: dict[str, list[dict[str, int]]]) -> None:
    cases = [case for _, case, _ in CASES]
    x = np.arange(len(cases))
    q50 = [quantile([r["fsz"] for r in all_rows[case]], 0.50) for case in cases]
    q90 = [quantile([r["fsz"] for r in all_rows[case]], 0.90) for case in cases]
    q99 = [quantile([r["fsz"] for r in all_rows[case]], 0.99) for case in cases]
    fmax = [max(r["fsz"] for r in all_rows[case]) for case in cases]

    fig, ax = plt.subplots(figsize=(17, 9), constrained_layout=True)
    ax.plot(x, q50, marker="o", label="p50")
    ax.plot(x, q90, marker="o", label="p90")
    ax.plot(x, q99, marker="o", label="p99")
    ax.plot(x, fmax, marker="o", label="max")
    ax.axhline(32, color="#555555", linestyle="--", linewidth=1, label="small/mid=32")
    ax.axhline(128, color="#999999", linestyle="--", linewidth=1, label="mid/big=128")
    ax.set_yscale("log", base=2)
    ax.set_xticks(x, [case_short_name(case) for case in cases], rotation=25, ha="right")
    ax.set_ylabel("front size (fsz, log2)")
    ax.set_title("Case-level front-size quantiles")
    ax.grid(axis="y", which="both", color="#dddddd", linewidth=0.8)
    ax.legend(ncols=3, frameon=False)
    fig.savefig(OUT_DIR / "front_size_case_quantiles.png", dpi=180)
    plt.close(fig)


def level_groups(rows: list[dict[str, int]]) -> dict[int, list[dict[str, int]]]:
    out: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        out[row["level"]].append(row)
    return dict(out)


def plot_level_heatmap(all_rows: dict[str, list[dict[str, int]]]) -> None:
    cases = [case for _, case, _ in CASES]
    max_level = max(max(r["level"] for r in all_rows[case]) for case in cases)
    arr = np.full((len(cases), max_level + 1), np.nan)
    for i, case in enumerate(cases):
        for level, rows in level_groups(all_rows[case]).items():
            arr[i, level] = max(r["fsz"] for r in rows)

    positive = arr[np.isfinite(arr) & (arr > 0)]
    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
    im = ax.imshow(
        arr,
        aspect="auto",
        interpolation="nearest",
        norm=LogNorm(vmin=max(1, float(np.nanmin(positive))), vmax=float(np.nanmax(positive))),
        cmap="viridis",
    )
    ax.set_yticks(np.arange(len(cases)), [case_short_name(case) for case in cases])
    ax.set_xlabel("etree level")
    ax.set_title("Max front size per level (log color)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("max fsz")
    fig.savefig(OUT_DIR / "level_max_front_size_heatmap.png", dpi=180)
    plt.close(fig)


def plot_per_case_levels(all_rows: dict[str, list[dict[str, int]]]) -> None:
    for _, case, _ in CASES:
        grouped = level_groups(all_rows[case])
        levels = sorted(grouped)
        counts = [len(grouped[level]) for level in levels]
        q50 = [quantile([r["fsz"] for r in grouped[level]], 0.50) for level in levels]
        q90 = [quantile([r["fsz"] for r in grouped[level]], 0.90) for level in levels]
        q99 = [quantile([r["fsz"] for r in grouped[level]], 0.99) for level in levels]
        fmax = [max(r["fsz"] for r in grouped[level]) for level in levels]

        fig, (ax0, ax1) = plt.subplots(
            2,
            1,
            figsize=(17, 11),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
            constrained_layout=True,
        )

        ax0.plot(levels, q50, marker="o", markersize=3, label="p50")
        ax0.plot(levels, q90, marker="o", markersize=3, label="p90")
        ax0.plot(levels, q99, marker="o", markersize=3, label="p99")
        ax0.plot(levels, fmax, marker="o", markersize=3, label="max")
        ax0.axhline(32, color="#555555", linestyle="--", linewidth=1)
        ax0.axhline(128, color="#999999", linestyle="--", linewidth=1)
        ax0.set_yscale("log", base=2)
        ax0.set_ylabel("front size (fsz)")
        ax0.set_title(f"{case}: front-size statistics by level")
        ax0.grid(axis="y", which="both", color="#dddddd", linewidth=0.8)
        ax0.legend(ncols=4, frameon=False)

        ax1.bar(levels, counts, color="#6baed6", edgecolor="white", linewidth=0.4)
        ax1.set_xlabel("etree level")
        ax1.set_ylabel("fronts")
        ax1.set_yscale("log")
        ax1.grid(axis="y", which="both", color="#eeeeee", linewidth=0.8)

        fig.savefig(BY_CASE_DIR / f"{case}_level_front_stats.png", dpi=180)
        plt.close(fig)


def text_color_for(color: object) -> str:
    r, g, b = to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.62 else "white"


def plot_level_distribution_bars(all_rows: dict[str, list[dict[str, int]]]) -> None:
    bucket_labels = [label for label, _, _ in POW2_BUCKETS]
    colors = plt.get_cmap("tab20").colors[: len(bucket_labels)]

    for _, case, _ in CASES:
        grouped = level_groups(all_rows[case])
        levels = sorted(grouped)
        height = max(8.0, 0.55 * len(levels) + 3.0)
        fig, ax = plt.subplots(figsize=(19, height), constrained_layout=True)

        left = np.zeros(len(levels))
        totals = np.array([len(grouped[level]) for level in levels], dtype=float)

        for i, bucket in enumerate(bucket_labels):
            counts = np.array([
                sum(1 for row in grouped[level] if bucket_of(row["fsz"]) == bucket)
                for level in levels
            ], dtype=float)
            widths = np.divide(counts * 100.0, totals, out=np.zeros_like(counts), where=totals > 0)
            ax.barh(
                levels,
                widths,
                left=left,
                label=bucket,
                color=colors[i],
                edgecolor="white",
                linewidth=0.45,
                height=0.78,
            )
            for y, x0, w, c in zip(levels, left, widths, counts):
                if c <= 0 or w < 5.0:
                    continue
                ax.text(
                    x0 + w / 2.0,
                    y,
                    str(int(c)),
                    ha="center",
                    va="center",
                    fontsize=IN_BAR_FONT_SIZE,
                    color=text_color_for(colors[i]),
                )
            left += widths

        for y, total in zip(levels, totals):
            ax.text(101.0, y, f"n={int(total)}", va="center", ha="left",
                    fontsize=N_LABEL_FONT_SIZE, color="#333333")

        ax.set_yticks(levels)
        ax.set_ylim(-0.75, max(levels) + 0.75)
        ax.set_xlim(0, 116)
        ax.set_xlabel("front-size bucket share within level (%)")
        ax.set_ylabel("etree level (level 0 at bottom)")
        ax.set_title(f"{case}: per-level front-size distribution")
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        ax.legend(ncols=1, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.savefig(BY_CASE_DIR / f"{case}_level_front_distribution.png", dpi=180)
        plt.close(fig)


def write_markdown() -> None:
    by_case_lines = [
        f"- `figures/by_case/{case}_level_front_stats.png`"
        for _, case, _ in CASES
    ]
    distribution_lines = [
        f"- `figures/by_case/{case}_level_front_distribution.png`"
        for _, case, _ in CASES
    ]
    text = [
        "# Front-Size Plots",
        "",
        "Generated by `scripts/plot_front_stats.py`.",
        "",
        "The histogram buckets use powers of two so the small-front region is visible:",
        "`<=4`, `5-8`, `9-16`, `17-32`, `33-64`, `65-128`, `129-256`, `>256`.",
        "",
        "## Figures",
        "",
        "### Case-Level Pow2 Histogram",
        "",
        "![case-level pow2 histogram](figures/front_size_pow2_distribution_pct.png)",
        "",
        "### Case-Level Quantiles",
        "",
        "![case-level quantiles](figures/front_size_case_quantiles.png)",
        "",
        "### Level Max Heatmap",
        "",
        "![level max front size heatmap](figures/level_max_front_size_heatmap.png)",
        "",
        "## Per-Case Level Plots",
        "",
        "### Stacked Level Distributions",
        "",
        "Each horizontal bar is one etree level. Level 0 is at the bottom. Bar segments show the front-size bucket mix within that level; numbers inside segments are front counts, and `n=...` at the right is the total front count for the level.",
        "",
        *distribution_lines,
        "",
        "### Quantiles And Level Counts",
        "",
        *by_case_lines,
        "",
        "## Data",
        "",
        "- `data/front_size_pow2_histogram.csv`: source table for the pow2 bucket plot.",
        "",
    ]
    (ROOT / "PLOTS.md").write_text("\n".join(text))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BY_CASE_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = {
        case: read_rows(FRONT_DIR / filename)
        for _, case, filename in CASES
    }
    hist_rows = write_pow2_histogram(all_rows)
    plot_pow2_stacked(hist_rows)
    plot_case_quantiles(all_rows)
    plot_level_heatmap(all_rows)
    plot_per_case_levels(all_rows)
    plot_level_distribution_bars(all_rows)
    write_markdown()


if __name__ == "__main__":
    main()
