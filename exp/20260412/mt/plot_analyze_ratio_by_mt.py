#!/usr/bin/env python3
"""Plot average analyze-time ratio by cuDSS host-thread MT level."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


MODE_FILES = {
    "end2end": "summary_end2end.csv",
    "operators": "summary_operators.csv",
}
TITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 15
LEGEND_FONT_SIZE = 15
POINT_LABEL_FONT_SIZE = 12
FOOTNOTE_FONT_SIZE = 13.5


def parse_mt_label(run_dir: Path) -> tuple[int, str] | None:
    name = run_dir.name
    if name.endswith("_no_mt"):
        return -1, "no_mt"

    match = re.search(r"_mt_(auto|\d+)$", name)
    if match is None:
        return None

    value = match.group(1)
    if value == "auto":
        return 999, "auto"
    return int(value), value


def collect_ratios(results_dir: Path, modes: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        parsed = parse_mt_label(run_dir)
        if parsed is None:
            continue
        mt_sort, mt_label = parsed

        for mode in modes:
            filename = MODE_FILES[mode]
            path = run_dir / filename
            if not path.exists():
                continue

            with path.open(newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    if row.get("success") != "True":
                        continue

                    elapsed_sec = float(row["elapsed_sec"])
                    analyze_sec = float(row["analyze_sec"])
                    if elapsed_sec <= 0.0:
                        continue

                    rows.append(
                        {
                            "measurement_mode": mode,
                            "mt_sort": mt_sort,
                            "mt_level": mt_label,
                            "case_name": row["case_name"],
                            "repeat_idx": row["repeat_idx"],
                            "analyze_ratio": analyze_sec / elapsed_sec,
                        }
                    )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        key = (
            str(row["measurement_mode"]),
            int(row["mt_sort"]),
            str(row["mt_level"]),
        )
        groups.setdefault(key, []).append(float(row["analyze_ratio"]))

    summary: list[dict[str, object]] = []
    for (mode, mt_sort, mt_level), values in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(
            {
                "measurement_mode": mode,
                "mt_sort": mt_sort,
                "mt_level": mt_level,
                "n": len(values),
                "analyze_ratio_mean": mean,
                "analyze_ratio_stdev": stdev,
                "analyze_ratio_percent": mean * 100.0,
                "analyze_ratio_stdev_percent": stdev * 100.0,
            }
        )
    return summary


def write_summary_csv(summary: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "measurement_mode",
        "mt_level",
        "n",
        "analyze_ratio_mean",
        "analyze_ratio_stdev",
        "analyze_ratio_percent",
        "analyze_ratio_stdev_percent",
    ]
    with output_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({field: row[field] for field in fieldnames})


def plot_summary(summary: list[dict[str, object]], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    modes = sorted({str(row["measurement_mode"]) for row in summary})
    labels = [row["mt_level"] for row in summary if row["measurement_mode"] == modes[0]]
    x_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    styles = {
        "end2end": {"color": "#1261a0", "marker": "o", "label": "end2end"},
        "operators": {"color": "#c43a31", "marker": "s", "label": "operators"},
    }

    for mode in modes:
        mode_rows = [row for row in summary if row["measurement_mode"] == mode]
        mode_rows.sort(key=lambda row: int(row["mt_sort"]))
        y_values = [float(row["analyze_ratio_mean"]) for row in mode_rows]
        ax.plot(
            x_positions,
            y_values,
            linewidth=2.2,
            markersize=6.5,
            **styles[mode],
        )
        for x, y in zip(x_positions, y_values):
            ax.annotate(
                f"{y * 100.0:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=POINT_LABEL_FONT_SIZE,
                color=styles[mode]["color"],
            )

    title_suffix = f" ({modes[0]})" if len(modes) == 1 else ""
    ax.set_title(f"Average Analyze-Time Share by MT Level{title_suffix}", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("MT level", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Analyze / elapsed time", fontsize=LABEL_FONT_SIZE)
    ax.set_xticks(x_positions, labels)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.84, 0.92)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left", fontsize=LEGEND_FONT_SIZE)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.22)
    fig.text(
        0.13,
        0.06,
        "Ratio = analyze_sec / elapsed_sec; averaged over successful end2end runs.",
        fontsize=FOOTNOTE_FONT_SIZE,
        color="#555555",
    )
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_FILES),
        default=["end2end"],
        help="Measurement mode(s) to include in the plot and summary CSV.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "analyze_ratio_by_mt.png",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "analyze_ratio_by_mt.csv",
    )
    args = parser.parse_args()

    rows = collect_ratios(args.results_dir, args.modes)
    if not rows:
        raise SystemExit(f"No successful rows found under {args.results_dir}")

    summary = summarize(rows)
    write_summary_csv(summary, args.output_csv)
    plot_summary(summary, args.output_png)
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
