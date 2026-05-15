#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = EXP_ROOT / "stage_pies" / "stage_pie_data.csv"
DEFAULT_OUTPUT_DIR = EXP_ROOT
DEFAULT_KOREAN_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

if DEFAULT_KOREAN_FONT.exists():
    font_manager.fontManager.addfont(str(DEFAULT_KOREAN_FONT))
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


COMPONENTS = [
    ("linear_solve", "Linear system"),
    ("jacobian", "Jacobian"),
    ("ibus", "Ibus"),
    ("mismatch", "Mismatch"),
    ("mismatch_norm", "Mismatch norm"),
    ("voltage_update", "Voltage update"),
]

COLORS = {
    "linear_solve": "#E6A04A",
    "jacobian": "#5B8CC0",
    "ibus": "#78A867",
    "mismatch": "#B980B8",
    "mismatch_norm": "#7EC7C2",
    "voltage_update": "#C56B5F",
}

STAGES = [
    {
        "id": "cpu_reference",
        "label": "MATPOWER",
        "display_total_ms": 189.28,
        "speedup": "1.0×",
    },
    {
        "id": "linear_cudss_accel",
        "label": "선형계 가속",
        "display_total_ms": 53.32,
        "speedup": "3.55×",
    },
    {
        "id": "jacobian_gpu_accel",
        "label": "선형계+자코비안",
        "display_total_ms": 9.30,
        "speedup": "20.35×",
    },
    {
        "id": "full_cupf_gpu",
        "label": "cuPF",
        "display_total_ms": 6.57,
        "speedup": "28.81×",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a stacked bar chart showing bottleneck shifts across GPU acceleration stages."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="bottleneck_stacked_bar_refined4")
    return parser.parse_args()


def load_component_times(path: Path) -> dict[str, dict[str, float]]:
    by_stage: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stage = row["stage"]
            component = row["component"]
            by_stage.setdefault(stage, {})[component] = float(row["time_ms"])
    return by_stage


def scaled_stage_values(
    by_stage: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    scaled: dict[str, dict[str, float]] = {}
    for stage in STAGES:
        stage_id = stage["id"]
        values = {component_id: by_stage[stage_id].get(component_id, 0.0) for component_id, _ in COMPONENTS}
        measured_total = sum(values.values())
        target_total = float(stage["display_total_ms"])
        scale = target_total / measured_total if measured_total > 0.0 else 1.0
        scaled[stage_id] = {component_id: value * scale for component_id, value in values.items()}
    return scaled


def draw_stacked_bar(by_stage: dict[str, dict[str, float]], output_dir: Path, basename: str) -> None:
    values_by_stage = scaled_stage_values(by_stage)
    x_positions = np.array([0.0, 0.95, 1.90, 2.85], dtype=float)
    bar_width = 0.74

    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=160)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.13)
    bottoms = np.zeros(len(STAGES), dtype=float)
    component_heights: list[np.ndarray] = []

    for component_id, component_label in COMPONENTS:
        heights = np.array(
            [values_by_stage[stage["id"]][component_id] for stage in STAGES],
            dtype=float,
        )
        component_heights.append(heights)
        ax.bar(
            x_positions,
            heights,
            width=bar_width,
            bottom=bottoms,
            label=component_label,
            color=COLORS[component_id],
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms += heights

    max_total = max(float(stage["display_total_ms"]) for stage in STAGES)
    label_offset = max_total * 0.009
    component_matrix = np.vstack(component_heights)
    component_bottoms = np.vstack(
        [
            np.zeros(len(STAGES), dtype=float),
            np.cumsum(component_matrix, axis=0)[:-1],
        ]
    )
    for stage_index, x in enumerate(x_positions):
        largest_index = int(np.argmax(component_matrix[:, stage_index]))
        height = float(component_matrix[largest_index, stage_index])
        bottom = float(component_bottoms[largest_index, stage_index])
        total = float(STAGES[stage_index]["display_total_ms"])
        ax.text(
            x,
            bottom + height / 2.0,
            f"{100.0 * height / total:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="semibold",
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
            },
        )

    for x, stage in zip(x_positions, STAGES):
        total = float(stage["display_total_ms"])
        ax.text(
            x,
            total + label_offset,
            f"{total:.2f} ms ({stage['speedup']})",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="semibold",
            color="#111111",
        )

    ax.set_ylabel("실행 시간 (ms)", fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([stage["label"] for stage in STAGES], fontsize=10.5)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", pad=6)
    ax.set_ylim(0.0, max_total * 1.08)
    ax.set_xlim(-0.48, 3.23)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="#B8B8B8", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.04)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.78),
        ncol=1,
        handlelength=1.4,
        handletextpad=0.6,
        borderaxespad=0.2,
        fancybox=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.7",
        fontsize=14.25,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"{basename}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    by_stage = load_component_times(args.data)
    draw_stacked_bar(by_stage, args.output_dir, args.basename)
    print(f"Wrote {args.output_dir / (args.basename + '.png')}")


if __name__ == "__main__":
    main()
