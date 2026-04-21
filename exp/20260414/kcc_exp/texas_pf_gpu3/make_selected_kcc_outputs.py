#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REPORT = ROOT / "KCC_SELECTED_5CASE_RESULTS.md"
DEFAULT_DPI = 600
BASE_FONT_SIZE = 21
FIGURE_PIXEL_SIZES = {
    "base_florida_pypower_newtonpf_pie.png": (4488, 3785),
    "cuda_edge_analyze_solve_stack_by_bus.png": (9152, 4861),
}
COMPRESSED_AXIS_START = 20.0
COMPRESSED_AXIS_END = 70.0
COMPRESSED_AXIS_SCALE = 0.2

ALL_CASES = [
    "case_ACTIVSg200",
    "case_ACTIVSg500",
    "MemphisCase2026_Mar7",
    "case_ACTIVSg2000",
    "Base_Florida_42GW",
    "Texas7k_20220923",
    "Base_Texas_66GW",
    "Base_MIOHIN_76GW",
    "Base_West_Interconnect_121GW",
    "case_ACTIVSg25k",
    "case_ACTIVSg70k",
    "Base_Eastern_Interconnect_515GW",
]

SELECTED_CASES = [
    "case_ACTIVSg200",
    "MemphisCase2026_Mar7",
    "Texas7k_20220923",
    "Base_West_Interconnect_121GW",
    "Base_Eastern_Interconnect_515GW",
]

PROFILE_LABELS = {
    "pypower": "PYPOWER",
    "cpp_naive": "CPP naive",
    "cpp": "CPP optimized",
    "cuda_edge": "CUDA edge",
    "cuda_vertex": "CUDA vertex",
    "cuda_wo_cudss": "w/o cuDSS",
    "cuda_wo_jacobian": "w/o Jacobian",
    "cuda_fp64_edge": "w/o mixed precision",
}

PASTEL = {
    "pink": "#f7b7c3",
    "mint": "#9dd9c5",
    "blue": "#a7c7e7",
    "peach": "#f7c59f",
    "lavender": "#c8b6ff",
    "yellow": "#f4e7a1",
    "green": "#b8e0a8",
    "coral": "#f4a6a6",
    "gray": "#cfd8dc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate selected KCC result tables and figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output figure DPI.")
    return parser.parse_args()


def figure_size_inches(path: Path, dpi: int) -> tuple[float, float]:
    width_px, height_px = FIGURE_PIXEL_SIZES[path.name]
    return width_px / dpi, height_px / dpi


def save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int,
    pad: float = 0.4,
    rect: tuple[float, float, float, float] | None = None,
    use_tight_layout: bool = True,
) -> None:
    if path.name in FIGURE_PIXEL_SIZES:
        fig.set_size_inches(*figure_size_inches(path, dpi), forward=True)
    if use_tight_layout:
        if rect is None:
            fig.tight_layout(pad=pad)
        else:
            fig.tight_layout(pad=pad, rect=rect)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_metadata(case: str) -> dict[str, Any]:
    path = ROOT / "cupf_dumps" / case / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    in_case_sizes = False
    for line in (ROOT / "KCC_EXTRACTED_RESULTS.md").read_text(encoding="utf-8").splitlines():
        if line == "## Case Sizes":
            in_case_sizes = True
            continue
        if in_case_sizes and line.startswith("## "):
            break
        if not in_case_sizes or not line.startswith("| "):
            continue

        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 5 or parts[0] in {"case", "---"} or parts[0] != case:
            continue
        return {
            "n_bus": int(parts[1]),
            "ybus_nnz": int(parts[2]),
            "n_pv": int(parts[3]),
            "n_pq": int(parts[4]),
        }

    raise FileNotFoundError(f"metadata not found for {case}")


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_x(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def fmt_sci(value: float | None) -> str:
    return "-" if value is None else f"{value:.3e}"


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def operator_row(
    rows: dict[tuple[str, str, str, str], dict[str, str]],
    case: str,
    profile: str,
    section: str,
    metric: str,
) -> dict[str, str]:
    return rows.get((case, profile, section, metric), {})


def load_end2end_rows() -> dict[tuple[str, str], dict[str, Any]]:
    rows = {(row["case_name"], row["profile"]): dict(row) for row in read_csv(TABLES / "end2end_main_chain.csv")}

    # The main KCC result file was updated to use this warmup-3 rerun for 25K+
    # CUDA edge end-to-end values. Keep the selected report consistent with it.
    warmup3 = RESULTS / "end2end_cuda_edge_25k_plus_warmup3" / "aggregates_end2end.csv"
    if warmup3.exists():
        for row in read_csv(warmup3):
            case = row["case_name"]
            if case not in {case for case, profile in rows if profile == "cuda_edge"}:
                continue
            updated = dict(rows[(case, "cuda_edge")])
            updated["elapsed_ms_mean"] = as_float(row["elapsed_sec_mean"]) * 1000.0
            updated["elapsed_ms_stdev"] = as_float(row["elapsed_sec_stdev"]) * 1000.0
            updated["analyze_ms_mean"] = as_float(row["analyze_sec_mean"]) * 1000.0
            updated["solve_ms_mean"] = as_float(row["solve_sec_mean"]) * 1000.0
            updated["iterations_mean"] = row["iterations_mean"]
            updated["final_mismatch_max"] = row["final_mismatch_max"]
            rows[(case, "cuda_edge")] = updated

    for case in ALL_CASES:
        pypower = as_float(rows[(case, "pypower")]["elapsed_ms_mean"])
        cpp = as_float(rows[(case, "cpp")]["elapsed_ms_mean"])
        cuda = rows.get((case, "cuda_edge"))
        if cuda is not None:
            cuda_elapsed = as_float(cuda["elapsed_ms_mean"])
            cuda["speedup_vs_pypower"] = (pypower / cuda_elapsed) if pypower and cuda_elapsed else None
            cuda["speedup_vs_cpp"] = (cpp / cuda_elapsed) if cpp and cuda_elapsed else None
    return rows


def make_pypower_pie(dpi: int) -> Path:
    rows = read_csv(TABLES / "pypower_operator_pie.csv")
    selected = [
        row for row in rows
        if row["case_name"] == "Base_Florida_42GW"
        and row["scope"] == "newtonpf"
        and row["operator_short"] != "init_index"
    ]
    selected.sort(key=lambda row: ["mismatch", "jacobian", "solve", "update_voltage"].index(row["operator_short"]))

    labels = {
        "mismatch": "Mismatch",
        "jacobian": "Jacobian",
        "solve": "Linear solve",
        "update_voltage": "Voltage update",
    }
    values = [as_float(row["ms_mean"]) or 0.0 for row in selected]
    names = [labels[row["operator_short"]] for row in selected]

    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "base_florida_pypower_newtonpf_pie.png"

    fig = plt.figure(figsize=figure_size_inches(path, dpi), dpi=dpi)
    ax = fig.add_axes([0.03, 0.08, 0.52, 0.84])
    colors = [PASTEL["blue"], PASTEL["mint"], PASTEL["peach"], PASTEL["pink"]]
    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.0 else "",
        startangle=90,
        counterclock=False,
        pctdistance=0.62,
        radius=0.86,
        wedgeprops={"edgecolor": "white", "linewidth": 1.4},
    )
    for text in autotexts:
        text.set_fontsize(BASE_FONT_SIZE)
        text.set_color("#263238")
    legend_ax = fig.add_axes([0.56, 0.12, 0.42, 0.76])
    legend_ax.axis("off")
    legend_ax.legend(
        wedges,
        names,
        loc="center left",
        frameon=False,
        fontsize=BASE_FONT_SIZE,
        handlelength=1.6,
        labelspacing=1.2,
    )
    ax.axis("equal")
    save_figure(fig, path, dpi=dpi, use_tight_layout=False)
    return path


def make_analyze_solve_stack(dpi: int) -> Path:
    rows = read_csv(TABLES / "cuda_edge_ablation_operator_breakdown.csv")
    by_key = {(row["case_name"], row["profile"], row["section"], row["metric"]): row for row in rows}
    raw = [
        row for row in read_csv(RESULTS / "cuda_edge_ablation_operators" / "summary_operators.csv")
        if row["profile"] == "cuda_edge"
    ]
    metadata = {case: read_metadata(case) for case in ALL_CASES}
    cases = sorted(ALL_CASES, key=lambda case: metadata[case]["n_bus"])
    x = np.arange(len(cases))

    def mean_raw_ms(case: str, key: str) -> float:
        values = [as_float(row.get(key)) for row in raw if row["case_name"] == case]
        values = [value for value in values if value is not None]
        return float(np.mean(values) * 1000.0) if values else 0.0

    solve_components = [
        ("Solve: mismatch", [as_float(operator_row(by_key, case, "cuda_edge", "nr", "mismatch").get("ms_mean")) or 0.0 for case in cases], PASTEL["mint"]),
        ("Solve: Jacobian", [as_float(operator_row(by_key, case, "cuda_edge", "nr", "jacobian").get("ms_mean")) or 0.0 for case in cases], PASTEL["yellow"]),
        ("Solve: linear solve", [as_float(operator_row(by_key, case, "cuda_edge", "nr", "linear_solve").get("ms_mean")) or 0.0 for case in cases], PASTEL["peach"]),
        ("Solve: voltage update", [as_float(operator_row(by_key, case, "cuda_edge", "nr", "voltage_update").get("ms_mean")) or 0.0 for case in cases], PASTEL["pink"]),
    ]

    analyze_components = [
        ("Analyze: Jacobian", [mean_raw_ms(case, "NR.analyze.jacobian_builder.total_sec") for case in cases], PASTEL["blue"]),
        (
            "Analyze: cuDSS setup/analysis",
            [
                mean_raw_ms(case, "CUDA.analyze.cudss32.setup.total_sec")
                + mean_raw_ms(case, "CUDA.analyze.cudss32.analysis.total_sec")
                for case in cases
            ],
            PASTEL["lavender"],
        ),
    ]

    all_components = analyze_components + solve_components

    def normalize_to_percent(components: list[tuple[str, list[float], str]],
                             totals: np.ndarray) -> list[tuple[str, list[float], str]]:
        normalized: list[tuple[str, list[float], str]] = []
        for label, values, color in components:
            percentages = [
                (value / total * 100.0) if total > 0.0 else 0.0
                for value, total in zip(values, totals)
            ]
            normalized.append((label, percentages, color))
        return normalized

    totals = np.zeros(len(cases))
    for _, values, _ in all_components:
        totals += np.array(values)
    analyze_components = normalize_to_percent(analyze_components, totals)
    solve_components = normalize_to_percent(solve_components, totals)

    path = FIGURES / "cuda_edge_analyze_solve_stack_by_bus.png"
    fig, ax = plt.subplots(figsize=figure_size_inches(path, dpi), dpi=dpi)

    def compress_percent(values: np.ndarray | float) -> np.ndarray | float:
        values_array = np.asarray(values, dtype=float)
        compressed = np.where(
            values_array <= COMPRESSED_AXIS_START,
            values_array,
            np.where(
                values_array <= COMPRESSED_AXIS_END,
                COMPRESSED_AXIS_START + (values_array - COMPRESSED_AXIS_START) * COMPRESSED_AXIS_SCALE,
                (
                    COMPRESSED_AXIS_START
                    + (COMPRESSED_AXIS_END - COMPRESSED_AXIS_START) * COMPRESSED_AXIS_SCALE
                    + (values_array - COMPRESSED_AXIS_END)
                ),
            ),
        )
        if np.isscalar(values):
            return float(compressed)
        return compressed

    bottom_raw = np.zeros(len(cases))
    for label, values, color in analyze_components:
        values_array = np.array(values)
        top_raw = bottom_raw + values_array
        bottom = compress_percent(bottom_raw)
        top = compress_percent(top_raw)
        ax.bar(x, top - bottom, 0.62, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.8)
        bottom_raw = top_raw

    analyze_top = compress_percent(bottom_raw)
    bottom_raw = np.zeros(len(cases))
    for _, values, _ in analyze_components:
        bottom_raw += np.array(values)
    for label, values, color in solve_components:
        values_array = np.array(values)
        top_raw = bottom_raw + values_array
        bottom = compress_percent(bottom_raw)
        top = compress_percent(top_raw)
        ax.bar(
            x,
            top - bottom,
            0.62,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="#607d8b",
            linewidth=0.7,
            hatch="//",
        )
        bottom_raw = top_raw

    for xpos, y in zip(x, analyze_top):
        ax.hlines(y, xpos - 0.31, xpos + 0.31, colors="#455a64", linewidth=1.5)

    compressed_start = compress_percent(COMPRESSED_AXIS_START)
    compressed_end = compress_percent(COMPRESSED_AXIS_END)
    ax.axhspan(compressed_start, compressed_end, color="#eceff1", alpha=0.35, zorder=0)
    ax.hlines(
        [compressed_start, compressed_end],
        x[0] - 0.75,
        x[-1] + 0.75,
        colors="#90a4ae",
        linestyles=(0, (4, 4)),
        linewidth=1.0,
        alpha=0.8,
    )
    ax.text(
        x[-1] + 0.72,
        (compressed_start + compressed_end) / 2.0,
        "20-70% compressed",
        ha="right",
        va="center",
        fontsize=BASE_FONT_SIZE * 0.82,
        color="#546e7a",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(metadata[case]["n_bus"]) for case in cases], rotation=35, ha="right", fontsize=BASE_FONT_SIZE)
    ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)
    ax.set_xlabel("Buses", fontsize=BASE_FONT_SIZE)
    ax.set_ylabel("Percent (%)", fontsize=BASE_FONT_SIZE, labelpad=6)
    raw_ticks = [0, 10, 20, 70, 80, 90, 100]
    ax.set_yticks([compress_percent(tick) for tick in raw_ticks])
    ax.set_yticklabels([f"{tick}%" for tick in raw_ticks], fontsize=BASE_FONT_SIZE)
    ax.set_ylim(0, compress_percent(100.0))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=BASE_FONT_SIZE,
        frameon=True,
        framealpha=0.96,
        edgecolor="#cfd8dc",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        handlelength=2.0,
        columnspacing=1.5,
    )
    save_figure(fig, path, dpi=dpi, pad=0.4, rect=(0.07, 0.02, 0.995, 0.84))
    return path


def write_report(pie_path: Path, stack_path: Path) -> None:
    metadata = {case: read_metadata(case) for case in ALL_CASES}
    end2end = load_end2end_rows()
    jac_rows = {(row["case_name"], row["profile"]): row for row in read_csv(TABLES / "jacobian_edge_vertex.csv")}
    ablation_agg = {
        (row["case_name"], row["profile"]): row
        for row in read_csv(RESULTS / "cuda_edge_ablation_operators" / "aggregates_operators.csv")
    }

    lines: list[str] = ["# KCC Selected Five-Case Results", ""]
    lines.append("- Accuracy is reported as max final mismatch across the measured repeats.")
    lines.append("- The `Base_Eastern_Interconnect_515GW` end-to-end CUDA edge value uses the warmup 3 rerun.")
    lines.append("")

    rows = []
    for case in SELECTED_CASES:
        meta = metadata[case]
        rows.append([case, str(meta["n_bus"]), str(meta["ybus_nnz"]), str(meta["n_pv"]), str(meta["n_pq"])])
    lines.append("## Selected Cases")
    lines.extend(md_table(["case", "buses", "Ybus nnz", "PV buses", "PQ buses"], rows))
    lines.append("")

    rows = []
    for case in SELECTED_CASES:
        pypower = end2end[(case, "pypower")]
        cpp_naive = end2end[(case, "cpp_naive")]
        cpp = end2end[(case, "cpp")]
        cuda = end2end[(case, "cuda_edge")]
        rows.append([
            case,
            str(metadata[case]["n_bus"]),
            fmt_ms(as_float(pypower["elapsed_ms_mean"])),
            fmt_ms(as_float(cpp_naive["elapsed_ms_mean"])),
            fmt_ms(as_float(cpp["elapsed_ms_mean"])),
            fmt_ms(as_float(cuda["elapsed_ms_mean"])),
            fmt_x(as_float(cuda["speedup_vs_pypower"])),
            fmt_x(as_float(cuda["speedup_vs_cpp"])),
        ])
    lines.append("## End-To-End Speed And Accuracy")
    lines.append("### Speed")
    lines.append("- Time values are milliseconds; speedups are baseline / CUDA edge, so larger is faster.")
    lines.append("")
    lines.extend(md_table(
        [
            "case",
            "buses",
            "PYPOWER ms",
            "CPP naive ms",
            "CPP optimized ms",
            "CUDA edge ms",
            "CUDA vs PYPOWER",
            "CUDA vs CPP opt",
        ],
        rows,
    ))
    lines.append("")

    rows = []
    for case in SELECTED_CASES:
        row = [case, str(metadata[case]["n_bus"])]
        for profile in ("pypower", "cpp_naive", "cpp", "cuda_edge"):
            datum = end2end[(case, profile)]
            row.append(fmt_sci(as_float(datum["final_mismatch_max"])))
        rows.append(row)
    lines.append("### Accuracy: Max Final Mismatch")
    lines.extend(md_table(
        [
            "case",
            "buses",
            "PYPOWER",
            "CPP naive",
            "CPP optimized",
            "CUDA edge",
        ],
        rows,
    ))
    lines.append("")

    rows = []
    for case in SELECTED_CASES:
        edge = jac_rows[(case, "cuda_edge")]
        vertex = jac_rows[(case, "cuda_vertex")]
        rows.append([
            case,
            str(metadata[case]["n_bus"]),
            fmt_ms(as_float(edge["jacobian_update_ms_mean"])),
            fmt_ms(as_float(vertex["jacobian_update_ms_mean"])),
            fmt_x(as_float(edge["edge_vs_vertex_jacobian_speedup"])),
        ])
    lines.append("## Edge Vs Vertex Jacobian Update")
    lines.extend(md_table(
        ["case", "buses", "edge J ms", "vertex J ms", "edge speedup"],
        rows,
    ))
    lines.append("")

    rows = []
    for case in SELECTED_CASES:
        row = [case, str(metadata[case]["n_bus"])]
        for profile in ("cuda_edge", "cuda_wo_cudss", "cuda_wo_jacobian", "cuda_fp64_edge"):
            elapsed = as_float(ablation_agg[(case, profile)]["elapsed_sec_mean"])
            row.append(fmt_ms(elapsed * 1000.0 if elapsed is not None else None))
        rows.append(row)
    lines.append("## CUDA Edge Ablation")
    lines.append("- Time values are elapsed milliseconds.")
    lines.append("")
    lines.extend(md_table(
        ["case", "buses", "full ms", "w/o cuDSS ms", "w/o Jacobian ms", "w/o mixed precision ms"],
        rows,
    ))
    lines.append("")

    lines.append("## Figures")
    lines.append(f"- Base Florida PYPOWER operator pie: `{pie_path.relative_to(ROOT)}`")
    lines.append(f"- CUDA edge analyze/solve stack by bus count: `{stack_path.relative_to(ROOT)}`")
    lines.append("- The pie chart uses PYPOWER `newtonpf` operator timing and excludes `init_index`.")
    lines.append("- The stacked graph uses combined analyze+solve percentages on the y-axis; the 20-70% y-axis section is compressed, solve components use hatching, and a separator line marks the analyze/solve boundary.")
    lines.append("")
    lines.append(f"![Base Florida PYPOWER operator pie]({pie_path.relative_to(ROOT)})")
    lines.append("")
    lines.append(f"![CUDA edge analyze solve stack]({stack_path.relative_to(ROOT)})")
    lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    FIGURES.mkdir(parents=True, exist_ok=True)
    pie_path = make_pypower_pie(args.dpi)
    stack_path = make_analyze_solve_stack(args.dpi)
    write_report(pie_path, stack_path)
    print(f"[OK] wrote {REPORT}")
    print(f"[OK] wrote {pie_path}")
    print(f"[OK] wrote {stack_path}")


if __name__ == "__main__":
    main()
