#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path("/workspace/exp/20260414/kcc_exp/texas_pf_gpu3")
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REPORT = ROOT / "KCC_SELECTED_5CASE_RESULTS.md"
DEFAULT_DPI = 600
BASE_FONT_SIZE = 14

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


def save_figure(fig: plt.Figure, path: Path, *, dpi: int, pad: float = 0.4, rect: tuple[float, float, float, float] | None = None) -> None:
    if rect is None:
        fig.tight_layout(pad=pad)
    else:
        fig.tight_layout(pad=pad, rect=rect)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_metadata(case: str) -> dict[str, Any]:
    return json.loads((ROOT / "cupf_dumps" / case / "metadata.json").read_text(encoding="utf-8"))


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

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=dpi)
    colors = [PASTEL["blue"], PASTEL["mint"], PASTEL["peach"], PASTEL["pink"]]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=names,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.0 else "",
        startangle=90,
        counterclock=False,
        pctdistance=0.58,
        labeldistance=1.28,
        radius=0.72,
        wedgeprops={"edgecolor": "white", "linewidth": 1.4},
    )
    for text in [*texts, *autotexts]:
        text.set_fontsize(BASE_FONT_SIZE)
        text.set_color("#263238")
    ax.axis("equal")
    save_figure(fig, path, dpi=dpi, pad=0.4)
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
    fig, ax = plt.subplots(figsize=(15.6, 8.2), dpi=dpi)

    bottom = np.zeros(len(cases))
    for label, values, color in analyze_components:
        ax.bar(x, values, 0.62, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.8)
        bottom += np.array(values)

    analyze_top = bottom.copy()
    bottom = np.zeros(len(cases))
    for _, values, _ in analyze_components:
        bottom += np.array(values)
    for label, values, color in solve_components:
        ax.bar(
            x,
            values,
            0.62,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="#607d8b",
            linewidth=0.7,
            hatch="//",
        )
        bottom += np.array(values)

    for xpos, y in zip(x, analyze_top):
        ax.hlines(y, xpos - 0.31, xpos + 0.31, colors="#455a64", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(metadata[case]["n_bus"]) for case in cases], rotation=35, ha="right", fontsize=BASE_FONT_SIZE)
    ax.set_xlabel("Buses", fontsize=BASE_FONT_SIZE)
    ax.set_ylabel("Percent (%)", fontsize=BASE_FONT_SIZE, labelpad=6)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
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
    save_figure(fig, path, dpi=dpi, pad=0.4, rect=(0.025, 0.0, 1.0, 0.86))
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
    lines.append("- The stacked graph uses combined analyze+solve percentages on the y-axis; solve components use hatching and a separator line marks the analyze/solve boundary.")
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
