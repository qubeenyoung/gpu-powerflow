#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_INPUT = EXP_ROOT / "results" / "shadow_dx_selected5_bs64_r1_i1_a0p9_requested.csv"
DEFAULT_RESULTS = EXP_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot shadow dx summary figures.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def find_column(fieldnames: Iterable[str], candidates: list[str]) -> str:
    by_normalized = {normalize_name(name): name for name in fieldnames}
    for candidate in candidates:
        found = by_normalized.get(normalize_name(candidate))
        if found is not None:
            return found
    raise KeyError(f"missing required column; tried {candidates}")


def parse_finite(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return None
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else None


def mean_of(rows: list[dict[str, float]], key: str) -> float:
    return mean(row[key] for row in rows)


def load_and_aggregate(path: Path) -> tuple[list[dict[str, float | str]], dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        columns = {
            "case": find_column(fieldnames, ["case", "case_name"]),
            "dx_norm_ratio": find_column(fieldnames, ["dx_norm_ratio"]),
            "theta_norm_ratio": find_column(fieldnames, ["theta_norm_ratio"]),
            "vmag_norm_ratio": find_column(fieldnames, ["vmag_norm_ratio", "v_norm_ratio"]),
            "gmres_nonlinear_ratio_inf": find_column(
                fieldnames, ["gmres_nonlinear_ratio_inf", "gmres_ratio_inf"]
            ),
            "cudss_nonlinear_ratio_inf": find_column(
                fieldnames, ["cudss_nonlinear_ratio_inf", "direct_nonlinear_ratio_inf"]
            ),
        }
        optional_columns = {}
        for key, candidates in {
            "mismatch_before_inf": ["mismatch_before_inf", "mismatch_inf_before"],
            "mismatch_after_gmres_inf": [
                "mismatch_after_gmres_inf",
                "shadow_mismatch_after_gmres_inf",
            ],
            "mismatch_after_cudss_inf": [
                "mismatch_after_cudss_inf",
                "shadow_mismatch_after_cudss_inf",
            ],
        }.items():
            try:
                optional_columns[key] = find_column(fieldnames, candidates)
            except KeyError:
                pass

        grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
        total_rows = 0
        excluded_rows = 0
        excluded_by_case: dict[str, int] = defaultdict(int)
        for raw in reader:
            total_rows += 1
            case = raw[columns["case"]]
            parsed: dict[str, float] = {}
            valid = True
            for key, column in columns.items():
                if key == "case":
                    continue
                value = parse_finite(raw[column])
                if value is None:
                    valid = False
                    break
                parsed[key] = value

            if not valid:
                excluded_rows += 1
                excluded_by_case[case] += 1
                continue

            # Prefer explicit ratio columns, but keep compatibility with schemas that
            # expose only before/after mismatch values.
            if not math.isfinite(parsed["gmres_nonlinear_ratio_inf"]):
                before = parse_finite(raw.get(optional_columns.get("mismatch_before_inf", ""), ""))
                after = parse_finite(raw.get(optional_columns.get("mismatch_after_gmres_inf", ""), ""))
                ratio = safe_ratio(after or math.nan, before or math.nan)
                if ratio is None:
                    excluded_rows += 1
                    excluded_by_case[case] += 1
                    continue
                parsed["gmres_nonlinear_ratio_inf"] = ratio
            if not math.isfinite(parsed["cudss_nonlinear_ratio_inf"]):
                before = parse_finite(raw.get(optional_columns.get("mismatch_before_inf", ""), ""))
                after = parse_finite(raw.get(optional_columns.get("mismatch_after_cudss_inf", ""), ""))
                ratio = safe_ratio(after or math.nan, before or math.nan)
                if ratio is None:
                    excluded_rows += 1
                    excluded_by_case[case] += 1
                    continue
                parsed["cudss_nonlinear_ratio_inf"] = ratio

            grouped[case].append(parsed)

    aggregates: list[dict[str, float | str]] = []
    for case, rows in grouped.items():
        aggregates.append(
            {
                "case": case,
                "row_count": len(rows),
                "dx_norm_ratio": mean_of(rows, "dx_norm_ratio"),
                "theta_norm_ratio": mean_of(rows, "theta_norm_ratio"),
                "vmag_norm_ratio": mean_of(rows, "vmag_norm_ratio"),
                "gmres_nonlinear_ratio_inf": mean_of(rows, "gmres_nonlinear_ratio_inf"),
                "cudss_nonlinear_ratio_inf": mean_of(rows, "cudss_nonlinear_ratio_inf"),
            }
        )

    aggregates.sort(key=lambda row: float(row["theta_norm_ratio"]), reverse=True)
    log = {
        "total_rows": total_rows,
        "used_rows": sum(int(row["row_count"]) for row in aggregates),
        "excluded_rows": excluded_rows,
    }
    for case, count in excluded_by_case.items():
        log[f"excluded_{case}"] = count
    return aggregates, log


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_dx_scale(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    y_positions = list(range(len(rows)))
    colors = {
        "overall": "#4B5563",
        "theta": "#2563EB",
        "vmag": "#D97706",
    }
    markers = {"overall": "o", "theta": "^", "vmag": "s"}
    offsets = {"overall": 0.22, "theta": 0.0, "vmag": -0.22}

    for y, row in zip(y_positions, rows):
        case = str(row["case"])
        if case == "case6468rte":
            ax.axhspan(y - 0.48, y + 0.48, color="#F8FAFC", zorder=0)

        values = {
            "overall": float(row["dx_norm_ratio"]),
            "theta": float(row["theta_norm_ratio"]),
            "vmag": float(row["vmag_norm_ratio"]),
        }
        for label, value in values.items():
            yy = y + offsets[label]
            ax.scatter(
                value,
                yy,
                s=52 if case == "case6468rte" else 42,
                marker=markers[label],
                color=colors[label],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            if label == "overall":
                text = "overall"
            elif label == "theta":
                text = f"theta {value:.3f}"
            else:
                text = f"|V| {value:.3f}"
            ax.text(
                value * 1.10,
                yy,
                text,
                va="center",
                ha="left",
                fontsize=8.5,
                color=colors[label],
            )

    ax.axvline(1.0, color="#111827", linewidth=1.1, linestyle="--", alpha=0.75)
    ax.text(1.04, -0.55, "cuDSS scale", fontsize=9, color="#111827")
    ax.set_xscale("log")
    ax.set_xlim(4.0e-4, 1.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(row["case"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("ratio to cuDSS ||dx||₂")
    ax.grid(axis="x", which="major", color="#CBD5E1", linewidth=0.7, alpha=0.7)
    ax.grid(axis="x", which="minor", color="#E2E8F0", linewidth=0.45, alpha=0.55)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.text(
        0.12,
        0.035,
        "Most GMRES corrections are much smaller than cuDSS; theta is usually the smallest component.",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save_figure(fig, output_dir, "fig_dx_scale_by_field")
    plt.close(fig)


def plot_step_effectiveness(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    y_positions = list(range(len(rows)))
    gmres_color = "#2563EB"
    cudss_color = "#111827"

    for y, row in zip(y_positions, rows):
        case = str(row["case"])
        gmres_ratio = float(row["gmres_nonlinear_ratio_inf"])
        cudss_ratio = float(row["cudss_nonlinear_ratio_inf"])
        theta_ratio = float(row["theta_norm_ratio"])

        if case == "case6468rte":
            ax.axhspan(y - 0.48, y + 0.48, color="#F8FAFC", zorder=0)

        ax.plot(
            [cudss_ratio, gmres_ratio],
            [y, y],
            color="#94A3B8",
            linewidth=1.8,
            zorder=1,
        )
        ax.scatter(cudss_ratio, y, s=48, color=cudss_color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.scatter(gmres_ratio, y, s=48, color=gmres_color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(gmres_ratio * 1.08, y - 0.12, "GMRES", fontsize=8.5, color=gmres_color)
        ax.text(cudss_ratio * 1.08, y + 0.12, "cuDSS", fontsize=8.5, color=cudss_color)
        ax.text(
            1.02,
            y,
            f"theta ratio={theta_ratio:.3f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=9,
            color="#475569",
            clip_on=False,
        )
        if case == "case2383wp" and cudss_ratio > 1.0:
            ax.scatter(
                cudss_ratio,
                y,
                s=92,
                facecolor="none",
                edgecolor="#DC2626",
                linewidth=1.2,
                zorder=4,
            )
            ax.text(
                cudss_ratio * 1.10,
                y + 0.33,
                "full direct step can overshoot here",
                fontsize=8.5,
                color="#DC2626",
                ha="left",
            )

    ax.axvline(1.0, color="#64748B", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.text(1.05, -0.55, "no reduction", fontsize=9, color="#64748B")
    ax.set_xscale("log")
    ax.set_xlim(8.0e-6, 4.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(row["case"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("mismatch ratio after one step (lower is better)")
    ax.grid(axis="x", which="major", color="#CBD5E1", linewidth=0.7, alpha=0.7)
    ax.grid(axis="x", which="minor", color="#E2E8F0", linewidth=0.45, alpha=0.55)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.text(
        0.12,
        0.035,
        "Smaller theta correction usually coincides with much weaker mismatch reduction.",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.07, 0.82, 0.98))
    save_figure(fig, output_dir, "fig_step_effectiveness")
    plt.close(fig)


def write_summary_markdown(rows: list[dict[str, float | str]], log: dict[str, int], input_path: Path, output_dir: Path) -> None:
    by_case = {str(row["case"]): row for row in rows}
    case6468 = by_case.get("case6468rte")
    case2383 = by_case.get("case2383wp")

    def case_line(row: dict[str, float | str] | None, fallback: str) -> str:
        if row is None:
            return fallback
        return (
            f"`{row['case']}`: theta ratio={float(row['theta_norm_ratio']):.3f}, "
            f"GMRES mismatch ratio={float(row['gmres_nonlinear_ratio_inf']):.3f}, "
            f"cuDSS mismatch ratio={float(row['cudss_nonlinear_ratio_inf']):.3g}."
        )

    lines = [
        "# Shadow Dx Figure Summary",
        "",
        f"- Input CSV: `{input_path}`",
        f"- Rows used: {log['used_rows']} / {log['total_rows']} "
        f"(excluded non-finite rows: {log['excluded_rows']})",
        "",
        "## Figures",
        "",
        "- `fig_dx_scale_by_field`: case-averaged GMRES dx scale relative to cuDSS, split into overall, theta, and |V| components.",
        "- `fig_step_effectiveness`: case-averaged one-step mismatch ratio for GMRES versus cuDSS, with theta ratio shown beside each case.",
        "",
        "## Key Observations",
        "",
        "- GMRES corrections are much smaller than cuDSS corrections across all five cases.",
        "- The theta component is usually smaller than the |V| component, which weakens the Newton direction.",
        "- GMRES one-step mismatch reduction is generally much weaker than cuDSS.",
        "- `case6468rte` is the closest to break-even because its theta ratio is relatively large.",
        "",
        "## Case Notes",
        "",
        f"- {case_line(case6468, '`case6468rte`: not present in input.')}",
        f"- {case_line(case2383, '`case2383wp`: not present in input.')} The cuDSS shadow step can overshoot here.",
        "",
        "## Columns Used",
        "",
        "- Figure 1: `dx_norm_ratio`, `theta_norm_ratio`, `vmag_norm_ratio`",
        "- Figure 2: `gmres_nonlinear_ratio_inf`, `cudss_nonlinear_ratio_inf`, `theta_norm_ratio`",
    ]
    (output_dir / "fig_shadow_dx_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, log = load_and_aggregate(args.input)
    setup_matplotlib()
    plot_dx_scale(rows, args.output_dir)
    plot_step_effectiveness(rows, args.output_dir)
    write_summary_markdown(rows, log, args.input, args.output_dir)

    print(f"input={args.input}")
    print(f"used_rows={log['used_rows']} total_rows={log['total_rows']} excluded_rows={log['excluded_rows']}")
    for row in rows:
        print(
            f"{row['case']}: theta={float(row['theta_norm_ratio']):.4f} "
            f"vmag={float(row['vmag_norm_ratio']):.4f} "
            f"gmres_ratio={float(row['gmres_nonlinear_ratio_inf']):.4f} "
            f"cudss_ratio={float(row['cudss_nonlinear_ratio_inf']):.4g}"
        )
    print(f"wrote={args.output_dir / 'fig_dx_scale_by_field.png'}")
    print(f"wrote={args.output_dir / 'fig_step_effectiveness.png'}")
    print(f"wrote={args.output_dir / 'fig_shadow_dx_summary.md'}")


if __name__ == "__main__":
    main()
