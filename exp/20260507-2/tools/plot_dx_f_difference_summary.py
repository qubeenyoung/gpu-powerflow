#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finite_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    mask = np.ones(len(out), dtype=bool)
    for column in columns:
        mask &= np.isfinite(out[column].to_numpy(dtype=float))
    return out.loc[mask].copy()


def norm2(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.dot(x, x)))


def cosine(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    denom = math.sqrt(float(np.dot(x, x))) * math.sqrt(float(np.dot(y, y)))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def summarize_j1(bus_df: pd.DataFrame, eq_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["case", "iteration", "block_size", "solver"]
    for key, group in bus_df.groupby(group_cols, dropna=False):
        case, iteration, block_size, solver = key
        eq_group = eq_df[
            (eq_df["case"] == case)
            & (eq_df["iteration"] == iteration)
            & (eq_df["block_size"] == block_size)
            & (eq_df["solver"] == solver)
        ]
        dx_direct_norm = math.sqrt(
            norm2(group["dx_cudss_theta"]) ** 2 + norm2(group["dx_cudss_vmag"]) ** 2
        )
        dx_iter_norm = math.sqrt(
            norm2(group["dx_iter_theta"]) ** 2 + norm2(group["dx_iter_vmag"]) ** 2
        )
        dx_error_norm = norm2(group["combined_dx_error_norm"])
        theta_direct = norm2(group["dx_cudss_theta"])
        theta_iter = norm2(group["dx_iter_theta"])
        vmag_direct = norm2(group["dx_cudss_vmag"])
        vmag_iter = norm2(group["dx_iter_vmag"])
        rows.append(
            {
                "case": case,
                "iteration": iteration,
                "block_size": block_size,
                "solver": solver,
                "dx_norm_ratio_iter_to_cudss": dx_iter_norm / max(dx_direct_norm, 1.0e-300),
                "dx_error_ratio": dx_error_norm / max(dx_direct_norm, 1.0e-300),
                "dx_cosine": cosine(
                    pd.concat([group["dx_iter_theta"], group["dx_iter_vmag"]], ignore_index=True),
                    pd.concat([group["dx_cudss_theta"], group["dx_cudss_vmag"]], ignore_index=True),
                ),
                "theta_norm_ratio_iter_to_cudss": theta_iter / max(theta_direct, 1.0e-300),
                "theta_error_ratio": norm2(group["theta_abs_error"]) / max(theta_direct, 1.0e-300),
                "theta_cosine": cosine(group["dx_iter_theta"], group["dx_cudss_theta"]),
                "vmag_norm_ratio_iter_to_cudss": vmag_iter / max(vmag_direct, 1.0e-300),
                "vmag_error_ratio": norm2(group["vmag_abs_error"]) / max(vmag_direct, 1.0e-300),
                "vmag_cosine": cosine(group["dx_iter_vmag"], group["dx_cudss_vmag"]),
                "linear_f_error_norm": norm2(eq_group["combined_PQ_error_norm"]),
                "p_error_norm": norm2(eq_group["P_error_abs"]),
                "q_error_norm": norm2(eq_group["Q_error_abs"]),
                "top_dx_error_bus": int(
                    group.sort_values("combined_dx_error_norm", ascending=False).iloc[0]["bus_id"]
                ),
                "top_f_error_bus": int(
                    eq_group.sort_values("combined_PQ_error_norm", ascending=False).iloc[0]["bus_id"]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_bus_error_map(bus_df: pd.DataFrame,
                       eq_df: pd.DataFrame,
                       case_name: str,
                       solver: str,
                       out_dir: Path) -> None:
    bus = bus_df[(bus_df["case"] == case_name) & (bus_df["solver"] == solver)].copy()
    eq = eq_df[(eq_df["case"] == case_name) & (eq_df["solver"] == solver)].copy()
    if bus.empty or eq.empty:
        raise RuntimeError(f"missing rows for {case_name} / {solver}")

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=True)
    x = bus["bus_id"].to_numpy(dtype=float)
    axes[0].scatter(x, bus["theta_abs_error"], s=5, alpha=0.45, color="#2458a4")
    axes[0].scatter(x, bus["vmag_abs_error"], s=5, alpha=0.45, color="#c45a2a")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("|dx_iter - dx_cuDSS|")
    axes[0].grid(True, which="both", axis="y", alpha=0.22)
    axes[0].text(0.99, 0.88, "theta", color="#2458a4", ha="right", transform=axes[0].transAxes)
    axes[0].text(0.99, 0.78, "|V|", color="#c45a2a", ha="right", transform=axes[0].transAxes)

    axes[1].scatter(eq["bus_id"], eq["P_error_abs"], s=5, alpha=0.45, color="#3d7d44")
    axes[1].scatter(eq["bus_id"], eq["Q_error_abs"], s=5, alpha=0.45, color="#6d4aa2")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("bus index")
    axes[1].set_ylabel("|J(dx_iter - dx_cuDSS)|")
    axes[1].grid(True, which="both", axis="y", alpha=0.22)
    axes[1].text(0.99, 0.88, "P rows", color="#3d7d44", ha="right", transform=axes[1].transAxes)
    axes[1].text(0.99, 0.78, "Q rows", color="#6d4aa2", ha="right", transform=axes[1].transAxes)

    top_bus = bus.nlargest(5, "combined_dx_error_norm")["bus_id"].astype(int).tolist()
    for axis in axes:
        for bus_id in top_bus:
            axis.axvline(bus_id, color="0.35", lw=0.7, alpha=0.18)
    axes[0].text(
        0.01,
        0.06,
        f"{case_name}, J1/F1, {solver}: largest dx-error buses are marked",
        transform=axes[0].transAxes,
        fontsize=10,
        color="0.25",
    )

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig_{case_name}_j1_bus_error_map.{ext}", dpi=220)
    plt.close(fig)


def plot_top_dx_pairs(bus_df: pd.DataFrame,
                      case_name: str,
                      solver: str,
                      out_dir: Path,
                      top_n: int = 14) -> None:
    bus = bus_df[(bus_df["case"] == case_name) & (bus_df["solver"] == solver)].copy()
    bus = bus.nlargest(top_n, "theta_abs_error").sort_values("theta_abs_error")
    labels = [f"bus {int(x)}" for x in bus["bus_id"]]
    y = np.arange(len(bus))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
    axes[0].hlines(y, bus["dx_iter_theta"], bus["dx_cudss_theta"], color="0.72", lw=1.2)
    axes[0].scatter(bus["dx_cudss_theta"], y, s=34, color="#222222")
    axes[0].scatter(bus["dx_iter_theta"], y, s=34, color="#2458a4")
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("theta correction")
    axes[0].grid(True, axis="x", alpha=0.22)
    axes[0].text(0.98, 0.92, "cuDSS", ha="right", color="#222222", transform=axes[0].transAxes)
    axes[0].text(0.98, 0.84, "iterative", ha="right", color="#2458a4", transform=axes[0].transAxes)

    axes[1].hlines(y, bus["dx_iter_vmag"], bus["dx_cudss_vmag"], color="0.72", lw=1.2)
    axes[1].scatter(bus["dx_cudss_vmag"], y, s=34, color="#222222")
    axes[1].scatter(bus["dx_iter_vmag"], y, s=34, color="#c45a2a")
    axes[1].set_xlabel("|V| correction")
    axes[1].grid(True, axis="x", alpha=0.22)
    axes[1].text(
        0.02,
        0.03,
        "Top buses by theta error.\nLong segments mean the iterative step misses the direct step.",
        transform=axes[1].transAxes,
        fontsize=9,
        color="0.25",
    )

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig_{case_name}_j1_top_dx_pairs.{ext}", dpi=220)
    plt.close(fig)


def plot_iteration_shadow(shadow_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    columns = [
        "nr_iter",
        "theta_norm_ratio",
        "vmag_norm_ratio",
        "gmres_nonlinear_ratio_inf",
        "cudss_nonlinear_ratio_inf",
        "theta_cosine",
        "vmag_cosine",
    ]
    shadow = finite_frame(shadow_df, columns)
    cases = list(dict.fromkeys(shadow["case"].tolist()))
    cases = [case for case in cases if case in {
        "case2383wp",
        "case3120sp",
        "case6468rte",
        "case9241pegase",
        "case13659pegase",
    }]
    fig, axes = plt.subplots(len(cases), 2, figsize=(12.0, 2.25 * len(cases)), sharex=False)
    if len(cases) == 1:
        axes = np.array([axes])

    rows: list[dict[str, object]] = []
    for row_idx, case in enumerate(cases):
        data = shadow[shadow["case"] == case].sort_values("nr_iter")
        ax = axes[row_idx, 0]
        ax.plot(data["nr_iter"], data["theta_norm_ratio"], marker="o", ms=3, color="#2458a4")
        ax.plot(data["nr_iter"], data["vmag_norm_ratio"], marker="o", ms=3, color="#c45a2a")
        ax.axhline(1.0, color="0.25", lw=0.8, alpha=0.35)
        ax.set_yscale("log")
        ax.set_ylabel(case)
        ax.grid(True, which="both", axis="y", alpha=0.2)
        if row_idx == 0:
            ax.text(0.03, 0.88, "theta ratio", color="#2458a4", transform=ax.transAxes)
            ax.text(0.03, 0.72, "|V| ratio", color="#c45a2a", transform=ax.transAxes)
        if row_idx == len(cases) - 1:
            ax.set_xlabel("NR iteration")

        ax2 = axes[row_idx, 1]
        ax2.plot(data["nr_iter"], data["gmres_nonlinear_ratio_inf"], marker="o", ms=3,
                 color="#2458a4")
        ax2.plot(data["nr_iter"], data["cudss_nonlinear_ratio_inf"], marker="o", ms=3,
                 color="#222222")
        ax2.axhline(1.0, color="0.25", lw=0.8, alpha=0.35)
        ax2.set_yscale("log")
        ax2.grid(True, which="both", axis="y", alpha=0.2)
        if row_idx == 0:
            ax2.text(0.03, 0.88, "iterative F-after / F-before", color="#2458a4",
                     transform=ax2.transAxes)
            ax2.text(0.03, 0.72, "cuDSS F-after / F-before", color="#222222",
                     transform=ax2.transAxes)
        if row_idx == len(cases) - 1:
            ax2.set_xlabel("NR iteration")

        rows.append(
            {
                "case": case,
                "middle_steps": len(data),
                "avg_theta_norm_ratio": data["theta_norm_ratio"].mean(),
                "avg_theta_cosine": data["theta_cosine"].mean(),
                "avg_vmag_norm_ratio": data["vmag_norm_ratio"].mean(),
                "avg_vmag_cosine": data["vmag_cosine"].mean(),
                "avg_iterative_nonlinear_ratio_inf": data["gmres_nonlinear_ratio_inf"].mean(),
                "avg_cudss_nonlinear_ratio_inf": data["cudss_nonlinear_ratio_inf"].mean(),
            }
        )

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig_iteration_dx_f_gap.{ext}", dpi=220)
    plt.close(fig)
    return pd.DataFrame(rows)


def write_markdown(out_dir: Path,
                   input_dir: Path,
                   shadow_path: Path,
                   case_name: str,
                   summary: pd.DataFrame,
                   iter_summary: pd.DataFrame,
                   nr_summary_path: Path) -> None:
    bilu = summary[summary["solver"].str.contains("block_ilu0", regex=False)].copy()
    bilu = bilu.sort_values("dx_error_ratio", ascending=False)
    nr_summary = pd.read_csv(nr_summary_path) if nr_summary_path.exists() else pd.DataFrame()

    lines: list[str] = []
    lines.append("# dx/F Difference Visual Summary\n")
    lines.append("## Inputs\n")
    lines.append(f"- J1/F1 localization CSV: `{input_dir}`")
    lines.append(f"- NR shadow CSV: `{shadow_path}`")
    lines.append("- J1 means the second Newton linear system in the existing dump naming.\n")
    lines.append("## What The Figures Show\n")
    lines.append(
        "- `fig_case13659pegase_j1_bus_error_map.*` maps where the iterative `dx` differs "
        "from cuDSS and where that difference appears in the linear equation rows "
        "`J(dx_iter-dx_cuDSS)`."
    )
    lines.append(
        "- `fig_case13659pegase_j1_top_dx_pairs.*` shows direct-vs-iterative correction "
        "values on the buses with the largest theta error."
    )
    lines.append(
        "- `fig_iteration_dx_f_gap.*` shows how the dx scale gap and nonlinear mismatch "
        "reduction gap evolve across middle NR iterations.\n"
    )
    lines.append("## Case Selection\n")
    lines.append(
        f"- Main visual case: `{case_name}`. It has the largest J1 block-ILU0 dx error "
        "among the selected cases, so it is the clearest failure-mode picture."
    )
    lines.append(
        "- The comparison set keeps `case2383wp`, `case3120sp`, `case6468rte`, "
        "`case9241pegase`, and `case13659pegase` so mild, failed, and severe cases are all visible.\n"
    )
    lines.append("## J1/F1 Block-ILU0 Summary\n")
    cols = [
        "case",
        "dx_norm_ratio_iter_to_cudss",
        "dx_error_ratio",
        "dx_cosine",
        "theta_norm_ratio_iter_to_cudss",
        "theta_error_ratio",
        "theta_cosine",
        "vmag_norm_ratio_iter_to_cudss",
        "vmag_error_ratio",
        "vmag_cosine",
        "top_dx_error_bus",
        "top_f_error_bus",
    ]
    lines.append(bilu[cols].to_markdown(index=False, floatfmt=".4g"))
    lines.append("\n## NR Shadow Summary\n")
    if not nr_summary.empty:
        keep = [
            "case_name",
            "converged",
            "nr_iters",
            "pure_full_cudss_calls",
            "hybrid_required_full_cudss_calls",
            "gmres_calls",
            "fallback_calls",
            "final_mismatch_inf",
        ]
        keep = [col for col in keep if col in nr_summary.columns]
        lines.append(nr_summary[keep].to_markdown(index=False, floatfmt=".4g"))
        lines.append("")
    lines.append(iter_summary.to_markdown(index=False, floatfmt=".4g"))
    lines.append("\n## Key Observations\n")
    worst = bilu.iloc[0]
    lines.append(
        f"- Worst J1 case is `{worst['case']}`: dx error ratio is "
        f"{worst['dx_error_ratio']:.3g}, theta norm ratio is "
        f"{worst['theta_norm_ratio_iter_to_cudss']:.3g}, and theta cosine is "
        f"{worst['theta_cosine']:.3g}."
    )
    low_theta = bilu.sort_values("theta_norm_ratio_iter_to_cudss").iloc[0]
    lines.append(
        f"- Smallest theta scale appears in `{low_theta['case']}`: iterative theta norm is "
        f"{low_theta['theta_norm_ratio_iter_to_cudss']:.3g} of cuDSS."
    )
    if not iter_summary.empty:
        weak = iter_summary.sort_values("avg_iterative_nonlinear_ratio_inf", ascending=False).iloc[0]
        lines.append(
            f"- Weakest nonlinear middle-step reduction is `{weak['case']}`: average iterative "
            f"mismatch ratio is {weak['avg_iterative_nonlinear_ratio_inf']:.3g}, while the "
            f"shadow cuDSS ratio is {weak['avg_cudss_nonlinear_ratio_inf']:.3g}."
        )
    lines.append(
        "- `J(dx_iter-dx_cuDSS)` is the linear F-row difference; large P/Q spikes identify "
        "where the missed correction reappears in the mismatch equations."
    )

    (out_dir / "dx_f_difference_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=Path("results/dx_f_difference_visual/gmres_j1_bs16"))
    parser.add_argument("--shadow-csv", type=Path,
                        default=Path("results/field_gain_j11/A_baseline_shadow.csv"))
    parser.add_argument("--nr-summary-csv", type=Path,
                        default=Path("results/field_gain_j11/A_baseline_summary.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dx_f_difference_visual"))
    parser.add_argument("--case", default="case13659pegase")
    parser.add_argument("--solver", default="gmres_block_ilu0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bus_path = args.input_dir / "dx_diff_bus_errors.csv"
    eq_path = args.input_dir / "dx_diff_equation_errors.csv"
    bus_df = pd.read_csv(bus_path)
    eq_df = pd.read_csv(eq_path)
    bus_df = finite_frame(
        bus_df,
        [
            "bus_id",
            "theta_abs_error",
            "vmag_abs_error",
            "combined_dx_error_norm",
            "dx_cudss_theta",
            "dx_iter_theta",
            "dx_cudss_vmag",
            "dx_iter_vmag",
        ],
    )
    eq_df = finite_frame(eq_df, ["bus_id", "P_error_abs", "Q_error_abs", "combined_PQ_error_norm"])

    summary = summarize_j1(bus_df, eq_df)
    summary.to_csv(args.output_dir / "j1_dx_f_difference_case_summary.csv", index=False)

    plot_bus_error_map(bus_df, eq_df, args.case, args.solver, args.output_dir)
    plot_top_dx_pairs(bus_df, args.case, args.solver, args.output_dir)

    shadow_df = pd.read_csv(args.shadow_csv)
    iter_summary = plot_iteration_shadow(shadow_df, args.output_dir)
    iter_summary.to_csv(args.output_dir / "iteration_dx_f_gap_summary.csv", index=False)

    write_markdown(args.output_dir,
                   args.input_dir,
                   args.shadow_csv,
                   args.case,
                   summary,
                   iter_summary,
                   args.nr_summary_csv)

    print(f"[done] wrote figures and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
