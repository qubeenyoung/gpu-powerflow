#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS = Path("results")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return 0.0


def mean_by_case(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(f(row, metric))
    return {case: mean(values) for case, values in grouped.items() if values}


def mean_metric(rows: list[dict[str, str]], metric: str) -> float:
    values = [f(row, metric) for row in rows]
    return mean(values) if values else 0.0


def fmt(value: float) -> str:
    return f"{value:.3g}"


def main() -> None:
    entry = read_csv(RESULTS / "metis_coupling_retention_entry.csv")
    effect = read_csv(RESULTS / "metis_coupling_retention_effect.csv")
    buspair = read_csv(RESULTS / "metis_coupling_retention_buspair.csv")
    drift = read_csv(RESULTS / "jacobian_numeric_drift.csv")

    cases = sorted({row["case"] for row in entry})

    entry_nnz = mean_by_case(entry, "offblock_nnz_ratio")
    entry_abs = mean_by_case(entry, "offblock_abs_ratio")
    entry_fro = mean_by_case(entry, "offblock_fro_ratio")
    effect_all = mean_by_case(effect, "offblock_effect_ratio")
    effect_fields = {
        name: mean_by_case(effect, f"{name}_offblock_effect_ratio")
        for name in ["J11", "J12", "J21", "J22"]
    }
    top5_effect = mean_by_case(buspair, "top5_effect_bus_edges_kept_ratio")
    top5_coupling = mean_by_case(buspair, "top5_coupling_norm_bus_edges_kept_ratio")
    split = {
        row["case"]: int(float(row["same_bus_theta_vm_split_count"]))
        for row in buspair
        if row["iteration"] == "0"
    }

    drift_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in drift:
        drift_by_case[row["case"]].append(row)

    lines: list[str] = []
    lines.append("# METIS Coupling Retention and Jacobian Drift Diagnostic")
    lines.append("")
    lines.append("## 1. Does current METIS cut strong coupling?")
    lines.append("")
    lines.append(
        f"- Mean off-block NNZ ratio is `{fmt(mean_metric(entry, 'offblock_nnz_ratio'))}`, "
        f"but mean off-block abs ratio is only `{fmt(mean_metric(entry, 'offblock_abs_ratio'))}` "
        f"and mean off-block Frobenius ratio is `{fmt(mean_metric(entry, 'offblock_fro_ratio'))}`."
    )
    lines.append(
        "- This means many entries are outside the diagonal blocks, but most large-magnitude coupling remains inside."
    )
    lines.append("")
    lines.append("| case | offblock nnz | offblock abs | offblock fro |")
    lines.append("|---|---:|---:|---:|")
    for case in cases:
        lines.append(
            f"| {case} | {fmt(entry_nnz[case])} | {fmt(entry_abs[case])} | {fmt(entry_fro[case])} |"
        )

    lines.append("")
    lines.append("## 2. Where is the cut coupling concentrated?")
    lines.append("")
    lines.append(
        "- By cuDSS-dx effect, the largest off-block fractions are in the cross terms `J12` and `J21`."
    )
    lines.append(
        f"- Mean field effect ratios: J11 `{fmt(mean_metric(effect, 'J11_offblock_effect_ratio'))}`, "
        f"J12 `{fmt(mean_metric(effect, 'J12_offblock_effect_ratio'))}`, "
        f"J21 `{fmt(mean_metric(effect, 'J21_offblock_effect_ratio'))}`, "
        f"J22 `{fmt(mean_metric(effect, 'J22_offblock_effect_ratio'))}`."
    )
    lines.append("")
    lines.append("| case | J11 effect | J12 effect | J21 effect | J22 effect |")
    lines.append("|---|---:|---:|---:|---:|")
    for case in cases:
        lines.append(
            f"| {case} | {fmt(effect_fields['J11'][case])} | {fmt(effect_fields['J12'][case])} | "
            f"{fmt(effect_fields['J21'][case])} | {fmt(effect_fields['J22'][case])} |"
        )

    lines.append("")
    lines.append("## 3. Is important cuDSS-dx coupling off-block?")
    lines.append("")
    lines.append(
        f"- Overall mean offblock_effect_ratio is `{fmt(mean_metric(effect, 'offblock_effect_ratio'))}`."
    )
    lines.append(
        "- It is not large enough to say that unknown-level METIS is discarding most of the correction-driving coupling."
    )
    lines.append("")
    lines.append("| case | offblock effect | top 5% effect kept | top 5% coupling kept | theta/V split |")
    lines.append("|---|---:|---:|---:|---:|")
    for case in cases:
        lines.append(
            f"| {case} | {fmt(effect_all[case])} | {fmt(top5_effect[case])} | "
            f"{fmt(top5_coupling[case])} | {split.get(case, 0)} |"
        )

    lines.append("")
    lines.append("## 4. Are top bus-pair couplings preserved?")
    lines.append("")
    lines.append(
        f"- Mean top-5% coupling kept ratio is `{fmt(mean_metric(buspair, 'top5_coupling_norm_bus_edges_kept_ratio'))}`."
    )
    lines.append(
        f"- Mean top-5% effect kept ratio is `{fmt(mean_metric(buspair, 'top5_effect_bus_edges_kept_ratio'))}`."
    )
    lines.append(
        "- The lowest top-5% effect kept case is `case3120sp`, but even there the mean is about `0.906`."
    )

    lines.append("")
    lines.append("## 5. How much does J change from J0 to J2?")
    lines.append("")
    lines.append("| case | rel J0->J1 | rel J1->J2 | offblock J0->J1 | offblock J1->J2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for case in cases:
        rows = sorted(drift_by_case[case], key=lambda row: int(row["from_iter"]))
        values = {
            int(row["from_iter"]): row
            for row in rows
        }
        lines.append(
            f"| {case} | {fmt(f(values[0], 'rel_change_all'))} | {fmt(f(values[1], 'rel_change_all'))} | "
            f"{fmt(f(values[0], 'rel_change_offblock'))} | {fmt(f(values[1], 'rel_change_offblock'))} |"
        )
    lines.append("")
    lines.append(
        "- Drift is case-dependent: `case2383wp` and `case3120sp` move noticeably early, while `case6468rte` and `case9241pegase` are nearly static."
    )
    lines.append(
        "- Off-block drift is not systematically larger than in-block drift, so the static partition assumption is not the main suspect from this diagnostic alone."
    )

    lines.append("")
    lines.append("## 6. Does this justify bus-aware weighted METIS?")
    lines.append("")
    lines.append(
        "- Weakly. Same-bus theta/V splits are common, and J12/J21 off-block effect is high, so the idea was worth checking."
    )
    lines.append(
        "- But top effect bus-pair retention is already high and overall offblock_effect_ratio is modest, so the diagnostic does not strongly support the hypothesis that unknown-level METIS is cutting the most important bus couplings."
    )
    lines.append(
        "- This matches the follow-up bus-weighted experiment: structural retention improved, but MR1 dx quality and fallback behavior did not."
    )

    (RESULTS / "metis_coupling_drift_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
