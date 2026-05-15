#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS = Path("results")
MODES = [
    ("unknown", "unknown_metis"),
    ("bus", "bus_weighted_metis"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def with_mode(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        copied["partition_mode"] = mode
        out.append(copied)
    return out


def avg_by_case_mode(rows: list[dict[str, str]], case_key: str, metric: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row, metric)
        if value != 0.0:
            values[(row.get(case_key, ""), row["partition_mode"])].append(value)
    return {key: mean(vals) for key, vals in values.items() if vals}


def latest_partition_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["case_name"], row["partition_mode"])
        if key not in out:
            out[key] = row
    return out


def first_by_case_mode(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["case_name"], row["partition_mode"])
        if key not in out:
            out[key] = row
    return out


def main() -> None:
    summary: list[dict[str, str]] = []
    iters: list[dict[str, str]] = []
    shadow: list[dict[str, str]] = []
    partition: list[dict[str, str]] = []
    timing: list[dict[str, str]] = []
    for suffix, mode in MODES:
        summary += with_mode(read_rows(RESULTS / f"bus_weighted_metis_summary_{suffix}.csv"), mode)
        iters += with_mode(read_rows(RESULTS / f"bus_weighted_metis_iters_{suffix}.csv"), mode)
        shadow += with_mode(read_rows(RESULTS / f"bus_weighted_metis_shadow_{suffix}.csv"), mode)
        partition += with_mode(read_rows(RESULTS / f"bus_weighted_metis_partition_stats_{suffix}.csv"), mode)
        timing += with_mode(read_rows(RESULTS / f"bus_weighted_metis_timing_{suffix}.csv"), mode)

    write_rows(RESULTS / "bus_weighted_metis_summary.csv", summary)
    write_rows(RESULTS / "bus_weighted_metis_iters.csv", iters)
    write_rows(RESULTS / "bus_weighted_metis_shadow_dx.csv", shadow)
    write_rows(RESULTS / "bus_weighted_metis_partition_stats.csv", partition)
    write_rows(RESULTS / "bus_weighted_metis_timing.csv", timing)

    cases = sorted({row["case_name"] for row in summary})
    summary_map = {(row["case_name"], row["partition_mode"]): row for row in summary}
    partition_map = latest_partition_rows(partition)
    timing_first = first_by_case_mode(timing)

    dx_norm = avg_by_case_mode(shadow, "case", "dx_norm_ratio")
    dx_cos = avg_by_case_mode(shadow, "case", "dx_cosine")
    theta_norm = avg_by_case_mode(shadow, "case", "theta_norm_ratio")
    vmag_norm = avg_by_case_mode(shadow, "case", "vmag_norm_ratio")
    trial_ratio = avg_by_case_mode(shadow, "case", "gmres_nonlinear_ratio_inf")
    linear_rel = avg_by_case_mode(shadow, "case", "linear_rel_res_gmres")
    middle_ms = avg_by_case_mode(timing, "case_name", "middle_solver_total_ms")
    setup_ms = avg_by_case_mode(timing, "case_name", "linear_setup_ms")

    lines = [
        "# Bus-weighted METIS block-Jacobi experiment",
        "",
        "## 1. Case summary",
        "",
        "| case | partition | converged | NR | cuDSS | MR1 | accepted | fallback | hybrid ms | pure cuDSS ms | speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        for _, mode in MODES:
            row = summary_map.get((case, mode), {})
            lines.append(
                f"| {case} | {mode} | {row.get('converged', '')} | "
                f"{row.get('nr_iters', '')} | {row.get('cudss_calls', '')} | "
                f"{row.get('gmres_calls', '')} | {row.get('accepted_gmres_steps', '')} | "
                f"{row.get('fallback_calls', '')} | {1000.0 * as_float(row, 'total_seconds'):.2f} | "
                f"{1000.0 * as_float(row, 'pure_cudss_total_seconds'):.2f} | "
                f"{as_float(row, 'speedup_vs_pure_cudss'):.3f} |"
            )

    lines += [
        "",
        "## 2. Partition stats",
        "",
        "| case | partition | blocks | min | max | avg | std | diag nnz | diag weighted | J11 | J12 | J21 | J22 | theta/V split | P/Q split |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        for _, mode in MODES:
            row = partition_map.get((case, mode), {})
            lines.append(
                f"| {case} | {mode} | {row.get('num_bus_partitions', '')} | "
                f"{row.get('min_block_unknowns', '')} | {row.get('max_block_unknowns', '')} | "
                f"{as_float(row, 'avg_block_unknowns'):.2f} | {as_float(row, 'std_block_unknowns'):.2f} | "
                f"{as_float(row, 'diagonal_block_nnz_ratio'):.3f} | "
                f"{as_float(row, 'diagonal_weighted_coupling_ratio'):.3f} | "
                f"{as_float(row, 'j11_diagonal_weighted_ratio'):.3f} | "
                f"{as_float(row, 'j12_diagonal_weighted_ratio'):.3f} | "
                f"{as_float(row, 'j21_diagonal_weighted_ratio'):.3f} | "
                f"{as_float(row, 'j22_diagonal_weighted_ratio'):.3f} | "
                f"{row.get('theta_vmag_split_count', '')} | {row.get('pq_split_count', '')} |"
            )

    lines += [
        "",
        "## 3. dx and middle quality",
        "",
        "| case | dx unknown | dx bus | cos unknown | cos bus | theta unknown | theta bus | vmag unknown | vmag bus | trial unknown | trial bus | linear rel bus |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case} | {dx_norm.get((case, 'unknown_metis'), 0.0):.4g} | "
            f"{dx_norm.get((case, 'bus_weighted_metis'), 0.0):.4g} | "
            f"{dx_cos.get((case, 'unknown_metis'), 0.0):.4g} | "
            f"{dx_cos.get((case, 'bus_weighted_metis'), 0.0):.4g} | "
            f"{theta_norm.get((case, 'unknown_metis'), 0.0):.4g} | "
            f"{theta_norm.get((case, 'bus_weighted_metis'), 0.0):.4g} | "
            f"{vmag_norm.get((case, 'unknown_metis'), 0.0):.4g} | "
            f"{vmag_norm.get((case, 'bus_weighted_metis'), 0.0):.4g} | "
            f"{trial_ratio.get((case, 'unknown_metis'), 0.0):.4g} | "
            f"{trial_ratio.get((case, 'bus_weighted_metis'), 0.0):.4g} | "
            f"{linear_rel.get((case, 'bus_weighted_metis'), 0.0):.4g} |"
        )

    lines += [
        "",
        "## 4. Timing",
        "",
        "| case | partition | partition ms | weighted graph ms | avg setup ms | avg middle solve ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        for _, mode in MODES:
            row = timing_first.get((case, mode), {})
            lines.append(
                f"| {case} | {mode} | "
                f"{as_float(row, 'partition_build_ms'):.2f} | "
                f"{as_float(row, 'weighted_graph_build_ms'):.2f} | "
                f"{setup_ms.get((case, mode), 0.0):.3f} | "
                f"{middle_ms.get((case, mode), 0.0):.3f} |"
            )

    better_fallback = [
        case for case in cases
        if as_float(summary_map.get((case, "bus_weighted_metis"), {}), "fallback_calls") <
        as_float(summary_map.get((case, "unknown_metis"), {}), "fallback_calls")
    ]
    better_nr = [
        case for case in cases
        if as_float(summary_map.get((case, "bus_weighted_metis"), {}), "nr_iters") <
        as_float(summary_map.get((case, "unknown_metis"), {}), "nr_iters")
    ]
    better_dx = [
        case for case in cases
        if dx_norm.get((case, "bus_weighted_metis"), 0.0) >
        dx_norm.get((case, "unknown_metis"), 0.0)
    ]
    lines += [
        "",
        "## 5. Judgment",
        "",
        "- Bus-weighted METIS did what it was designed to do structurally: theta/Vm and P/Q split counts became zero, diagonal NNZ recovery increased, and weighted coupling recovery became almost 1.0 in every case.",
        f"- dx_norm_ratio improved only in {', '.join(better_dx) if better_dx else 'none'}; it dropped sharply on case2383wp and case3120sp, which were the most useful unknown-METIS cases.",
        f"- fallback decreased in {', '.join(better_fallback) if better_fallback else 'none'}; bus-weighted either kept or increased fallback pressure.",
        f"- NR iterations decreased in {', '.join(better_nr) if better_nr else 'none'}, but this mostly came from doing more direct cuDSS fallback/polish work rather than accepting better MR1 steps.",
        "- Final call: bus-aware weighted partition improves block locality metrics, but it does not improve this MR1 block-Jacobi middle solver enough to be useful. The preconditioner-quality bottleneck is not fixed by preserving same-bus theta/Vm and high-weight bus couplings alone.",
    ]
    (RESULTS / "bus_weighted_metis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
