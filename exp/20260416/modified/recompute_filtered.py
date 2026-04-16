#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_RESULT_ROOT = (
    Path(__file__).resolve().parent / "results" / "texas_gpu3_mt_auto_r10"
)

TOP_LEVEL_METRICS = ("elapsed_sec", "analyze_sec", "solve_sec")
OPERATOR_METRICS = (
    "NR.iteration.mismatch.total_sec",
    "NR.iteration.jacobian.total_sec",
    "NR.iteration.linear.total_sec",
    "NR.iteration.linear_factorize.total_sec",
    "NR.iteration.linear_solve.total_sec",
    "NR.iteration.voltage_update.total_sec",
    "CUDA.solve.factorization32.total_sec",
    "CUDA.solve.refactorization32.total_sec",
    "CUDA.solve.rhsPrepare.total_sec",
    "CUDA.solve.solve32.total_sec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute modified benchmark summaries after high-spike repeat filtering."
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=1.20,
        help="A value must exceed median by this ratio before it can be filtered.",
    )
    parser.add_argument(
        "--iqr-scale",
        type=float,
        default=1.5,
        help="High Tukey fence scale: Q3 + scale * IQR.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def high_fence(values: list[float], *, iqr_scale: float, min_ratio: float) -> tuple[float, float]:
    median = statistics.median(values)
    if len(values) < 4:
        return median * min_ratio, median

    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    q1, q3 = quartiles[0], quartiles[2]
    iqr = q3 - q1
    tukey = q3 + iqr_scale * iqr if iqr > 0.0 else median * min_ratio
    return max(tukey, median * min_ratio), median


def repeat_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["measurement_mode"],
        row["case_name"],
        row["profile"],
        row["repeat_idx"],
    )


def mark_outliers(
    rows: list[dict[str, str]],
    *,
    iqr_scale: float,
    min_ratio: float,
) -> tuple[set[tuple[str, str, str, str]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(
            (row["measurement_mode"], row["case_name"], row["profile"]),
            [],
        ).append(row)

    outlier_keys: set[tuple[str, str, str, str]] = set()
    outliers: list[dict[str, Any]] = []
    for (mode, case_name, profile), group in sorted(groups.items()):
        for metric in TOP_LEVEL_METRICS:
            values = [fnum(row.get(metric)) for row in group]
            numeric = [value for value in values if value is not None]
            if len(numeric) < 5:
                continue
            fence, median = high_fence(numeric, iqr_scale=iqr_scale, min_ratio=min_ratio)
            for row in group:
                value = fnum(row.get(metric))
                if value is None or value <= fence:
                    continue
                key = repeat_key(row)
                outlier_keys.add(key)
                outliers.append({
                    "measurement_mode": mode,
                    "case_name": case_name,
                    "profile": profile,
                    "repeat_idx": row["repeat_idx"],
                    "metric": metric,
                    "value": value,
                    "median": median,
                    "high_fence": fence,
                    "value_over_median": value / median if median else "",
                })
    return outlier_keys, outliers


def add_numeric_stats(out: dict[str, Any], rows: list[dict[str, str]], key: str) -> None:
    values = [fnum(row.get(key)) for row in rows]
    numeric = [value for value in values if value is not None]
    if not numeric:
        out[f"{key}_mean"] = ""
        out[f"{key}_median"] = ""
        out[f"{key}_min"] = ""
        out[f"{key}_max"] = ""
        out[f"{key}_stdev"] = ""
        return
    out[f"{key}_mean"] = statistics.mean(numeric)
    out[f"{key}_median"] = statistics.median(numeric)
    out[f"{key}_min"] = min(numeric)
    out[f"{key}_max"] = max(numeric)
    out[f"{key}_stdev"] = statistics.stdev(numeric) if len(numeric) > 1 else 0.0


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(
            (row["measurement_mode"], row["profile"], row["case_name"]),
            [],
        ).append(row)

    aggregates: list[dict[str, Any]] = []
    for (mode, profile, case_name), group in sorted(groups.items()):
        iterations = [fnum(row.get("iterations")) for row in group]
        iterations = [value for value in iterations if value is not None]
        final_mismatches = [fnum(row.get("final_mismatch")) for row in group]
        final_mismatches = [value for value in final_mismatches if value is not None]
        aggregate: dict[str, Any] = {
            "measurement_mode": mode,
            "profile": profile,
            "case_name": case_name,
            "source_profile": group[0].get("source_profile", ""),
            "implementation": group[0].get("implementation", ""),
            "backend": group[0].get("backend", ""),
            "compute": group[0].get("compute", ""),
            "jacobian": group[0].get("jacobian", ""),
            "algorithm": group[0].get("algorithm", "standard"),
            "runs_used": len(group),
            "success_all": all(row.get("success") == "True" or row.get("success") == "true" for row in group),
            "iterations_mean": statistics.mean(iterations) if iterations else "",
            "final_mismatch_max": max(final_mismatches) if final_mismatches else "",
        }
        for metric in TOP_LEVEL_METRICS:
            add_numeric_stats(aggregate, group, metric)
        aggregates.append(aggregate)
    return aggregates


def build_profile_comparison(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["measurement_mode"], row["case_name"], row["profile"]): row
        for row in aggregates
    }
    rows: list[dict[str, Any]] = []
    for mode in ("end2end", "operators"):
        cases = sorted({row["case_name"] for row in aggregates if row["measurement_mode"] == mode})
        for case_name in cases:
            standard = by_key.get((mode, case_name, "cuda_edge"))
            modified = by_key.get((mode, case_name, "cuda_edge_modified"))
            if standard is None or modified is None:
                continue
            row: dict[str, Any] = {
                "measurement_mode": mode,
                "case_name": case_name,
                "standard_runs_used": standard["runs_used"],
                "modified_runs_used": modified["runs_used"],
                "standard_success_all": standard["success_all"],
                "modified_success_all": modified["success_all"],
                "standard_iterations_mean": standard["iterations_mean"],
                "modified_iterations_mean": modified["iterations_mean"],
                "standard_final_mismatch_max": standard["final_mismatch_max"],
                "modified_final_mismatch_max": modified["final_mismatch_max"],
            }
            for metric in TOP_LEVEL_METRICS:
                standard_value = fnum(standard.get(f"{metric}_mean"))
                modified_value = fnum(modified.get(f"{metric}_mean"))
                row[f"standard_{metric}_mean"] = standard_value if standard_value is not None else ""
                row[f"modified_{metric}_mean"] = modified_value if modified_value is not None else ""
                row[f"{metric}_speedup_standard_over_modified"] = (
                    standard_value / modified_value
                    if standard_value is not None and modified_value not in (None, 0.0)
                    else ""
                )
                row[f"{metric}_delta_modified_minus_standard"] = (
                    modified_value - standard_value
                    if standard_value is not None and modified_value is not None
                    else ""
                )
            rows.append(row)
    return rows


def mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [fnum(row.get(key)) for row in rows]
    numeric = [value for value in values if value is not None]
    return statistics.mean(numeric) if numeric else None


def build_operator_comparison(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["measurement_mode"] != "operators":
            continue
        groups.setdefault((row["case_name"], row["profile"]), []).append(row)

    out: list[dict[str, Any]] = []
    cases = sorted({case_name for case_name, _ in groups})
    for case_name in cases:
        standard = groups.get((case_name, "cuda_edge"), [])
        modified = groups.get((case_name, "cuda_edge_modified"), [])
        if not standard or not modified:
            continue
        for metric in OPERATOR_METRICS:
            standard_value = mean(standard, metric)
            modified_value = mean(modified, metric)
            if standard_value is None and modified_value is None:
                continue
            count_key = metric.removesuffix(".total_sec") + ".count"
            standard_count = mean(standard, count_key)
            modified_count = mean(modified, count_key)
            out.append({
                "case_name": case_name,
                "metric": metric,
                "standard_sec_mean": standard_value if standard_value is not None else "",
                "modified_sec_mean": modified_value if modified_value is not None else "",
                "standard_ms_mean": standard_value * 1000.0 if standard_value is not None else "",
                "modified_ms_mean": modified_value * 1000.0 if modified_value is not None else "",
                "speedup_standard_over_modified": (
                    standard_value / modified_value
                    if standard_value is not None and modified_value not in (None, 0.0)
                    else ""
                ),
                "delta_ms_modified_minus_standard": (
                    (modified_value - standard_value) * 1000.0
                    if standard_value is not None and modified_value is not None
                    else ""
                ),
                "standard_count_mean": standard_count if standard_count is not None else "",
                "modified_count_mean": modified_count if modified_count is not None else "",
                "standard_runs_used": len(standard),
                "modified_runs_used": len(modified),
            })
    return out


def write_markdown(
    path: Path,
    comparison: list[dict[str, Any]],
    operator_comparison: list[dict[str, Any]],
    outliers: list[dict[str, Any]],
) -> None:
    def fmt_ms(value: Any) -> str:
        number = fnum(value)
        return "" if number is None else f"{number * 1000.0:.3f}"

    def fmt_speed(value: Any) -> str:
        number = fnum(value)
        return "" if number is None else f"{number:.3f}x"

    lines = [
        "# Filtered Standard vs Modified Comparison",
        "",
        "Filtering rule: within each `(mode, case, profile)` group, a repeat is removed when any of `elapsed_sec`, `analyze_sec`, or `solve_sec` is above both `Q3 + 1.5 * IQR` and `1.20 * median`.",
        "",
        f"Removed metric hits: {len(outliers)}",
        f"Removed repeat rows: {len({(r['measurement_mode'], r['case_name'], r['profile'], r['repeat_idx']) for r in outliers})}",
        "",
        "## Average Speedup",
        "",
        "| mode | elapsed speedup std/mod | solve speedup std/mod |",
        "|---|---:|---:|",
    ]
    for mode in ("end2end", "operators"):
        subset = [row for row in comparison if row["measurement_mode"] == mode]
        elapsed = [
            fnum(row["elapsed_sec_speedup_standard_over_modified"])
            for row in subset
        ]
        solve = [
            fnum(row["solve_sec_speedup_standard_over_modified"])
            for row in subset
        ]
        elapsed = [value for value in elapsed if value is not None]
        solve = [value for value in solve if value is not None]
        lines.append(f"| {mode} | {statistics.mean(elapsed):.3f}x | {statistics.mean(solve):.3f}x |")

    lines += [
        "",
        "Values are standard divided by modified. Greater than 1 means modified is faster.",
        "",
        "## End2End By Case",
        "",
        "| case | std ms | mod ms | speedup | std solve ms | mod solve ms | solve speedup | std runs | mod runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [item for item in comparison if item["measurement_mode"] == "end2end"]:
        lines.append(
            f"| {row['case_name']} | "
            f"{fmt_ms(row['standard_elapsed_sec_mean'])} | "
            f"{fmt_ms(row['modified_elapsed_sec_mean'])} | "
            f"{fmt_speed(row['elapsed_sec_speedup_standard_over_modified'])} | "
            f"{fmt_ms(row['standard_solve_sec_mean'])} | "
            f"{fmt_ms(row['modified_solve_sec_mean'])} | "
            f"{fmt_speed(row['solve_sec_speedup_standard_over_modified'])} | "
            f"{row['standard_runs_used']} | {row['modified_runs_used']} |"
        )

    by_operator = {
        (row["case_name"], row["metric"]): row
        for row in operator_comparison
    }
    lines += [
        "",
        "## Operator Solve By Case",
        "",
        "| case | std solve ms | mod solve ms | solve speedup | std factorize count | mod factorize count | std solve count | mod solve count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [item for item in comparison if item["measurement_mode"] == "operators"]:
        case_name = row["case_name"]
        factorize = by_operator.get((case_name, "NR.iteration.linear_factorize.total_sec"), {})
        solve = by_operator.get((case_name, "NR.iteration.linear_solve.total_sec"), {})
        lines.append(
            f"| {case_name} | "
            f"{fmt_ms(row['standard_solve_sec_mean'])} | "
            f"{fmt_ms(row['modified_solve_sec_mean'])} | "
            f"{fmt_speed(row['solve_sec_speedup_standard_over_modified'])} | "
            f"{factorize.get('standard_count_mean', '')} | "
            f"{factorize.get('modified_count_mean', '')} | "
            f"{solve.get('standard_count_mean', '')} | "
            f"{solve.get('modified_count_mean', '')} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.result_root
    rows = read_csv(root / "summary.csv")
    outlier_keys, outliers = mark_outliers(
        rows,
        iqr_scale=args.iqr_scale,
        min_ratio=args.min_ratio,
    )
    kept_rows = [row for row in rows if repeat_key(row) not in outlier_keys]
    aggregates = aggregate_rows(kept_rows)
    comparison = build_profile_comparison(aggregates)
    operator_comparison = build_operator_comparison(kept_rows)

    write_csv(root / "filtered_outliers.csv", outliers)
    write_csv(root / "filtered_summary.csv", kept_rows)
    write_csv(root / "filtered_aggregates.csv", aggregates)
    write_csv(root / "filtered_profile_comparison.csv", comparison)
    write_csv(root / "filtered_operator_comparison.csv", operator_comparison)
    write_markdown(root / "FILTERED_COMPARISON.md", comparison, operator_comparison, outliers)

    print(f"result_root={root}")
    print(f"removed_repeat_rows={len(outlier_keys)}")
    print(f"removed_metric_hits={len(outliers)}")
    print(f"kept_rows={len(kept_rows)} / {len(rows)}")


if __name__ == "__main__":
    main()
