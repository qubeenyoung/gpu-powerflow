"""Aggregate benchmark results written as ``benchmark/results/<run>/<variant>``.

The output is a long-form ``summary.csv`` plus a compact ``summary.md`` with
per-variant headline metrics and per-case pivots.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


def _to_float(value: object):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_runs(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.fmean(values), statistics.pstdev(values))


def _field(row: dict, new: str, old: str = "") -> str:
    value = row.get(new, "")
    if value != "" or not old:
        return value
    return row.get(old, "")


def collect(run_dir: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    skipped: list[str] = []
    for variant_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        variant_name = variant_dir.name
        if (variant_dir / "SKIPPED.txt").exists():
            skipped.append(variant_name)
            continue
        runs_csv = variant_dir / "runs.csv"
        if not runs_csv.exists():
            continue

        by_case: dict[str, list[dict]] = {}
        for row in _read_runs(runs_csv):
            if not _is_truthy(row.get("success", "1")):
                continue
            if _is_truthy(row.get("warmup", "0")):
                continue
            by_case.setdefault(row.get("case_name", ""), []).append(row)

        for case_name, case_rows in by_case.items():
            init = [v for v in (_to_float(r.get("initialize_ms")) for r in case_rows) if v is not None]
            solve = [v for v in (_to_float(r.get("solve_ms")) for r in case_rows) if v is not None]
            total = [v for v in (_to_float(r.get("total_ms")) for r in case_rows) if v is not None]
            resid = [v for v in (_to_float(r.get("output_mismatch")) for r in case_rows) if v is not None]
            iterations = [v for v in (_to_float(_field(r, "iterations", "cupf_iterations")) for r in case_rows) if v is not None]
            last = case_rows[-1]
            variant = last.get("variant") or variant_name
            mean_init, std_init = _mean_std(init)
            mean_solve, std_solve = _mean_std(solve)
            mean_total, std_total = _mean_std(total)
            rows.append(
                {
                    "variant": variant,
                    "mode": last.get("mode", ""),
                    "case_name": case_name,
                    "n_bus": last.get("n_bus", ""),
                    "ybus_nnz": last.get("ybus_nnz", ""),
                    "mean_init_ms": mean_init,
                    "std_init_ms": std_init,
                    "mean_solve_ms": mean_solve,
                    "std_solve_ms": std_solve,
                    "mean_total_ms": mean_total,
                    "std_total_ms": std_total,
                    "mean_iterations": statistics.fmean(iterations) if iterations else math.nan,
                    "worst_residual": max(resid) if resid else math.nan,
                    "converged": _is_truthy(_field(last, "converged", "cupf_converged")),
                    "success_count": len(case_rows),
                }
            )
    return rows, skipped


def _fmt(value, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def write_summary_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "variant",
        "mode",
        "case_name",
        "n_bus",
        "ybus_nnz",
        "mean_init_ms",
        "std_init_ms",
        "mean_solve_ms",
        "std_solve_ms",
        "mean_total_ms",
        "std_total_ms",
        "mean_iterations",
        "worst_residual",
        "converged",
        "success_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pivot_md(rows: list[dict], variants: list[str], cases: list[str], value_key: str, spec: str) -> list[str]:
    index = {(r["variant"], r["case_name"]): r for r in rows}
    header = "| case | n_bus | " + " | ".join(variants) + " |"
    sep = "|" + "---|" * (len(variants) + 2)
    lines = [header, sep]
    for case in cases:
        n_bus = next((index[(v, case)]["n_bus"] for v in variants if (v, case) in index), "")
        cells = [_fmt(index[(v, case)][value_key], spec) if (v, case) in index else "-" for v in variants]
        lines.append(f"| {case} | {n_bus} | " + " | ".join(cells) + " |")
    return lines


def _geomean(values: list[float]) -> float:
    pos = [v for v in values if v and v > 0 and not math.isnan(v)]
    if not pos:
        return math.nan
    return math.exp(statistics.fmean(math.log(v) for v in pos))


def _case_size(rows: list[dict], case: str) -> int:
    for row in rows:
        if row["case_name"] == case:
            try:
                return int(row["n_bus"])
            except (TypeError, ValueError):
                return 0
    return 0


def write_summary_md(rows: list[dict], skipped: list[str], run_dir: Path, path: Path) -> None:
    variants = sorted({r["variant"] for r in rows})
    cases = sorted({r["case_name"] for r in rows}, key=lambda c: (_case_size(rows, c), c))
    out: list[str] = [f"# Benchmark summary - {run_dir.name}", ""]
    if skipped:
        out += ["Skipped variants: " + ", ".join(f"`{s}`" for s in skipped), ""]

    out += ["## Per-variant overview", ""]
    out += ["| variant | cases | converged | init_ms mean | solve_ms geomean | worst residual |", "|---|---:|---:|---:|---:|---:|"]
    for variant in variants:
        vr = [r for r in rows if r["variant"] == variant]
        conv = sum(1 for r in vr if bool(r["converged"]))
        init_mean = statistics.fmean([r["mean_init_ms"] for r in vr if not math.isnan(r["mean_init_ms"])]) if vr else math.nan
        solve_gm = _geomean([r["mean_solve_ms"] for r in vr])
        worst = max((r["worst_residual"] for r in vr if not math.isnan(r["worst_residual"])), default=math.nan)
        out.append(f"| `{variant}` | {len(vr)} | {conv}/{len(vr)} | {_fmt(init_mean)} | {_fmt(solve_gm)} | {_fmt(worst, '.2e')} |")
    out.append("")

    out += ["## solve_ms by case x variant", ""]
    out += _pivot_md(rows, variants, cases, "mean_solve_ms", ".3f")
    out += ["", "## output_mismatch by case x variant", ""]
    out += _pivot_md(rows, variants, cases, "worst_residual", ".2e")
    out += [""]
    path.write_text("\n".join(out), encoding="utf-8")


def aggregate(run_dir: Path) -> tuple[Path, Path]:
    rows, skipped = collect(run_dir)
    summary_csv = run_dir / "summary.csv"
    summary_md = run_dir / "summary.md"
    write_summary_csv(rows, summary_csv)
    write_summary_md(rows, skipped, run_dir, summary_md)
    print(f"[aggregate] {len(rows)} rows -> {summary_csv}, {summary_md}", flush=True)
    return summary_csv, summary_md


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark results into summary.csv / summary.md.")
    parser.add_argument("run_dir", type=Path, help="benchmark/results/<run-name> directory")
    args = parser.parse_args(argv)
    aggregate(args.run_dir)


if __name__ == "__main__":
    main()
