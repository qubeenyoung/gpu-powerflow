#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parent
DEFAULT_RESULTS_ROOT = EXP_ROOT / "results"
DEFAULT_EXISTING_RUN_DIR = EXP_ROOT.parent / "modified" / "results" / "texas_gpu3_mt_auto_r10"
DEFAULT_ALGORITHMS = ["DEFAULT", "ALG_1", "ALG_2", "ALG_3"]

METRICS = {
    "elapsed_ms": "elapsed_sec",
    "analyze_ms": "analyze_sec",
    "solve_ms": "solve_sec",
    "nr_linear_factorize_ms": "NR.iteration.linear_factorize.avg_sec",
    "nr_linear_solve_ms": "NR.iteration.linear_solve.avg_sec",
    "cuda_factorization_ms": "CUDA.solve.factorization32.avg_sec",
    "cuda_refactorization_ms": "CUDA.solve.refactorization32.avg_sec",
    "cuda_rhs_prepare_ms": "CUDA.solve.rhsPrepare.avg_sec",
    "cuda_solve_ms": "CUDA.solve.solve32.avg_sec",
    "cudss_analysis_ms": "CUDA.analyze.cudss32.analysis.avg_sec",
    "cudss_setup_ms": "CUDA.analyze.cudss32.setup.avg_sec",
}

COUNT_METRICS = {
    "nr_linear_factorize_count": "NR.iteration.linear_factorize.count",
    "nr_linear_solve_count": "NR.iteration.linear_solve.count",
    "cuda_factorization_count": "CUDA.solve.factorization32.count",
    "cuda_refactorization_count": "CUDA.solve.refactorization32.count",
    "cuda_solve_count": "CUDA.solve.solve32.count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize standard cuda_edge refactorization timing across cuDSS reordering algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument(
        "--default-run-dir",
        type=Path,
        default=DEFAULT_EXISTING_RUN_DIR,
        help="Existing benchmark run to use for cuDSS DEFAULT reordering.",
    )
    parser.add_argument("--profile", default="cuda_edge")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def fmt_ms(value: Any) -> str:
    value = as_float(value)
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_x(value: Any) -> str:
    value = as_float(value)
    return "-" if value is None else f"{value:.2f}x"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def algorithm_from_manifest(run_dir: Path) -> str | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("cudss_reordering_alg")


def run_row_count(run_dir: Path) -> int:
    summary_path = run_dir / "summary_operators.csv"
    if not summary_path.exists():
        return -1
    with summary_path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def prefer_run(candidate: Path, current: Path | None) -> Path:
    if current is None:
        return candidate
    candidate_score = run_row_count(candidate)
    current_score = run_row_count(current)
    if candidate_score != current_score:
        return candidate if candidate_score > current_score else current
    return candidate if candidate.name > current.name else current


def collect_run_dirs(results_root: Path, algorithms: set[str], default_run_dir: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    if "DEFAULT" in algorithms and default_run_dir.exists():
        algorithm = algorithm_from_manifest(default_run_dir)
        if algorithm == "DEFAULT":
            selected["DEFAULT"] = default_run_dir
    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        algorithm = algorithm_from_manifest(run_dir)
        if algorithm in algorithms:
            selected[algorithm] = prefer_run(run_dir, selected.get(algorithm))
    missing = algorithms.difference(selected)
    if missing:
        raise FileNotFoundError(f"Missing result runs for algorithms: {', '.join(sorted(missing))}")
    return selected


def summarize_rows(algorithm: str, rows: list[dict[str, str]], profile: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("measurement_mode") != "operators" or row.get("profile") != profile:
            continue
        grouped.setdefault(row["case_name"], []).append(row)

    summary: list[dict[str, Any]] = []
    for case_name, group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "case_name": case_name,
            "cudss_reordering_alg": algorithm,
            "profile": profile,
            "runs": len(group),
            "success_all": all(row.get("success") == "True" for row in group),
            "iterations_mean": mean([value for row in group if (value := as_float(row.get("iterations"))) is not None]),
        }
        for out_key, source_key in METRICS.items():
            values = [value * 1000.0 for row in group if (value := as_float(row.get(source_key))) is not None]
            out[out_key] = mean(values)
        for out_key, source_key in COUNT_METRICS.items():
            values = [value for row in group if (value := as_float(row.get(source_key))) is not None]
            out[out_key] = mean(values)
        out["refactorization_vs_factorization_ratio"] = safe_ratio(
            out.get("cuda_refactorization_ms"),
            out.get("cuda_factorization_ms"),
        )
        out["refactorization_vs_cudss_solve_ratio"] = safe_ratio(
            out.get("cuda_refactorization_ms"),
            out.get("cuda_solve_ms"),
        )
        out["refactorization_share_of_nr_factorize_pct"] = (
            safe_ratio(out.get("cuda_refactorization_ms"), out.get("nr_linear_factorize_ms")) or 0.0
        ) * 100.0
        summary.append(out)
    return summary


def add_comparisons(rows: list[dict[str, Any]], algorithms: list[str]) -> list[dict[str, Any]]:
    by_case_alg = {(row["case_name"], row["cudss_reordering_alg"]): row for row in rows}
    if len(algorithms) < 2:
        return []
    baseline_alg = algorithms[0]
    candidate_algs = algorithms[1:]
    comparison: list[dict[str, Any]] = []
    cases = sorted({row["case_name"] for row in rows})
    metrics = (
        "elapsed_ms",
        "analyze_ms",
        "solve_ms",
        "cudss_analysis_ms",
        "cuda_factorization_ms",
        "cuda_refactorization_ms",
        "cuda_solve_ms",
        "nr_linear_factorize_ms",
        "nr_linear_solve_ms",
    )
    for candidate_alg in candidate_algs:
        for case_name in cases:
            baseline = by_case_alg.get((case_name, baseline_alg), {})
            candidate = by_case_alg.get((case_name, candidate_alg), {})
            out: dict[str, Any] = {
                "case_name": case_name,
                "baseline_alg": baseline_alg,
                "candidate_alg": candidate_alg,
                "baseline_runs": baseline.get("runs"),
                "candidate_runs": candidate.get("runs"),
                "baseline_success_all": baseline.get("success_all"),
                "candidate_success_all": candidate.get("success_all"),
            }
            for metric in metrics:
                baseline_value = as_float(baseline.get(metric))
                candidate_value = as_float(candidate.get(metric))
                out[f"baseline_{metric}"] = baseline_value
                out[f"candidate_{metric}"] = candidate_value
                out[f"candidate_vs_baseline_{metric}_ratio"] = safe_ratio(candidate_value, baseline_value)
                out[f"candidate_vs_baseline_{metric}_speedup"] = safe_ratio(baseline_value, candidate_value)
            comparison.append(out)
    return comparison


def geomean(values: list[float]) -> float | None:
    filtered = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not filtered:
        return None
    return math.exp(statistics.mean(math.log(value) for value in filtered))


def write_summary(summary_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], run_dirs: dict[str, Path], algorithms: list[str]) -> None:
    by_case_alg = {(row["case_name"], row["cudss_reordering_alg"]): row for row in summary_rows}
    cases = sorted({row["case_name"] for row in summary_rows})

    lines = [
        "# Refactorization Reorder Summary",
        "",
        "- Dataset root: `/workspace/datasets/texas_univ_cases/cuPF_datasets`",
        "- Profile: `cuda_edge` (`cuda_mixed_edge`, standard algorithm)",
        "- Measurement mode: `operators`",
        f"- Algorithms: {', '.join(f'`{alg}`' for alg in algorithms)}",
        "- cuDSS MT: enabled, host threads `AUTO`",
        "",
        "## Run Directories",
        "",
    ]
    for algorithm in algorithms:
        lines.append(f"- `{algorithm}`: `{run_dirs[algorithm]}`")
    lines.append("")
    lines.extend([
        "## Raw Operator Directories",
        "",
    ])
    for algorithm in algorithms:
        lines.append(f"- `{algorithm}`: `{run_dirs[algorithm] / 'raw' / 'operators' / 'cuda_edge'}`")
    lines.append("")

    lines.extend([
        "## Refactorization Focus",
        "",
        *md_table(
            [
                "case",
                *[f"{alg} refactor ms" for alg in algorithms],
                *[f"{alg} factor ms" for alg in algorithms],
                *[f"{alg} solve32 ms" for alg in algorithms],
            ],
            [
                [
                    case,
                    *[fmt_ms(by_case_alg.get((case, alg), {}).get("cuda_refactorization_ms")) for alg in algorithms],
                    *[fmt_ms(by_case_alg.get((case, alg), {}).get("cuda_factorization_ms")) for alg in algorithms],
                    *[fmt_ms(by_case_alg.get((case, alg), {}).get("cuda_solve_ms")) for alg in algorithms],
                ]
                for case in cases
            ],
        ),
        "",
    ])

    if comparison_rows:
        baseline_alg = algorithms[0]
        candidate_algs = algorithms[1:]
        lines.extend([
            f"## Reorder Ratios vs {baseline_alg}",
            "",
        ])
        for candidate_alg in candidate_algs:
            candidate_rows = [
                row for row in comparison_rows
                if row.get("candidate_alg") == candidate_alg
            ]
            lines.extend([
                f"### {candidate_alg} / {baseline_alg}",
                "",
                *md_table(
                    [
                        "case",
                        "elapsed ratio",
                        "analyze ratio",
                        "solve ratio",
                        "refactor ratio",
                        "factor ratio",
                        "cudss solve ratio",
                    ],
                    [
                        [
                            row["case_name"],
                            fmt_x(row.get("candidate_vs_baseline_elapsed_ms_ratio")),
                            fmt_x(row.get("candidate_vs_baseline_analyze_ms_ratio")),
                            fmt_x(row.get("candidate_vs_baseline_solve_ms_ratio")),
                            fmt_x(row.get("candidate_vs_baseline_cuda_refactorization_ms_ratio")),
                            fmt_x(row.get("candidate_vs_baseline_cuda_factorization_ms_ratio")),
                            fmt_x(row.get("candidate_vs_baseline_cuda_solve_ms_ratio")),
                        ]
                        for row in candidate_rows
                    ],
                ),
                "",
            ])
            for metric in ("elapsed_ms", "analyze_ms", "solve_ms", "cuda_refactorization_ms"):
                ratios = [
                    value
                    for row in candidate_rows
                    if (value := as_float(row.get(f"candidate_vs_baseline_{metric}_ratio"))) is not None
                ]
                gm = geomean(ratios)
                lines.append(f"- Geomean `{candidate_alg}/{baseline_alg}` {metric}: {fmt_x(gm)}")
            lines.append("")
        lines.append("")

    lines.extend([
        "## Files",
        "",
        "- `operator_refactor_comparison.csv`: per-case operator timing means for each algorithm",
        "- `reorder_alg_comparison.csv`: pairwise candidate vs DEFAULT ratios",
        "- `combined_summary_operators.csv`: raw operator rows with algorithm labels",
    ])
    (EXP_ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    algorithms = list(args.algorithms)
    run_dirs = collect_run_dirs(args.results_root, set(algorithms), args.default_run_dir)

    combined_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for algorithm in algorithms:
        rows = read_csv(run_dirs[algorithm] / "summary_operators.csv")
        for row in rows:
            tagged = dict(row)
            tagged["cudss_reordering_alg"] = algorithm
            combined_rows.append(tagged)
        summary_rows.extend(summarize_rows(algorithm, rows, args.profile))

    comparison_rows = add_comparisons(summary_rows, algorithms)
    write_csv(EXP_ROOT / "combined_summary_operators.csv", combined_rows)
    write_csv(EXP_ROOT / "operator_refactor_comparison.csv", summary_rows)
    write_csv(EXP_ROOT / "reorder_alg_comparison.csv", comparison_rows)
    write_summary(summary_rows, comparison_rows, run_dirs, algorithms)
    print(f"[OK] wrote {EXP_ROOT / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
