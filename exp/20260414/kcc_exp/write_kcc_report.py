#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(os.environ.get("KCC_EXP_ROOT", "/workspace/exp/20260414/kcc_exp"))
TABLES = ROOT / "tables"
OUT = ROOT / os.environ.get("KCC_EXP_REPORT_FILENAME", "KCC_EXTRACTED_RESULTS.md")

DEFAULT_CASES = [
    "case30_ieee",
    "case118_ieee",
    "case793_goc",
    "case1354_pegase",
    "case2746wop_k",
    "case4601_goc",
    "case8387_pegase",
    "case9241_pegase",
]
CASES = [
    case.strip()
    for case in os.environ.get("KCC_EXP_CASES", ",".join(DEFAULT_CASES)).split(",")
    if case.strip()
]
SNAPSHOT_CASES = [
    case.strip()
    for case in os.environ.get("KCC_EXP_SNAPSHOT_CASES", ",".join(CASES[-3:])).split(",")
    if case.strip()
]
REPORT_TITLE = os.environ.get("KCC_EXP_REPORT_TITLE", "KCC Extracted Results")
DATASET_NOTE = os.environ.get("KCC_EXP_DATASET_NOTE", "Dataset: `/workspace/datasets/cuPF_benchmark_dumps`")
CUDA_NOTE = os.environ.get(
    "KCC_EXP_CUDA_NOTE",
    "CUDA runs: `CUDA_VISIBLE_DEVICES=1`, cuDSS MT enabled, host threads `AUTO`",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_ms(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def fmt_x(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}x"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.{digits}f}%"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def main() -> None:
    pypower = read_csv(TABLES / "pypower_operator_pie.csv")
    end2end = read_csv(TABLES / "end2end_main_chain.csv")
    ablation = read_csv(TABLES / "cuda_edge_ablation_operator_breakdown.csv")
    jacobian = read_csv(TABLES / "jacobian_edge_vertex.csv")
    validation = read_csv(TABLES / "validation_summary.csv")

    end_by_case_profile = {(row["case_name"], row["profile"]): row for row in end2end}
    ab_by_case_profile_metric = {
        (row["case_name"], row["profile"], row["section"], row["metric"]): row for row in ablation
    }
    jac_by_case_profile = {(row["case_name"], row["profile"]): row for row in jacobian}

    lines: list[str] = []
    lines.append(f"# {REPORT_TITLE}")
    lines.append("")
    lines.append(f"- Result root: `{ROOT / 'results'}`")
    lines.append(f"- {DATASET_NOTE}")
    lines.append("- Cases: `" + "`, `".join(CASES) + "`")
    lines.append("- Measurement: warmup 1, repeats 10")
    lines.append(f"- {CUDA_NOTE}")
    lines.append("- ND level: CLI option was not passed for the regenerated GPU runs")
    lines.append("")

    lines.append("## PYPOWER Newtonpf Operator Profile")
    rows: list[list[str]] = []
    for case in CASES:
        case_rows = {
            row["operator_short"]: row
            for row in pypower
            if row["case_name"] == case and row["scope"] == "newtonpf"
        }
        rows.append([
            case,
            fmt_ms(f(case_rows["mismatch"], "ms_mean")),
            fmt_pct(f(case_rows["mismatch"], "share")),
            fmt_ms(f(case_rows["jacobian"], "ms_mean")),
            fmt_pct(f(case_rows["jacobian"], "share")),
            fmt_ms(f(case_rows["solve"], "ms_mean")),
            fmt_pct(f(case_rows["solve"], "share")),
            fmt_ms(f(case_rows["update_voltage"], "ms_mean")),
            fmt_pct(f(case_rows["update_voltage"], "share")),
        ])
    lines.extend(md_table(
        [
            "case",
            "mismatch ms",
            "mismatch %",
            "jacobian ms",
            "jacobian %",
            "solve ms",
            "solve %",
            "update ms",
            "update %",
        ],
        rows,
    ))
    lines.append("")
    lines.append("Note: this table excludes `init_index`; the full runpf/newtonpf breakdown is in `tables/pypower_operator_pie.csv`.")
    lines.append("")

    lines.append("## End-To-End Chain")
    rows = []
    for case in CASES:
        py = end_by_case_profile[(case, "pypower")]
        naive = end_by_case_profile[(case, "cpp_naive")]
        cpp = end_by_case_profile[(case, "cpp")]
        cuda = end_by_case_profile[(case, "cuda_edge")]
        rows.append([
            case,
            fmt_ms(f(py, "elapsed_ms_mean")),
            fmt_ms(f(naive, "elapsed_ms_mean")),
            fmt_ms(f(cpp, "elapsed_ms_mean")),
            fmt_ms(f(cuda, "elapsed_ms_mean")),
            fmt_x(f(cuda, "speedup_vs_pypower")),
            fmt_x(f(cuda, "speedup_vs_cpp")),
        ])
    lines.extend(md_table(
        [
            "case",
            "pypower ms",
            "cpp naive ms",
            "cpp optimized ms",
            "cuda edge ms",
            "cuda vs pypower",
            "cuda vs cpp opt",
        ],
        rows,
    ))
    lines.append("")

    lines.append("## CUDA Edge Ablation")
    rows = []
    profiles = [
        ("cuda_edge", "full"),
        ("cuda_wo_cudss", "w/o cuDSS"),
        ("cuda_wo_jacobian", "w/o Jacobian"),
        ("cuda_fp64_edge", "w/o mixed precision"),
    ]
    for case in CASES:
        full_elapsed = f(ab_by_case_profile_metric[(case, "cuda_edge", "top_level", "elapsed")], "ms_mean")
        full_solve = f(ab_by_case_profile_metric[(case, "cuda_edge", "top_level", "solve")], "ms_mean")
        for profile, label in profiles:
            elapsed = f(ab_by_case_profile_metric[(case, profile, "top_level", "elapsed")], "ms_mean")
            solve = f(ab_by_case_profile_metric[(case, profile, "top_level", "solve")], "ms_mean")
            rows.append([
                case,
                label,
                fmt_ms(elapsed),
                fmt_ms(solve),
                fmt_x((elapsed / full_elapsed) if elapsed is not None and full_elapsed else None),
                fmt_x((solve / full_solve) if solve is not None and full_solve else None),
            ])
    lines.extend(md_table(
        ["case", "profile", "elapsed mean ms", "solve mean ms", "elapsed / full", "solve / full"],
        rows,
    ))
    lines.append("")

    lines.append("## CUDA Ablation Operator Snapshot")
    rows = []
    metrics = [
        ("nr", "jacobian", "jacobian ms"),
        ("nr", "linear_solve", "linear solve ms"),
        ("cudss", "factorization", "factor ms"),
        ("cudss", "refactorization", "refactor ms"),
        ("cudss", "solve", "cudss solve ms"),
        ("ablation_bridge", "cpu_naive_jacobian", "cpu naive J ms"),
        ("cpu_reference", "cpu_superlu", "cpu SuperLU ms"),
    ]
    for case in SNAPSHOT_CASES:
        for profile, label in profiles:
            row = [case, label]
            for section, metric, _ in metrics:
                row.append(fmt_ms(f(ab_by_case_profile_metric[(case, profile, section, metric)], "ms_mean")))
            rows.append(row)
    lines.extend(md_table(["case", "profile"] + [label for _, _, label in metrics], rows))
    lines.append("")
    lines.append("Full operator trace for all cases/profiles is in `tables/cuda_edge_ablation_operator_breakdown.csv`.")
    lines.append("")

    lines.append("## Edge Vs Vertex Jacobian Update")
    rows = []
    for case in CASES:
        edge = jac_by_case_profile[(case, "cuda_edge")]
        vertex = jac_by_case_profile[(case, "cuda_vertex")]
        rows.append([
            case,
            fmt_ms(f(edge, "jacobian_update_ms_mean")),
            fmt_ms(f(vertex, "jacobian_update_ms_mean")),
            fmt_x(f(edge, "edge_vs_vertex_jacobian_speedup")),
            fmt_ms(f(edge, "jacobian_update_per_call_ms")),
            fmt_ms(f(vertex, "jacobian_update_per_call_ms")),
        ])
    lines.extend(md_table(
        [
            "case",
            "edge J ms",
            "vertex J ms",
            "edge speedup",
            "edge per call ms",
            "vertex per call ms",
        ],
        rows,
    ))
    lines.append("")

    lines.append("## Validation")
    rows = [
        [
            row["run_name"],
            row["profile"],
            row["success_all"],
            row["rows"],
            row["final_mismatch_max"],
        ]
        for row in validation
    ]
    lines.extend(md_table(["run", "profile", "success", "rows", "final mismatch max"], rows))
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
