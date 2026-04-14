#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(os.environ.get("KCC_EXP_ROOT", "/workspace/exp/20260414/kcc_exp"))
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"

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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def mean(values: Iterable[float | None]) -> float | str:
    xs = [value for value in values if value is not None]
    return statistics.mean(xs) if xs else ""


def stdev(values: Iterable[float | None]) -> float | str:
    xs = [value for value in values if value is not None]
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else "")


def sec_to_ms(value: float | str | None) -> float | str:
    return value * 1000.0 if isinstance(value, float) else ""


def speedup(base: float | None, candidate: float | None) -> float | str:
    if base is None or candidate is None or candidate == 0.0:
        return ""
    return base / candidate


def group_by(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    return grouped


def metric_stats(rows: list[dict[str, str]], key: str) -> dict[str, float | str]:
    values = [f(row.get(key)) for row in rows]
    m = mean(values)
    s = stdev(values)
    return {
        "sec_mean": m,
        "sec_stdev": s,
        "ms_mean": sec_to_ms(m),
        "ms_stdev": sec_to_ms(s),
    }


def summarize_pypower_operator_pie() -> list[dict[str, Any]]:
    rows = read_rows(RESULTS / "pypower_operator_profile" / "summary_operators.csv")
    grouped = group_by(rows, ("case_name",))
    scopes = {
        "runpf": [
            "runpf.load_case_data",
            "runpf.bus_indexing",
            "runpf.build_Ybus",
            "runpf.build_Sbus",
            "runpf.newtonpf",
        ],
        "newtonpf": [
            "newtonpf.init_index",
            "newtonpf.mismatch",
            "newtonpf.jacobian",
            "newtonpf.solve",
            "newtonpf.update_voltage",
        ],
    }

    out: list[dict[str, Any]] = []
    for case_name in CASES:
        case_rows = grouped.get((case_name,), [])
        for scope, metrics in scopes.items():
            means = {metric: mean(f(row.get(f"{metric}.total_sec")) for row in case_rows) for metric in metrics}
            scope_total = sum(value for value in means.values() if isinstance(value, float))
            for metric in metrics:
                stats = metric_stats(case_rows, f"{metric}.total_sec")
                value = means[metric]
                out.append({
                    "case_name": case_name,
                    "scope": scope,
                    "operator": metric,
                    "operator_short": metric.split(".", 1)[1],
                    "sec_mean": stats["sec_mean"],
                    "sec_stdev": stats["sec_stdev"],
                    "ms_mean": stats["ms_mean"],
                    "ms_stdev": stats["ms_stdev"],
                    "share": (value / scope_total) if isinstance(value, float) and scope_total else "",
                    "runs": len(case_rows),
                    "success_all": all(row.get("success") == "True" for row in case_rows),
                })
    return out


def summarize_end2end_chain() -> list[dict[str, Any]]:
    rows = read_rows(RESULTS / "end2end_main_chain" / "aggregates_end2end.csv")
    by_case_profile = {(row["case_name"], row["profile"]): row for row in rows}
    profile_order = ["pypower", "cpp_naive", "cpp", "cuda_edge"]

    out: list[dict[str, Any]] = []
    for case_name in CASES:
        pypower_elapsed = f(by_case_profile.get((case_name, "pypower"), {}).get("elapsed_sec_mean"))
        cpp_elapsed = f(by_case_profile.get((case_name, "cpp"), {}).get("elapsed_sec_mean"))
        for profile in profile_order:
            row = by_case_profile.get((case_name, profile), {})
            elapsed = f(row.get("elapsed_sec_mean"))
            solve = f(row.get("solve_sec_mean"))
            analyze = f(row.get("analyze_sec_mean"))
            out.append({
                "case_name": case_name,
                "profile": profile,
                "implementation": row.get("implementation", ""),
                "backend": row.get("backend", ""),
                "compute": row.get("compute", ""),
                "jacobian": row.get("jacobian", ""),
                "elapsed_ms_mean": sec_to_ms(elapsed),
                "elapsed_ms_stdev": sec_to_ms(f(row.get("elapsed_sec_stdev"))),
                "analyze_ms_mean": sec_to_ms(analyze),
                "solve_ms_mean": sec_to_ms(solve),
                "speedup_vs_pypower": speedup(pypower_elapsed, elapsed),
                "speedup_vs_cpp": speedup(cpp_elapsed, elapsed),
                "iterations_mean": row.get("iterations_mean", ""),
                "final_mismatch_max": row.get("final_mismatch_max", ""),
                "runs": row.get("runs", ""),
                "success_all": row.get("success_all", ""),
            })
    return out


def available_metric(row: dict[str, str], names: list[str], suffix: str) -> str:
    for name in names:
        key = f"{name}.{suffix}"
        if row.get(key) not in (None, ""):
            return key
    return f"{names[0]}.{suffix}"


def summarize_cuda_ablation() -> list[dict[str, Any]]:
    rows = read_rows(RESULTS / "cuda_edge_ablation_operators" / "summary_operators.csv")
    grouped = group_by(rows, ("case_name", "profile"))
    profiles = ["cuda_edge", "cuda_wo_cudss", "cuda_wo_jacobian", "cuda_fp64_edge"]
    metrics: list[tuple[str, str, list[str]]] = [
        ("top_level", "elapsed", ["elapsed_sec"]),
        ("top_level", "analyze", ["analyze_sec"]),
        ("top_level", "solve", ["solve_sec"]),
        ("nr", "upload", ["NR.solve.upload"]),
        ("nr", "mismatch", ["NR.iteration.mismatch"]),
        ("nr", "jacobian", ["NR.iteration.jacobian"]),
        ("nr", "linear_solve", ["NR.iteration.linear_solve"]),
        ("nr", "voltage_update", ["NR.iteration.voltage_update"]),
        ("nr", "download", ["NR.solve.download"]),
        ("cudss", "rhs_prepare", ["CUDA.solve.rhsPrepare", "CUDA.solve.rhsPrepare64"]),
        ("cudss", "factorization", ["CUDA.solve.factorization32", "CUDA.solve.factorization64"]),
        ("cudss", "refactorization", ["CUDA.solve.refactorization32", "CUDA.solve.refactorization64"]),
        ("cudss", "solve", ["CUDA.solve.solve32", "CUDA.solve.solve64"]),
        ("ablation_bridge", "cuda_to_cpu_voltage", ["ABLATION.cuda_to_cpu_voltage"]),
        ("ablation_bridge", "cpu_naive_jacobian", ["ABLATION.cpu_naive_jacobian"]),
        ("ablation_bridge", "cpu_jacobian_to_cuda_csr", ["ABLATION.cpu_jacobian_to_cuda_csr"]),
        ("ablation_bridge", "cuda_F_to_cpu", ["ABLATION.cuda_F_to_cpu"]),
        ("ablation_bridge", "cuda_J_to_cpu_csc", ["ABLATION.cuda_J_to_cpu_csc"]),
        ("ablation_bridge", "cpu_dx_to_cuda", ["ABLATION.cpu_dx_to_cuda"]),
        ("cpu_reference", "cpu_naive_jacobian_dS_dVm", ["CPU.naive.jacobian.dS_dVm"]),
        ("cpu_reference", "cpu_naive_jacobian_dS_dVa", ["CPU.naive.jacobian.dS_dVa"]),
        ("cpu_reference", "cpu_naive_jacobian_assemble", ["CPU.naive.jacobian.assemble"]),
        ("cpu_reference", "cpu_superlu", ["CPU.naive.solve.superlu"]),
    ]

    full_elapsed = {
        case_name: mean(f(row.get("elapsed_sec")) for row in grouped.get((case_name, "cuda_edge"), []))
        for case_name in CASES
    }
    full_solve = {
        case_name: mean(f(row.get("solve_sec")) for row in grouped.get((case_name, "cuda_edge"), []))
        for case_name in CASES
    }

    out: list[dict[str, Any]] = []
    for case_name in CASES:
        for profile in profiles:
            case_rows = grouped.get((case_name, profile), [])
            solve_mean = mean(f(row.get("solve_sec")) for row in case_rows)
            elapsed_mean = mean(f(row.get("elapsed_sec")) for row in case_rows)
            for section, metric, names in metrics:
                if not case_rows:
                    continue
                if len(names) == 1 and names[0].endswith("_sec"):
                    key = names[0]
                else:
                    key = available_metric(case_rows[0], names, "total_sec")
                stats = metric_stats(case_rows, key)
                value = stats["sec_mean"]
                count_key = key[:-len(".total_sec")] + ".count" if key.endswith(".total_sec") else ""
                count_mean = mean(f(row.get(count_key)) for row in case_rows) if count_key else ""
                out.append({
                    "case_name": case_name,
                    "profile": profile,
                    "section": section,
                    "metric": metric,
                    "source_column": key,
                    "sec_mean": stats["sec_mean"],
                    "sec_stdev": stats["sec_stdev"],
                    "ms_mean": stats["ms_mean"],
                    "ms_stdev": stats["ms_stdev"],
                    "count_mean": count_mean,
                    "share_of_profile_solve": (value / solve_mean) if isinstance(value, float) and isinstance(solve_mean, float) and solve_mean else "",
                    "share_of_profile_elapsed": (value / elapsed_mean) if isinstance(value, float) and isinstance(elapsed_mean, float) and elapsed_mean else "",
                    "profile_elapsed_speedup_vs_cuda_edge": speedup(full_elapsed.get(case_name) if isinstance(full_elapsed.get(case_name), float) else None, elapsed_mean if isinstance(elapsed_mean, float) else None),
                    "profile_solve_speedup_vs_cuda_edge": speedup(full_solve.get(case_name) if isinstance(full_solve.get(case_name), float) else None, solve_mean if isinstance(solve_mean, float) else None),
                    "runs": len(case_rows),
                    "success_all": all(row.get("success") == "True" for row in case_rows),
                })
    return out


def summarize_jacobian_edge_vertex() -> list[dict[str, Any]]:
    rows = read_rows(RESULTS / "jacobian_edge_vs_vertex" / "summary_operators.csv")
    grouped = group_by(rows, ("case_name", "profile"))
    out: list[dict[str, Any]] = []
    for case_name in CASES:
        edge_rows = grouped.get((case_name, "cuda_edge"), [])
        vertex_rows = grouped.get((case_name, "cuda_vertex"), [])
        edge_j = mean(f(row.get("NR.iteration.jacobian.total_sec")) for row in edge_rows)
        vertex_j = mean(f(row.get("NR.iteration.jacobian.total_sec")) for row in vertex_rows)
        for profile in ("cuda_edge", "cuda_vertex"):
            case_rows = grouped.get((case_name, profile), [])
            jac_total = mean(f(row.get("NR.iteration.jacobian.total_sec")) for row in case_rows)
            jac_count = mean(f(row.get("NR.iteration.jacobian.count")) for row in case_rows)
            jac_per_call = (jac_total / jac_count) if isinstance(jac_total, float) and isinstance(jac_count, float) and jac_count else ""
            out.append({
                "case_name": case_name,
                "profile": profile,
                "jacobian_update_ms_mean": sec_to_ms(jac_total),
                "jacobian_update_ms_stdev": sec_to_ms(stdev(f(row.get("NR.iteration.jacobian.total_sec")) for row in case_rows)),
                "jacobian_update_per_call_ms": sec_to_ms(jac_per_call),
                "jacobian_count_mean": jac_count,
                "analyze_jacobian_builder_ms_mean": sec_to_ms(mean(f(row.get("NR.analyze.jacobian_builder.total_sec")) for row in case_rows)),
                "solve_ms_mean": sec_to_ms(mean(f(row.get("solve_sec")) for row in case_rows)),
                "elapsed_ms_mean": sec_to_ms(mean(f(row.get("elapsed_sec")) for row in case_rows)),
                "edge_vs_vertex_jacobian_speedup": speedup(vertex_j if isinstance(vertex_j, float) else None, edge_j if isinstance(edge_j, float) else None),
                "profile_jacobian_speedup_vs_vertex": speedup(vertex_j if isinstance(vertex_j, float) else None, jac_total if isinstance(jac_total, float) else None),
                "runs": len(case_rows),
                "success_all": all(row.get("success") == "True" for row in case_rows),
            })
    return out


def summarize_validation() -> list[dict[str, Any]]:
    runs = {
        "pypower_operator_profile": "summary_operators.csv",
        "end2end_main_chain": "summary_end2end.csv",
        "cuda_edge_ablation_operators": "summary_operators.csv",
        "jacobian_edge_vs_vertex": "summary_operators.csv",
    }
    out: list[dict[str, Any]] = []
    for run_name, filename in runs.items():
        rows = read_rows(RESULTS / run_name / filename)
        grouped = group_by(rows, ("profile",))
        for (profile,), group in sorted(grouped.items()):
            mismatches = [f(row.get("final_mismatch")) for row in group]
            out.append({
                "run_name": run_name,
                "profile": profile,
                "rows": len(group),
                "success_all": all(row.get("success") == "True" for row in group),
                "final_mismatch_max": max((value for value in mismatches if value is not None), default=""),
                "cases": ",".join(sorted({row.get("case_name", "") for row in group})),
            })
    return out


def main() -> None:
    outputs = {
        "pypower_operator_pie.csv": summarize_pypower_operator_pie(),
        "end2end_main_chain.csv": summarize_end2end_chain(),
        "cuda_edge_ablation_operator_breakdown.csv": summarize_cuda_ablation(),
        "jacobian_edge_vertex.csv": summarize_jacobian_edge_vertex(),
        "validation_summary.csv": summarize_validation(),
    }
    for filename, rows in outputs.items():
        write_rows(TABLES / filename, rows)
        print(f"[OK] wrote {TABLES / filename} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
