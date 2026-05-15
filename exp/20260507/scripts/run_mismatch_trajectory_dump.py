#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "exp" / "20260507" / "results" / "mismatch_trends"
DEFAULT_BINARY = (
    Path("/workspace/gpu-powerflow-master")
    / "cuPF"
    / "build"
    / "bench-end2end-superlu-cudss-mt-auto-dump"
    / "benchmarks"
    / "cupf_case_benchmark"
)
DEFAULT_CUDSS_THREADING_LIB = Path(
    "/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so"
)
RESIDUAL_RE = re.compile(r"residual_iter(\d+)\.txt$")


@dataclass(frozen=True)
class VectorStats:
    iteration: int
    dim: int
    norm_inf: float
    norm_l2: float
    norm_l1: float
    max_abs_index: int
    max_abs_value: float
    topk_indices: set[int]
    values: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump and summarize Newton mismatch trajectories.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("all_matpower_b1_%Y%m%d_%H%M%S"))
    parser.add_argument("--benchmark-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--cases", nargs="*", help="Optional case names. Defaults to every dump directory.")
    parser.add_argument("--profile", default="cuda_mixed_edge")
    parser.add_argument("--tolerance", default="1e-8")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cudss-threading-lib", type=Path, default=DEFAULT_CUDSS_THREADING_LIB)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def benchmark_env(cudss_threading_lib: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDSS_THREADING_LIB"] = str(cudss_threading_lib)
    preload_entries = [entry for entry in env.get("LD_PRELOAD", "").split(":") if entry]
    threading_path = str(cudss_threading_lib)
    if threading_path not in preload_entries:
        env["LD_PRELOAD"] = ":".join([threading_path, *preload_entries])
    return env


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def case_names(dataset_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(
        path.name for path in dataset_root.iterdir()
        if path.is_dir() and path.joinpath("dump_Ybus.mtx").exists()
    )


def parse_run_line(text: str) -> dict[str, str] | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("RUN "):
            continue
        parsed: dict[str, str] = {}
        for token in line[len("RUN "):].split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[key] = value
        return parsed
    return None


def size_bin(buses: int) -> str:
    if buses < 100:
        return "<100"
    if buses < 1000:
        return "100-999"
    if buses < 10000:
        return "1k-9,999"
    if buses < 50000:
        return "10k-49,999"
    return ">=50k"


def read_vector(path: Path) -> list[float]:
    values: list[float] = []
    with path.open(encoding="utf-8") as f:
        in_values = False
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "values":
                in_values = True
                continue
            if not in_values:
                continue
            _idx, value = stripped.split(maxsplit=1)
            values.append(float(value))
    return values


def topk_count(dim: int) -> int:
    return min(100, max(10, math.ceil(0.01 * dim)))


def vector_stats(iteration: int, values: list[float]) -> VectorStats:
    dim = len(values)
    if dim == 0:
        raise ValueError("empty residual vector")
    abs_values = [abs(v) for v in values]
    max_abs_index, max_abs_value = max(enumerate(abs_values), key=lambda item: item[1])
    k = topk_count(dim)
    topk = {
        idx for idx, _value in sorted(
            enumerate(abs_values), key=lambda item: item[1], reverse=True
        )[:k]
    }
    norm_l1 = math.fsum(abs_values)
    norm_l2 = math.sqrt(math.fsum(v * v for v in values))
    return VectorStats(
        iteration=iteration,
        dim=dim,
        norm_inf=max_abs_value,
        norm_l2=norm_l2,
        norm_l1=norm_l1,
        max_abs_index=max_abs_index,
        max_abs_value=values[max_abs_index],
        topk_indices=topk,
        values=values,
    )


def cosine(lhs: list[float], rhs: list[float], lhs_norm_l2: float, rhs_norm_l2: float) -> float | str:
    if lhs_norm_l2 == 0.0 or rhs_norm_l2 == 0.0:
        return ""
    dot = math.fsum(a * b for a, b in zip(lhs, rhs))
    value = dot / (lhs_norm_l2 * rhs_norm_l2)
    return max(-1.0, min(1.0, value))


def relative_delta(curr: list[float], prev: list[float], prev_norm_l2: float) -> float | str:
    if prev_norm_l2 == 0.0:
        return ""
    delta_l2 = math.sqrt(math.fsum((a - b) * (a - b) for a, b in zip(curr, prev)))
    return delta_l2 / prev_norm_l2


def parse_residuals(dump_dir: Path, case_name: str, run_meta: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: list[VectorStats] = []
    for path in sorted(dump_dir.glob("residual_iter*.txt")):
        match = RESIDUAL_RE.search(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        stats.append(vector_stats(iteration, read_vector(path)))
    stats.sort(key=lambda item: item.iteration)
    if not stats:
        raise RuntimeError(f"no residual_iter*.txt files under {dump_dir}")

    rows: list[dict[str, Any]] = []
    for idx, stat in enumerate(stats):
        prev = stats[idx - 1] if idx > 0 else None
        cos_prev: float | str = ""
        angle_deg: float | str = ""
        topk_overlap: float | str = ""
        ratio_inf: float | str = ""
        ratio_l2: float | str = ""
        rel_delta: float | str = ""
        if prev is not None:
            cos_prev = cosine(stat.values, prev.values, stat.norm_l2, prev.norm_l2)
            if cos_prev != "":
                angle_deg = math.degrees(math.acos(cos_prev))
            topk_union = stat.topk_indices | prev.topk_indices
            topk_overlap = (
                len(stat.topk_indices & prev.topk_indices) / len(topk_union)
                if topk_union else ""
            )
            ratio_inf = stat.norm_inf / prev.norm_inf if prev.norm_inf != 0.0 else ""
            ratio_l2 = stat.norm_l2 / prev.norm_l2 if prev.norm_l2 != 0.0 else ""
            rel_delta = relative_delta(stat.values, prev.values, prev.norm_l2)

        rows.append({
            "case_name": case_name,
            "size_bin": size_bin(int(run_meta["buses"])),
            "iteration": stat.iteration,
            "dimF": stat.dim,
            "topk_count": topk_count(stat.dim),
            "norm_inf": stat.norm_inf,
            "norm_l2": stat.norm_l2,
            "norm_l1": stat.norm_l1,
            "ratio_inf_prev": ratio_inf,
            "ratio_l2_prev": ratio_l2,
            "cos_prev": cos_prev,
            "angle_deg_prev": angle_deg,
            "relative_delta_l2_prev": rel_delta,
            "topk_jaccard_prev": topk_overlap,
            "max_abs_index": stat.max_abs_index,
            "max_abs_value": stat.max_abs_value,
            "buses": int(run_meta["buses"]),
            "pv": int(run_meta["pv"]),
            "pq": int(run_meta["pq"]),
            "success": run_meta["success"],
            "solver_iterations": int(run_meta["iterations"]),
            "final_mismatch": float(run_meta["final_mismatch"]),
        })

    cosine_values = [float(row["cos_prev"]) for row in rows if row["cos_prev"] != ""]
    ratio_inf_values = [float(row["ratio_inf_prev"]) for row in rows if row["ratio_inf_prev"] != ""]
    ratio_l2_values = [float(row["ratio_l2_prev"]) for row in rows if row["ratio_l2_prev"] != ""]
    topk_values = [float(row["topk_jaccard_prev"]) for row in rows if row["topk_jaccard_prev"] != ""]
    summary = {
        "case_name": case_name,
        "size_bin": size_bin(int(run_meta["buses"])),
        "buses": int(run_meta["buses"]),
        "pv": int(run_meta["pv"]),
        "pq": int(run_meta["pq"]),
        "dimF": stats[0].dim,
        "success": run_meta["success"],
        "iterations": int(run_meta["iterations"]),
        "final_mismatch": float(run_meta["final_mismatch"]),
        "initial_norm_inf": stats[0].norm_inf,
        "final_norm_inf": stats[-1].norm_inf,
        "initial_norm_l2": stats[0].norm_l2,
        "final_norm_l2": stats[-1].norm_l2,
        "total_norm_inf_reduction": stats[0].norm_inf / stats[-1].norm_inf if stats[-1].norm_inf != 0 else "",
        "total_norm_l2_reduction": stats[0].norm_l2 / stats[-1].norm_l2 if stats[-1].norm_l2 != 0 else "",
        "min_cos_prev": min(cosine_values) if cosine_values else "",
        "mean_cos_prev": math.fsum(cosine_values) / len(cosine_values) if cosine_values else "",
        "negative_cos_steps": sum(1 for value in cosine_values if value < 0.0),
        "min_ratio_inf_prev": min(ratio_inf_values) if ratio_inf_values else "",
        "max_ratio_inf_prev": max(ratio_inf_values) if ratio_inf_values else "",
        "mean_ratio_inf_prev": math.fsum(ratio_inf_values) / len(ratio_inf_values) if ratio_inf_values else "",
        "max_ratio_l2_prev": max(ratio_l2_values) if ratio_l2_values else "",
        "monotone_inf_steps": sum(1 for value in ratio_inf_values if value < 1.0),
        "total_steps": len(ratio_inf_values),
        "mean_topk_jaccard_prev": math.fsum(topk_values) / len(topk_values) if topk_values else "",
    }
    return rows, summary


def write_markdown_report(run_root: Path,
                          summaries: list[dict[str, Any]],
                          errors: list[dict[str, Any]]) -> None:
    success_count = sum(1 for row in summaries if row["success"] == "true")
    total_steps = sum(int(row["total_steps"]) for row in summaries)
    monotone_steps = sum(int(row["monotone_inf_steps"]) for row in summaries)
    nonmonotone_cases = [
        row["case_name"] for row in summaries
        if int(row["monotone_inf_steps"]) < int(row["total_steps"])
    ]
    lines = [
        "# Mismatch Direction Batch 1 Analysis",
        "",
        f"- Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Run root: `{run_root}`",
        f"- Cases: `{len(summaries)}`",
        f"- Success: `{success_count}/{len(summaries)}`",
        "- Batch size: `1`",
        "- Profile: `cuda_mixed_edge`",
        "",
        "## Key Findings",
        "",
        f"- Overall success: `{success_count}/{len(summaries)}` cases.",
        f"- L_inf mismatch decreased in `{monotone_steps}/{total_steps}` consecutive iteration steps.",
        f"- Cases with any L_inf increase: `{', '.join(nonmonotone_cases) if nonmonotone_cases else 'none'}`.",
        "- Negative cosine steps indicate that residual direction can flip even while the norm decreases.",
        "- This supports a guarded hybrid policy: approximate middle iterations are plausible, but they should be checked by actual mismatch reduction.",
        "",
        "## Summary By Size",
        "",
        "| size bin | cases | success | iter mean | min cosine median | mean cosine median | max L_inf ratio median | monotone steps/total |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    bins = ["<100", "100-999", "1k-9,999", "10k-49,999", ">=50k"]
    for bin_name in bins:
        rows = [row for row in summaries if row["size_bin"] == bin_name]
        if not rows:
            continue
        def median(values: list[float]) -> float:
            values = sorted(values)
            mid = len(values) // 2
            if len(values) % 2:
                return values[mid]
            return 0.5 * (values[mid - 1] + values[mid])
        min_cos = [float(row["min_cos_prev"]) for row in rows if row["min_cos_prev"] != ""]
        mean_cos = [float(row["mean_cos_prev"]) for row in rows if row["mean_cos_prev"] != ""]
        max_ratio = [float(row["max_ratio_inf_prev"]) for row in rows if row["max_ratio_inf_prev"] != ""]
        monotone = sum(int(row["monotone_inf_steps"]) for row in rows)
        steps = sum(int(row["total_steps"]) for row in rows)
        lines.append(
            f"| `{bin_name}` | {len(rows)} | {sum(1 for row in rows if row['success'] == 'true')} | "
            f"{sum(int(row['iterations']) for row in rows) / len(rows):.2f} | "
            f"{median(min_cos):.4g} | {median(mean_cos):.4g} | {median(max_ratio):.4g} | "
            f"{monotone}/{steps} |"
        )

    lines.extend([
        "",
        "## Case Highlights",
        "",
        "| case | buses | iter | final mismatch | total L_inf reduction | min cos | mean cos | max L_inf ratio | monotone steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in sorted(summaries, key=lambda item: int(item["buses"])):
        lines.append(
            f"| `{row['case_name']}` | {row['buses']} | {row['iterations']} | "
            f"{float(row['final_mismatch']):.3e} | {float(row['total_norm_inf_reduction']):.3e} | "
            f"{float(row['min_cos_prev']) if row['min_cos_prev'] != '' else 0.0:.4g} | "
            f"{float(row['mean_cos_prev']) if row['mean_cos_prev'] != '' else 0.0:.4g} | "
            f"{float(row['max_ratio_inf_prev']) if row['max_ratio_inf_prev'] != '' else 0.0:.4g} | "
            f"{row['monotone_inf_steps']}/{row['total_steps']} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `norm_inf`, `norm_l2`, `norm_l1`은 반복별 mismatch vector 크기다.",
        "- `cos_prev`는 연속 반복 mismatch vector 사이의 방향 유사도다.",
        "- `ratio_inf_prev < 1`이면 수렴 판정 기준인 L_inf mismatch가 감소한 step이다.",
        "- `topk_jaccard_prev`는 큰 mismatch component 위치가 반복 사이에 얼마나 유지되는지 보는 보조 지표다.",
        "- raw vector는 `raw/<case>/dumps/repeat_00/residual_iter*.txt`에 남아 있다.",
    ])
    if errors:
        lines.extend(["", "## Errors", "", "| case | return code | stderr tail |", "| --- | ---: | --- |"])
        for error in errors:
            tail = str(error.get("stderr_tail", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{error['case_name']}` | {error['returncode']} | `{tail}` |")
    lines.extend([
        "",
        "## Files",
        "",
        "- `trajectory_summary.csv`: one row per case",
        "- `vector_metrics.csv`: one row per case/iteration",
        "- `run_summary.csv`: benchmark stdout summary",
        "- `raw/`: benchmark stdout/stderr, command, and residual vector dumps",
    ])
    run_root.joinpath("MISMATCH_DIRECTION_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("experiment 2 is defined for --batch-size 1")
    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=False)

    names = case_names(args.dataset_root, args.cases)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root),
        "benchmark_binary": str(args.benchmark_binary),
        "cases": names,
        "profile": args.profile,
        "batch_size": args.batch_size,
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "repeats": 1,
        "warmup": 0,
        "metrics": [
            "norm_inf",
            "norm_l2",
            "norm_l1",
            "ratio_inf_prev",
            "ratio_l2_prev",
            "cos_prev",
            "angle_deg_prev",
            "relative_delta_l2_prev",
            "topk_jaccard_prev",
        ],
    }
    write_json(run_root / "manifest.json", manifest)

    env = benchmark_env(args.cudss_threading_lib)
    run_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, case_name in enumerate(names, start=1):
        case_dir = args.dataset_root / case_name
        raw_dir = run_root / "raw" / case_name
        dump_dir = raw_dir / "dumps"
        raw_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(args.benchmark_binary),
            "--case-dir", str(case_dir),
            "--profile", args.profile,
            "--warmup", "0",
            "--repeats", "1",
            "--batch-size", str(args.batch_size),
            "--tolerance", args.tolerance,
            "--max-iter", str(args.max_iter),
            "--cudss-matching-alg", "DEFAULT",
            "--cudss-pivot-epsilon", "AUTO",
            "--dump-residuals",
            "--dump-newton-diagnostics",
            "--dump-dir", str(dump_dir),
        ]
        print(f"[{idx}/{len(names)}] mismatch trajectory {case_name}", flush=True)
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        raw_dir.joinpath("stdout.txt").write_text(completed.stdout, encoding="utf-8")
        raw_dir.joinpath("stderr.txt").write_text(completed.stderr, encoding="utf-8")
        write_json(raw_dir / "command.json", {"cmd": cmd})

        if completed.returncode != 0:
            error = {
                "case_name": case_name,
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-1000:],
            }
            errors.append(error)
            if args.continue_on_error:
                continue
            write_json(run_root / "errors.json", errors)
            raise RuntimeError(f"benchmark failed for {case_name}")

        run_meta = parse_run_line(completed.stdout)
        if run_meta is None:
            raise RuntimeError(f"missing RUN line for {case_name}")
        run_rows.append({
            "case_name": case_name,
            "size_bin": size_bin(int(run_meta["buses"])),
            "success": run_meta["success"],
            "iterations": int(run_meta["iterations"]),
            "final_mismatch": float(run_meta["final_mismatch"]),
            "elapsed_sec": float(run_meta["total_sec"]),
            "analyze_sec": float(run_meta["analyze_sec"]),
            "solve_sec": float(run_meta["solve_sec"]),
            "buses": int(run_meta["buses"]),
            "pv": int(run_meta["pv"]),
            "pq": int(run_meta["pq"]),
            "dimF": int(run_meta["pv"]) + 2 * int(run_meta["pq"]),
        })
        case_vector_rows, case_summary = parse_residuals(dump_dir / "repeat_00", case_name, run_meta)
        vector_rows.extend(case_vector_rows)
        summaries.append(case_summary)

        write_csv(run_root / "run_summary.csv", run_rows)
        write_csv(run_root / "vector_metrics.csv", vector_rows)
        write_csv(run_root / "trajectory_summary.csv", summaries)

    write_csv(run_root / "run_summary.csv", run_rows)
    write_csv(run_root / "vector_metrics.csv", vector_rows)
    write_csv(run_root / "trajectory_summary.csv", summaries)
    if errors:
        write_json(run_root / "errors.json", errors)
    write_markdown_report(run_root, summaries, errors)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
