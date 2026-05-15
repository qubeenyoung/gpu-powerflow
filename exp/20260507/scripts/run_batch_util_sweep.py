#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "exp" / "20260507" / "results" / "batch_util"
DEFAULT_BINARY = (
    Path("/workspace/gpu-powerflow-master")
    / "cuPF"
    / "build"
    / "bench-end2end-superlu-cudss-mt-auto"
    / "benchmarks"
    / "cupf_case_benchmark"
)
DEFAULT_CUDSS_THREADING_LIB = Path(
    "/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so"
)


@dataclass(frozen=True)
class RunSpec:
    case_name: str
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cuPF CUDA mixed batch-size sweep and summarize throughput/utilization."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--run-name",
        default=datetime.now(timezone.utc).strftime("batch_util_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--benchmark-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", default="cuda_mixed_edge")
    parser.add_argument("--tolerance", default="1e-8")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--cudss-threading-lib", type=Path, default=DEFAULT_CUDSS_THREADING_LIB)
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("--gpu-sample-ms", type=int, default=100)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


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


def parse_run_line(line: str) -> dict[str, str] | None:
    if not line.startswith("RUN "):
        return None
    parsed: dict[str, str] = {}
    for token in line[len("RUN "):].strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def parse_gpu_samples(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("timestamp") or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            samples.append({
                "timestamp": parts[0],
                "utilization_gpu_pct": float(parts[1]),
                "utilization_memory_pct": float(parts[2]),
                "memory_used_mib": float(parts[3]),
                "power_draw_w": float(parts[4]),
            })
        except ValueError:
            continue
    return samples


def gpu_sample_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "gpu_sample_count": 0,
            "gpu_util_mean_pct": "",
            "gpu_util_max_pct": "",
            "mem_util_mean_pct": "",
            "mem_used_max_mib": "",
            "power_mean_w": "",
        }
    gpu_utils = [float(row["utilization_gpu_pct"]) for row in samples]
    mem_utils = [float(row["utilization_memory_pct"]) for row in samples]
    mem_used = [float(row["memory_used_mib"]) for row in samples]
    power = [float(row["power_draw_w"]) for row in samples]
    return {
        "gpu_sample_count": len(samples),
        "gpu_util_mean_pct": statistics.mean(gpu_utils),
        "gpu_util_max_pct": max(gpu_utils),
        "mem_util_mean_pct": statistics.mean(mem_utils),
        "mem_used_max_mib": max(mem_used),
        "power_mean_w": statistics.mean(power),
    }


def run_command_with_optional_gpu_sampling(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    sample_gpu: bool,
    gpu_sample_ms: int,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    sampler: subprocess.Popen[str] | None = None
    if sample_gpu:
        sampler_cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,nounits",
            "-lms",
            str(gpu_sample_ms),
        ]
        sampler = subprocess.Popen(
            sampler_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.2)

    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    samples: list[dict[str, Any]] = []
    if sampler is not None:
        sampler.terminate()
        try:
            stdout, _ = sampler.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            sampler.kill()
            stdout, _ = sampler.communicate()
        samples = parse_gpu_samples(stdout)
    return completed, samples


def benchmark_env(cudss_threading_lib: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDSS_THREADING_LIB"] = str(cudss_threading_lib)
    preload_entries = [entry for entry in env.get("LD_PRELOAD", "").split(":") if entry]
    threading_path = str(cudss_threading_lib)
    if threading_path not in preload_entries:
        env["LD_PRELOAD"] = ":".join([threading_path, *preload_entries])
    return env


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def numeric_mean(rows: list[dict[str, Any]], key: str) -> float | str:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return statistics.mean(values) if values else ""


def numeric_max(rows: list[dict[str, Any]], key: str) -> float | str:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return max(values) if values else ""


def summarize_by_case_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault((str(row["case_name"]), int(row["batch_size"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (case_name, batch_size), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        elapsed_mean = float(numeric_mean(group, "elapsed_sec"))
        solve_mean = float(numeric_mean(group, "solve_sec"))
        analyze_mean = float(numeric_mean(group, "analyze_sec"))
        buses = int(group[0]["buses"])
        summary.append({
            "case_name": case_name,
            "size_bin": size_bin(buses),
            "buses": buses,
            "batch_size": batch_size,
            "runs": len(group),
            "success_all": all(str(row["success"]).lower() == "true" for row in group),
            "iterations_mean": numeric_mean(group, "iterations"),
            "final_mismatch_max": numeric_max(group, "final_mismatch"),
            "elapsed_sec_mean": elapsed_mean,
            "analyze_sec_mean": analyze_mean,
            "solve_sec_mean": solve_mean,
            "elapsed_ms_mean": elapsed_mean * 1000.0,
            "solve_ms_mean": solve_mean * 1000.0,
            "elapsed_ms_per_scenario": elapsed_mean * 1000.0 / batch_size,
            "solve_ms_per_scenario": solve_mean * 1000.0 / batch_size,
            "scenario_per_sec_elapsed": batch_size / elapsed_mean if elapsed_mean > 0 else "",
            "scenario_per_sec_solve": batch_size / solve_mean if solve_mean > 0 else "",
            "gpu_util_mean_pct": numeric_mean(group, "gpu_util_mean_pct"),
            "gpu_util_max_pct": numeric_max(group, "gpu_util_max_pct"),
            "mem_used_max_mib": numeric_max(group, "mem_used_max_mib"),
        })
    return summary


def summarize_by_size_bin(case_batch_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in case_batch_summary:
        grouped.setdefault((str(row["size_bin"]), int(row["batch_size"])), []).append(row)
    rows: list[dict[str, Any]] = []
    for (bin_name, batch_size), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        rows.append({
            "size_bin": bin_name,
            "batch_size": batch_size,
            "cases": len(group),
            "success_cases": sum(1 for row in group if row["success_all"]),
            "elapsed_ms_per_scenario_mean": numeric_mean(group, "elapsed_ms_per_scenario"),
            "solve_ms_per_scenario_mean": numeric_mean(group, "solve_ms_per_scenario"),
            "scenario_per_sec_elapsed_mean": numeric_mean(group, "scenario_per_sec_elapsed"),
            "scenario_per_sec_solve_mean": numeric_mean(group, "scenario_per_sec_solve"),
            "gpu_util_mean_pct": numeric_mean(group, "gpu_util_mean_pct"),
            "gpu_util_max_pct": numeric_max(group, "gpu_util_max_pct"),
            "mem_used_max_mib": numeric_max(group, "mem_used_max_mib"),
        })
    return rows


def recommended_batches(case_batch_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in case_batch_summary:
        by_case.setdefault(str(row["case_name"]), []).append(row)

    recommendations: list[dict[str, Any]] = []
    for case_name, rows in sorted(by_case.items()):
        valid = [row for row in rows if row["success_all"]]
        if not valid:
            continue
        by_elapsed = min(valid, key=lambda row: float(row["elapsed_ms_per_scenario"]))
        by_solve = min(valid, key=lambda row: float(row["solve_ms_per_scenario"]))
        buses = int(valid[0]["buses"])
        recommendations.append({
            "case_name": case_name,
            "size_bin": size_bin(buses),
            "buses": buses,
            "best_elapsed_batch": by_elapsed["batch_size"],
            "best_elapsed_ms_per_scenario": by_elapsed["elapsed_ms_per_scenario"],
            "best_solve_batch": by_solve["batch_size"],
            "best_solve_ms_per_scenario": by_solve["solve_ms_per_scenario"],
            "max_success_batch": max(int(row["batch_size"]) for row in valid),
        })
    return recommendations


def write_markdown_summary(run_root: Path, recommendations: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    lines = [
        "# Batch Utilization Sweep Summary",
        "",
        f"- Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Run root: `{run_root}`",
        "- Note: GPU utilization values are coarse `nvidia-smi` samples and should be confirmed with Nsight for final claims.",
        "",
        "## Best Batch By Case",
        "",
        "| case | size bin | buses | best elapsed batch | elapsed ms/scenario | best solve batch | solve ms/scenario | max success batch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in recommendations:
        lines.append(
            "| {case_name} | {size_bin} | {buses} | {best_elapsed_batch} | {best_elapsed_ms_per_scenario:.6g} | "
            "{best_solve_batch} | {best_solve_ms_per_scenario:.6g} | {max_success_batch} |".format(**row)
        )
    if errors:
        lines.extend([
            "",
            "## Errors",
            "",
            "| case | batch | return code | stderr tail |",
            "| --- | ---: | ---: | --- |",
        ])
        for row in errors:
            stderr_tail = str(row.get("stderr_tail", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['case_name']} | {row['batch_size']} | {row['returncode']} | `{stderr_tail}` |")
    run_root.joinpath("SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.benchmark_binary.exists():
        raise FileNotFoundError(f"benchmark binary not found: {args.benchmark_binary}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=False)
    env = benchmark_env(args.cudss_threading_lib)
    specs = [RunSpec(case_name=case_name, batch_size=batch) for case_name in args.cases for batch in args.batches]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root),
        "benchmark_binary": str(args.benchmark_binary),
        "cases": args.cases,
        "batches": args.batches,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile": args.profile,
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "sample_gpu": args.sample_gpu,
        "gpu_sample_ms": args.gpu_sample_ms,
        "cudss_threading_lib": str(args.cudss_threading_lib),
    }
    write_json(run_root / "manifest.json", manifest)

    run_rows: list[dict[str, Any]] = []
    gpu_sample_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, spec in enumerate(specs, start=1):
        case_dir = args.dataset_root / spec.case_name
        cmd = [
            str(args.benchmark_binary),
            "--case-dir", str(case_dir),
            "--profile", args.profile,
            "--warmup", str(args.warmup),
            "--repeats", str(args.repeats),
            "--batch-size", str(spec.batch_size),
            "--tolerance", args.tolerance,
            "--max-iter", str(args.max_iter),
            "--cudss-matching-alg", "DEFAULT",
            "--cudss-pivot-epsilon", "AUTO",
        ]
        print(f"[{idx}/{len(specs)}] {spec.case_name} batch={spec.batch_size}", flush=True)
        started = datetime.now(timezone.utc).isoformat()
        completed, samples = run_command_with_optional_gpu_sampling(
            cmd,
            cwd=Path("/workspace/gpu-powerflow-master"),
            env=env,
            sample_gpu=args.sample_gpu,
            gpu_sample_ms=args.gpu_sample_ms,
        )
        ended = datetime.now(timezone.utc).isoformat()
        raw_dir = run_root / "raw" / spec.case_name / f"b{spec.batch_size}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.joinpath("stdout.txt").write_text(completed.stdout, encoding="utf-8")
        raw_dir.joinpath("stderr.txt").write_text(completed.stderr, encoding="utf-8")
        write_json(raw_dir / "command.json", {"cmd": cmd, "started_utc": started, "ended_utc": ended})

        for sample_idx, sample in enumerate(samples):
            gpu_sample_rows.append({
                "case_name": spec.case_name,
                "batch_size": spec.batch_size,
                "sample_idx": sample_idx,
                **sample,
            })

        sample_summary = gpu_sample_summary(samples)
        if completed.returncode != 0:
            error = {
                "case_name": spec.case_name,
                "batch_size": spec.batch_size,
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-500:],
            }
            errors.append(error)
            run_rows.append({
                "status": "error",
                **error,
                **sample_summary,
            })
            write_csv(run_root / "summary.csv", run_rows)
            write_csv(run_root / "gpu_util_samples.csv", gpu_sample_rows)
            if args.continue_on_error:
                continue
            write_json(run_root / "errors.json", errors)
            raise RuntimeError(f"benchmark failed for {spec.case_name} batch={spec.batch_size}")

        parsed_count = 0
        for line in completed.stdout.splitlines():
            parsed = parse_run_line(line.strip())
            if parsed is None:
                continue
            parsed_count += 1
            buses = int(parsed["buses"])
            batch_size = int(parsed.get("batch_size", spec.batch_size))
            elapsed_sec = float(parsed["total_sec"])
            solve_sec = float(parsed["solve_sec"])
            run_rows.append({
                "status": "ok",
                "case_name": spec.case_name,
                "size_bin": size_bin(buses),
                "batch_size": batch_size,
                "repeat_idx": int(parsed["repeat"]),
                "success": parsed["success"],
                "iterations": int(parsed["iterations"]),
                "final_mismatch": float(parsed["final_mismatch"]),
                "elapsed_sec": elapsed_sec,
                "analyze_sec": float(parsed["analyze_sec"]),
                "solve_sec": solve_sec,
                "elapsed_ms_per_scenario": elapsed_sec * 1000.0 / batch_size,
                "solve_ms_per_scenario": solve_sec * 1000.0 / batch_size,
                "scenario_per_sec_elapsed": batch_size / elapsed_sec if elapsed_sec > 0 else "",
                "scenario_per_sec_solve": batch_size / solve_sec if solve_sec > 0 else "",
                "buses": buses,
                "pv": int(parsed["pv"]),
                "pq": int(parsed["pq"]),
                "profile": parsed["profile"],
                "compute": parsed["compute"],
                "implementation": parsed["implementation"],
                **sample_summary,
            })
        if parsed_count != args.repeats:
            raise RuntimeError(
                f"expected {args.repeats} RUN lines for {spec.case_name} batch={spec.batch_size}, got {parsed_count}"
            )

        write_csv(run_root / "summary.csv", run_rows)
        write_csv(run_root / "gpu_util_samples.csv", gpu_sample_rows)

    case_batch_summary = summarize_by_case_batch(run_rows)
    size_summary = summarize_by_size_bin(case_batch_summary)
    recommendations = recommended_batches(case_batch_summary)
    write_csv(run_root / "case_batch_summary.csv", case_batch_summary)
    write_csv(run_root / "size_bin_summary.csv", size_summary)
    write_csv(run_root / "recommendations.csv", recommendations)
    if errors:
        write_json(run_root / "errors.json", errors)
    write_markdown_summary(run_root, recommendations, errors)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
