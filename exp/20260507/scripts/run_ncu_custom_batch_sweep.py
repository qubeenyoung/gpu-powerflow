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
CUSTOM_KERNEL_REGEX = (
    r".*(compute_ibus_kernel|compute_mismatch_from_ibus_kernel|reduce_mismatch_norm_kernel|"
    r"fill_jacobian_gpu_kernel|prepare_rhs_kernel|apply_voltage_update_kernel|"
    r"reconstruct_voltage_kernel).*"
)


@dataclass(frozen=True)
class RunSpec:
    case_name: str
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ncu basic set over batch sizes.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("ncu_custom_%Y%m%d_%H%M%S"))
    parser.add_argument("--benchmark-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", default="cuda_mixed_edge")
    parser.add_argument("--tolerance", default="1e-8")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--cudss-threading-lib", type=Path, default=DEFAULT_CUDSS_THREADING_LIB)
    parser.add_argument(
        "--kernel-scope",
        choices=("custom", "all"),
        default="custom",
        help="custom profiles only cuPF custom kernels; all profiles every CUDA kernel, including cuDSS.",
    )
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


def operator_from_kernel(kernel_name: str) -> str:
    if "cudss::" in kernel_name:
        return "cudss"
    if (
        "dependency_map_ker" in kernel_name
        or "offsets_par_ker" in kernel_name
        or "radix_sort_ker" in kernel_name
        or "xadj_ker" in kernel_name
        or "adjncy_ker" in kernel_name
    ):
        return "cudss_aux"
    if "compute_ibus_kernel" in kernel_name:
        return "ibus"
    if "compute_mismatch_from_ibus_kernel" in kernel_name:
        return "mismatch"
    if "reduce_mismatch_norm_kernel" in kernel_name:
        return "mismatch_norm"
    if "fill_jacobian_gpu_kernel" in kernel_name:
        return "jacobian_fill"
    if "prepare_rhs_kernel" in kernel_name:
        return "prepare_rhs"
    if "apply_voltage_update_kernel" in kernel_name:
        return "voltage_update_apply"
    if "reconstruct_voltage_kernel" in kernel_name:
        return "voltage_reconstruct"
    return "other"


def parse_numeric(value: str) -> float | str:
    if value == "":
        return ""
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return value


def parse_ncu_csv(path: Path, case_name: str, batch_size: int) -> list[dict[str, Any]]:
    rows_for_csv: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith('"'):
                rows_for_csv.append(line)
    if not rows_for_csv:
        return []

    reader = csv.DictReader(rows_for_csv)
    by_launch: dict[str, dict[str, Any]] = {}
    interesting = {
        "Duration": "duration_ns",
        "Compute (SM) Throughput": "compute_sm_pct",
        "Memory Throughput": "memory_throughput_pct",
        "DRAM Throughput": "dram_throughput_pct",
        "Achieved Occupancy": "achieved_occupancy_pct",
        "Grid Size": "grid_size",
        "Block Size": "block_size",
        "Threads": "threads",
        "Waves Per SM": "waves_per_sm",
    }
    for metric in reader:
        launch_id = metric["ID"]
        kernel_name = metric["Kernel Name"]
        launch = by_launch.setdefault(launch_id, {
            "case_name": case_name,
            "batch_size": batch_size,
            "launch_id": int(launch_id),
            "operator": operator_from_kernel(kernel_name),
            "kernel_name": kernel_name,
        })
        metric_name = metric["Metric Name"]
        out_key = interesting.get(metric_name)
        if out_key:
            launch[out_key] = parse_numeric(metric["Metric Value"])

    return [by_launch[key] for key in sorted(by_launch, key=lambda value: int(value))]


def mean(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def aggregate_launches(launch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in launch_rows:
        grouped.setdefault((row["case_name"], int(row["batch_size"]), row["operator"]), []).append(row)

    out: list[dict[str, Any]] = []
    for (case_name, batch_size, operator), rows in sorted(grouped.items(), key=lambda item: item[0]):
        duration_values = [float(row["duration_ns"]) for row in rows if row.get("duration_ns") not in ("", None)]
        compute_values = [float(row["compute_sm_pct"]) for row in rows if row.get("compute_sm_pct") not in ("", None)]
        memory_values = [float(row["memory_throughput_pct"]) for row in rows if row.get("memory_throughput_pct") not in ("", None)]
        dram_values = [float(row["dram_throughput_pct"]) for row in rows if row.get("dram_throughput_pct") not in ("", None)]
        occ_values = [float(row["achieved_occupancy_pct"]) for row in rows if row.get("achieved_occupancy_pct") not in ("", None)]
        waves_values = [float(row["waves_per_sm"]) for row in rows if row.get("waves_per_sm") not in ("", None)]
        out.append({
            "case_name": case_name,
            "batch_size": batch_size,
            "operator": operator,
            "launches": len(rows),
            "duration_ns_sum": sum(duration_values) if duration_values else "",
            "duration_ns_mean": mean(duration_values),
            "compute_sm_pct_mean": mean(compute_values),
            "compute_sm_pct_max": max(compute_values) if compute_values else "",
            "memory_throughput_pct_mean": mean(memory_values),
            "memory_throughput_pct_max": max(memory_values) if memory_values else "",
            "dram_throughput_pct_mean": mean(dram_values),
            "dram_throughput_pct_max": max(dram_values) if dram_values else "",
            "achieved_occupancy_pct_mean": mean(occ_values),
            "achieved_occupancy_pct_max": max(occ_values) if occ_values else "",
            "waves_per_sm_mean": mean(waves_values),
            "waves_per_sm_max": max(waves_values) if waves_values else "",
        })
    return out


def aggregate_by_case_batch(launch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in launch_rows:
        grouped.setdefault((row["case_name"], int(row["batch_size"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (case_name, batch_size), rows in sorted(grouped.items(), key=lambda item: item[0]):
        durations = [float(row["duration_ns"]) for row in rows if row.get("duration_ns") not in ("", None)]
        compute = [float(row["compute_sm_pct"]) for row in rows if row.get("compute_sm_pct") not in ("", None)]
        memory = [float(row["memory_throughput_pct"]) for row in rows if row.get("memory_throughput_pct") not in ("", None)]
        occ = [float(row["achieved_occupancy_pct"]) for row in rows if row.get("achieved_occupancy_pct") not in ("", None)]
        waves = [float(row["waves_per_sm"]) for row in rows if row.get("waves_per_sm") not in ("", None)]
        out.append({
            "case_name": case_name,
            "batch_size": batch_size,
            "profiled_kernel_launches": len(rows),
            "duration_ns_sum": sum(durations) if durations else "",
            "compute_sm_pct_mean": mean(compute),
            "compute_sm_pct_max": max(compute) if compute else "",
            "memory_throughput_pct_mean": mean(memory),
            "memory_throughput_pct_max": max(memory) if memory else "",
            "achieved_occupancy_pct_mean": mean(occ),
            "achieved_occupancy_pct_max": max(occ) if occ else "",
            "waves_per_sm_mean": mean(waves),
            "waves_per_sm_max": max(waves) if waves else "",
        })
    return out


def write_markdown_summary(run_root: Path,
                           run_summaries: list[dict[str, Any]],
                           case_batch_summary: list[dict[str, Any]],
                           errors: list[dict[str, Any]],
                           scope_label: str) -> None:
    lines = [
        "# NCU Batch Sweep",
        "",
        f"- Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Run root: `{run_root}`",
        "- Batch sizes: `1, 2, 4, 8, 16, 32, 64, 128, 256`",
        f"- Scope: `{scope_label}`",
        "",
        "## Case/Batch Summary",
        "",
        "| case | batch | launches | duration ms sum | SM mean % | SM max % | Mem mean % | Occ mean % | waves/SM max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in case_batch_summary:
        duration_ms = float(row["duration_ns_sum"]) / 1e6 if row["duration_ns_sum"] != "" else 0.0
        lines.append(
            f"| `{row['case_name']}` | {row['batch_size']} | {row['profiled_kernel_launches']} | "
            f"{duration_ms:.4g} | {float(row['compute_sm_pct_mean']):.3g} | "
            f"{float(row['compute_sm_pct_max']):.3g} | {float(row['memory_throughput_pct_mean']):.3g} | "
            f"{float(row['achieved_occupancy_pct_mean']):.3g} | {float(row['waves_per_sm_max']):.3g} |"
        )
    if errors:
        lines.extend(["", "## Errors", "", "| case | batch | return code | stderr tail |", "| --- | ---: | ---: | --- |"])
        for error in errors:
            tail = str(error.get("stderr_tail", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{error['case_name']}` | {error['batch_size']} | {error['returncode']} | `{tail}` |")
    lines.extend([
        "",
        "## Files",
        "",
        "- `launch_metrics.csv`: one row per profiled kernel launch",
        "- `operator_summary.csv`: aggregate by case, batch, operator",
        "- `case_batch_summary.csv`: aggregate by case and batch",
        "- `run_summary.csv`: benchmark stdout summary per case/batch",
        "- `raw/`: original ncu CSV and process stdout/stderr",
    ])
    run_root.joinpath("SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=False)

    env = benchmark_env(args.cudss_threading_lib)
    specs = [RunSpec(case_name=case_name, batch_size=batch) for case_name in args.cases for batch in args.batches]
    scope_label = (
        "cuPF custom kernels only; cuDSS internal kernels are excluded"
        if args.kernel_scope == "custom"
        else "all CUDA kernels; includes cuDSS internal kernels"
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root),
        "benchmark_binary": str(args.benchmark_binary),
        "cases": args.cases,
        "batches": args.batches,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile": args.profile,
        "kernel_scope": args.kernel_scope,
        "kernel_regex": CUSTOM_KERNEL_REGEX if args.kernel_scope == "custom" else "",
        "ncu_set": "basic",
        "scope": scope_label,
    }
    write_json(run_root / "manifest.json", manifest)

    run_summaries: list[dict[str, Any]] = []
    launch_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, spec in enumerate(specs, start=1):
        case_dir = args.dataset_root / spec.case_name
        raw_dir = run_root / "raw" / spec.case_name / f"b{spec.batch_size}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        ncu_csv = raw_dir / ("ncu_custom_basic.csv" if args.kernel_scope == "custom" else "ncu_all_basic.csv")
        cmd = [
            "ncu",
            "--target-processes", "all",
            "--set", "basic",
            "--kernel-name-base", "demangled",
        ]
        if args.kernel_scope == "custom":
            cmd += ["--kernel-name", f"regex:{CUSTOM_KERNEL_REGEX}"]
        cmd += [
            "--csv",
            "--log-file", str(ncu_csv),
            "--force-overwrite",
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
        print(f"[{idx}/{len(specs)}] ncu {args.kernel_scope} {spec.case_name} batch={spec.batch_size}", flush=True)
        completed = subprocess.run(
            cmd,
            cwd="/workspace/gpu-powerflow-master",
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
                "case_name": spec.case_name,
                "batch_size": spec.batch_size,
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-500:],
            }
            errors.append(error)
            if args.continue_on_error:
                continue
            write_json(run_root / "errors.json", errors)
            raise RuntimeError(f"ncu failed for {spec.case_name} batch={spec.batch_size}")

        parsed_run = parse_run_line(completed.stdout)
        if parsed_run is None:
            raise RuntimeError(f"missing RUN line for {spec.case_name} batch={spec.batch_size}")
        buses = int(parsed_run["buses"])
        run_summaries.append({
            "case_name": spec.case_name,
            "size_bin": size_bin(buses),
            "batch_size": int(parsed_run["batch_size"]),
            "success": parsed_run["success"],
            "iterations": int(parsed_run["iterations"]),
            "final_mismatch": float(parsed_run["final_mismatch"]),
            "elapsed_sec": float(parsed_run["total_sec"]),
            "analyze_sec": float(parsed_run["analyze_sec"]),
            "solve_sec": float(parsed_run["solve_sec"]),
            "buses": buses,
            "pv": int(parsed_run["pv"]),
            "pq": int(parsed_run["pq"]),
        })
        launch_rows.extend(parse_ncu_csv(ncu_csv, spec.case_name, spec.batch_size))

        write_csv(run_root / "run_summary.csv", run_summaries)
        write_csv(run_root / "launch_metrics.csv", launch_rows)
        write_csv(run_root / "operator_summary.csv", aggregate_launches(launch_rows))
        write_csv(run_root / "case_batch_summary.csv", aggregate_by_case_batch(launch_rows))

    operator_summary = aggregate_launches(launch_rows)
    case_batch_summary = aggregate_by_case_batch(launch_rows)
    write_csv(run_root / "run_summary.csv", run_summaries)
    write_csv(run_root / "launch_metrics.csv", launch_rows)
    write_csv(run_root / "operator_summary.csv", operator_summary)
    write_csv(run_root / "case_batch_summary.csv", case_batch_summary)
    if errors:
        write_json(run_root / "errors.json", errors)
    write_markdown_summary(run_root, run_summaries, case_batch_summary, errors, scope_label)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
