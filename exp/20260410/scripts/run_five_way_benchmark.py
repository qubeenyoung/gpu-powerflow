#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = WORKSPACE_ROOT / "exp" / "20260410"
RESULTS_ROOT = EXP_ROOT / "results"
BENCHMARK_SCRIPT_PATH = WORKSPACE_ROOT / "cuPF" / "benchmarks" / "run_benchmarks.py"
DEFAULT_BENCHMARK_BINARY = WORKSPACE_ROOT / "cuPF" / "build" / "bench-cuda-timing" / "cupf_case_benchmark"

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from python.converters.common import TARGET_CASES, case_metadata, case_stem, preprocess_case, save_cupf_dump, write_json


def load_benchmark_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("cupf_run_benchmarks", BENCHMARK_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load benchmark helpers from {BENCHMARK_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_benchmark_helpers()


IMPLEMENTATIONS = (
    {
        "implementation": "cpp_pypowerlike",
        "backend": "cpu",
        "algorithm": "pypower_like",
        "jacobian": "edge_based",
        "repeat_kind": "cpu",
    },
    {
        "implementation": "cpp_optimized",
        "backend": "cpu",
        "algorithm": "optimized",
        "jacobian": "edge_based",
        "repeat_kind": "cpu",
    },
    {
        "implementation": "cpp_cuda_edge",
        "backend": "cuda",
        "algorithm": "optimized",
        "jacobian": "edge_based",
        "repeat_kind": "gpu",
    },
    {
        "implementation": "cpp_cuda_vertex",
        "backend": "cuda",
        "algorithm": "optimized",
        "jacobian": "vertex_based",
        "repeat_kind": "gpu",
    },
)


def parse_args() -> argparse.Namespace:
    default_run_name = datetime.now(timezone.utc).strftime("selected_cases_20260410_gpu3_%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Run the 2026-04-10 five-way cuPF benchmark.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--run-name", default=default_run_name)
    parser.add_argument("--dump-root", type=Path, default=bench.DEFAULT_DUMP_ROOT)
    parser.add_argument("--benchmark-binary", type=Path, default=DEFAULT_BENCHMARK_BINARY)
    parser.add_argument("--cases", nargs="*", default=list(TARGET_CASES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu-repeats", type=int, default=10)
    parser.add_argument("--gpu-repeats", type=int, default=10)
    parser.add_argument("--gpu-device", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    return parser.parse_args()


def collect_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpus": []}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,uuid",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception as exc:
        info["error"] = str(exc)
        return info

    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        info["gpus"].append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total": parts[2],
                "driver_version": parts[3],
                "uuid": parts[4],
            }
        )
    return info


def run_cpp_case(
    *,
    binary: Path,
    dump_dir: Path,
    case_name: str,
    case_stem_value: str,
    implementation: str,
    backend: str,
    algorithm: str,
    jacobian: str,
    warmup: int,
    repeats: int,
    tolerance: float,
    max_iter: int,
    run_root: Path,
    command_log: list[dict[str, Any]],
    gpu_device: int | None,
) -> list[dict[str, Any]]:
    cmd = [
        str(binary),
        "--case-dir",
        str(dump_dir),
        "--backend",
        backend,
        "--jacobian",
        jacobian,
        "--algorithm",
        algorithm,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--tolerance",
        str(tolerance),
        "--max-iter",
        str(max_iter),
    ]

    env = os.environ.copy()
    env_overrides: dict[str, str] = {}
    if backend == "cuda" and gpu_device is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        env.update(env_overrides)

    command_log.append(
        {
            "name": f"run_{implementation}",
            "cmd": cmd,
            "cwd": str(WORKSPACE_ROOT),
            "env_overrides": env_overrides,
        }
    )

    completed = bench.run_command(cmd, cwd=WORKSPACE_ROOT, env=env)

    non_run_lines: list[str] = []
    parsed_runs: dict[int, dict[str, str]] = {}
    metric_summaries: dict[int, dict[str, dict[str, float | int]]] = {}
    metric_entries: dict[int, list[dict[str, Any]]] = {}

    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("RUN "):
            parsed = bench.parse_cpp_run_line(line)
            parsed_runs[int(parsed["repeat"])] = parsed
            continue
        if line.startswith("METRIC "):
            parsed = bench.parse_cpp_metric_line(line)
            repeat_idx = int(parsed["repeat"])
            name = parsed["name"]
            count = int(parsed["count"])
            total_sec = float(parsed["total_sec"])
            avg_sec = float(parsed["avg_sec"])
            metric_summaries.setdefault(repeat_idx, {})[name] = {
                "count": count,
                "total_sec": total_sec,
                "avg_sec": avg_sec,
            }
            tag, op_name = bench.metric_name_parts(name)
            metric_entries.setdefault(repeat_idx, []).append(
                {
                    "tag": tag,
                    "op_name": op_name,
                    "iter_idx": repeat_idx,
                    "count": count,
                    "elapsed_sec": total_sec,
                    "avg_sec": avg_sec,
                }
            )
            continue
        if line.startswith("["):
            non_run_lines.append(line)
            continue
        raise RuntimeError(f"Unexpected benchmark output line: {line}")

    rows: list[dict[str, Any]] = []
    for repeat_idx in sorted(parsed_runs):
        parsed = parsed_runs[repeat_idx]
        analyze_sec = float(parsed["analyze_sec"])
        solve_sec = float(parsed["solve_sec"])
        total_sec = float(parsed["total_sec"])
        summary = metric_summaries.get(repeat_idx)
        if not summary:
            summary = bench.cpp_timing_summary(analyze_sec, solve_sec, total_sec)

        timing_entries = metric_entries.get(repeat_idx)
        if timing_entries is None:
            timing_entries = [
                {
                    "tag": "benchmark",
                    "op_name": "analyze",
                    "iter_idx": repeat_idx,
                    "elapsed_sec": analyze_sec,
                },
                {
                    "tag": "benchmark",
                    "op_name": "solve",
                    "iter_idx": repeat_idx,
                    "elapsed_sec": solve_sec,
                },
                {
                    "tag": "benchmark",
                    "op_name": "total",
                    "iter_idx": repeat_idx,
                    "elapsed_sec": total_sec,
                },
            ]

        row = {
            "implementation": implementation,
            "case_name": case_name,
            "case_stem": case_stem_value,
            "dump_case_name": parsed["case"],
            "repeat_idx": repeat_idx,
            "backend": parsed["backend"],
            "jacobian": parsed["jacobian"],
            "algorithm": parsed["algorithm"],
            "success": parsed["success"] == "true",
            "iterations": int(parsed["iterations"]),
            "final_mismatch": float(parsed["final_mismatch"]),
            "elapsed_sec": total_sec,
            "analyze_sec": analyze_sec,
            "solve_sec": solve_sec,
            "max_abs_v_delta_from_v0": float(parsed["max_abs_v_delta_from_v0"]),
            "buses": int(parsed["buses"]),
            "pv": int(parsed["pv"]),
            "pq": int(parsed["pq"]),
            **bench.flatten_summary(summary),
        }

        if backend == "cuda" and gpu_device is not None:
            row["gpu_device_requested"] = gpu_device

        rows.append(row)

        raw_path = run_root / "raw" / implementation / case_stem_value / f"run_{repeat_idx:02d}.json"
        write_json(
            raw_path,
            {
                "implementation": implementation,
                "summary": row,
                "info_lines": non_run_lines,
                "timing_entries": timing_entries,
            },
        )

    if len(rows) != repeats:
        raise RuntimeError(
            f"{implementation} for {case_name} produced {len(rows)} RUN lines, expected {repeats}."
        )

    return rows


def build_aggregate_map(aggregates: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    aggregate_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in aggregates:
        aggregate_map[(str(row["case_stem"]), str(row["implementation"]))] = row
    return aggregate_map


def write_run_readme(run_root: Path, manifest: dict[str, Any], aggregates: list[dict[str, Any]]) -> None:
    cpu_summary = manifest["environment"]["cpu"].get("summary", {})
    software = manifest["environment"]["software"]

    lines = [
        f"# Benchmark Run `{manifest['run_name']}`",
        "",
        bench.format_env_line("Created (UTC)", manifest["created_at_utc"]),
        bench.format_env_line("Cases", ", ".join(manifest["cases"])),
        bench.format_env_line("Implementations", ", ".join(manifest["implementations"])),
        bench.format_env_line("Warmup", manifest["warmup"]),
        bench.format_env_line("CPU repeats", manifest["cpu_repeats"]),
        bench.format_env_line("GPU repeats", manifest["gpu_repeats"]),
        bench.format_env_line("Requested GPU index", manifest["gpu_device"]),
        "",
        "## Environment",
        "",
        bench.format_env_line("OS", manifest["environment"]["os"].get("platform")),
        bench.format_env_line("CPU model", cpu_summary.get("cpu_model")),
        bench.format_env_line("Logical CPUs", cpu_summary.get("logical_cpus")),
        bench.format_env_line("Python", software["python"]["version"].splitlines()[0]),
        bench.format_env_line("NumPy", software["modules"].get("numpy")),
        bench.format_env_line("SciPy", software["modules"].get("scipy")),
        "",
        "## Files",
        "",
        "- `manifest.json`: full environment and command log",
        "- `SUMMARY.md`: compact comparison tables",
        "- `summary.csv`: one row per run",
        "- `aggregates.csv`: grouped statistics per implementation/case",
        "- `raw/`: per-run raw payloads",
        "",
        "## Aggregate Snapshot",
        "",
    ]

    if aggregates:
        lines.append("| implementation | case | runs | elapsed mean (s) | analyze mean (s) | solve mean (s) |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in aggregates:
            lines.append(
                f"| {row['implementation']} | {row['case_stem']} | {row['runs']} | "
                f"{bench.fmt_seconds(row.get('elapsed_sec_mean'))} | "
                f"{bench.fmt_seconds(row.get('analyze_sec_mean'))} | "
                f"{bench.fmt_seconds(row.get('solve_sec_mean'))} |"
            )
    else:
        lines.append("No aggregate rows were generated.")

    (run_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_summary_markdown(run_root: Path, manifest: dict[str, Any], aggregates: list[dict[str, Any]]) -> None:
    aggregate_map = build_aggregate_map(aggregates)

    def get(case_name: str, implementation: str) -> dict[str, Any]:
        return aggregate_map.get((case_stem(case_name), implementation), {})

    lines = [
        f"# Result Summary `{manifest['run_name']}`",
        "",
        "## Setup",
        "",
        bench.format_env_line("Created (UTC)", manifest["created_at_utc"]),
        bench.format_env_line("Cases", ", ".join(manifest["cases"])),
        bench.format_env_line("Warmup", manifest["warmup"]),
        bench.format_env_line("CPU repeats", manifest["cpu_repeats"]),
        bench.format_env_line("GPU repeats", manifest["gpu_repeats"]),
        bench.format_env_line("Requested GPU index", manifest["gpu_device"]),
        "",
        "## Elapsed Time",
        "",
        "| case | pypower (s) | cpu naive (s) | cpu optimized (s) | cuda edge (s) | cuda vertex (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for case_name in manifest["cases"]:
        pypower = get(case_name, "pypower")
        naive = get(case_name, "cpp_pypowerlike")
        optimized = get(case_name, "cpp_optimized")
        cuda_edge = get(case_name, "cpp_cuda_edge")
        cuda_vertex = get(case_name, "cpp_cuda_vertex")

        lines.append(
            f"| {case_stem(case_name)} | "
            f"{bench.fmt_seconds(pypower.get('elapsed_sec_mean'))} | "
            f"{bench.fmt_seconds(naive.get('elapsed_sec_mean'))} | "
            f"{bench.fmt_seconds(optimized.get('elapsed_sec_mean'))} | "
            f"{bench.fmt_seconds(cuda_edge.get('elapsed_sec_mean'))} | "
            f"{bench.fmt_seconds(cuda_vertex.get('elapsed_sec_mean'))} |"
        )

    lines.extend(
        [
            "",
            "## Speedup",
            "",
            "| case | cpu naive vs pypower | cpu optimized vs pypower | cuda edge vs pypower | cuda vertex vs pypower | cuda edge vs cpu optimized | cuda vertex vs cpu optimized |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for case_name in manifest["cases"]:
        pypower = get(case_name, "pypower")
        naive = get(case_name, "cpp_pypowerlike")
        optimized = get(case_name, "cpp_optimized")
        cuda_edge = get(case_name, "cpp_cuda_edge")
        cuda_vertex = get(case_name, "cpp_cuda_vertex")

        lines.append(
            f"| {case_stem(case_name)} | "
            f"{bench.fmt_speedup(naive.get('elapsed_sec_mean'), pypower.get('elapsed_sec_mean'))} | "
            f"{bench.fmt_speedup(optimized.get('elapsed_sec_mean'), pypower.get('elapsed_sec_mean'))} | "
            f"{bench.fmt_speedup(cuda_edge.get('elapsed_sec_mean'), pypower.get('elapsed_sec_mean'))} | "
            f"{bench.fmt_speedup(cuda_vertex.get('elapsed_sec_mean'), pypower.get('elapsed_sec_mean'))} | "
            f"{bench.fmt_speedup(cuda_edge.get('elapsed_sec_mean'), optimized.get('elapsed_sec_mean'))} | "
            f"{bench.fmt_speedup(cuda_vertex.get('elapsed_sec_mean'), optimized.get('elapsed_sec_mean'))} |"
        )

    lines.extend(
        [
            "",
            "## C++ Breakdown",
            "",
            "| case | cpu naive analyze (s) | cpu naive solve (s) | cpu optimized analyze (s) | cpu optimized solve (s) | cuda edge analyze (s) | cuda edge solve (s) | cuda vertex analyze (s) | cuda vertex solve (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for case_name in manifest["cases"]:
        naive = get(case_name, "cpp_pypowerlike")
        optimized = get(case_name, "cpp_optimized")
        cuda_edge = get(case_name, "cpp_cuda_edge")
        cuda_vertex = get(case_name, "cpp_cuda_vertex")

        lines.append(
            f"| {case_stem(case_name)} | "
            f"{bench.fmt_seconds(naive.get('analyze_sec_mean'))} | "
            f"{bench.fmt_seconds(naive.get('solve_sec_mean'))} | "
            f"{bench.fmt_seconds(optimized.get('analyze_sec_mean'))} | "
            f"{bench.fmt_seconds(optimized.get('solve_sec_mean'))} | "
            f"{bench.fmt_seconds(cuda_edge.get('analyze_sec_mean'))} | "
            f"{bench.fmt_seconds(cuda_edge.get('solve_sec_mean'))} | "
            f"{bench.fmt_seconds(cuda_vertex.get('analyze_sec_mean'))} | "
            f"{bench.fmt_seconds(cuda_vertex.get('solve_sec_mean'))} |"
        )

    lines.extend(
        [
            "",
            "## Correctness Snapshot",
            "",
            "| case | pypower success | cpu naive success | cpu optimized success | cuda edge success | cuda vertex success | pypower iter | cpu naive iter | cpu optimized iter | cuda edge iter | cuda vertex iter |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for case_name in manifest["cases"]:
        pypower = get(case_name, "pypower")
        naive = get(case_name, "cpp_pypowerlike")
        optimized = get(case_name, "cpp_optimized")
        cuda_edge = get(case_name, "cpp_cuda_edge")
        cuda_vertex = get(case_name, "cpp_cuda_vertex")

        lines.append(
            f"| {case_stem(case_name)} | "
            f"{bench.to_bool(pypower.get('success_all'))} | "
            f"{bench.to_bool(naive.get('success_all'))} | "
            f"{bench.to_bool(optimized.get('success_all'))} | "
            f"{bench.to_bool(cuda_edge.get('success_all'))} | "
            f"{bench.to_bool(cuda_vertex.get('success_all'))} | "
            f"{pypower.get('iterations_mean', 'n/a')} | "
            f"{naive.get('iterations_mean', 'n/a')} | "
            f"{optimized.get('iterations_mean', 'n/a')} | "
            f"{cuda_edge.get('iterations_mean', 'n/a')} | "
            f"{cuda_vertex.get('iterations_mean', 'n/a')} |"
        )

    (run_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.cpu_repeats <= 0:
        raise ValueError("cpu-repeats must be > 0")
    if args.gpu_repeats <= 0:
        raise ValueError("gpu-repeats must be > 0")
    if not args.benchmark_binary.exists():
        raise FileNotFoundError(f"Benchmark binary not found: {args.benchmark_binary}")

    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    args.dump_root.mkdir(parents=True, exist_ok=True)

    command_log: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    case_names = list(args.cases)

    for case_name in case_names:
        print(f"[case] preparing {case_name}", flush=True)
        case_data = preprocess_case(case_name)
        dump_dir = save_cupf_dump(case_data, output_root=args.dump_root)
        write_json(run_root / "cases" / f"{case_data.case_stem}.json", case_metadata(case_data))

        print(f"[run] pypower case={case_name} warmup={args.warmup} repeats={args.cpu_repeats}", flush=True)
        rows.extend(bench.run_python_case(case_name, run_root, args.warmup, args.cpu_repeats))

        for implementation in IMPLEMENTATIONS:
            repeats = args.cpu_repeats if implementation["repeat_kind"] == "cpu" else args.gpu_repeats
            print(
                f"[run] {implementation['implementation']} case={case_name} "
                f"backend={implementation['backend']} jacobian={implementation['jacobian']} "
                f"warmup={args.warmup} repeats={repeats}",
                flush=True,
            )
            rows.extend(
                run_cpp_case(
                    binary=args.benchmark_binary,
                    dump_dir=dump_dir,
                    case_name=case_name,
                    case_stem_value=case_data.case_stem,
                    implementation=implementation["implementation"],
                    backend=implementation["backend"],
                    algorithm=implementation["algorithm"],
                    jacobian=implementation["jacobian"],
                    warmup=args.warmup,
                    repeats=repeats,
                    tolerance=args.tolerance,
                    max_iter=args.max_iter,
                    run_root=run_root,
                    command_log=command_log,
                    gpu_device=args.gpu_device,
                )
            )

    aggregates = bench.aggregate_rows(rows)
    bench.write_csv(run_root / "summary.csv", rows)
    bench.write_csv(run_root / "aggregates.csv", aggregates)

    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "workspace_root": str(WORKSPACE_ROOT),
        "results_root": str(run_root),
        "dump_root": str(args.dump_root),
        "benchmark_binary": str(args.benchmark_binary),
        "cases": case_names,
        "warmup": args.warmup,
        "cpu_repeats": args.cpu_repeats,
        "gpu_repeats": args.gpu_repeats,
        "gpu_device": args.gpu_device,
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "implementations": [
            "pypower",
            "cpp_pypowerlike",
            "cpp_optimized",
            "cpp_cuda_edge",
            "cpp_cuda_vertex",
        ],
        "environment": {
            "os": bench.collect_os_info(),
            "cpu": bench.collect_cpu_info(),
            "gpu": collect_gpu_info(),
            "software": bench.collect_software_info(),
            "git": bench.collect_git_info(WORKSPACE_ROOT),
        },
        "commands": command_log,
    }
    write_json(run_root / "manifest.json", manifest)
    write_run_readme(run_root, manifest, aggregates)
    write_run_summary_markdown(run_root, manifest, aggregates)

    print(f"[OK] benchmark run created at {run_root}", flush=True)


if __name__ == "__main__":
    main()
