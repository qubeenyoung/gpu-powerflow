from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CUPF_ROOT = WORKSPACE_ROOT / "cuPF"
DEFAULT_RESULTS_ROOT = CUPF_ROOT / "benchmarks" / "results"
DEFAULT_DUMP_ROOT = WORKSPACE_ROOT / "datasets" / "cuPF_benchmark_dumps"
DEFAULT_BUILD_DIR = CUPF_ROOT / "build" / "bench-release"
DEFAULT_BENCHMARK_BINARY = DEFAULT_BUILD_DIR / "cupf_case_benchmark"
SCRIPT_PATH = Path(__file__).resolve()

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from python.converters.common import TARGET_CASES, case_metadata, case_stem, preprocess_case, save_cupf_dump, write_json
from python.pypower.runpf import my_runpf
from python.pypower.timer import summarize_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PYPOWER and cuPF CPU benchmarks with a unified result format.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--benchmark-binary", type=Path, default=DEFAULT_BENCHMARK_BINARY)
    parser.add_argument("--cases", nargs="*", default=list(TARGET_CASES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--jacobian", default="edge_based")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def maybe_run_command(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return run_command(cmd, cwd=cwd).stdout.strip()
    except Exception:
        return None


def flatten_summary(summary: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for key, values in summary.items():
        for metric_name, metric_value in values.items():
            flattened[f"{key}.{metric_name}"] = metric_value
    return flattened


def module_version(module_name: str) -> str | None:
    try:
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return None
        return getattr(module, "__version__", None)


def collect_os_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }

    os_release = Path("/etc/os-release")
    if os_release.exists():
        parsed: dict[str, str] = {}
        for line in os_release.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            parsed[key] = value.strip().strip('"')
        info["os_release"] = parsed

    return info


def collect_cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {}

    lscpu_json = maybe_run_command(["lscpu", "-J"])
    if lscpu_json:
        info["lscpu_json"] = json.loads(lscpu_json)
        selected: dict[str, str] = {}
        for entry in info["lscpu_json"].get("lscpu", []):
            field = entry.get("field", "").rstrip(":")
            data = entry.get("data", "")
            if field:
                selected[field] = data
        info["summary"] = {
            "architecture": selected.get("Architecture"),
            "cpu_op_modes": selected.get("CPU op-mode(s)"),
            "cpu_model": selected.get("Model name"),
            "vendor_id": selected.get("Vendor ID"),
            "logical_cpus": selected.get("CPU(s)"),
            "online_cpus": selected.get("On-line CPU(s) list"),
            "threads_per_core": selected.get("Thread(s) per core"),
            "cores_per_socket": selected.get("Core(s) per socket"),
            "sockets": selected.get("Socket(s)"),
            "cpu_max_mhz": selected.get("CPU max MHz"),
            "cpu_min_mhz": selected.get("CPU min MHz"),
        }

    mem_total_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    mem_total_kib = int(parts[1])
                break
    info["mem_total_kib"] = mem_total_kib

    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
            info["sched_getaffinity"] = affinity
            info["sched_getaffinity_count"] = len(affinity)
        except Exception:
            pass

    return info


def collect_software_info() -> dict[str, Any]:
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "modules": {
            "numpy": module_version("numpy"),
            "scipy": module_version("scipy"),
            "pypower": module_version("pypower"),
        },
        "tools": {
            "cmake": maybe_run_command(["cmake", "--version"]),
            "cxx": maybe_run_command(["c++", "--version"]),
        },
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
            if os.environ.get(key) is not None
        },
    }


def collect_git_info(repo_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"repo_root": str(repo_root)}
    info["branch"] = maybe_run_command(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    info["commit"] = maybe_run_command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    info["status_short"] = maybe_run_command(["git", "-C", str(repo_root), "status", "--short"])
    return info


def ensure_cpp_benchmark_binary(args: argparse.Namespace, command_log: list[dict[str, Any]]) -> Path:
    build_dir = args.build_dir
    binary = args.benchmark_binary
    if binary == DEFAULT_BENCHMARK_BINARY:
        binary = build_dir / "cupf_case_benchmark"

    if args.skip_build and binary.exists():
        return binary

    configure_cmd = [
        "cmake",
        "-S",
        str(CUPF_ROOT),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_NAIVE_CPU=ON",
        "-DBUILD_BENCHMARKS=ON",
        "-DWITH_CUDA=OFF",
    ]
    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "cupf_case_benchmark",
        "-j",
        str(os.cpu_count() or 1),
    ]

    command_log.append({"name": "cmake_configure", "cmd": configure_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(configure_cmd, cwd=WORKSPACE_ROOT)

    command_log.append({"name": "cmake_build", "cmd": build_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(build_cmd, cwd=WORKSPACE_ROOT)

    if not binary.exists():
        raise FileNotFoundError(f"Expected benchmark binary was not built: {binary}")

    return binary


def run_python_case(case_name: str,
                    run_root: Path,
                    warmup: int,
                    repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    implementation = "pypower"

    for _ in range(warmup):
        my_runpf(
            casedata=case_name,
            log_pf=True,
            log_newtonpf=True,
            print_results=False,
            emit_timing_log=False,
            emit_status=False,
        )

    for repeat_idx in range(repeats):
        result = my_runpf(
            casedata=case_name,
            log_pf=True,
            log_newtonpf=True,
            print_results=False,
            emit_timing_log=False,
            emit_status=False,
        )
        summary = summarize_entries(result.timing_entries)
        row = {
            "implementation": implementation,
            "case_name": case_name,
            "case_stem": result.case_stem,
            "repeat_idx": repeat_idx,
            "success": result.success,
            "iterations": result.iterations,
            "final_mismatch": result.final_mismatch,
            "elapsed_sec": result.elapsed_sec,
            **flatten_summary(summary),
        }
        rows.append(row)

        raw_path = run_root / "raw" / implementation / result.case_stem / f"run_{repeat_idx:02d}.json"
        write_json(
            raw_path,
            {
                "implementation": implementation,
                "case_name": case_name,
                "summary": row,
                "timing_entries": [
                    {
                        "tag": entry.tag,
                        "op_name": entry.op_name,
                        "iter_idx": entry.iter_idx,
                        "elapsed_sec": entry.elapsed_sec,
                    }
                    for entry in result.timing_entries
                ],
            },
        )

    return rows


def parse_cpp_run_line(line: str) -> dict[str, str]:
    if not line.startswith("RUN "):
        raise ValueError(f"Unexpected benchmark output line: {line}")
    parsed: dict[str, str] = {}
    for token in line[4:].strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def parse_cpp_metric_line(line: str) -> dict[str, str]:
    if not line.startswith("METRIC "):
        raise ValueError(f"Unexpected benchmark metric line: {line}")
    parsed: dict[str, str] = {}
    for token in line[7:].strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def metric_name_parts(name: str) -> tuple[str, str]:
    if "." not in name:
        return "timing", name
    tag, op_name = name.split(".", 1)
    return tag, op_name


def cpp_timing_summary(analyze_sec: float, solve_sec: float, total_sec: float) -> dict[str, dict[str, float | int]]:
    return {
        "benchmark.analyze": {"count": 1, "total_sec": analyze_sec, "avg_sec": analyze_sec},
        "benchmark.solve": {"count": 1, "total_sec": solve_sec, "avg_sec": solve_sec},
        "benchmark.total": {"count": 1, "total_sec": total_sec, "avg_sec": total_sec},
    }


def run_cpp_case(binary: Path,
                 dump_dir: Path,
                 case_name: str,
                 case_stem: str,
                 implementation: str,
                 algorithm: str,
                 jacobian: str,
                 warmup: int,
                 repeats: int,
                 tolerance: float,
                 max_iter: int,
                 run_root: Path,
                 command_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cmd = [
        str(binary),
        "--case-dir", str(dump_dir),
        "--backend", "cpu",
        "--jacobian", jacobian,
        "--algorithm", algorithm,
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--tolerance", str(tolerance),
        "--max-iter", str(max_iter),
    ]
    command_log.append({
        "name": f"run_{implementation}",
        "cmd": cmd,
        "cwd": str(WORKSPACE_ROOT),
    })
    completed = run_command(cmd, cwd=WORKSPACE_ROOT)

    parsed_runs: dict[int, dict[str, str]] = {}
    metric_summaries: dict[int, dict[str, dict[str, float | int]]] = {}
    metric_entries: dict[int, list[dict[str, Any]]] = {}

    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("RUN "):
            parsed = parse_cpp_run_line(line)
            parsed_runs[int(parsed["repeat"])] = parsed
            continue
        if line.startswith("METRIC "):
            parsed = parse_cpp_metric_line(line)
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
            tag, op_name = metric_name_parts(name)
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

    rows: list[dict[str, Any]] = []
    for repeat_idx in sorted(parsed_runs):
        parsed = parsed_runs[repeat_idx]
        analyze_sec = float(parsed["analyze_sec"])
        solve_sec = float(parsed["solve_sec"])
        total_sec = float(parsed["total_sec"])
        summary = metric_summaries.get(repeat_idx)
        if not summary:
            summary = cpp_timing_summary(analyze_sec, solve_sec, total_sec)

        timing_entries = metric_entries.get(repeat_idx)
        if timing_entries is None:
            timing_entries = [
                {"tag": "benchmark", "op_name": "analyze", "iter_idx": repeat_idx, "elapsed_sec": analyze_sec},
                {"tag": "benchmark", "op_name": "solve", "iter_idx": repeat_idx, "elapsed_sec": solve_sec},
                {"tag": "benchmark", "op_name": "total", "iter_idx": repeat_idx, "elapsed_sec": total_sec},
            ]

        row = {
            "implementation": implementation,
            "case_name": case_name,
            "case_stem": case_stem,
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
            **flatten_summary(summary),
        }
        rows.append(row)

        raw_path = run_root / "raw" / implementation / case_stem / f"run_{repeat_idx:02d}.json"
        write_json(
            raw_path,
            {
                "implementation": implementation,
                "summary": row,
                "timing_entries": timing_entries,
            },
        )

    return rows


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["implementation"]), str(row["case_stem"])), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (implementation, case_stem), group_rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "implementation": implementation,
            "case_stem": case_stem,
            "runs": len(group_rows),
            "success_all": all(bool(row["success"]) for row in group_rows),
            "iterations_mean": statistics.mean(numeric_values(group_rows, "iterations")),
            "final_mismatch_max": max(numeric_values(group_rows, "final_mismatch")),
        }

        for key in ("elapsed_sec", "analyze_sec", "solve_sec"):
            values = numeric_values(group_rows, key)
            if not values:
                continue
            aggregate[f"{key}_mean"] = statistics.mean(values)
            aggregate[f"{key}_median"] = statistics.median(values)
            aggregate[f"{key}_min"] = min(values)
            aggregate[f"{key}_max"] = max(values)
            aggregate[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0

        aggregates.append(aggregate)

    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_env_line(title: str, value: Any) -> str:
    return f"- {title}: {value if value is not None else 'n/a'}"


def write_run_readme(run_root: Path, manifest: dict[str, Any], aggregates: list[dict[str, Any]]) -> None:
    cpu_summary = manifest["environment"]["cpu"].get("summary", {})
    software = manifest["environment"]["software"]
    implementations = ", ".join(manifest["implementations"])

    lines = [
        f"# Benchmark Run `{manifest['run_name']}`",
        "",
        format_env_line("Created (UTC)", manifest["created_at_utc"]),
        format_env_line("Cases", ", ".join(manifest["cases"])),
        format_env_line("Implementations", implementations),
        format_env_line("Warmup", manifest["warmup"]),
        format_env_line("Repeats", manifest["repeats"]),
        "",
        "## Environment",
        "",
        format_env_line("OS", manifest["environment"]["os"].get("platform")),
        format_env_line("CPU model", cpu_summary.get("cpu_model")),
        format_env_line("Logical CPUs", cpu_summary.get("logical_cpus")),
        format_env_line("Threads per core", cpu_summary.get("threads_per_core")),
        format_env_line("Cores per socket", cpu_summary.get("cores_per_socket")),
        format_env_line("Sockets", cpu_summary.get("sockets")),
        format_env_line("Memory (KiB)", manifest["environment"]["cpu"].get("mem_total_kib")),
        format_env_line("Python", software["python"]["version"].splitlines()[0]),
        format_env_line("NumPy", software["modules"].get("numpy")),
        format_env_line("SciPy", software["modules"].get("scipy")),
        "",
        "## Files",
        "",
        "- `manifest.json`: full environment and command log",
        "- `SUMMARY.md`: benchmark result summary with speedup tables",
        "- `summary.csv`: one row per run",
        "- `aggregates.csv`: grouped statistics per implementation/case",
        "- `raw/`: per-run raw payloads",
        "",
        "## Aggregate Snapshot",
        "",
    ]

    if aggregates:
        lines.append("| implementation | case | elapsed mean (s) | analyze mean (s) | solve mean (s) |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in aggregates:
            lines.append(
                f"| {row['implementation']} | {row['case_stem']} | "
                f"{row.get('elapsed_sec_mean', 'n/a')} | "
                f"{row.get('analyze_sec_mean', 'n/a')} | "
                f"{row.get('solve_sec_mean', 'n/a')} |"
            )
    else:
        lines.append("No aggregate rows were generated.")

    (run_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def fmt_seconds(value: Any) -> str:
    parsed = to_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.6f}"


def fmt_speedup(numerator: Any, denominator: Any) -> str:
    num = to_float(numerator)
    den = to_float(denominator)
    if num is None or den is None or num == 0.0:
        return "n/a"
    return f"{den / num:.2f}x"


def write_run_summary_markdown(run_root: Path, manifest: dict[str, Any], aggregates: list[dict[str, Any]]) -> None:
    aggregate_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in aggregates:
        aggregate_map[(str(row["case_stem"]), str(row["implementation"]))] = row

    cpu_summary = manifest["environment"]["cpu"].get("summary", {})
    software = manifest["environment"]["software"]

    lines = [
        f"# Result Summary `{manifest['run_name']}`",
        "",
        "## Setup",
        "",
        format_env_line("Created (UTC)", manifest["created_at_utc"]),
        format_env_line("Cases", ", ".join(manifest["cases"])),
        format_env_line("Warmup", manifest["warmup"]),
        format_env_line("Repeats", manifest["repeats"]),
        format_env_line("CPU model", cpu_summary.get("cpu_model")),
        format_env_line("Logical CPUs", cpu_summary.get("logical_cpus")),
        format_env_line("Pinned thread env", manifest["environment"]["software"].get("thread_env") or "not set"),
        format_env_line("Python", software["python"]["version"].splitlines()[0]),
        format_env_line("NumPy", software["modules"].get("numpy")),
        format_env_line("SciPy", software["modules"].get("scipy")),
        "",
        "## Elapsed Time",
        "",
        "| case | pypower (s) | cpp_pypowerlike (s) | cpp_optimized (s) | pypowerlike speedup vs pypower | optimized speedup vs pypower | optimized speedup vs pypowerlike |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for case_name in manifest["cases"]:
        stem = case_stem(case_name)
        pypower = aggregate_map.get((stem, "pypower"), {})
        naive = aggregate_map.get((stem, "cpp_pypowerlike"), {})
        optimized = aggregate_map.get((stem, "cpp_optimized"), {})

        lines.append(
            f"| {stem} | "
            f"{fmt_seconds(pypower.get('elapsed_sec_mean'))} | "
            f"{fmt_seconds(naive.get('elapsed_sec_mean'))} | "
            f"{fmt_seconds(optimized.get('elapsed_sec_mean'))} | "
            f"{fmt_speedup(naive.get('elapsed_sec_mean'), pypower.get('elapsed_sec_mean'))} | "
            f"{fmt_speedup(optimized.get('elapsed_sec_mean'), pypower.get('elapsed_sec_mean'))} | "
            f"{fmt_speedup(optimized.get('elapsed_sec_mean'), naive.get('elapsed_sec_mean'))} |"
        )

    lines.extend([
        "",
        "## C++ Breakdown",
        "",
        "| case | cpp_pypowerlike analyze (s) | cpp_pypowerlike solve (s) | cpp_optimized analyze (s) | cpp_optimized solve (s) |",
        "|---|---:|---:|---:|---:|",
    ])

    for case_name in manifest["cases"]:
        stem = case_stem(case_name)
        naive = aggregate_map.get((stem, "cpp_pypowerlike"), {})
        optimized = aggregate_map.get((stem, "cpp_optimized"), {})
        lines.append(
            f"| {stem} | "
            f"{fmt_seconds(naive.get('analyze_sec_mean'))} | "
            f"{fmt_seconds(naive.get('solve_sec_mean'))} | "
            f"{fmt_seconds(optimized.get('analyze_sec_mean'))} | "
            f"{fmt_seconds(optimized.get('solve_sec_mean'))} |"
        )

    lines.extend([
        "",
        "## Correctness Snapshot",
        "",
        "| case | pypower success | cpp_pypowerlike success | cpp_optimized success | pypower iterations | cpp_pypowerlike iterations | cpp_optimized iterations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for case_name in manifest["cases"]:
        stem = case_stem(case_name)
        pypower = aggregate_map.get((stem, "pypower"), {})
        naive = aggregate_map.get((stem, "cpp_pypowerlike"), {})
        optimized = aggregate_map.get((stem, "cpp_optimized"), {})
        lines.append(
            f"| {stem} | "
            f"{to_bool(pypower.get('success_all'))} | "
            f"{to_bool(naive.get('success_all'))} | "
            f"{to_bool(optimized.get('success_all'))} | "
            f"{pypower.get('iterations_mean', 'n/a')} | "
            f"{naive.get('iterations_mean', 'n/a')} | "
            f"{optimized.get('iterations_mean', 'n/a')} |"
        )

    lines.extend([
        "",
        "Raw data lives in `summary.csv`, `aggregates.csv`, and `raw/`.",
    ])

    (run_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")

    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    args.dump_root.mkdir(parents=True, exist_ok=True)

    command_log: list[dict[str, Any]] = []
    cpp_binary = ensure_cpp_benchmark_binary(args, command_log)

    rows: list[dict[str, Any]] = []
    case_names = list(args.cases)

    for case_name in case_names:
        case_data = preprocess_case(case_name)
        dump_dir = save_cupf_dump(case_data, output_root=args.dump_root)
        write_json(run_root / "cases" / f"{case_data.case_stem}.json", case_metadata(case_data))

        rows.extend(run_python_case(case_name, run_root, args.warmup, args.repeats))
        rows.extend(
            run_cpp_case(
                cpp_binary,
                dump_dir,
                case_name,
                case_data.case_stem,
                "cpp_pypowerlike",
                "pypower_like",
                args.jacobian,
                args.warmup,
                args.repeats,
                args.tolerance,
                args.max_iter,
                run_root,
                command_log,
            )
        )
        rows.extend(
            run_cpp_case(
                cpp_binary,
                dump_dir,
                case_name,
                case_data.case_stem,
                "cpp_optimized",
                "optimized",
                args.jacobian,
                args.warmup,
                args.repeats,
                args.tolerance,
                args.max_iter,
                run_root,
                command_log,
            )
        )

    aggregates = aggregate_rows(rows)
    write_csv(run_root / "summary.csv", rows)
    write_csv(run_root / "aggregates.csv", aggregates)

    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(SCRIPT_PATH),
        "workspace_root": str(WORKSPACE_ROOT),
        "results_root": str(run_root),
        "dump_root": str(args.dump_root),
        "benchmark_binary": str(cpp_binary),
        "cases": case_names,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "jacobian": args.jacobian,
        "implementations": ["pypower", "cpp_pypowerlike", "cpp_optimized"],
        "environment": {
            "os": collect_os_info(),
            "cpu": collect_cpu_info(),
            "software": collect_software_info(),
            "git": collect_git_info(WORKSPACE_ROOT),
        },
        "commands": command_log,
    }
    write_json(run_root / "manifest.json", manifest)
    write_run_readme(run_root, manifest, aggregates)
    write_run_summary_markdown(run_root, manifest, aggregates)

    print(f"[OK] benchmark run created at {run_root}")


if __name__ == "__main__":
    main()
