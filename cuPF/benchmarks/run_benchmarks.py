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
import platform
import statistics
import subprocess
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
CUPF_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = CUPF_ROOT.parent

DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "cuPF_benchmark_dumps"
DEFAULT_RESULTS_ROOT = CUPF_ROOT / "benchmarks" / "results"
DEFAULT_END2END_BUILD_DIR = CUPF_ROOT / "build" / "bench-end2end"
DEFAULT_OPERATORS_BUILD_DIR = CUPF_ROOT / "build" / "bench-operators"
DEFAULT_PROFILES = ["pypower", "cpp_naive", "cpp", "cuda_edge", "cuda_vertex"]
MEASUREMENT_MODES = ("end2end", "operators")


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    runner: str
    cupf_profile: str | None = None
    implementation: str | None = None
    backend: str | None = None
    compute: str | None = None
    jacobian: str | None = None


PROFILE_SPECS: dict[str, ProfileSpec] = {
    "pypower": ProfileSpec("pypower", "pypower", implementation="pypower", backend="python", compute="fp64", jacobian="pypower"),
    "cpp_naive": ProfileSpec("cpp_naive", "cupf", "cpp_pypowerlike", "cpp_naive", "cpu", "fp64", "pypower_like"),
    "cpp_pypowerlike": ProfileSpec("cpp_naive", "cupf", "cpp_pypowerlike", "cpp_naive", "cpu", "fp64", "pypower_like"),
    "cpp": ProfileSpec("cpp", "cupf", "cpu_fp64_edge", "cpp", "cpu", "fp64", "edge_based"),
    "cpu_fp64_edge": ProfileSpec("cpp", "cupf", "cpu_fp64_edge", "cpp", "cpu", "fp64", "edge_based"),
    "cuda_edge": ProfileSpec("cuda_edge", "cupf", "cuda_mixed_edge", "cuda_edge", "cuda", "mixed", "edge_based"),
    "cuda_mixed_edge": ProfileSpec("cuda_edge", "cupf", "cuda_mixed_edge", "cuda_edge", "cuda", "mixed", "edge_based"),
    "cuda_edge_modified": ProfileSpec("cuda_edge_modified", "cupf", "cuda_mixed_edge_modified", "cuda_edge_modified", "cuda", "mixed", "edge_based"),
    "cuda_mixed_edge_modified": ProfileSpec("cuda_edge_modified", "cupf", "cuda_mixed_edge_modified", "cuda_edge_modified", "cuda", "mixed", "edge_based"),
    "cuda_vertex": ProfileSpec("cuda_vertex", "cupf", "cuda_mixed_vertex", "cuda_vertex", "cuda", "mixed", "vertex_based"),
    "cuda_mixed_vertex": ProfileSpec("cuda_vertex", "cupf", "cuda_mixed_vertex", "cuda_vertex", "cuda", "mixed", "vertex_based"),
    "cuda_vertex_modified": ProfileSpec("cuda_vertex_modified", "cupf", "cuda_mixed_vertex_modified", "cuda_vertex_modified", "cuda", "mixed", "vertex_based"),
    "cuda_mixed_vertex_modified": ProfileSpec("cuda_vertex_modified", "cupf", "cuda_mixed_vertex_modified", "cuda_vertex_modified", "cuda", "mixed", "vertex_based"),
    "cuda_wo_jacobian": ProfileSpec("cuda_wo_jacobian", "cupf", "cuda_mixed_edge_cpu_naive_jacobian", "cuda_wo_jacobian", "cuda", "mixed", "cpu_naive_pypower_like"),
    "cuda_mixed_edge_cpu_naive_jacobian": ProfileSpec("cuda_wo_jacobian", "cupf", "cuda_mixed_edge_cpu_naive_jacobian", "cuda_wo_jacobian", "cuda", "mixed", "cpu_naive_pypower_like"),
    "cuda_wo_cudss": ProfileSpec("cuda_wo_cudss", "cupf", "cuda_mixed_edge_cpu_superlu", "cuda_wo_cudss", "cuda", "mixed", "edge_based"),
    "cuda_mixed_edge_cpu_superlu": ProfileSpec("cuda_wo_cudss", "cupf", "cuda_mixed_edge_cpu_superlu", "cuda_wo_cudss", "cuda", "mixed", "edge_based"),
    "cuda_fp64_edge": ProfileSpec("cuda_fp64_edge", "cupf", "cuda_fp64_edge", "cuda_fp64_edge", "cuda", "fp64", "edge_based"),
    "cuda_fp64_vertex": ProfileSpec("cuda_fp64_vertex", "cupf", "cuda_fp64_vertex", "cuda_fp64_vertex", "cuda", "fp64", "vertex_based"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cuPF benchmark profiles over dump datasets.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--mode", choices=("end2end", "operators", "both"), default="both")
    parser.add_argument("--build-dir", type=Path, help="Compatibility shortcut for a single selected mode.")
    parser.add_argument("--end2end-build-dir", type=Path, default=DEFAULT_END2END_BUILD_DIR)
    parser.add_argument("--operators-build-dir", type=Path, default=DEFAULT_OPERATORS_BUILD_DIR)
    parser.add_argument("--benchmark-binary", type=Path, help="Compatibility shortcut for a single selected mode.")
    parser.add_argument("--end2end-binary", type=Path)
    parser.add_argument("--operators-binary", type=Path)
    parser.add_argument("--cases", nargs="*", default=["case30_ieee"])
    parser.add_argument("--case-list", type=Path, help="Text file with one case name per line.")
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--with-cuda", action="store_true", help="Force CUDA build.")
    parser.add_argument(
        "--cudss-reordering-alg",
        choices=("DEFAULT", "ALG_1", "ALG_2"),
        default="DEFAULT",
        help="cuDSS reordering algorithm used for CUDA benchmark builds.",
    )
    parser.add_argument("--cudss-enable-mt", action="store_true", help="Enable cuDSS multi-threaded mode.")
    parser.add_argument(
        "--cudss-host-nthreads",
        default="AUTO",
        help="cuDSS CUDSS_CONFIG_HOST_NTHREADS value: AUTO or an integer >= 1.",
    )
    parser.add_argument(
        "--cudss-threading-lib",
        type=Path,
        help="Threading layer library for cuDSS MT mode. Used as CUDSS_THREADING_LIB.",
    )
    parser.add_argument(
        "--cudss-nd-nlevels",
        default="AUTO",
        help="cuDSS CUDSS_CONFIG_ND_NLEVELS value: AUTO or an integer >= 0.",
    )
    parser.add_argument(
        "--cudss-use-matching",
        "--cudss-matching",
        dest="cudss_use_matching",
        action="store_true",
        help="Enable cuDSS CUDSS_CONFIG_USE_MATCHING.",
    )
    parser.add_argument(
        "--cudss-matching-alg",
        choices=("DEFAULT", "ALG_1", "ALG_2", "ALG_3", "ALG_4", "ALG_5"),
        default="DEFAULT",
        help="cuDSS CUDSS_CONFIG_MATCHING_ALG value used when matching is enabled.",
    )
    parser.add_argument(
        "--cudss-pivot-epsilon",
        "--cudss-epsilon",
        dest="cudss_pivot_epsilon",
        default="AUTO",
        help="cuDSS CUDSS_CONFIG_PIVOT_EPSILON value: AUTO or a non-negative float.",
    )
    parser.add_argument("--enable-dump", action="store_true", help="Build benchmark with dump utilities enabled.")
    parser.add_argument(
        "--dump-residuals",
        "--dump-newton-diagnostics",
        dest="dump_residuals",
        action="store_true",
        help="Dump per-iteration residual vectors plus cuDSS linear-system diagnostics.",
    )
    parser.add_argument(
        "--residual-dump-root",
        type=Path,
        help="Root directory for residual dumps. Defaults to <run-root>/residuals.",
    )
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str],
                *,
                cwd: Path | None = None,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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


def read_case_list(path: Path) -> list[str]:
    cases: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cases.append(stripped)
    return cases


def validate_auto_or_int(value: str, *, name: str, minimum: int) -> str:
    normalized = value.upper()
    if normalized == "AUTO":
        return "AUTO"
    if not value.isdigit():
        raise ValueError(f"{name} must be AUTO or an integer >= {minimum}: {value}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be AUTO or an integer >= {minimum}: {value}")
    return value


def validate_auto_or_nonnegative_float(value: str, *, name: str) -> str:
    normalized = value.upper()
    if normalized == "AUTO":
        return "AUTO"
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be AUTO or a non-negative float: {value}") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be AUTO or a non-negative float: {value}")
    return value


def benchmark_env(args: argparse.Namespace) -> dict[str, str] | None:
    if not args.cudss_enable_mt:
        return None

    env = os.environ.copy()
    threading_lib = args.cudss_threading_lib
    if threading_lib is not None:
        env["CUDSS_THREADING_LIB"] = str(threading_lib)
    elif not env.get("CUDSS_THREADING_LIB"):
        raise ValueError("--cudss-enable-mt requires --cudss-threading-lib or CUDSS_THREADING_LIB")
    threading_lib_path = env["CUDSS_THREADING_LIB"]
    ld_preload = env.get("LD_PRELOAD", "")
    preload_entries = [entry for entry in ld_preload.split(":") if entry]
    if threading_lib_path not in preload_entries:
        env["LD_PRELOAD"] = ":".join([threading_lib_path, *preload_entries])
    return env


def selected_modes(mode: str) -> list[str]:
    if mode == "both":
        return list(MEASUREMENT_MODES)
    return [mode]


def resolve_profiles(profile_names: list[str]) -> list[ProfileSpec]:
    specs: list[ProfileSpec] = []
    seen: set[str] = set()
    for profile_name in profile_names:
        spec = PROFILE_SPECS.get(profile_name)
        if spec is None:
            raise ValueError(f"Unknown benchmark profile: {profile_name}")
        if spec.name in seen:
            continue
        specs.append(spec)
        seen.add(spec.name)
    return specs


def cupf_profiles(profile_specs: list[ProfileSpec]) -> list[ProfileSpec]:
    return [spec for spec in profile_specs if spec.runner == "cupf"]


def profile_requires_cuda(profile: ProfileSpec) -> bool:
    return bool(profile.cupf_profile and profile.cupf_profile.startswith("cuda_"))


def build_dir_for_mode(args: argparse.Namespace, mode: str) -> Path:
    if args.build_dir is not None:
        if args.mode == "both":
            raise ValueError("--build-dir can only be used when --mode is end2end or operators")
        return args.build_dir
    if mode == "end2end":
        return args.end2end_build_dir
    if mode == "operators":
        return args.operators_build_dir
    raise ValueError(f"Unknown measurement mode: {mode}")


def binary_for_mode(args: argparse.Namespace, mode: str, build_dir: Path) -> Path:
    if args.benchmark_binary is not None:
        if args.mode == "both":
            raise ValueError("--benchmark-binary can only be used when --mode is end2end or operators")
        return args.benchmark_binary
    if mode == "end2end" and args.end2end_binary is not None:
        return args.end2end_binary
    if mode == "operators" and args.operators_binary is not None:
        return args.operators_binary
    return build_dir / "benchmarks" / "cupf_case_benchmark"


def ensure_benchmark_binary(mode: str,
                            args: argparse.Namespace,
                            profile_specs: list[ProfileSpec],
                            command_log: list[dict[str, Any]]) -> Path | None:
    if not cupf_profiles(profile_specs):
        return None

    build_dir = build_dir_for_mode(args, mode)
    binary = binary_for_mode(args, mode, build_dir)

    if args.skip_build:
        if binary.exists():
            return binary
        raise FileNotFoundError(f"Benchmark binary does not exist and --skip-build was set: {binary}")

    with_cuda = args.with_cuda or any(profile_requires_cuda(profile) for profile in cupf_profiles(profile_specs))
    timing_enabled = mode == "operators"
    configure_cmd = [
        "cmake",
        "-S", str(CUPF_ROOT),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DWITH_CUDA={'ON' if with_cuda else 'OFF'}",
        "-DENABLE_LOG=OFF",
        f"-DENABLE_DUMP={'ON' if (args.enable_dump or args.dump_residuals) else 'OFF'}",
        f"-DENABLE_TIMING={'ON' if timing_enabled else 'OFF'}",
        f"-DENABLE_NVTX={'ON' if timing_enabled else 'OFF'}",
        f"-DCUPF_CUDSS_REORDERING_ALG={args.cudss_reordering_alg}",
        f"-DCUPF_CUDSS_ENABLE_MT={'ON' if args.cudss_enable_mt else 'OFF'}",
        f"-DCUPF_CUDSS_HOST_NTHREADS={args.cudss_host_nthreads}",
        f"-DCUPF_CUDSS_ND_NLEVELS={args.cudss_nd_nlevels}",
        "-DBUILD_BENCHMARKS=ON",
        "-DBUILD_TESTING=OFF",
        "-DBUILD_PYTHON_BINDINGS=OFF",
    ]
    build_cmd = [
        "cmake",
        "--build", str(build_dir),
        "--target", "cupf_case_benchmark",
        "-j", str(os.cpu_count() or 1),
    ]

    command_log.append({"name": f"cmake_configure_{mode}", "cmd": configure_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(configure_cmd, cwd=WORKSPACE_ROOT)
    command_log.append({"name": f"cmake_build_{mode}", "cmd": build_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(build_cmd, cwd=WORKSPACE_ROOT)

    if not binary.exists():
        raise FileNotFoundError(f"Benchmark binary was not built: {binary}")
    return binary


def parse_key_value_line(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise ValueError(f"Expected {prefix!r} line, got: {line}")
    parsed: dict[str, str] = {}
    for token in line[len(prefix):].strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def flatten_metrics(metrics: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for name, values in metrics.items():
        for metric, value in values.items():
            flattened[f"{name}.{metric}"] = value
    return flattened


def run_cupf_profile_case(binary: Path,
                          dataset_root: Path,
                          run_root: Path,
                          measurement_mode: str,
                          case_name: str,
                          profile: ProfileSpec,
                          warmup: int,
                          repeats: int,
                          batch_size: int,
                          tolerance: float,
                          max_iter: int,
                          command_log: list[dict[str, Any]],
                          dump_residuals: bool = False,
                          residual_dump_root: Path | None = None,
                          cudss_use_matching: bool = False,
                          cudss_matching_alg: str = "DEFAULT",
                          cudss_pivot_epsilon: str = "AUTO",
                          env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    if profile.cupf_profile is None:
        raise ValueError(f"Profile is not a cuPF profile: {profile.name}")

    case_dir = dataset_root / case_name
    if not case_dir.exists():
        raise FileNotFoundError(f"Dataset case not found: {case_dir}")

    cmd = [
        str(binary),
        "--case-dir", str(case_dir),
        "--profile", profile.cupf_profile,
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--batch-size", str(batch_size),
        "--tolerance", str(tolerance),
        "--max-iter", str(max_iter),
    ]
    if cudss_use_matching:
        cmd.append("--cudss-use-matching")
    cmd.extend([
        "--cudss-matching-alg", cudss_matching_alg,
        "--cudss-pivot-epsilon", cudss_pivot_epsilon,
    ])
    if dump_residuals:
        if residual_dump_root is None:
            raise ValueError("residual_dump_root must be provided when dump_residuals is enabled")
        dump_dir = residual_dump_root / measurement_mode / profile.name / case_name
        cmd.extend(["--dump-residuals", "--dump-dir", str(dump_dir)])
    command_entry: dict[str, Any] = {
        "name": f"run_{measurement_mode}_{profile.name}_{case_name}",
        "cmd": cmd,
        "cwd": str(WORKSPACE_ROOT),
    }
    if env is not None:
        env_entry = {}
        for key in ("CUDSS_THREADING_LIB", "LD_PRELOAD"):
            if key in env:
                env_entry[key] = env[key]
        if env_entry:
            command_entry["env"] = env_entry
    command_log.append(command_entry)
    completed = run_command(cmd, cwd=WORKSPACE_ROOT, env=env)

    parsed_runs: dict[int, dict[str, str]] = {}
    metric_summaries: dict[int, dict[str, dict[str, float | int]]] = {}
    metric_entries: dict[int, list[dict[str, Any]]] = {}

    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("RUN "):
            parsed = parse_key_value_line(line, "RUN ")
            parsed_runs[int(parsed["repeat"])] = parsed
        elif line.startswith("METRIC "):
            parsed = parse_key_value_line(line, "METRIC ")
            repeat = int(parsed["repeat"])
            name = parsed["name"]
            count = int(parsed["count"])
            total_sec = float(parsed["total_sec"])
            avg_sec = float(parsed["avg_sec"])
            metric_summaries.setdefault(repeat, {})[name] = {
                "count": count,
                "total_sec": total_sec,
                "avg_sec": avg_sec,
            }
            metric_entries.setdefault(repeat, []).append({
                "name": name,
                "count": count,
                "total_sec": total_sec,
                "avg_sec": avg_sec,
            })

    rows: list[dict[str, Any]] = []
    for repeat in sorted(parsed_runs):
        parsed = parsed_runs[repeat]
        metrics = metric_summaries.get(repeat, {})
        row = {
            "measurement_mode": measurement_mode,
            "case_name": case_name,
            "dump_case_name": parsed["case"],
            "profile": profile.name,
            "source_profile": parsed["profile"],
            "implementation": profile.implementation or parsed["implementation"],
            "backend": profile.backend or parsed["backend"],
            "compute": profile.compute or parsed["compute"],
            "jacobian": profile.jacobian or parsed["jacobian"],
            "repeat_idx": repeat,
            "batch_size": int(parsed.get("batch_size", batch_size)),
            "success": parsed["success"] == "true",
            "iterations": int(parsed["iterations"]),
            "final_mismatch": float(parsed["final_mismatch"]),
            "elapsed_sec": float(parsed["total_sec"]),
            "analyze_sec": float(parsed["analyze_sec"]),
            "solve_sec": float(parsed["solve_sec"]),
            "max_abs_v_delta_from_v0": float(parsed["max_abs_v_delta_from_v0"]),
            "buses": int(parsed["buses"]),
            "pv": int(parsed["pv"]),
            "pq": int(parsed["pq"]),
            **flatten_metrics(metrics),
        }
        rows.append(row)

        raw_path = run_root / "raw" / measurement_mode / profile.name / case_name / f"run_{repeat:02d}.json"
        write_json(raw_path, {
            "summary": row,
            "timing_entries": metric_entries.get(repeat, []),
        })

    if len(rows) != repeats:
        raise RuntimeError(f"Expected {repeats} runs for {measurement_mode}/{profile.name}/{case_name}, got {len(rows)}")
    return rows


def import_pypower_helpers():
    workspace_root = str(WORKSPACE_ROOT)
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)

    from python.converters.common import case_stem  # pylint: disable=import-outside-toplevel
    from python.pypower.runpf import my_runpf  # pylint: disable=import-outside-toplevel
    from python.pypower.timer import summarize_entries  # pylint: disable=import-outside-toplevel

    return case_stem, my_runpf, summarize_entries


def run_pypower_case(run_root: Path,
                     measurement_mode: str,
                     case_name: str,
                     profile: ProfileSpec,
                     warmup: int,
                     repeats: int,
                     command_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_stem, my_runpf, summarize_entries = import_pypower_helpers()
    resolved_case_stem = case_stem(case_name)
    collect_operator_timing = measurement_mode == "operators"

    command_log.append({
        "name": f"run_{measurement_mode}_{profile.name}_{case_name}",
        "cmd": [
            "python.pypower.runpf",
            case_name,
            f"warmup={warmup}",
            f"repeats={repeats}",
            f"operator_timing={collect_operator_timing}",
        ],
        "cwd": str(WORKSPACE_ROOT),
    })

    for _ in range(warmup):
        my_runpf(
            casedata=case_name,
            log_pf=collect_operator_timing,
            log_newtonpf=collect_operator_timing,
            print_results=False,
            emit_timing_log=False,
            emit_status=False,
        )

    rows: list[dict[str, Any]] = []
    for repeat_idx in range(repeats):
        result = my_runpf(
            casedata=case_name,
            log_pf=collect_operator_timing,
            log_newtonpf=collect_operator_timing,
            print_results=False,
            emit_timing_log=False,
            emit_status=False,
        )
        metric_summaries = summarize_entries(result.timing_entries) if collect_operator_timing else {}
        solve_sec = metric_summaries.get("runpf.newtonpf", {}).get("total_sec", "")

        row = {
            "measurement_mode": measurement_mode,
            "case_name": case_name,
            "dump_case_name": "",
            "case_stem": resolved_case_stem,
            "profile": profile.name,
            "source_profile": "pypower",
            "implementation": profile.implementation or "pypower",
            "backend": profile.backend or "python",
            "compute": profile.compute or "fp64",
            "jacobian": profile.jacobian or "pypower",
            "repeat_idx": repeat_idx,
            "batch_size": 1,
            "success": result.success,
            "iterations": result.iterations,
            "final_mismatch": result.final_mismatch,
            "elapsed_sec": result.elapsed_sec,
            "analyze_sec": "",
            "solve_sec": solve_sec,
            "max_abs_v_delta_from_v0": "",
            "buses": "",
            "pv": "",
            "pq": "",
            **flatten_metrics(metric_summaries),
        }
        rows.append(row)

        raw_path = run_root / "raw" / measurement_mode / profile.name / case_name / f"run_{repeat_idx:02d}.json"
        write_json(raw_path, {
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
        })

    return rows


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def add_numeric_aggregate(aggregate: dict[str, Any], group: list[dict[str, Any]], key: str) -> None:
    values = numeric_values(group, key)
    if not values:
        aggregate[f"{key}_mean"] = ""
        aggregate[f"{key}_median"] = ""
        aggregate[f"{key}_min"] = ""
        aggregate[f"{key}_max"] = ""
        aggregate[f"{key}_stdev"] = ""
        return

    aggregate[f"{key}_mean"] = statistics.mean(values)
    aggregate[f"{key}_median"] = statistics.median(values)
    aggregate[f"{key}_min"] = min(values)
    aggregate[f"{key}_max"] = max(values)
    aggregate[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["measurement_mode"]), str(row["profile"]), str(row["case_name"])), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (measurement_mode, profile, case_name), group in sorted(grouped.items()):
        iterations = numeric_values(group, "iterations")
        final_mismatches = numeric_values(group, "final_mismatch")
        aggregate: dict[str, Any] = {
            "measurement_mode": measurement_mode,
            "profile": profile,
            "case_name": case_name,
            "source_profile": group[0].get("source_profile", ""),
            "implementation": group[0]["implementation"],
            "backend": group[0]["backend"],
            "compute": group[0]["compute"],
            "jacobian": group[0]["jacobian"],
            "batch_size": group[0].get("batch_size", 1),
            "runs": len(group),
            "success_all": all(bool(row["success"]) for row in group),
            "iterations_mean": statistics.mean(iterations) if iterations else "",
            "final_mismatch_max": max(final_mismatches) if final_mismatches else "",
        }
        for key in ("elapsed_sec", "analyze_sec", "solve_sec"):
            add_numeric_aggregate(aggregate, group, key)
        aggregates.append(aggregate)
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_environment() -> dict[str, Any]:
    return {
        "created_on": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cmake": maybe_run_command(["cmake", "--version"]),
        "cxx": maybe_run_command(["c++", "--version"]),
        "nvcc": maybe_run_command(["nvcc", "--version"]),
        "nvidia_smi": maybe_run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "git": {
            "branch": maybe_run_command(["git", "-C", str(CUPF_ROOT), "rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": maybe_run_command(["git", "-C", str(CUPF_ROOT), "rev-parse", "HEAD"]),
            "status_short": maybe_run_command(["git", "-C", str(CUPF_ROOT), "status", "--short"]),
        },
    }


def fmt_seconds(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_speedup(baseline: Any, candidate: Any) -> str:
    try:
        base = float(baseline)
        cand = float(candidate)
    except (TypeError, ValueError):
        return "n/a"
    if cand == 0.0:
        return "n/a"
    return f"{base / cand:.2f}x"


def write_summary_markdown(run_root: Path, manifest: dict[str, Any], aggregates: list[dict[str, Any]]) -> None:
    by_case_profile = {
        (str(row["measurement_mode"]), str(row["case_name"]), str(row["profile"])): row
        for row in aggregates
    }
    baseline_profile = "cpp" if "cpp" in manifest["profiles"] else "cpu_fp64_edge"
    cuda_profiles = [profile for profile in manifest["profiles"] if str(profile).startswith("cuda")]
    nsight_profile = cuda_profiles[0] if cuda_profiles else "cuda_mixed_vertex"
    nsight_case = "case118_ieee" if "case118_ieee" in manifest["cases"] else manifest["cases"][0]
    nsight_case_dir = Path(str(manifest["dataset_root"])) / nsight_case
    benchmark_binaries = manifest.get("benchmark_binaries", {})
    operators_binary = benchmark_binaries.get("operators") or benchmark_binaries.get("end2end") or "cupf_case_benchmark"
    cupf_profile_by_name = {
        str(spec["name"]): spec.get("cupf_profile")
        for spec in manifest.get("profile_specs", [])
        if spec.get("cupf_profile")
    }
    nsight_cupf_profile = cupf_profile_by_name.get(str(nsight_profile), nsight_profile)

    lines = [
        f"# cuPF Benchmark `{manifest['run_name']}`",
        "",
        "## Setup",
        "",
        f"- Created UTC: {manifest['created_at_utc']}",
        f"- Dataset root: `{manifest['dataset_root']}`",
        f"- Cases: {', '.join(manifest['cases'])}",
        f"- Profiles: {', '.join(manifest['profiles'])}",
        f"- Measurement modes: {', '.join(manifest['measurement_modes'])}",
        f"- Warmup: {manifest['warmup']}",
        f"- Repeats: {manifest['repeats']}",
        f"- Batch size: {manifest['batch_size']}",
        f"- cuDSS reordering algorithm: {manifest['cudss_reordering_alg']}",
        f"- cuDSS MT mode: {manifest['cudss_enable_mt']}",
        f"- cuDSS host threads: {manifest['cudss_host_nthreads']}",
        f"- cuDSS ND_NLEVELS: {manifest['cudss_nd_nlevels']}",
        f"- cuDSS matching: {manifest['cudss_use_matching']}",
        f"- cuDSS matching algorithm: {manifest['cudss_matching_alg']}",
        f"- cuDSS pivot epsilon: {manifest['cudss_pivot_epsilon']}",
        f"- cuDSS LD_PRELOAD: {manifest.get('cudss_ld_preload', '')}",
        "",
    ]

    for measurement_mode in manifest["measurement_modes"]:
        lines.extend([
            f"## {measurement_mode} aggregate timing",
            "",
            f"| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs {baseline_profile} | iterations mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for case_name in manifest["cases"]:
            baseline = by_case_profile.get((measurement_mode, case_name, baseline_profile), {})
            baseline_elapsed = baseline.get("elapsed_sec_mean")
            for profile in manifest["profiles"]:
                row = by_case_profile.get((measurement_mode, case_name, profile), {})
                lines.append(
                    f"| {case_name} | {profile} | {row.get('success_all', 'n/a')} | "
                    f"{fmt_seconds(row.get('elapsed_sec_mean'))} | "
                    f"{fmt_seconds(row.get('analyze_sec_mean'))} | "
                    f"{fmt_seconds(row.get('solve_sec_mean'))} | "
                    f"{fmt_speedup(baseline_elapsed, row.get('elapsed_sec_mean'))} | "
                    f"{row.get('iterations_mean', 'n/a')} |"
                )
        lines.append("")

    lines.extend([
        "## Files",
        "",
        "- `manifest.json`: run configuration and environment",
        "- `summary.csv`: one row per measured run across all measurement modes",
        "- `aggregates.csv`: grouped statistics by mode/case/profile",
        "- `summary_<mode>.csv` and `aggregates_<mode>.csv`: mode-specific views",
        "- `raw/<mode>/`: per-run timing payload",
        "",
        "## Nsight Hints",
        "",
    ])
    if cuda_profiles:
        lines.extend([
            "Use the operators benchmark binary directly for profiling. Prefer `--warmup 1` to remove one-time CUDA setup from measured repeats.",
            "",
            "```bash",
            "nsys profile --trace=cuda,nvtx -o cupf_nsys \\",
            f"  {operators_binary} \\",
            f"  --case-dir {nsight_case_dir} \\",
            f"  --profile {nsight_cupf_profile} --warmup 1 --repeats 1",
            "```",
            "",
            "```bash",
            "ncu --set full -o cupf_ncu \\",
            f"  {operators_binary} \\",
            f"  --case-dir {nsight_case_dir} \\",
            f"  --profile {nsight_cupf_profile} --warmup 1 --repeats 1",
            "```",
        ])
    else:
        lines.extend([
            "This run did not include CUDA profiles. Build a CUDA benchmark run before using Nsight, for example:",
            "",
            "```bash",
            f"python3 {SCRIPT_PATH} \\",
            f"  --dataset-root {manifest['dataset_root']} \\",
            f"  --cases {nsight_case} \\",
            "  --profiles cuda_mixed_vertex \\",
            "  --with-cuda --warmup 1 --repeats 1",
            "```",
        ])

    (run_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(run_root: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# Benchmark Run `{manifest['run_name']}`",
        "",
        "This directory was generated by `cuPF/benchmarks/run_benchmarks.py`.",
        "",
        f"- Dataset root: `{manifest['dataset_root']}`",
        f"- Measurement modes: {', '.join(manifest['measurement_modes'])}",
        f"- Benchmark binaries: `{manifest['benchmark_binaries']}`",
        f"- Profiles: {', '.join(manifest['profiles'])}",
        f"- Batch size: {manifest['batch_size']}",
        f"- cuDSS reordering algorithm: {manifest['cudss_reordering_alg']}",
        f"- cuDSS MT mode: {manifest['cudss_enable_mt']}",
        f"- cuDSS host threads: {manifest['cudss_host_nthreads']}",
        f"- cuDSS ND_NLEVELS: {manifest['cudss_nd_nlevels']}",
        f"- cuDSS matching: {manifest['cudss_use_matching']}",
        f"- cuDSS matching algorithm: {manifest['cudss_matching_alg']}",
        f"- cuDSS pivot epsilon: {manifest['cudss_pivot_epsilon']}",
        f"- cuDSS LD_PRELOAD: {manifest.get('cudss_ld_preload', '')}",
        "",
        "See `SUMMARY.md` for the human-readable result table.",
    ]
    (run_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.cudss_host_nthreads = validate_auto_or_int(
        args.cudss_host_nthreads, name="--cudss-host-nthreads", minimum=1)
    args.cudss_nd_nlevels = validate_auto_or_int(
        args.cudss_nd_nlevels, name="--cudss-nd-nlevels", minimum=0)
    args.cudss_pivot_epsilon = validate_auto_or_nonnegative_float(
        args.cudss_pivot_epsilon, name="--cudss-pivot-epsilon")
    if args.cudss_use_matching and args.cudss_reordering_alg in ("ALG_1", "ALG_2"):
        raise ValueError("--cudss-use-matching is not supported with --cudss-reordering-alg ALG_1 or ALG_2")
    run_env = benchmark_env(args)

    cases = read_case_list(args.case_list) if args.case_list else list(args.cases)
    modes = selected_modes(args.mode)
    profile_specs = resolve_profiles(list(args.profiles))
    if not cases:
        raise ValueError("No benchmark cases were provided")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    command_log: list[dict[str, Any]] = []
    benchmark_binaries: dict[str, str] = {}
    mode_binaries: dict[str, Path | None] = {}
    for mode in modes:
        binary = ensure_benchmark_binary(mode, args, profile_specs, command_log)
        mode_binaries[mode] = binary
        if binary is not None:
            benchmark_binaries[mode] = str(binary)

    rows: list[dict[str, Any]] = []
    for mode in modes:
        binary = mode_binaries[mode]
        for case_name in cases:
            for profile in profile_specs:
                if profile.runner == "pypower":
                    if args.batch_size != 1:
                        raise ValueError("PYPOWER benchmark profiles support only --batch-size 1")
                    rows.extend(run_pypower_case(
                        run_root,
                        mode,
                        case_name,
                        profile,
                        args.warmup,
                        args.repeats,
                        command_log,
                    ))
                    continue

                if binary is None:
                    raise RuntimeError(f"No cuPF benchmark binary is available for {mode}/{profile.name}")
                if args.batch_size != 1 and not (profile.backend == "cuda" and profile.compute == "mixed"):
                    raise ValueError("--batch-size > 1 is currently supported only by CUDA mixed profiles")
                rows.extend(run_cupf_profile_case(
                    binary,
                    args.dataset_root,
                    run_root,
                    mode,
                    case_name,
                    profile,
                    args.warmup,
                    args.repeats,
                    args.batch_size,
                    args.tolerance,
                    args.max_iter,
                    command_log,
                    args.dump_residuals,
                    args.residual_dump_root or (run_root / "residuals"),
                    args.cudss_use_matching,
                    args.cudss_matching_alg,
                    args.cudss_pivot_epsilon,
                    run_env,
                ))

    aggregates = aggregate_rows(rows)
    write_csv(run_root / "summary.csv", rows)
    write_csv(run_root / "aggregates.csv", aggregates)
    for mode in modes:
        mode_rows = [row for row in rows if row["measurement_mode"] == mode]
        mode_aggregates = [row for row in aggregates if row["measurement_mode"] == mode]
        write_csv(run_root / f"summary_{mode}.csv", mode_rows)
        write_csv(run_root / f"aggregates_{mode}.csv", mode_aggregates)

    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(SCRIPT_PATH),
        "workspace_root": str(WORKSPACE_ROOT),
        "dataset_root": str(args.dataset_root),
        "results_root": str(run_root),
        "measurement_modes": modes,
        "benchmark_binaries": benchmark_binaries,
        "build_dirs": {
            "end2end": str(build_dir_for_mode(args, "end2end")) if "end2end" in modes else "",
            "operators": str(build_dir_for_mode(args, "operators")) if "operators" in modes else "",
        },
        "cases": cases,
        "profiles": [profile.name for profile in profile_specs],
        "profile_specs": [
            {
                "name": profile.name,
                "runner": profile.runner,
                "cupf_profile": profile.cupf_profile,
                "implementation": profile.implementation,
                "backend": profile.backend,
                "compute": profile.compute,
                "jacobian": profile.jacobian,
            }
            for profile in profile_specs
        ],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "cudss_reordering_alg": args.cudss_reordering_alg,
        "cudss_enable_mt": args.cudss_enable_mt,
        "cudss_host_nthreads": args.cudss_host_nthreads,
        "cudss_nd_nlevels": args.cudss_nd_nlevels,
        "cudss_use_matching": args.cudss_use_matching,
        "cudss_matching_alg": args.cudss_matching_alg,
        "cudss_pivot_epsilon": args.cudss_pivot_epsilon,
        "cudss_threading_lib": run_env.get("CUDSS_THREADING_LIB", "") if run_env is not None else "",
        "cudss_ld_preload": run_env.get("LD_PRELOAD", "") if run_env is not None else "",
        "dump_residuals": args.dump_residuals,
        "residual_dump_root": str(args.residual_dump_root or (run_root / "residuals")) if args.dump_residuals else "",
        "environment": collect_environment(),
        "commands": command_log,
    }
    write_json(run_root / "manifest.json", manifest)
    write_readme(run_root, manifest)
    write_summary_markdown(run_root, manifest, aggregates)

    print(f"[OK] benchmark run created at {run_root}")


if __name__ == "__main__":
    main()
