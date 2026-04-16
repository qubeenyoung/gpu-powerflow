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
EXP_ROOT = SCRIPT_PATH.parent
WORKSPACE_ROOT = EXP_ROOT.parents[2]
CUPF_ROOT = WORKSPACE_ROOT / "cuPF"
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "texas_univ_cases" / "cuPF_datasets"
DEFAULT_RESULTS_ROOT = EXP_ROOT / "results"
DEFAULT_BUILD_ROOT = EXP_ROOT / "build"
DEFAULT_CASES = [
    "case_ACTIVSg200",
    "case_ACTIVSg500",
    "MemphisCase2026_Mar7",
    "case_ACTIVSg2000",
    "Base_Florida_42GW",
    "Texas7k_20220923",
    "Base_Texas_66GW",
    "Base_MIOHIN_76GW",
    "Base_West_Interconnect_121GW",
    "case_ACTIVSg25k",
    "case_ACTIVSg70k",
    "Base_Eastern_Interconnect_515GW",
]
DEFAULT_PROFILES = ["cuda_edge"]
MEASUREMENT_MODES = ("end2end", "operators")

PROFILE_TO_CUPF = {
    "cuda_edge": "cuda_mixed_edge",
    "cuda_mixed_edge": "cuda_mixed_edge",
    "cuda_edge_modified": "cuda_mixed_edge_modified",
    "cuda_mixed_edge_modified": "cuda_mixed_edge_modified",
    "cuda_vertex": "cuda_mixed_vertex",
    "cuda_mixed_vertex": "cuda_mixed_vertex",
    "cuda_vertex_modified": "cuda_mixed_vertex_modified",
    "cuda_mixed_vertex_modified": "cuda_mixed_vertex_modified",
    "cuda_fp64_edge": "cuda_fp64_edge",
    "cuda_fp64_edge_modified": "cuda_fp64_edge_modified",
    "cuda_fp64_vertex": "cuda_fp64_vertex",
    "cuda_fp64_vertex_modified": "cuda_fp64_vertex_modified",
}

PROFILE_LABEL = {
    "cuda_edge": "CUDA mixed edge",
    "cuda_mixed_edge": "CUDA mixed edge",
    "cuda_edge_modified": "CUDA mixed edge modified",
    "cuda_mixed_edge_modified": "CUDA mixed edge modified",
    "cuda_vertex": "CUDA mixed vertex",
    "cuda_mixed_vertex": "CUDA mixed vertex",
    "cuda_vertex_modified": "CUDA mixed vertex modified",
    "cuda_mixed_vertex_modified": "CUDA mixed vertex modified",
    "cuda_fp64_edge": "CUDA FP64 edge",
    "cuda_fp64_edge_modified": "CUDA FP64 edge modified",
    "cuda_fp64_vertex": "CUDA FP64 vertex",
    "cuda_fp64_vertex_modified": "CUDA FP64 vertex modified",
}

IMPORTANT_OPERATOR_METRICS = [
    "elapsed_sec",
    "analyze_sec",
    "solve_sec",
    "NR.analyze.total.total_sec",
    "NR.analyze.jacobian_builder.total_sec",
    "NR.analyze.storage_prepare.total_sec",
    "NR.analyze.linear_solve.total_sec",
    "CUDA.analyze.cudss32.setup.total_sec",
    "CUDA.analyze.cudss32.analysis.total_sec",
    "CUDA.analyze.cudss64.setup.total_sec",
    "CUDA.analyze.cudss64.analysis.total_sec",
    "NR.solve.total.total_sec",
    "NR.solve.upload.total_sec",
    "NR.iteration.total.total_sec",
    "NR.iteration.mismatch.total_sec",
    "NR.iteration.jacobian.total_sec",
    "NR.iteration.linear.total_sec",
    "NR.iteration.linear_factorize.total_sec",
    "NR.iteration.linear_solve.total_sec",
    "NR.iteration.voltage_update.total_sec",
    "NR.solve.download.total_sec",
    "CUDA.solve.rhsPrepare.total_sec",
    "CUDA.solve.factorization32.total_sec",
    "CUDA.solve.refactorization32.total_sec",
    "CUDA.solve.solve32.total_sec",
    "CUDA.solve.rhsPrepare64.total_sec",
    "CUDA.solve.factorization64.total_sec",
    "CUDA.solve.refactorization64.total_sec",
    "CUDA.solve.solve64.total_sec",
]


@dataclass(frozen=True)
class Variant:
    name: str
    mg_enabled: bool


VARIANTS = {
    "baseline": Variant("baseline", False),
    "mg": Variant("mg", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cuPF Newton solver timing before/after cuDSS MG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--mode", choices=("end2end", "operators", "both"), default="both")
    parser.add_argument("--variants", nargs="*", choices=sorted(VARIANTS), default=["baseline", "mg"])
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES, choices=sorted(PROFILE_TO_CUPF))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--cudss-reordering-alg", choices=("DEFAULT", "ALG_1", "ALG_2"), default="DEFAULT")
    parser.add_argument("--cudss-enable-mt", action="store_true")
    parser.add_argument("--cudss-host-nthreads", default="AUTO")
    parser.add_argument("--cudss-threading-lib", type=Path)
    parser.add_argument("--cudss-nd-nlevels", default="AUTO")
    parser.add_argument("--mg-device-indices", default=os.environ.get("CUPF_CUDSS_MG_DEVICE_INDICES", "0,1"))
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES for all builds and runs.")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def selected_modes(mode: str) -> list[str]:
    if mode == "both":
        return list(MEASUREMENT_MODES)
    return [mode]


def read_case_list(path: Path) -> list[str]:
    cases: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            cases.append(stripped)
    return cases


def validate_auto_or_int(value: str, *, name: str, minimum: int) -> str:
    if value.upper() == "AUTO":
        return "AUTO"
    if not value.isdigit() or int(value) < minimum:
        raise ValueError(f"{name} must be AUTO or an integer >= {minimum}: {value}")
    return value


def normalize_device_indices(value: str) -> str:
    normalized = value.replace(";", ",").replace(" ", "")
    if not normalized:
        raise ValueError("--mg-device-indices must contain at least one device index")
    parts = normalized.split(",")
    if any(not part.isdigit() for part in parts):
        raise ValueError(f"--mg-device-indices must be comma-separated non-negative integers: {value}")
    return ",".join(str(int(part)) for part in parts)


def default_threading_lib() -> Path | None:
    candidates = [
        Path("/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0"),
        Path("/usr/lib/x86_64-linux-gnu/libcudss_mtlayer_gomp.so.0"),
        Path("/usr/local/lib/libcudss_mtlayer_gomp.so.0"),
    ]
    return next((path for path in candidates if path.exists()), None)


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    cudss_lib_dir = Path("/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib")
    if cudss_lib_dir.exists():
        existing = env.get("LD_LIBRARY_PATH", "")
        entries = [entry for entry in existing.split(":") if entry]
        if str(cudss_lib_dir) not in entries:
            env["LD_LIBRARY_PATH"] = ":".join([str(cudss_lib_dir), *entries])

    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.cudss_enable_mt:
        threading_lib = args.cudss_threading_lib or default_threading_lib()
        if threading_lib is None:
            raise ValueError("--cudss-enable-mt requires --cudss-threading-lib or an installed libcudss_mtlayer_gomp.so.0")
        env["CUDSS_THREADING_LIB"] = str(threading_lib)
        preload_entries = [entry for entry in env.get("LD_PRELOAD", "").split(":") if entry]
        if str(threading_lib) not in preload_entries:
            env["LD_PRELOAD"] = ":".join([str(threading_lib), *preload_entries])
    return env


def run_command(cmd: list[str],
                *,
                cwd: Path | None = None,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {joined}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def maybe_run_command(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return run_command(cmd, cwd=cwd).stdout.strip()
    except Exception:
        return None


def build_binary(mode: str,
                 variant: Variant,
                 args: argparse.Namespace,
                 env: dict[str, str],
                 command_log: list[dict[str, Any]]) -> Path:
    build_dir = args.build_root / variant.name / mode
    binary = build_dir / "benchmarks" / "cupf_case_benchmark"
    if args.skip_build:
        if binary.exists():
            return binary
        raise FileNotFoundError(f"Benchmark binary missing with --skip-build: {binary}")

    timing_enabled = mode == "operators"
    configure_cmd = [
        "cmake",
        "-S", str(CUPF_ROOT),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DWITH_CUDA=ON",
        "-DENABLE_LOG=OFF",
        "-DENABLE_DUMP=OFF",
        f"-DENABLE_TIMING={'ON' if timing_enabled else 'OFF'}",
        f"-DENABLE_NVTX={'ON' if timing_enabled else 'OFF'}",
        f"-DCUPF_CUDSS_REORDERING_ALG={args.cudss_reordering_alg}",
        f"-DCUPF_CUDSS_ENABLE_MT={'ON' if args.cudss_enable_mt else 'OFF'}",
        f"-DCUPF_CUDSS_HOST_NTHREADS={args.cudss_host_nthreads}",
        f"-DCUPF_CUDSS_ND_NLEVELS={args.cudss_nd_nlevels}",
        f"-DCUPF_CUDSS_ENABLE_MG={'ON' if variant.mg_enabled else 'OFF'}",
        f"-DCUPF_CUDSS_MG_DEVICE_INDICES={args.mg_device_indices}",
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

    command_log.append({"name": f"configure_{variant.name}_{mode}", "cmd": configure_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(configure_cmd, cwd=WORKSPACE_ROOT, env=env)
    command_log.append({"name": f"build_{variant.name}_{mode}", "cmd": build_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(build_cmd, cwd=WORKSPACE_ROOT, env=env)
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


def run_case(binary: Path,
             mode: str,
             variant: Variant,
             case_name: str,
             profile: str,
             args: argparse.Namespace,
             env: dict[str, str],
             command_log: list[dict[str, Any]],
             run_root: Path) -> list[dict[str, Any]]:
    cupf_profile = PROFILE_TO_CUPF[profile]
    case_dir = args.dataset_root / case_name
    if not case_dir.exists():
        raise FileNotFoundError(f"Dataset case not found: {case_dir}")

    cmd = [
        str(binary),
        "--case-dir", str(case_dir),
        "--profile", cupf_profile,
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--tolerance", str(args.tolerance),
        "--max-iter", str(args.max_iter),
    ]
    command_log.append({
        "name": f"run_{variant.name}_{mode}_{profile}_{case_name}",
        "cmd": cmd,
        "cwd": str(WORKSPACE_ROOT),
    })
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
            "measurement_mode": mode,
            "variant": variant.name,
            "mg_enabled": variant.mg_enabled,
            "mg_device_indices": args.mg_device_indices if variant.mg_enabled else "",
            "case_name": case_name,
            "dump_case_name": parsed["case"],
            "profile": profile,
            "profile_label": PROFILE_LABEL.get(profile, profile),
            "source_profile": parsed["profile"],
            "implementation": parsed["implementation"],
            "backend": parsed["backend"],
            "compute": parsed["compute"],
            "jacobian": parsed["jacobian"],
            "algorithm": parsed.get("algorithm", "standard"),
            "repeat_idx": repeat,
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

        raw_path = run_root / "raw" / mode / variant.name / profile / case_name / f"run_{repeat:02d}.json"
        write_json(raw_path, {
            "summary": row,
            "timing_entries": metric_entries.get(repeat, []),
            "stdout": completed.stdout,
        })

    if len(rows) != args.repeats:
        raise RuntimeError(f"Expected {args.repeats} runs for {variant.name}/{mode}/{profile}/{case_name}, got {len(rows)}")
    return rows


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = finite_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def add_numeric_stats(out: dict[str, Any], rows: list[dict[str, Any]], key: str) -> None:
    values = numeric_values(rows, key)
    if not values:
        out[f"{key}_mean"] = ""
        out[f"{key}_median"] = ""
        out[f"{key}_min"] = ""
        out[f"{key}_max"] = ""
        out[f"{key}_stdev"] = ""
        return
    out[f"{key}_mean"] = statistics.mean(values)
    out[f"{key}_median"] = statistics.median(values)
    out[f"{key}_min"] = min(values)
    out[f"{key}_max"] = max(values)
    out[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((
            str(row["measurement_mode"]),
            str(row["variant"]),
            str(row["profile"]),
            str(row["case_name"]),
        ), []).append(row)

    metric_keys = sorted(
        key for key in {key for row in rows for key in row}
        if key in {"elapsed_sec", "analyze_sec", "solve_sec", "iterations", "final_mismatch"}
        or key.endswith(".total_sec")
        or key.endswith(".avg_sec")
        or key.endswith(".count")
    )

    aggregates: list[dict[str, Any]] = []
    for (mode, variant, profile, case_name), group in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "measurement_mode": mode,
            "variant": variant,
            "mg_enabled": group[0]["mg_enabled"],
            "mg_device_indices": group[0]["mg_device_indices"],
            "profile": profile,
            "profile_label": group[0]["profile_label"],
            "case_name": case_name,
            "source_profile": group[0]["source_profile"],
            "implementation": group[0]["implementation"],
            "backend": group[0]["backend"],
            "compute": group[0]["compute"],
            "jacobian": group[0]["jacobian"],
            "algorithm": group[0].get("algorithm", "standard"),
            "buses": group[0]["buses"],
            "pv": group[0]["pv"],
            "pq": group[0]["pq"],
            "runs": len(group),
            "success_all": all(bool(row["success"]) for row in group),
        }
        for key in metric_keys:
            add_numeric_stats(aggregate, group, key)
        aggregates.append(aggregate)
    return aggregates


def speedup(baseline: Any, candidate: Any) -> float | str:
    base = finite_float(baseline)
    cand = finite_float(candidate)
    if base is None or cand is None or cand == 0.0:
        return ""
    return base / cand


def build_top_level_comparison(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["measurement_mode"], row["profile"], row["case_name"], row["variant"]): row
        for row in aggregates
    }
    rows: list[dict[str, Any]] = []
    groups = sorted({(row["measurement_mode"], row["profile"], row["case_name"]) for row in aggregates})
    for mode, profile, case_name in groups:
        baseline = by_key.get((mode, profile, case_name, "baseline"))
        mg = by_key.get((mode, profile, case_name, "mg"))
        if baseline is None or mg is None:
            continue
        row: dict[str, Any] = {
            "measurement_mode": mode,
            "profile": profile,
            "case_name": case_name,
            "buses": baseline.get("buses", ""),
            "baseline_success_all": baseline.get("success_all", ""),
            "mg_success_all": mg.get("success_all", ""),
            "baseline_iterations_mean": baseline.get("iterations_mean", ""),
            "mg_iterations_mean": mg.get("iterations_mean", ""),
            "baseline_final_mismatch_mean": baseline.get("final_mismatch_mean", ""),
            "mg_final_mismatch_mean": mg.get("final_mismatch_mean", ""),
        }
        for metric in ("elapsed_sec", "analyze_sec", "solve_sec"):
            base = baseline.get(f"{metric}_mean", "")
            cand = mg.get(f"{metric}_mean", "")
            row[f"baseline_{metric}_mean"] = base
            row[f"mg_{metric}_mean"] = cand
            row[f"{metric}_speedup_baseline_over_mg"] = speedup(base, cand)
            base_f = finite_float(base)
            cand_f = finite_float(cand)
            row[f"{metric}_delta_mg_minus_baseline"] = (cand_f - base_f) if base_f is not None and cand_f is not None else ""
        rows.append(row)
    return rows


def build_operator_comparison(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["measurement_mode"], row["profile"], row["case_name"], row["variant"]): row
        for row in aggregates
    }
    metric_names = sorted({
        key.removesuffix("_mean")
        for row in aggregates
        for key in row
        if key.endswith("_mean")
        and (
            key.removesuffix("_mean") in {"elapsed_sec", "analyze_sec", "solve_sec"}
            or key.removesuffix("_mean").endswith(".total_sec")
        )
    })

    rows: list[dict[str, Any]] = []
    groups = sorted({(row["measurement_mode"], row["profile"], row["case_name"]) for row in aggregates})
    for mode, profile, case_name in groups:
        baseline = by_key.get((mode, profile, case_name, "baseline"))
        mg = by_key.get((mode, profile, case_name, "mg"))
        if baseline is None or mg is None:
            continue
        for metric in metric_names:
            base = baseline.get(f"{metric}_mean", "")
            cand = mg.get(f"{metric}_mean", "")
            if base == "" and cand == "":
                continue
            base_f = finite_float(base)
            cand_f = finite_float(cand)
            rows.append({
                "measurement_mode": mode,
                "profile": profile,
                "case_name": case_name,
                "metric": metric,
                "baseline_sec_mean": base,
                "mg_sec_mean": cand,
                "baseline_ms_mean": base_f * 1000.0 if base_f is not None else "",
                "mg_ms_mean": cand_f * 1000.0 if cand_f is not None else "",
                "speedup_baseline_over_mg": speedup(base, cand),
                "delta_ms_mg_minus_baseline": (cand_f - base_f) * 1000.0 if base_f is not None and cand_f is not None else "",
                "baseline_count_mean": baseline.get(f"{metric.removesuffix('.total_sec')}.count_mean", ""),
                "mg_count_mean": mg.get(f"{metric.removesuffix('.total_sec')}.count_mean", ""),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_environment(env: dict[str, str]) -> dict[str, Any]:
    return {
        "created_on": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cmake": maybe_run_command(["cmake", "--version"]),
        "cxx": maybe_run_command(["c++", "--version"]),
        "nvcc": maybe_run_command(["nvcc", "--version"]),
        "nvidia_smi": maybe_run_command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"]),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "ld_library_path": env.get("LD_LIBRARY_PATH", ""),
        "ld_preload": env.get("LD_PRELOAD", ""),
        "cudss_threading_lib": env.get("CUDSS_THREADING_LIB", ""),
        "git": {
            "branch": maybe_run_command(["git", "-C", str(WORKSPACE_ROOT), "rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": maybe_run_command(["git", "-C", str(WORKSPACE_ROOT), "rev-parse", "HEAD"]),
            "status_short": maybe_run_command(["git", "-C", str(WORKSPACE_ROOT), "status", "--short"]),
        },
    }


def fmt_ms_from_sec(value: Any) -> str:
    numeric = finite_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric * 1000.0:.3f}"


def fmt_ms(value: Any) -> str:
    numeric = finite_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.3f}"


def fmt_speedup(value: Any) -> str:
    numeric = finite_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.3f}x"


def write_summary_markdown(run_root: Path,
                           manifest: dict[str, Any],
                           top_comparison: list[dict[str, Any]],
                           operator_comparison: list[dict[str, Any]]) -> None:
    lines = [
        f"# cuDSS MG Newton Solver Benchmark `{manifest['run_name']}`",
        "",
        "## Setup",
        "",
        f"- Created UTC: {manifest['created_at_utc']}",
        f"- Dataset root: `{manifest['dataset_root']}`",
        f"- Cases: {', '.join(manifest['cases'])}",
        f"- Profiles: {', '.join(manifest['profiles'])}",
        f"- Modes: {', '.join(manifest['measurement_modes'])}",
        f"- Warmup: {manifest['warmup']}",
        f"- Repeats: {manifest['repeats']}",
        f"- cuDSS MG indices: `{manifest['mg_device_indices']}`",
        f"- cuDSS reordering: `{manifest['cudss_reordering_alg']}`",
        f"- cuDSS MT: {manifest['cudss_enable_mt']}",
        f"- cuDSS host threads: `{manifest['cudss_host_nthreads']}`",
        f"- cuDSS ND_NLEVELS: `{manifest['cudss_nd_nlevels']}`",
        f"- CUDA_VISIBLE_DEVICES: `{manifest.get('cuda_visible_devices', '')}`",
        "",
    ]

    for mode in manifest["measurement_modes"]:
        rows = [row for row in top_comparison if row["measurement_mode"] == mode]
        if not rows:
            continue
        lines.extend([
            f"## {mode}",
            "",
            "| case | profile | baseline elapsed ms | MG elapsed ms | speedup | baseline solve ms | MG solve ms | solve speedup | success | iterations |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row['case_name']} | {row['profile']} | "
                f"{fmt_ms_from_sec(row.get('baseline_elapsed_sec_mean'))} | "
                f"{fmt_ms_from_sec(row.get('mg_elapsed_sec_mean'))} | "
                f"{fmt_speedup(row.get('elapsed_sec_speedup_baseline_over_mg'))} | "
                f"{fmt_ms_from_sec(row.get('baseline_solve_sec_mean'))} | "
                f"{fmt_ms_from_sec(row.get('mg_solve_sec_mean'))} | "
                f"{fmt_speedup(row.get('solve_sec_speedup_baseline_over_mg'))} | "
                f"{row.get('baseline_success_all')}/{row.get('mg_success_all')} | "
                f"{row.get('baseline_iterations_mean', 'n/a')}/{row.get('mg_iterations_mean', 'n/a')} |"
            )
        lines.append("")

    operator_rows = [
        row for row in operator_comparison
        if row["measurement_mode"] == "operators" and row["metric"] in IMPORTANT_OPERATOR_METRICS
    ]
    if operator_rows:
        lines.extend([
            "## Operator Metrics",
            "",
            "| case | profile | metric | baseline ms | MG ms | speedup | delta ms |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for row in operator_rows:
            lines.append(
                f"| {row['case_name']} | {row['profile']} | `{row['metric']}` | "
                f"{fmt_ms(row.get('baseline_ms_mean'))} | "
                f"{fmt_ms(row.get('mg_ms_mean'))} | "
                f"{fmt_speedup(row.get('speedup_baseline_over_mg'))} | "
                f"{fmt_ms(row.get('delta_ms_mg_minus_baseline'))} |"
            )
        lines.append("")

    lines.extend([
        "## Files",
        "",
        "- `manifest.json`: configuration, commands, and environment",
        "- `summary.csv`: one row per repeat",
        "- `aggregates.csv`: grouped timing statistics",
        "- `mg_comparison.csv`: top-level baseline vs MG timing",
        "- `operator_comparison.csv`: all collected timers, baseline vs MG",
        "- `raw/`: per-repeat parsed payload and stdout",
        "",
    ])
    (run_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.cudss_host_nthreads = validate_auto_or_int(args.cudss_host_nthreads, name="--cudss-host-nthreads", minimum=1)
    args.cudss_nd_nlevels = validate_auto_or_int(args.cudss_nd_nlevels, name="--cudss-nd-nlevels", minimum=0)
    args.mg_device_indices = normalize_device_indices(args.mg_device_indices)
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")

    cases = read_case_list(args.case_list) if args.case_list else list(args.cases)
    if not cases:
        raise ValueError("No cases selected")
    modes = selected_modes(args.mode)
    variants = [VARIANTS[name] for name in args.variants]
    env = base_env(args)

    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    command_log: list[dict[str, Any]] = []
    binaries: dict[tuple[str, str], Path] = {}
    for variant in variants:
        for mode in modes:
            binaries[(variant.name, mode)] = build_binary(mode, variant, args, env, command_log)

    rows: list[dict[str, Any]] = []
    for mode in modes:
        for variant in variants:
            binary = binaries[(variant.name, mode)]
            for case_name in cases:
                for profile in args.profiles:
                    rows.extend(run_case(binary, mode, variant, case_name, profile, args, env, command_log, run_root))

    aggregates = aggregate_rows(rows)
    top_comparison = build_top_level_comparison(aggregates)
    operator_comparison = build_operator_comparison(aggregates)

    write_csv(run_root / "summary.csv", rows)
    write_csv(run_root / "aggregates.csv", aggregates)
    write_csv(run_root / "mg_comparison.csv", top_comparison)
    write_csv(run_root / "operator_comparison.csv", operator_comparison)
    for mode in modes:
        write_csv(run_root / f"summary_{mode}.csv", [row for row in rows if row["measurement_mode"] == mode])
        write_csv(run_root / f"aggregates_{mode}.csv", [row for row in aggregates if row["measurement_mode"] == mode])

    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(SCRIPT_PATH),
        "workspace_root": str(WORKSPACE_ROOT),
        "cupf_root": str(CUPF_ROOT),
        "dataset_root": str(args.dataset_root),
        "results_root": str(run_root),
        "build_root": str(args.build_root),
        "measurement_modes": modes,
        "variants": [variant.name for variant in variants],
        "benchmark_binaries": {f"{variant}_{mode}": str(path) for (variant, mode), path in binaries.items()},
        "cases": cases,
        "profiles": list(args.profiles),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "cudss_reordering_alg": args.cudss_reordering_alg,
        "cudss_enable_mt": args.cudss_enable_mt,
        "cudss_host_nthreads": args.cudss_host_nthreads,
        "cudss_nd_nlevels": args.cudss_nd_nlevels,
        "mg_device_indices": args.mg_device_indices,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "environment": collect_environment(env),
        "commands": command_log,
    }
    write_json(run_root / "manifest.json", manifest)
    write_summary_markdown(run_root, manifest, top_comparison, operator_comparison)
    print(f"[OK] cuDSS MG benchmark run created at {run_root}")


if __name__ == "__main__":
    main()
