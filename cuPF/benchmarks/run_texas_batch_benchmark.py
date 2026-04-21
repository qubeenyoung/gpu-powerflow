#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
CUPF_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_DATASET_ROOT = Path("/workspace/gpu-powerflow/datasets/texas_univ_cases/cuPF_datasets")
DEFAULT_RESULTS_ROOT = CUPF_ROOT / "benchmarks" / "results"
DEFAULT_BATCH_SIZES = [1, 4, 16, 64, 256]
DEFAULT_PROFILE = "cuda_mixed_edge"
DEFAULT_NCU_KERNEL_REGEX = (
    "regex:(compute_ibus_batch_fp32_kernel|compute_mismatch_batch_f64_kernel|"
    "reduce_norm_batch_f64_kernel|fill_jacobian_edge_offdiag_fp32_kernel|"
    "fill_jacobian_diag_from_ibus_fp32_kernel|cast_rhs_f64_to_f32_kernel|"
    "update_voltage_mixed_kernel|reconstruct_voltage_kernel)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CUDA Mixed batch benchmarks on Texas Univ cuPF dump datasets.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default="texas_batch_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--cases", nargs="*", help="Case directory names. Defaults to all case directories.")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--run-ncu", action="store_true")
    parser.add_argument("--ncu-set", default="basic")
    parser.add_argument("--ncu-timeout-sec", type=int, default=900)
    parser.add_argument("--ncu-kernel-name", default=DEFAULT_NCU_KERNEL_REGEX)
    parser.add_argument("--ncu-launch-count", type=int, default=8)
    parser.add_argument("--ncu-cases", nargs="*", help="Optional subset of cases for NCU.")
    parser.add_argument("--ncu-batch-sizes", nargs="*", type=int, help="Optional subset of batch sizes for NCU.")
    return parser.parse_args()


def run_command(cmd: list[str],
                *,
                cwd: Path = CUPF_ROOT,
                timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def require_success(completed: subprocess.CompletedProcess[str], label: str) -> None:
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def configure_and_build(build_dir: Path, *, timing: bool) -> Path:
    cmake_cmd = [
        "cmake",
        "-S", str(CUPF_ROOT),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DWITH_CUDA=ON",
        "-DBUILD_BENCHMARKS=ON",
        "-DBUILD_PYTHON_BINDINGS=OFF",
        "-DENABLE_LOG=OFF",
        f"-DENABLE_TIMING={'ON' if timing else 'OFF'}",
        f"-DENABLE_NVTX={'ON' if timing else 'OFF'}",
    ]
    require_success(run_command(cmake_cmd), f"configure {build_dir.name}")
    require_success(
        run_command(["cmake", "--build", str(build_dir), "--target", "cupf_case_benchmark", "-j2"]),
        f"build {build_dir.name}")
    return build_dir / "benchmarks" / "cupf_case_benchmark"


def parse_key_value_line(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line[len(prefix):].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


def flatten_metrics(metrics: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for name, values in metrics.items():
        safe = name.replace(".", "_").replace("/", "_")
        flat[f"metric_{safe}_count"] = values["count"]
        flat[f"metric_{safe}_total_sec"] = values["total_sec"]
        flat[f"metric_{safe}_avg_sec"] = values["avg_sec"]
    return flat


def run_benchmark(binary: Path,
                  *,
                  mode: str,
                  dataset_root: Path,
                  case_name: str,
                  batch_size: int,
                  profile: str,
                  warmup: int,
                  repeats: int,
                  tolerance: float,
                  max_iter: int) -> tuple[list[dict[str, Any]], str, str, int]:
    case_dir = dataset_root / case_name
    cmd = [
        str(binary),
        "--case-dir", str(case_dir),
        "--profile", profile,
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--batch-size", str(batch_size),
        "--tolerance", str(tolerance),
        "--max-iter", str(max_iter),
    ]
    completed = run_command(cmd)

    parsed_runs: dict[int, dict[str, str]] = {}
    metric_summaries: dict[int, dict[str, dict[str, float | int]]] = {}
    for raw in completed.stdout.splitlines():
        line = raw.strip()
        if line.startswith("RUN "):
            parsed = parse_key_value_line(line, "RUN ")
            parsed_runs[int(parsed["repeat"])] = parsed
        elif line.startswith("METRIC "):
            parsed = parse_key_value_line(line, "METRIC ")
            repeat = int(parsed["repeat"])
            metric_summaries.setdefault(repeat, {})[parsed["name"]] = {
                "count": int(parsed["count"]),
                "total_sec": float(parsed["total_sec"]),
                "avg_sec": float(parsed["avg_sec"]),
            }

    rows: list[dict[str, Any]] = []
    for repeat in sorted(parsed_runs):
        parsed = parsed_runs[repeat]
        metrics = metric_summaries.get(repeat, {})
        row = {
            "mode": mode,
            "case": case_name,
            "profile": parsed.get("profile", profile),
            "batch_size": int(parsed.get("batch_size", batch_size)),
            "repeat": repeat,
            "process_returncode": completed.returncode,
            "success": parsed.get("success") == "true",
            "iterations": int(parsed.get("iterations", 0)),
            "final_mismatch": float(parsed.get("final_mismatch", "nan")),
            "analyze_sec": float(parsed.get("analyze_sec", "nan")),
            "solve_sec": float(parsed.get("solve_sec", "nan")),
            "total_sec": float(parsed.get("total_sec", "nan")),
            "max_abs_v_delta_from_v0": float(parsed.get("max_abs_v_delta_from_v0", "nan")),
            "buses": int(parsed.get("buses", 0)),
            "pv": int(parsed.get("pv", 0)),
            "pq": int(parsed.get("pq", 0)),
            **flatten_metrics(metrics),
        }
        rows.append(row)

    if completed.returncode != 0 or not rows:
        rows.append({
            "mode": mode,
            "case": case_name,
            "profile": profile,
            "batch_size": batch_size,
            "repeat": 0,
            "process_returncode": completed.returncode,
            "success": False,
            "iterations": 0,
            "final_mismatch": "",
            "analyze_sec": "",
            "solve_sec": "",
            "total_sec": "",
            "max_abs_v_delta_from_v0": "",
            "buses": "",
            "pv": "",
            "pq": "",
            "error": (completed.stderr or completed.stdout).strip().replace("\n", " | "),
        })

    return rows, completed.stdout, completed.stderr, completed.returncode


def run_ncu(binary: Path,
            *,
            result_dir: Path,
            dataset_root: Path,
            case_name: str,
            batch_size: int,
            profile: str,
            tolerance: float,
            max_iter: int,
            ncu_set: str,
            ncu_kernel_name: str,
            ncu_launch_count: int,
            timeout_sec: int) -> dict[str, Any]:
    report_base = result_dir / "ncu" / f"{case_name}_b{batch_size}"
    report_base.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ncu",
        "--set", ncu_set,
        "--kernel-name-base", "function",
        "--kernel-name", ncu_kernel_name,
        "--launch-count", str(ncu_launch_count),
        "--target-processes", "all",
        "--force-overwrite",
        "--export", str(report_base),
        str(binary),
        "--case-dir", str(dataset_root / case_name),
        "--profile", profile,
        "--warmup", "0",
        "--repeats", "1",
        "--batch-size", str(batch_size),
        "--tolerance", str(tolerance),
        "--max-iter", str(max_iter),
    ]
    try:
        completed = run_command(cmd, timeout=timeout_sec)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        return {
            "case": case_name,
            "batch_size": batch_size,
            "profile": profile,
            "ncu_set": ncu_set,
            "ncu_kernel_name": ncu_kernel_name,
            "ncu_launch_count": ncu_launch_count,
            "returncode": "timeout",
            "timed_out": True,
            "report": str(report_base) + ".ncu-rep",
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }

    (report_base.with_suffix(".stdout.txt")).write_text(completed.stdout, encoding="utf-8")
    (report_base.with_suffix(".stderr.txt")).write_text(completed.stderr, encoding="utf-8")
    return {
        "case": case_name,
        "batch_size": batch_size,
        "profile": profile,
        "ncu_set": ncu_set,
        "ncu_kernel_name": ncu_kernel_name,
        "ncu_launch_count": ncu_launch_count,
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "report": str(report_base) + ".ncu-rep",
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


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


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("mode") != "end2end":
            continue
        groups.setdefault((str(row["case"]), str(row["profile"]), int(row["batch_size"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (case, profile, batch_size), group in sorted(groups.items()):
        def nums(key: str) -> list[float]:
            values: list[float] = []
            for item in group:
                try:
                    values.append(float(item[key]))
                except (KeyError, TypeError, ValueError):
                    pass
            return values

        total = nums("total_sec")
        solve = nums("solve_sec")
        mismatch = nums("final_mismatch")
        out.append({
            "case": case,
            "profile": profile,
            "batch_size": batch_size,
            "runs": len(group),
            "success_all": all(bool(item.get("success")) for item in group),
            "iterations_max": max(nums("iterations") or [0]),
            "final_mismatch_max": max(mismatch) if mismatch else "",
            "total_sec_mean": statistics.mean(total) if total else "",
            "solve_sec_mean": statistics.mean(solve) if solve else "",
            "per_case_total_sec_mean": statistics.mean(total) / batch_size if total else "",
        })
    return out


def write_summary(path: Path,
                  *,
                  run_name: str,
                  dataset_root: Path,
                  batches: list[int],
                  cases: list[str],
                  timing_aggregate: list[dict[str, Any]],
                  ncu_rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Texas Batch Benchmark `{run_name}`",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Cases: {', '.join(cases)}",
        f"- Batch sizes: {', '.join(map(str, batches))}",
        "",
        "## End-to-End Timing",
        "",
        "| case | batch | converged | iter max | final mismatch max | total mean (s) | per-case mean (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in timing_aggregate:
        lines.append(
            f"| {row['case']} | {row['batch_size']} | {row['success_all']} | "
            f"{row['iterations_max']} | {row['final_mismatch_max']} | "
            f"{row['total_sec_mean']} | {row['per_case_total_sec_mean']} |")

    lines.extend([
        "",
        "## NCU",
        "",
        "| case | batch | returncode | report |",
        "|---|---:|---:|---|",
    ])
    for row in ncu_rows:
        lines.append(
            f"| {row['case']} | {row['batch_size']} | {row['returncode']} | `{row['report']}` |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")

    cases = args.cases or sorted(path.name for path in args.dataset_root.iterdir() if path.is_dir())
    result_dir = args.results_root / args.run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_build:
        end2end_binary = CUPF_ROOT / "build" / "texas-batch-end2end" / "benchmarks" / "cupf_case_benchmark"
        operators_binary = CUPF_ROOT / "build" / "texas-batch-operators" / "benchmarks" / "cupf_case_benchmark"
    else:
        end2end_binary = configure_and_build(CUPF_ROOT / "build" / "texas-batch-end2end", timing=False)
        operators_binary = configure_and_build(CUPF_ROOT / "build" / "texas-batch-operators", timing=True)

    rows: list[dict[str, Any]] = []
    raw_dir = result_dir / "raw"
    for mode, binary in (("end2end", end2end_binary), ("operators", operators_binary)):
        for case in cases:
            for batch in args.batch_sizes:
                bench_rows, stdout, stderr, returncode = run_benchmark(
                    binary,
                    mode=mode,
                    dataset_root=args.dataset_root,
                    case_name=case,
                    batch_size=batch,
                    profile=args.profile,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    tolerance=args.tolerance,
                    max_iter=args.max_iter)
                rows.extend(bench_rows)
                stem = raw_dir / mode / f"{case}_b{batch}"
                stem.parent.mkdir(parents=True, exist_ok=True)
                stem.with_suffix(".stdout.txt").write_text(stdout, encoding="utf-8")
                stem.with_suffix(".stderr.txt").write_text(stderr, encoding="utf-8")
                print(f"[{mode}] {case} B={batch} rc={returncode}")

    ncu_rows: list[dict[str, Any]] = []
    if args.run_ncu:
        ncu_cases = args.ncu_cases or cases
        ncu_batches = args.ncu_batch_sizes or args.batch_sizes
        for case in ncu_cases:
            for batch in ncu_batches:
                row = run_ncu(
                    end2end_binary,
                    result_dir=result_dir,
                    dataset_root=args.dataset_root,
                    case_name=case,
                    batch_size=batch,
                    profile=args.profile,
                    tolerance=args.tolerance,
                    max_iter=args.max_iter,
                    ncu_set=args.ncu_set,
                    ncu_kernel_name=args.ncu_kernel_name,
                    ncu_launch_count=args.ncu_launch_count,
                    timeout_sec=args.ncu_timeout_sec)
                ncu_rows.append(row)
                print(f"[ncu] {case} B={batch} rc={row['returncode']}")

    timing_aggregate = aggregate(rows)
    write_csv(result_dir / "summary.csv", rows)
    write_csv(result_dir / "aggregate.csv", timing_aggregate)
    write_csv(result_dir / "ncu_summary.csv", ncu_rows)
    write_json(result_dir / "manifest.json", {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root),
        "cases": cases,
        "batch_sizes": args.batch_sizes,
        "profile": args.profile,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "end2end_binary": str(end2end_binary),
        "operators_binary": str(operators_binary),
        "run_ncu": args.run_ncu,
        "ncu_set": args.ncu_set,
        "ncu_kernel_name": args.ncu_kernel_name,
        "ncu_launch_count": args.ncu_launch_count,
    })
    write_summary(
        result_dir / "SUMMARY.md",
        run_name=args.run_name,
        dataset_root=args.dataset_root,
        batches=args.batch_sizes,
        cases=cases,
        timing_aggregate=timing_aggregate,
        ncu_rows=ncu_rows)
    print(f"[OK] wrote benchmark results to {result_dir}")


if __name__ == "__main__":
    main()
