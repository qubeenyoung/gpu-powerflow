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
import statistics
import subprocess
import sys
from typing import Any

import numpy as np
from pypower.ppoption import ppoption
from scipy.io import mmread


WORKSPACE_ROOT = Path("/workspace/gpu-powerflow")
DEFAULT_CASE_DIR = WORKSPACE_ROOT / "datasets/matpower8.1/cupf_all_dumps/case9241pegase"
DEFAULT_CUPF_BINARY = (
    WORKSPACE_ROOT
    / "build/bench-operators/benchmarks/cupf_case_benchmark"
)
DEFAULT_JAC_ASM_BINARY = WORKSPACE_ROOT / "exp/20260423/jac_asm/build/jac_asm_bench"


@dataclass(frozen=True)
class Method:
    method: str
    label: str
    profile: str
    backend: str
    compute: str
    jacobian_basis: str
    timer_scope: str


METHODS = (
    Method(
        "pypower_bus",
        "PyPower bus",
        "pypower",
        "python",
        "fp64",
        "bus",
        "python_wall_timer",
    ),
    Method(
        "c_bus",
        "C bus",
        "cpp_pypowerlike",
        "cpu",
        "fp64",
        "bus",
        "cupf_scoped_timer",
    ),
    Method(
        "c_edge",
        "C edge",
        "cpu_fp64_edge",
        "cpu",
        "fp64",
        "edge",
        "cupf_scoped_timer",
    ),
    Method(
        "cuda_edge",
        "CUDA edge",
        "cuda_mixed_edge",
        "cuda",
        "mixed",
        "edge",
        "cupf_scoped_timer_cuda_launch_blocking",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Jacobian assembly paths for MATPOWER case9241pegase."
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--cupf-binary", type=Path, default=DEFAULT_CUPF_BINARY)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--jac-asm-binary", type=Path, default=DEFAULT_JAC_ASM_BINARY)
    parser.add_argument("--jac-asm-warmup", type=int, default=10)
    parser.add_argument("--jac-asm-iters", type=int, default=1000)
    parser.add_argument(
        "--skip-cuda-event-fill",
        action="store_true",
        help="Skip standalone CUDA event timing for bus/edge Jacobian fill kernels.",
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set CUDA_LAUNCH_BLOCKING=1 for CUDA profiles so scoped timers include kernel completion.",
    )
    return parser.parse_args()


def parse_key_value_line(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise ValueError(f"line does not start with {prefix!r}: {line}")
    parsed: dict[str, str] = {}
    for item in line[len(prefix) :].split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def load_complex_txt(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    return values[:, 0].astype(np.float64) + 1j * values[:, 1].astype(np.float64)


def load_int_txt(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64)
    return np.atleast_1d(values).astype(np.int64)


def case_metadata(case_dir: Path) -> dict[str, Any]:
    metadata_path = case_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        nan = float("nan")
        return {
            "mean": nan,
            "median": nan,
            "stdev": nan,
            "min": nan,
            "max": nan,
        }
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def add_row(rows: list[dict[str, Any]],
            *,
            method: Method,
            repeat_idx: int,
            success: bool,
            iterations: int,
            final_mismatch: float,
            jacobian_updates: int,
            jacobian_total_sec: float,
            jacobian_avg_sec: float,
            elapsed_sec: float | None,
            analyze_sec: float | None,
            solve_sec: float | None,
            n_bus: int,
            n_pv: int,
            n_pq: int,
            ybus_nnz: int) -> None:
    rows.append({
        "method": method.method,
        "label": method.label,
        "profile": method.profile,
        "backend": method.backend,
        "compute": method.compute,
        "jacobian_basis": method.jacobian_basis,
        "timer_scope": method.timer_scope,
        "repeat_idx": repeat_idx,
        "success": success,
        "iterations": iterations,
        "final_mismatch": final_mismatch,
        "jacobian_updates": jacobian_updates,
        "jacobian_total_sec": jacobian_total_sec,
        "jacobian_avg_sec": jacobian_avg_sec,
        "jacobian_ms": jacobian_avg_sec * 1000.0,
        "elapsed_sec": "" if elapsed_sec is None else elapsed_sec,
        "analyze_sec": "" if analyze_sec is None else analyze_sec,
        "solve_sec": "" if solve_sec is None else solve_sec,
        "n_bus": n_bus,
        "n_pv": n_pv,
        "n_pq": n_pq,
        "ybus_nnz": ybus_nnz,
    })


def run_pypower(case_dir: Path,
                warmup: int,
                repeats: int,
                tolerance: float,
                max_iter: int,
                rows: list[dict[str, Any]]) -> dict[str, Any]:
    if str(CUPF_WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(CUPF_WORKSPACE_ROOT))

    from python.pypower.newtonpf import my_newtonpf  # pylint: disable=import-outside-toplevel
    from python.pypower.timer import TimingLog, summarize_entries  # pylint: disable=import-outside-toplevel

    method = METHODS[0]
    ybus = mmread(case_dir / "dump_Ybus.mtx").tocsr()
    v0 = load_complex_txt(case_dir / "dump_V.txt")
    sbus = load_complex_txt(case_dir / "dump_Sbus.txt")
    pv = load_int_txt(case_dir / "dump_pv.txt")
    pq = load_int_txt(case_dir / "dump_pq.txt")
    ref = np.setdiff1d(np.arange(ybus.shape[0], dtype=np.int64), np.r_[pv, pq])
    options = ppoption(PF_TOL=tolerance, PF_MAX_IT=max_iter)

    for _ in range(warmup):
        my_newtonpf(
            ybus, sbus, v0, ref, pv, pq,
            ppopt=options,
            timing_log=None,
            emit_status=False,
        )

    for repeat_idx in range(repeats):
        timing_log = TimingLog(True, emit_log=False)
        result = my_newtonpf(
            ybus, sbus, v0, ref, pv, pq,
            ppopt=options,
            timing_log=timing_log,
            emit_status=False,
        )
        metrics = summarize_entries(timing_log.entries)
        jac = metrics["newtonpf.jacobian"]
        add_row(
            rows,
            method=method,
            repeat_idx=repeat_idx,
            success=bool(result.converged and result.final_mismatch <= tolerance),
            iterations=int(result.iterations),
            final_mismatch=float(result.final_mismatch),
            jacobian_updates=int(jac["count"]),
            jacobian_total_sec=float(jac["total_sec"]),
            jacobian_avg_sec=float(jac["avg_sec"]),
            elapsed_sec=sum(entry.elapsed_sec for entry in timing_log.entries),
            analyze_sec=None,
            solve_sec=None,
            n_bus=int(ybus.shape[0]),
            n_pv=int(len(pv)),
            n_pq=int(len(pq)),
            ybus_nnz=int(ybus.nnz),
        )

    return {
        "n_bus": int(ybus.shape[0]),
        "n_pv": int(len(pv)),
        "n_pq": int(len(pq)),
        "ybus_nnz": int(ybus.nnz),
    }


def run_cupf(method: Method,
             case_dir: Path,
             cupf_binary: Path,
             out_dir: Path,
             warmup: int,
             repeats: int,
             tolerance: float,
             max_iter: int,
             batch_size: int,
             cuda_launch_blocking: bool,
             rows: list[dict[str, Any]]) -> None:
    cmd = [
        str(cupf_binary),
        "--case-dir", str(case_dir),
        "--profile", method.profile,
        "--warmup", str(warmup),
        "--repeats", str(repeats),
        "--batch-size", str(batch_size),
        "--tolerance", str(tolerance),
        "--max-iter", str(max_iter),
    ]
    env = os.environ.copy()
    cuda_env: dict[str, str] = {}
    if method.backend == "cuda" and cuda_launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        cuda_env["CUDA_LAUNCH_BLOCKING"] = "1"
    completed = subprocess.run(
        cmd,
        cwd=str(WORKSPACE_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    command_record = {
        "cmd": cmd,
        "cwd": str(WORKSPACE_ROOT),
        "env": cuda_env,
    }
    (out_dir / f"command_{method.method}.json").write_text(
        json.dumps(command_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / f"stdout_{method.method}.txt").write_text(completed.stdout, encoding="utf-8")
    (out_dir / f"stderr_{method.method}.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{method.profile} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )

    runs: dict[int, dict[str, str]] = {}
    jac_metrics: dict[int, dict[str, str]] = {}
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("RUN "):
            parsed = parse_key_value_line(line, "RUN ")
            runs[int(parsed["repeat"])] = parsed
        elif line.startswith("METRIC "):
            parsed = parse_key_value_line(line, "METRIC ")
            if parsed.get("name") == "NR.iteration.jacobian":
                jac_metrics[int(parsed["repeat"])] = parsed

    if len(runs) != repeats:
        raise RuntimeError(f"{method.profile}: expected {repeats} RUN rows, got {len(runs)}")
    if len(jac_metrics) != repeats:
        raise RuntimeError(
            f"{method.profile}: expected {repeats} NR.iteration.jacobian rows, "
            f"got {len(jac_metrics)}"
        )

    for repeat_idx in sorted(runs):
        run = runs[repeat_idx]
        jac = jac_metrics[repeat_idx]
        add_row(
            rows,
            method=method,
            repeat_idx=repeat_idx,
            success=run["success"] == "true",
            iterations=int(run["iterations"]),
            final_mismatch=float(run["final_mismatch"]),
            jacobian_updates=int(jac["count"]),
            jacobian_total_sec=float(jac["total_sec"]),
            jacobian_avg_sec=float(jac["avg_sec"]),
            elapsed_sec=float(run["total_sec"]),
            analyze_sec=float(run["analyze_sec"]),
            solve_sec=float(run["solve_sec"]),
            n_bus=int(run["buses"]),
            n_pv=int(run["pv"]),
            n_pq=int(run["pq"]),
            ybus_nnz=-1,
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("no rows to write")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_cuda_event_fill(case_dir: Path,
                        jac_asm_binary: Path,
                        out_dir: Path,
                        warmup: int,
                        iters: int) -> list[dict[str, Any]]:
    cmd = [
        str(jac_asm_binary),
        "--data", str(case_dir.parent),
        "--case", case_dir.name,
        "--mode", "both",
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(WORKSPACE_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    csv_path = out_dir / f"cuda_event_jac_asm_{case_dir.name}.csv"
    csv_path.write_text(completed.stdout, encoding="utf-8")
    (out_dir / "stderr_cuda_event_jac_asm.txt").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    (out_dir / "command_cuda_event_jac_asm.json").write_text(
        json.dumps(
            {
                "cmd": cmd,
                "cwd": str(WORKSPACE_ROOT),
                "timer": "cuda_event",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"jac_asm_bench failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one jac_asm row, got {len(rows)}")

    row = rows[0]
    edge_ms = float(row["edge_fill_ms"])
    bus_ms = float(row["vertex_thread_fill_ms"])
    event_rows = [
        {
            "method": "cuda_bus_event",
            "label": "CUDA bus",
            "profile": "jac_asm_vertex_thread",
            "backend": "cuda",
            "compute": "fp32",
            "jacobian_basis": "bus",
            "timer_scope": "cuda_event_standalone_fill",
            "warmup": warmup,
            "iters": iters,
            "fill_ms": bus_ms,
            "speedup_vs_cuda_bus": 1.0,
            "n_bus": int(row["n_bus"]),
            "n_pv": int(row["n_pv"]),
            "n_pq": int(row["n_pq"]),
            "ybus_nnz": int(row["ybus_nnz"]),
            "jac_dim": int(row["jac_dim"]),
            "jac_nnz": int(row["jac_nnz"]),
        },
        {
            "method": "cuda_edge_event",
            "label": "CUDA edge",
            "profile": "jac_asm_edge",
            "backend": "cuda",
            "compute": "fp32",
            "jacobian_basis": "edge",
            "timer_scope": "cuda_event_standalone_fill",
            "warmup": warmup,
            "iters": iters,
            "fill_ms": edge_ms,
            "speedup_vs_cuda_bus": bus_ms / edge_ms if edge_ms else float("nan"),
            "n_bus": int(row["n_bus"]),
            "n_pv": int(row["n_pv"]),
            "n_pq": int(row["n_pq"]),
            "ybus_nnz": int(row["ybus_nnz"]),
            "jac_dim": int(row["jac_dim"]),
            "jac_nnz": int(row["jac_nnz"]),
        },
    ]
    write_csv(out_dir / "cuda_event_summary.csv", event_rows)
    return event_rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    pypower_mean = summarize([
        float(row["jacobian_ms"])
        for row in by_method.get("pypower_bus", [])
        if row["success"]
    ])["mean"]

    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = by_method.get(method.method, [])
        values = [float(row["jacobian_ms"]) for row in method_rows if row["success"]]
        stats = summarize(values)
        speedup = pypower_mean / stats["mean"] if stats["mean"] and math.isfinite(stats["mean"]) else float("nan")
        summary_rows.append({
            "method": method.method,
            "label": method.label,
            "profile": method.profile,
            "backend": method.backend,
            "compute": method.compute,
            "jacobian_basis": method.jacobian_basis,
            "timer_scope": method.timer_scope,
            "runs": len(method_rows),
            "success_all": all(bool(row["success"]) for row in method_rows),
            "jacobian_ms_mean": stats["mean"],
            "jacobian_ms_median": stats["median"],
            "jacobian_ms_stdev": stats["stdev"],
            "jacobian_ms_min": stats["min"],
            "jacobian_ms_max": stats["max"],
            "speedup_vs_pypower_mean": speedup,
            "iterations_mean": summarize([float(row["iterations"]) for row in method_rows])["mean"],
            "jacobian_updates_mean": summarize([float(row["jacobian_updates"]) for row in method_rows])["mean"],
        })

    write_csv(path, summary_rows)
    return summary_rows


def write_readme(path: Path,
                 *,
                 case_dir: Path,
                 cupf_binary: Path,
                 out_dir: Path,
                 warmup: int,
                 repeats: int,
                 event_rows: list[dict[str, Any]],
                 summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# case9241pegase Jacobian Assembly Measurement",
        "",
        f"- Generated UTC: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Case dir: `{case_dir}`",
        f"- cuPF operators binary: `{cupf_binary}`",
        f"- Warmup: `{warmup}`",
        f"- Repeats: `{repeats}`",
        "- Metric: average Jacobian assembly time per Newton Jacobian update.",
        "- Python/PyPower uses the dump Ybus/Sbus/V/pv/pq and the repository's PyPower Newton wrapper, which calls PYPOWER `dSbus_dV`.",
        "- C/CUDA use `METRIC name=NR.iteration.jacobian` from the cuPF operators benchmark.",
        "- CUDA profiles are run with `CUDA_LAUNCH_BLOCKING=1`, so the scoped timer waits for kernel completion.",
    ]
    if event_rows:
        lines.append(
            "- `cuda_event_jac_asm_*.csv` is standalone CUDA-event timing for bus/edge fill kernels."
        )
        lines.append(
            "- CUDA bus is measured with the standalone `jac_asm_vertex_thread` kernel because the cuPF solver benchmark exposes CUDA solver profiles only for edge-based Jacobian assembly."
        )
    lines.extend([
        "",
        "## Summary",
        "",
        "| Method | Profile | Mean ms | Median ms | Std ms | Speedup vs PyPower |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | `{row['profile']}` | "
            f"{float(row['jacobian_ms_mean']):.6f} | "
            f"{float(row['jacobian_ms_median']):.6f} | "
            f"{float(row['jacobian_ms_stdev']):.6f} | "
            f"{float(row['speedup_vs_pypower_mean']):.2f}x |"
        )
    if event_rows:
        lines.extend([
            "",
            "## CUDA Event Fill Summary",
            "",
            "| Method | Kernel/Profile | Basis | Fill ms | Speedup vs CUDA bus |",
            "|---|---|---|---:|---:|",
        ])
        for row in event_rows:
            lines.append(
                f"| {row['label']} | `{row['profile']}` | "
                f"{row['jacobian_basis']} | "
                f"{float(row['fill_ms']):.6f} | "
                f"{float(row['speedup_vs_cuda_bus']):.2f}x |"
            )
    lines.extend([
        "",
        "## Files",
        "",
        f"- `raw.csv`: per-repeat rows.",
        f"- `summary.csv`: method-level aggregate rows.",
        f"- `cuda_event_summary.csv`: standalone CUDA event aggregate rows, when enabled.",
        f"- `cuda_event_jac_asm_*.csv`: raw standalone CUDA event timing, when enabled.",
        f"- `stdout_*.txt` / `stderr_*.txt`: raw cuPF benchmark output.",
        f"- Output directory: `{out_dir}`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.out_dir = (
            WORKSPACE_ROOT
            / f"exp/20260423/jac_asm_9000/results/case9241pegase_{stamp}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    meta = case_metadata(args.case_dir)
    common = run_pypower(
        args.case_dir,
        args.warmup,
        args.repeats,
        args.tolerance,
        args.max_iter,
        rows,
    )

    for method in METHODS[1:]:
        run_cupf(
            method,
            args.case_dir,
            args.cupf_binary,
            args.out_dir,
            args.warmup,
            args.repeats,
            args.tolerance,
            args.max_iter,
            args.batch_size,
            args.cuda_launch_blocking,
            rows,
        )

    for row in rows:
        if int(row["ybus_nnz"]) < 0:
            row["ybus_nnz"] = common["ybus_nnz"]

    write_csv(args.out_dir / "raw.csv", rows)
    summary_rows = write_summary(args.out_dir / "summary.csv", rows)
    event_rows: list[dict[str, Any]] = []
    if not args.skip_cuda_event_fill:
        event_rows = run_cuda_event_fill(
            args.case_dir,
            args.jac_asm_binary,
            args.out_dir,
            args.jac_asm_warmup,
            args.jac_asm_iters,
        )
    write_readme(
        args.out_dir / "README.md",
        case_dir=args.case_dir,
        cupf_binary=args.cupf_binary,
        out_dir=args.out_dir,
        warmup=args.warmup,
        repeats=args.repeats,
        event_rows=event_rows,
        summary_rows=summary_rows,
    )
    manifest = {
        "case_dir": str(args.case_dir),
        "metadata": meta,
        "cupf_binary": str(args.cupf_binary),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "tolerance": args.tolerance,
        "max_iter": args.max_iter,
        "batch_size": args.batch_size,
        "cuda_launch_blocking": args.cuda_launch_blocking,
        "jac_asm_binary": str(args.jac_asm_binary),
        "jac_asm_warmup": args.jac_asm_warmup,
        "jac_asm_iters": args.jac_asm_iters,
        "cuda_event_fill_enabled": not args.skip_cuda_event_fill,
        "outputs": {
            "raw": str(args.out_dir / "raw.csv"),
            "summary": str(args.out_dir / "summary.csv"),
            "readme": str(args.out_dir / "README.md"),
            "cuda_event_summary": str(args.out_dir / "cuda_event_summary.csv"),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
