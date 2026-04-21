#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = EXP_ROOT.parents[2]
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "exp" / "20260414" / "amgx" / "cupf_dumps"
DEFAULT_BUILD_DIR = EXP_ROOT / "build" / "operator_probe"
DEFAULT_RESULTS_ROOT = EXP_ROOT / "results" / "operator_speed"
DEFAULT_CASE_LIST = EXP_ROOT / "cases_ncu_speed.txt"
DEFAULT_MODES = ["edge_atomic", "edge_noatomic", "vertex"]
LANE_UTIL_BY_CASE = {
    "case_ACTIVSg25k": {
        "label": "ACTIVSg",
        "order": 0,
        "vertex_lane_util_percent": 49.27,
        "edge_lane_util_percent": 95.80,
    },
    "Base_Eastern_Interconnect_515GW": {
        "label": "Base Eastern",
        "order": 1,
        "vertex_lane_util_percent": 49.27,
        "edge_lane_util_percent": 96.66,
    },
    "Base_West_Interconnect_121GW": {
        "label": "Base West",
        "order": 2,
        "vertex_lane_util_percent": 49.27,
        "edge_lane_util_percent": 96.59,
    },
    "MemphisCase2026_Mar7": {
        "label": "Memphis",
        "order": 3,
        "vertex_lane_util_percent": 49.27,
        "edge_lane_util_percent": 96.61,
    },
    "Texas7k_20220923": {
        "label": "Texas7K",
        "order": 4,
        "vertex_lane_util_percent": 49.27,
        "edge_lane_util_percent": 96.30,
    },
}


def read_case_list(path: Path) -> list[str]:
    cases: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            cases.append(stripped)
    return cases


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("[RUN]", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def configure_and_build(build_dir: Path, cuda_arch: str | None, command_log: list[dict[str, Any]]) -> Path:
    configure_cmd = [
        "cmake",
        "-S", str(EXP_ROOT),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if cuda_arch:
        configure_cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
    build_cmd = [
        "cmake",
        "--build", str(build_dir),
        "--target", "jacobian_operator_probe",
        "-j", str(os.cpu_count() or 1),
    ]
    command_log.append({"name": "cmake_configure", "cmd": configure_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(configure_cmd, cwd=WORKSPACE_ROOT)
    command_log.append({"name": "cmake_build", "cmd": build_cmd, "cwd": str(WORKSPACE_ROOT)})
    run_command(build_cmd, cwd=WORKSPACE_ROOT)
    binary = build_dir / "jacobian_operator_probe"
    if not binary.exists():
        raise FileNotFoundError(f"probe binary was not built: {binary}")
    return binary


def percentile_nearest(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["case"], row["mode"]), []).append(row)

    out: list[dict[str, Any]] = []
    for (case, mode), group in sorted(grouped.items()):
        elapsed = [float(row["elapsed_ms"]) for row in group]
        mean = statistics.mean(elapsed) if elapsed else 0.0
        out.append({
            "case": case,
            "mode": mode,
            "repeats": len(elapsed),
            "mean_ms": mean,
            "median_ms": statistics.median(elapsed) if elapsed else 0.0,
            "stdev_ms": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
            "min_ms": min(elapsed, default=0.0),
            "p95_ms": percentile_nearest(elapsed, 0.95),
            "max_ms": max(elapsed, default=0.0),
            "n_bus": group[0]["n_bus"],
            "ybus_nnz": group[0]["ybus_nnz"],
            "jacobian_dim": group[0]["jacobian_dim"],
            "jacobian_nnz": group[0]["jacobian_nnz"],
        })
    return out


def geometric_mean(values: list[float]) -> float:
    positive = [value for value in values if value > 0.0]
    if not positive:
        return 0.0
    return math.prod(positive) ** (1.0 / len(positive))


def compare_modes(summary_rows: list[dict[str, Any]], case_order: list[str]) -> list[dict[str, Any]]:
    by_case_mode = {
        (str(row["case"]), str(row["mode"])): row
        for row in summary_rows
    }
    out: list[dict[str, Any]] = []
    for case in case_order:
        required = [
            by_case_mode.get((case, "edge_atomic")),
            by_case_mode.get((case, "edge_noatomic")),
            by_case_mode.get((case, "vertex")),
        ]
        if any(row is None for row in required):
            continue
        edge_atomic, edge_noatomic, vertex = required
        edge_atomic_ms = float(edge_atomic["mean_ms"])
        edge_noatomic_ms = float(edge_noatomic["mean_ms"])
        vertex_ms = float(vertex["mean_ms"])
        edge_atomic_us = edge_atomic_ms * 1000.0
        edge_noatomic_us = edge_noatomic_ms * 1000.0
        vertex_us = vertex_ms * 1000.0
        raw_atomic_time_share = (
            (edge_atomic_ms - edge_noatomic_ms) / edge_atomic_ms
            if edge_atomic_ms
            else 0.0
        )
        atomic_time_share = max(0.0, raw_atomic_time_share)
        lane_util = LANE_UTIL_BY_CASE.get(case, {})
        out.append({
            "case_label": lane_util.get("label", case),
            "case": case,
            "n_bus": edge_atomic["n_bus"],
            "ybus_nnz": edge_atomic["ybus_nnz"],
            "edge_atomic_mean_us": edge_atomic_us,
            "edge_noatomic_mean_us": edge_noatomic_us,
            "vertex_mean_us": vertex_us,
            "edge_atomic_speedup_vs_vertex": vertex_ms / edge_atomic_ms if edge_atomic_ms else 0.0,
            "edge_noatomic_speedup_vs_vertex": vertex_ms / edge_noatomic_ms if edge_noatomic_ms else 0.0,
            "atomic_time_share_estimate_percent": atomic_time_share * 100.0,
            "raw_atomic_time_share_estimate_percent": raw_atomic_time_share * 100.0,
            "vertex_lane_util_percent": lane_util.get("vertex_lane_util_percent"),
            "edge_lane_util_percent": lane_util.get("edge_lane_util_percent"),
        })
    case_index = {case: index for index, case in enumerate(case_order)}
    return sorted(
        out,
        key=lambda row: (
            LANE_UTIL_BY_CASE.get(str(row["case"]), {}).get("order", len(LANE_UTIL_BY_CASE)),
            case_index.get(str(row["case"]), len(case_index)),
        ),
    )


def format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def write_markdown_summary(path: Path, rows: list[dict[str, Any]], warmup: int, repeats: int) -> None:
    lines = [
        "# Jacobian Operator Speed Probe",
        "",
        f"This run used `jacobian_operator_probe` with `warmup={warmup}` and `repeats={repeats}`.",
        "Timing is CUDA event elapsed time for one Jacobian operator run, including the",
        "Jacobian value-buffer zeroing done by each operator.",
        "",
        "`edge_noatomic` is not a valid solver path. It is the same edge work with",
        "plain load/add/store replacing `atomicAdd`, so the numbers are only an",
        "upper-bound style probe for atomic overhead.",
        "",
    ]

    if rows:
        lines.extend([
            "`edge/vertex speedup` is `vertex mean us / edge_atomic mean us`.",
            "`atomic time share` estimates the fraction of edge_atomic time attributable",
            "to atomic updates: `(edge_atomic mean us - edge_noatomic mean us) / edge_atomic mean us * 100`.",
            "Negative estimates from timing noise are reported as 0.0%.",
            "",
            "| Case | Edge atomic (us) | Vertex (us) | Edge/Vertex speedup | Edge no-atomic (us) | Atomic time share | Vertex lane util | Edge lane util |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                "| {case_label} | {edge_atomic_mean_us:.3f} | {vertex_mean_us:.3f} | "
                "{edge_atomic_speedup_vs_vertex:.3f}x | {edge_noatomic_mean_us:.3f} | "
                "{atomic_time_share_estimate_percent:.1f}% | "
                "{vertex_lane_util} | {edge_lane_util} |".format(
                    vertex_lane_util=format_percent(row["vertex_lane_util_percent"]),
                    edge_lane_util=format_percent(row["edge_lane_util_percent"]),
                    **row,
                )
            )

        edge_speedups = [float(row["edge_atomic_speedup_vs_vertex"]) for row in rows]
        atomic_time_shares = [float(row["atomic_time_share_estimate_percent"]) for row in rows]
        lines.extend([
            "",
            f"Across these {len(rows)} cases, the geometric-mean edge_atomic speedup over vertex is",
            f"{geometric_mean(edge_speedups):.3f}x. The arithmetic-mean atomic time share estimate is",
            f"{statistics.mean(atomic_time_shares):.1f}%. Small cases can be dominated by timing granularity noise.",
            "",
        ])
    else:
        lines.extend([
            "No comparison table was generated because the selected modes did not include",
            "`edge_atomic`, `edge_noatomic`, and `vertex` for the same cases.",
            "",
        ])

    lines.extend([
        "Raw repeat data:",
        "",
        "- `operator_speed_raw.csv`",
        "- `operator_speed_summary.csv`",
        "- `operator_speed_comparison.csv`",
        "- `raw/*.csv`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Jacobian operator-only speed probe for edge atomic/no-atomic and vertex modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--case-list", type=Path, default=DEFAULT_CASE_LIST)
    parser.add_argument("--cases", nargs="*", help="Override case-list with explicit case names.")
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cuda-visible-devices", help="Set CUDA_VISIBLE_DEVICES for the probe.")
    parser.add_argument("--cuda-arch", help="Override CMAKE_CUDA_ARCHITECTURES, for example 86.")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = args.cases if args.cases is not None else read_case_list(args.case_list)
    if not cases:
        raise SystemExit("no cases selected")

    command_log: list[dict[str, Any]] = []
    binary = args.build_dir / "jacobian_operator_probe"
    if args.skip_build:
        if not binary.exists():
            raise FileNotFoundError(f"--skip-build was set but probe binary is missing: {binary}")
    else:
        binary = configure_and_build(args.build_dir, args.cuda_arch, command_log)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    for case in cases:
        case_dir = args.dataset_root / case
        if not case_dir.exists():
            raise FileNotFoundError(f"case directory not found: {case_dir}")
        for mode in args.modes:
            cmd = [
                str(binary),
                "--case-dir", str(case_dir),
                "--mode", mode,
                "--warmup", str(args.warmup),
                "--repeats", str(args.repeats),
            ]
            command_log.append({"name": f"probe_{case}_{mode}", "cmd": cmd, "cwd": str(WORKSPACE_ROOT)})
            completed = run_command(cmd, cwd=WORKSPACE_ROOT, env=env)
            raw_path = run_root / "raw" / f"{case}_{mode}.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(completed.stdout, encoding="utf-8")

            reader = csv.DictReader(io.StringIO(completed.stdout))
            all_rows.extend(reader)

    raw_fields = ["case", "mode", "repeat", "elapsed_ms", "n_bus", "n_pv", "n_pq", "ybus_nnz", "jacobian_dim", "jacobian_nnz"]
    summary_fields = [
        "case",
        "mode",
        "repeats",
        "mean_ms",
        "median_ms",
        "stdev_ms",
        "min_ms",
        "p95_ms",
        "max_ms",
        "n_bus",
        "ybus_nnz",
        "jacobian_dim",
        "jacobian_nnz",
    ]
    comparison_fields = [
        "case_label",
        "case",
        "n_bus",
        "ybus_nnz",
        "edge_atomic_mean_us",
        "edge_noatomic_mean_us",
        "vertex_mean_us",
        "edge_atomic_speedup_vs_vertex",
        "edge_noatomic_speedup_vs_vertex",
        "atomic_time_share_estimate_percent",
        "raw_atomic_time_share_estimate_percent",
        "vertex_lane_util_percent",
        "edge_lane_util_percent",
    ]
    summary_rows = summarize(all_rows)
    comparison_rows = compare_modes(summary_rows, cases)
    write_csv(run_root / "operator_speed_raw.csv", all_rows, raw_fields)
    write_csv(run_root / "operator_speed_summary.csv", summary_rows, summary_fields)
    write_csv(run_root / "operator_speed_comparison.csv", comparison_rows, comparison_fields)
    write_markdown_summary(run_root / "SUMMARY.md", comparison_rows, args.warmup, args.repeats)
    (run_root / "command_log.json").write_text(
        json.dumps({
            "dataset_root": str(args.dataset_root),
            "cases": cases,
            "modes": args.modes,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            "commands": command_log,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] wrote {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
