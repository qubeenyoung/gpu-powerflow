#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import scipy.sparse.linalg as spla


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_GPU_BINARY = REPO_ROOT / "cuPF" / "build" / "bench-operators" / "benchmarks" / "cupf_case_benchmark"
DEFAULT_KOREAN_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

if str(REPO_ROOT / "exp" / "20260511") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "exp" / "20260511"))

from benchmarks.utils import build_jacobian, load_case, mismatch_vector  # noqa: E402


if DEFAULT_KOREAN_FONT.exists():
    font_manager.fontManager.addfont(str(DEFAULT_KOREAN_FONT))
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False


COMPONENTS = [
    ("linear_solve", "Linear system"),
    ("jacobian", "Jacobian"),
    ("ibus", "Ibus"),
    ("mismatch", "Mismatch"),
    ("mismatch_norm", "Mismatch norm"),
    ("voltage_update", "Voltage update"),
]

COLORS = {
    "linear_solve": "#E6A04A",
    "jacobian": "#5B8CC0",
    "ibus": "#78A867",
    "mismatch": "#B980B8",
    "mismatch_norm": "#7EC7C2",
    "voltage_update": "#C56B5F",
}

STAGES = [
    ("cpu_reference", "MATPOWER", "Measured Python/SciPy reference"),
    ("linear_cudss_accel", "선형계 GPU 가속", "Synthetic: CPU reference with cuPF cuDSS linear bucket"),
    ("jacobian_gpu_accel", "선형계 + 자코비안 GPU 가속", "Synthetic: cuDSS plus cuPF GPU Jacobian bucket"),
    ("full_cupf_gpu", "cuPF 전체 구조", "Measured cuPF CUDA mixed path"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stage-by-stage pie charts for CPU reference and cuPF GPU acceleration buckets."
    )
    parser.add_argument("--case", default="case13659pegase", help="Case directory name under --dataset-dir.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--gpu-binary", type=Path, default=DEFAULT_GPU_BINARY)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT / "stage_pies")
    parser.add_argument(
        "--reuse-gpu-raw",
        action="store_true",
        help="Reuse raw/gpu_<case>.txt if it exists instead of rerunning cupf_case_benchmark.",
    )
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def timed_cpu_reference(case_dir: Path, *, repeats: int, tolerance: float, max_iter: int) -> dict[str, Any]:
    case = load_case(case_dir)
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        v = np.array(case.v0, dtype=np.complex128, copy=True)
        va = np.angle(v)
        vm = np.abs(v)
        pvpq = case.pvpq
        n_pvpq = int(pvpq.size)

        timers = {
            "mismatch": 0.0,
            "jacobian": 0.0,
            "linear_solve": 0.0,
            "voltage_update": 0.0,
        }
        converged = False
        final_mismatch = math.inf
        completed = 0
        solve_start = time.perf_counter()
        for iteration in range(int(max_iter)):
            t0 = time.perf_counter()
            f = mismatch_vector(case, v)
            timers["mismatch"] += time.perf_counter() - t0

            final_mismatch = float(np.max(np.abs(f))) if f.size else 0.0
            completed = iteration
            if np.isfinite(final_mismatch) and final_mismatch <= tolerance:
                converged = True
                break

            t0 = time.perf_counter()
            jac = build_jacobian(case, v)
            timers["jacobian"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            dx = spla.spsolve(jac, f)
            timers["linear_solve"] += time.perf_counter() - t0
            if not np.all(np.isfinite(dx)):
                break

            t0 = time.perf_counter()
            va[pvpq] -= dx[:n_pvpq]
            vm[case.pq] -= dx[n_pvpq:]
            v = vm * np.exp(1j * va)
            timers["voltage_update"] += time.perf_counter() - t0
            completed = iteration + 1

        solve_sec = time.perf_counter() - solve_start
        other_ops = timers["mismatch"] + timers["voltage_update"]
        accounted = timers["linear_solve"] + timers["jacobian"] + other_ops
        rows.append(
            {
                "repeat": repeat,
                "success": bool(converged),
                "iterations": int(completed),
                "final_mismatch": final_mismatch,
                "solve_ms": solve_sec * 1000.0,
                "linear_solve_ms": timers["linear_solve"] * 1000.0,
                "jacobian_ms": timers["jacobian"] * 1000.0,
                "ibus_ms": 0.0,
                "mismatch_ms": timers["mismatch"] * 1000.0,
                "mismatch_norm_ms": 0.0,
                "voltage_update_ms": timers["voltage_update"] * 1000.0,
                "other_ops_ms": other_ops * 1000.0,
                "outer_overhead_ms": max(0.0, (solve_sec - accounted) * 1000.0),
            }
        )
    return {"rows": rows}


def parse_kv_payload(line: str, prefix: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in line[len(prefix) :].strip().split():
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key] = value
    return parsed


def parse_gpu_raw(raw_text: str) -> dict[str, Any]:
    runs: dict[int, dict[str, str]] = {}
    metrics: dict[int, dict[str, float]] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if line.startswith("RUN "):
            run = parse_kv_payload(line, "RUN ")
            runs[int(run["repeat"])] = run
        elif line.startswith("METRIC "):
            metric = parse_kv_payload(line, "METRIC ")
            repeat = int(metric["repeat"])
            metrics.setdefault(repeat, {})[metric["name"]] = float(metric["total_sec"]) * 1000.0

    rows: list[dict[str, Any]] = []
    for repeat, run in sorted(runs.items()):
        metric = metrics.get(repeat, {})
        linear = (
            metric.get("NR.iteration.prepare_rhs", 0.0)
            + metric.get("NR.iteration.factorize", 0.0)
            + metric.get("NR.iteration.solve", 0.0)
        )
        jacobian = metric.get("NR.iteration.jacobian", 0.0)
        other_ops = (
            metric.get("NR.iteration.ibus", 0.0)
            + metric.get("NR.iteration.mismatch", 0.0)
            + metric.get("NR.iteration.mismatch_norm", 0.0)
            + metric.get("NR.iteration.voltage_update", 0.0)
        )
        solve_ms = float(run["solve_sec"]) * 1000.0
        accounted = linear + jacobian + other_ops
        rows.append(
            {
                "repeat": repeat,
                "success": run["success"].lower() == "true",
                "iterations": int(run["iterations"]),
                "final_mismatch": float(run["final_mismatch"]),
                "solve_ms": solve_ms,
                "linear_solve_ms": linear,
                "jacobian_ms": jacobian,
                "ibus_ms": metric.get("NR.iteration.ibus", 0.0),
                "mismatch_ms": metric.get("NR.iteration.mismatch", 0.0),
                "mismatch_norm_ms": metric.get("NR.iteration.mismatch_norm", 0.0),
                "voltage_update_ms": metric.get("NR.iteration.voltage_update", 0.0),
                "other_ops_ms": other_ops,
                "outer_overhead_ms": max(0.0, solve_ms - accounted),
            }
        )
    return {"rows": rows}


def run_gpu_benchmark(args: argparse.Namespace, raw_path: Path) -> dict[str, Any]:
    if args.reuse_gpu_raw and raw_path.exists():
        return parse_gpu_raw(raw_path.read_text(encoding="utf-8"))

    cmd = [
        str(args.gpu_binary),
        "--case-dir",
        str(args.dataset_dir / args.case),
        "--profile",
        "cuda_mixed_edge",
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--tolerance",
        str(args.tolerance),
        "--max-iter",
        str(args.max_iter),
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    raw_path.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        raw_path.with_suffix(".stderr.txt").write_text(completed.stderr, encoding="utf-8")
    return parse_gpu_raw(completed.stdout)


def mean_components(result: dict[str, Any]) -> dict[str, float]:
    rows = result["rows"]
    return {
        "linear_solve": summarize([row["linear_solve_ms"] for row in rows])["mean"],
        "jacobian": summarize([row["jacobian_ms"] for row in rows])["mean"],
        "ibus": summarize([row["ibus_ms"] for row in rows])["mean"],
        "mismatch": summarize([row["mismatch_ms"] for row in rows])["mean"],
        "mismatch_norm": summarize([row["mismatch_norm_ms"] for row in rows])["mean"],
        "voltage_update": summarize([row["voltage_update_ms"] for row in rows])["mean"],
        "other_ops": summarize([row["other_ops_ms"] for row in rows])["mean"],
        "outer_overhead": summarize([row["outer_overhead_ms"] for row in rows])["mean"],
        "solve": summarize([row["solve_ms"] for row in rows])["mean"],
    }


def build_stage_rows(case_name: str, cpu: dict[str, float], gpu: dict[str, float]) -> list[dict[str, Any]]:
    stage_values = {
        "cpu_reference": {
            "linear_solve": cpu["linear_solve"],
            "jacobian": cpu["jacobian"],
            "ibus": cpu["ibus"],
            "mismatch": cpu["mismatch"],
            "mismatch_norm": cpu["mismatch_norm"],
            "voltage_update": cpu["voltage_update"],
        },
        "linear_cudss_accel": {
            "linear_solve": gpu["linear_solve"],
            "jacobian": cpu["jacobian"],
            "ibus": cpu["ibus"],
            "mismatch": cpu["mismatch"],
            "mismatch_norm": cpu["mismatch_norm"],
            "voltage_update": cpu["voltage_update"],
        },
        "jacobian_gpu_accel": {
            "linear_solve": gpu["linear_solve"],
            "jacobian": gpu["jacobian"],
            "ibus": cpu["ibus"],
            "mismatch": cpu["mismatch"],
            "mismatch_norm": cpu["mismatch_norm"],
            "voltage_update": cpu["voltage_update"],
        },
        "full_cupf_gpu": {
            "linear_solve": gpu["linear_solve"],
            "jacobian": gpu["jacobian"],
            "ibus": gpu["ibus"],
            "mismatch": gpu["mismatch"],
            "mismatch_norm": gpu["mismatch_norm"],
            "voltage_update": gpu["voltage_update"],
        },
    }

    rows: list[dict[str, Any]] = []
    baseline_total = sum(stage_values["cpu_reference"].values())
    for stage_id, stage_label, note in STAGES:
        values = stage_values[stage_id]
        total = sum(values.values())
        for component_id, component_label in COMPONENTS:
            value = values[component_id]
            rows.append(
                {
                    "case": case_name,
                    "stage": stage_id,
                    "stage_label": stage_label,
                    "component": component_id,
                    "component_label": component_label,
                    "time_ms": value,
                    "share_pct": 100.0 * value / total if total > 0 else math.nan,
                    "stage_total_ms": total,
                    "speedup_vs_cpu_reference": baseline_total / total if total > 0 else math.nan,
                    "note": note,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_stage_pies(rows: list[dict[str, Any]], output_dir: Path, case_name: str) -> None:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_stage.setdefault(row["stage"], []).append(row)

    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.4), dpi=150)
    fig.subplots_adjust(left=0.02, right=0.82, bottom=0.24, top=0.96, wspace=0.05)
    legend_handles = None
    legend_labels = None

    for ax, (stage_id, stage_label, _) in zip(axes, STAGES):
        stage_rows = sorted(by_stage[stage_id], key=lambda row: [c[0] for c in COMPONENTS].index(row["component"]))
        values = [float(row["time_ms"]) for row in stage_rows]
        colors = [COLORS[row["component"]] for row in stage_rows]
        total = float(stage_rows[0]["stage_total_ms"])
        speedup = float(stage_rows[0]["speedup_vs_cpu_reference"])
        wedges, texts, autotexts = ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 5.0 else "",
            pctdistance=0.67,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
            textprops={"fontsize": 15},
        )
        for text in autotexts:
            text.set_fontsize(15)
        ax.text(
            0.5,
            -0.12,
            f"{total:.2f} ms",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=17,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "white",
                "edgecolor": "#404040",
                "linewidth": 0.8,
            },
        )
        ax.axis("equal")
        if legend_handles is None:
            legend_handles = wedges
            legend_labels = [
                f"{row['component_label']} ({row['time_ms']:.2f} ms in full GPU)"
                if row["stage"] == "full_cupf_gpu"
                else row["component_label"]
                for row in stage_rows
            ]

        bbox = ax.get_position()
        fig.text(
            bbox.x0 + bbox.width / 2.0,
            0.945,
            stage_label,
            ha="center",
            va="center",
            fontsize=17,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "#404040",
                "linewidth": 0.8,
            },
        )

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            [label.split(" (")[0] for label in legend_labels],
            loc="center right",
            bbox_to_anchor=(0.99, 0.5),
            frameon=False,
            fontsize=17,
        )

    base = output_dir / f"stage_pies_{case_name}"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def write_report(
    path: Path,
    *,
    case_name: str,
    cpu: dict[str, float],
    gpu: dict[str, float],
    rows: list[dict[str, Any]],
    gpu_binary: Path,
) -> None:
    totals: dict[str, tuple[float, float]] = {}
    for stage_id, stage_label, _ in STAGES:
        stage_rows = [row for row in rows if row["stage"] == stage_id]
        if not stage_rows:
            continue
        totals[stage_label] = (
            float(stage_rows[0]["stage_total_ms"]),
            float(stage_rows[0]["speedup_vs_cpu_reference"]),
        )

    lines = [
        f"# cuPF Stage Pie Check: {case_name}",
        "",
        "## Verdict",
        "",
        "- Feasible: yes, for measured CPU reference and full cuPF GPU buckets.",
        "- Intermediate pies for `+ cuDSS linear solve` and `+ GPU Jacobian` are compositional estimates: they replace one measured CPU bucket at a time with the corresponding measured cuPF bucket.",
        "- A true ablation pie would require dedicated `cuda_wo_cudss` / `cuda_wo_jacobian` benchmark paths in the current 20260511 flow.",
        "",
        "## Inputs",
        "",
        f"- CPU reference: Python/SciPy Newton path from `exp/20260511/benchmarks/utils.py` with local per-stage timers.",
        f"- Full cuPF GPU: `{gpu_binary}` profile `cuda_mixed_edge` with `ENABLE_TIMING=ON`.",
        "- Linear bucket: `prepare_rhs + factorize + solve`.",
        "- Expanded NR buckets: `ibus`, `mismatch`, `mismatch_norm`, and `voltage_update`; CPU reference has zero for `ibus` and `mismatch_norm` because those are not separately timed there.",
        "- `outer / transfer` is retained in timing CSVs as a diagnostic but excluded from the pies and stage totals.",
        "",
        "## Stage Totals",
        "",
        "| stage | total ms | speedup vs CPU reference |",
        "| --- | ---: | ---: |",
    ]
    for stage_label, (total, speedup) in totals.items():
        lines.append(f"| {stage_label} | {total:.3f} | {speedup:.2f}x |")

    lines.extend(
        [
            "",
            "## Component Means",
            "",
            "| source | linear ms | jacobian ms | ibus ms | mismatch ms | mismatch norm ms | voltage update ms | excluded outer/transfer ms | measured solve ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| CPU reference | {cpu['linear_solve']:.3f} | {cpu['jacobian']:.3f} | {cpu['ibus']:.3f} | {cpu['mismatch']:.3f} | {cpu['mismatch_norm']:.3f} | {cpu['voltage_update']:.3f} | {cpu['outer_overhead']:.3f} | {cpu['solve']:.3f} |",
            f"| Full cuPF GPU | {gpu['linear_solve']:.3f} | {gpu['jacobian']:.3f} | {gpu['ibus']:.3f} | {gpu['mismatch']:.3f} | {gpu['mismatch_norm']:.3f} | {gpu['voltage_update']:.3f} | {gpu['outer_overhead']:.3f} | {gpu['solve']:.3f} |",
            "",
            "## Outputs",
            "",
            f"- `stage_pies_{case_name}.png` / `.pdf`",
            "- `stage_pie_data.csv`",
            "- `cpu_reference_timing.csv`",
            "- `gpu_timing.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    case_dir = args.dataset_dir / args.case
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    if not args.gpu_binary.exists():
        raise FileNotFoundError(f"GPU benchmark binary not found: {args.gpu_binary}")

    cpu_result = timed_cpu_reference(
        case_dir,
        repeats=args.repeats,
        tolerance=args.tolerance,
        max_iter=args.max_iter,
    )
    gpu_result = run_gpu_benchmark(args, raw_dir / f"gpu_{args.case}.txt")

    cpu_rows = cpu_result["rows"]
    gpu_rows = gpu_result["rows"]
    write_csv(output_dir / "cpu_reference_timing.csv", cpu_rows)
    write_csv(output_dir / "gpu_timing.csv", gpu_rows)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "case": args.case,
                "dataset_dir": str(args.dataset_dir),
                "gpu_binary": str(args.gpu_binary),
                "warmup": args.warmup,
                "repeats": args.repeats,
                "tolerance": args.tolerance,
                "max_iter": args.max_iter,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    cpu = mean_components(cpu_result)
    gpu = mean_components(gpu_result)
    stage_rows = build_stage_rows(args.case, cpu, gpu)
    write_csv(output_dir / "stage_pie_data.csv", stage_rows)
    draw_stage_pies(stage_rows, output_dir, args.case)
    write_report(
        output_dir / "README.md",
        case_name=args.case,
        cpu=cpu,
        gpu=gpu,
        rows=stage_rows,
        gpu_binary=args.gpu_binary,
    )
    print(f"Wrote stage pie outputs to {output_dir}")


if __name__ == "__main__":
    main()
