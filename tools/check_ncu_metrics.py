#!/usr/bin/env python3
"""Resolve Nsight Compute metric aliases against collected CSVs and ncu query output."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class AliasSpec:
    alias: str
    group: str
    exact: tuple[str, ...] = ()
    regex: tuple[str, ...] = ()
    description: str = ""


def alias_specs() -> list[AliasSpec]:
    """Metric aliases used by the cuDSS bottleneck analyzer.

    The exact names are raw Nsight Compute metric names when known. Regexes are
    intentionally broad enough to handle Nsight Compute version and GPU changes.
    """

    dep = "dependency_scheduler"
    mem = "irregular_memory"
    atom = "atomic_reduction"
    return [
        AliasSpec("gpu_time", dep, ("gpu__time_duration.sum",), description="GPU kernel duration"),
        AliasSpec("active_warps_per_cycle", dep, ("smsp__warps_active.avg.per_cycle_active",)),
        AliasSpec("eligible_warps_per_cycle", dep, ("smsp__warps_eligible.avg.per_cycle_active",)),
        AliasSpec("issue_percent", dep, ("smsp__issue_active.avg.pct_of_peak_sustained_active",)),
        AliasSpec("issue_per_cycle", dep, ("smsp__issue_active.avg.per_cycle_active",)),
        AliasSpec(
            "long_scoreboard_percent",
            dep,
            ("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",),
            (r"long.*scoreboard",),
        ),
        AliasSpec(
            "short_scoreboard_percent",
            dep,
            ("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",),
            (r"short.*scoreboard",),
        ),
        AliasSpec(
            "wait_percent",
            dep,
            ("smsp__warp_issue_stalled_wait_per_warp_active.pct",),
            (r"warp.*stall.*wait|wait.*per_warp_active",),
        ),
        AliasSpec(
            "barrier_percent",
            dep,
            ("smsp__warp_issue_stalled_barrier_per_warp_active.pct",),
            (r"warp.*stall.*barrier|barrier.*per_warp_active",),
        ),
        AliasSpec(
            "not_selected_percent",
            dep,
            ("smsp__warp_issue_stalled_not_selected_per_warp_active.pct",),
            (r"not_selected",),
        ),
        AliasSpec(
            "no_instruction_percent",
            dep,
            ("smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",),
            (r"no_instruction",),
        ),
        AliasSpec(
            "compute_percent",
            dep,
            ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
            (r"^sm__throughput.*pct_of_peak",),
        ),
        AliasSpec(
            "memory_percent",
            dep,
            ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",),
            (r"gpu__compute_memory.*throughput.*pct_of_peak",),
        ),
        AliasSpec(
            "dram_percent",
            dep,
            ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",),
            (r"gpu__dram.*throughput.*pct_of_peak|dram__throughput.*pct_of_peak",),
        ),
        AliasSpec(
            "occupancy_percent",
            dep,
            ("sm__warps_active.avg.pct_of_peak_sustained_active",),
            (r"sm__warps_active.*pct_of_peak",),
        ),
        AliasSpec(
            "l1_global_load_requests",
            mem,
            ("l1tex__t_requests_pipe_lsu_mem_global_op_ld",),
            (r"l1tex.*(t_)?requests.*global.*op_ld", r"l1tex.*global.*ld.*request"),
        ),
        AliasSpec(
            "l1_global_load_sectors",
            mem,
            ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld",),
            (r"l1tex.*(t_)?sectors.*global.*op_ld", r"l1tex.*global.*ld.*sector"),
        ),
        AliasSpec(
            "l1_global_load_sectors_per_request",
            mem,
            ("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld",),
            (r"l1tex.*average.*sectors.*request.*global.*op_ld",),
        ),
        AliasSpec(
            "l1_global_store_requests",
            mem,
            ("l1tex__t_requests_pipe_lsu_mem_global_op_st",),
            (r"l1tex.*(t_)?requests.*global.*op_st", r"l1tex.*global.*st.*request"),
        ),
        AliasSpec(
            "l1_global_store_sectors",
            mem,
            ("l1tex__t_sectors_pipe_lsu_mem_global_op_st",),
            (r"l1tex.*(t_)?sectors.*global.*op_st", r"l1tex.*global.*st.*sector"),
        ),
        AliasSpec(
            "l1_global_store_sectors_per_request",
            mem,
            ("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st",),
            (r"l1tex.*average.*sectors.*request.*global.*op_st",),
        ),
        AliasSpec(
            "l1_hit_rate",
            mem,
            (),
            (r"l1tex.*hit_rate", r"l1tex.*sector.*hit"),
        ),
        AliasSpec(
            "l1_sector_misses_to_l2",
            mem,
            (),
            (r"l1tex.*sector.*miss.*l2", r"l1tex.*miss.*sector", r"l1tex.*xbar.*sector"),
        ),
        AliasSpec(
            "l2_read_requests",
            mem,
            ("lts__t_requests_op_read", "lts__t_requests_aperture_device_op_read"),
            (r"lts.*request.*read", r"l2.*request.*read", r"lts.*requests.*op_read"),
        ),
        AliasSpec(
            "l2_read_sectors",
            mem,
            ("lts__t_sectors_op_read", "lts__t_sectors_aperture_device_op_read"),
            (r"lts.*sector.*read", r"l2.*sector.*read", r"lts.*sectors.*op_read"),
        ),
        AliasSpec(
            "l2_write_requests",
            mem,
            ("lts__t_requests_op_write", "lts__t_requests_aperture_device_op_write"),
            (r"lts.*request.*write", r"l2.*request.*write", r"lts.*requests.*op_write"),
        ),
        AliasSpec(
            "l2_write_sectors",
            mem,
            ("lts__t_sectors_op_write", "lts__t_sectors_aperture_device_op_write"),
            (r"lts.*sector.*write", r"l2.*sector.*write", r"lts.*sectors.*op_write"),
        ),
        AliasSpec(
            "l2_hit_rate",
            mem,
            (),
            (r"lts.*hit_rate", r"l2.*hit_rate", r"lts.*sector.*hit"),
        ),
        AliasSpec(
            "l2_sector_misses_to_device",
            mem,
            (),
            (r"lts.*sector.*miss.*device", r"lts.*miss.*sector", r"lts.*dram.*sector"),
        ),
        AliasSpec(
            "ideal_sectors_if_available",
            mem,
            (),
            (r"ideal.*sector", r"source.*ideal.*sector"),
        ),
        AliasSpec(
            "excessive_sectors_if_available",
            mem,
            (),
            (r"excessive.*sector", r"uncoalesced", r"source.*excessive"),
        ),
        AliasSpec(
            "atomic_inst",
            atom,
            ("smsp__sass_inst_executed_op_global_atom", "sm__sass_inst_executed_op_global_atom"),
            (r"^(sm|smsp)__.*inst.*op_global_atom($|[._])", r"^(sm|smsp)__sass_inst_executed_op_atom$"),
        ),
        AliasSpec(
            "atomic_thread_inst",
            atom,
            (),
            (r"^(sm|smsp)__.*thread.*op_global_atom($|[._])",),
        ),
        AliasSpec(
            "reduction_inst",
            atom,
            (
                "smsp__sass_inst_executed_op_global_red",
                "smsp__inst_executed_op_global_red",
                "sm__sass_inst_executed_op_global_red",
            ),
            (r"^(sm|smsp)__.*inst.*op_global_red($|[._])", r"^(sm|smsp)__.*inst.*op_red($|[._])"),
        ),
        AliasSpec(
            "reduction_thread_inst",
            atom,
            (),
            (r"^(sm|smsp)__.*thread.*op_global_red($|[._])", r"thread.*reduction"),
        ),
        AliasSpec(
            "l1_atomic_requests",
            atom,
            ("l1tex__t_requests_pipe_lsu_mem_global_op_atom",),
            (r"l1tex.*request.*global.*atom", r"l1tex.*global_op_atom.*request"),
        ),
        AliasSpec(
            "l1_atomic_sectors",
            atom,
            ("l1tex__t_sectors_pipe_lsu_mem_global_op_atom",),
            (r"l1tex.*sector.*global.*atom", r"l1tex.*global_op_atom.*sector"),
        ),
        AliasSpec(
            "l1_atomic_sectors_per_request",
            atom,
            ("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom",),
            (r"l1tex.*average.*sector.*request.*global.*atom",),
        ),
        AliasSpec(
            "l1_reduction_requests",
            atom,
            ("l1tex__t_requests_pipe_lsu_mem_global_op_red",),
            (r"l1tex.*request.*global.*red", r"l1tex.*global_op_red.*request|reduction.*request"),
        ),
        AliasSpec(
            "l1_reduction_sectors",
            atom,
            ("l1tex__t_sectors_pipe_lsu_mem_global_op_red",),
            (r"l1tex.*sector.*global.*red", r"l1tex.*global_op_red.*sector|reduction.*sector"),
        ),
        AliasSpec(
            "l1_reduction_sectors_per_request",
            atom,
            ("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_red",),
            (r"l1tex.*average.*sector.*request.*global.*red",),
        ),
        AliasSpec(
            "l2_atomic_requests",
            atom,
            ("lts__t_requests_op_atom", "lts__t_requests_aperture_device_op_atom"),
            (r"lts.*request.*atom", r"l2.*request.*atom"),
        ),
        AliasSpec(
            "l2_atomic_sectors",
            atom,
            ("lts__t_sectors_op_atom", "lts__t_sectors_aperture_device_op_atom"),
            (r"lts.*sector.*atom", r"l2.*sector.*atom"),
        ),
        AliasSpec(
            "l2_reduction_requests",
            atom,
            ("lts__t_requests_op_red", "lts__t_requests_aperture_device_op_red"),
            (r"lts.*request.*op_red($|[._])", r"l2.*request.*red"),
        ),
        AliasSpec(
            "l2_reduction_sectors",
            atom,
            ("lts__t_sectors_op_red", "lts__t_sectors_aperture_device_op_red"),
            (r"lts.*sector.*op_red($|[._])", r"l2.*sector.*red"),
        ),
    ]


def clean_metric_name(name: object) -> str:
    return str(name).strip().strip('"')


def read_csv_metrics(paths: Iterable[Path]) -> set[str]:
    metrics: set[str] = set()
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path, comment="=")
        except Exception:
            continue
        if "Metric Name" not in df.columns:
            continue
        metrics.update(clean_metric_name(v) for v in df["Metric Name"].dropna())
    return {m for m in metrics if m}


def read_query_metrics(paths: Iterable[Path]) -> set[str]:
    metrics: set[str] = set()
    metric_line = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*__[A-Za-z0-9_.$]+)\s+")
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            m = metric_line.match(line)
            if m:
                metrics.add(m.group(1).strip())
    return metrics


def query_ncu_metrics(ncu_bin: str, out_file: Path) -> set[str]:
    if shutil.which(ncu_bin) is None:
        return set()
    try:
        proc = subprocess.run(
            [ncu_bin, "--query-metrics"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        return set()
    out_file.write_text(proc.stdout)
    return read_query_metrics([out_file])


def find_candidates(spec: AliasSpec, metrics: set[str]) -> list[str]:
    lower_to_original = {m.lower(): m for m in metrics}
    found: list[str] = []
    for exact in spec.exact:
        if exact.lower() in lower_to_original:
            found.append(lower_to_original[exact.lower()])
    for pattern in spec.regex:
        rx = re.compile(pattern, re.IGNORECASE)
        for metric in sorted(metrics):
            if rx.search(metric) and metric not in found:
                found.append(metric)
    return found


def fuzzy_suggestions(spec: AliasSpec, metrics: set[str], limit: int = 8) -> list[str]:
    if not metrics:
        return []
    words = re.split(r"[_\W]+", spec.alias)
    probe = ".*".join(re.escape(w) for w in words if w)
    suggestions: list[str] = []
    if probe:
        rx = re.compile(probe, re.IGNORECASE)
        suggestions.extend(m for m in sorted(metrics) if rx.search(m))
    names = sorted(metrics)
    suggestions.extend(difflib.get_close_matches(spec.alias, names, n=limit, cutoff=0.25))
    out: list[str] = []
    for value in suggestions:
        if value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def resolve_metric_aliases(
    collected_metrics: set[str],
    query_metrics: set[str],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    resolved: dict[str, dict[str, object]] = {}
    missing: list[dict[str, object]] = []
    all_metrics = collected_metrics | query_metrics
    for spec in alias_specs():
        collected_candidates = find_candidates(spec, collected_metrics)
        query_candidates = find_candidates(spec, query_metrics)
        metric = collected_candidates[0] if collected_candidates else None
        status = "collected" if metric else ("supported_not_collected" if query_candidates else "missing")
        row = {
            "alias": spec.alias,
            "group": spec.group,
            "metric": metric,
            "status": status,
            "collected_candidates": collected_candidates[:20],
            "query_candidates": query_candidates[:20],
            "exact": list(spec.exact),
            "regex": list(spec.regex),
            "description": spec.description,
        }
        resolved[spec.alias] = row
        if status != "collected":
            missing.append(
                {
                    "alias": spec.alias,
                    "group": spec.group,
                    "status": status,
                    "reason": "not_collected_in_csv" if query_candidates else "not_found_in_csv_or_query",
                    "query_candidates": ";".join(query_candidates[:12]),
                    "fuzzy_suggestions": ";".join(fuzzy_suggestions(spec, all_metrics)),
                }
            )
    return resolved, missing


def write_missing(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["alias", "group", "status", "reason", "query_candidates", "fuzzy_suggestions"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_csvs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.csv or []:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.csv")))
        else:
            paths.append(p)
    for item in args.exports or []:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.csv")))
    return sorted(dict.fromkeys(paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="*", help="Raw/details/source CSV files or directories.")
    parser.add_argument("--exports", nargs="*", default=["exports"], help="Export directories to scan.")
    parser.add_argument("--query-file", nargs="*", default=[], help="Saved ncu --query-metrics output files.")
    parser.add_argument("--query-ncu", action="store_true", help="Run ncu --query-metrics and include the result.")
    parser.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    parser.add_argument("--out", default="out", help="Output directory.")
    parser.add_argument("--missing", default=None, help="Missing metric CSV path.")
    parser.add_argument("--aliases", default=None, help="Resolved alias JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = discover_csvs(args)
    collected_metrics = read_csv_metrics(csv_paths)
    query_paths = [Path(p) for p in args.query_file]
    query_metrics = read_query_metrics(query_paths)
    if args.query_ncu:
        query_metrics |= query_ncu_metrics(args.ncu_bin, out_dir / "ncu_query_metrics.txt")
    resolved, missing = resolve_metric_aliases(collected_metrics, query_metrics)
    aliases_path = Path(args.aliases) if args.aliases else out_dir / "metric_aliases_resolved.json"
    missing_path = Path(args.missing) if args.missing else out_dir / "missing_metrics.csv"
    aliases_path.write_text(json.dumps(resolved, indent=2, sort_keys=True))
    write_missing(missing_path, missing)
    print(f"collected_metrics={len(collected_metrics)}")
    print(f"query_metrics={len(query_metrics)}")
    print(f"resolved_collected={sum(1 for v in resolved.values() if v['status'] == 'collected')}")
    print(f"missing_or_not_collected={len(missing)}")
    print(f"aliases={aliases_path}")
    print(f"missing={missing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
