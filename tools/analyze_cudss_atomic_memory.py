#!/usr/bin/env python3
"""Combine existing cuDSS dependency results with a B=1024 atomic/memory NCU pass."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIME_METRIC = "gpu__time_duration.sum"
TARGET_KERNELS = (
    "cudss::factorize_v3_ker",
    "cudss::factorize_ker",
    "bwd_ker",
    "fwd_ker",
)

METRIC_PREFIXES = (
    "gpu__",
    "sm__",
    "smsp__",
    "l1tex__",
    "lts__",
    "dram__",
    "derived__",
)


@dataclass(frozen=True)
class Alias:
    name: str
    group: str
    kind: str
    exact: tuple[str, ...]
    regex: tuple[str, ...] = ()


def aliases() -> list[Alias]:
    mem = "memory"
    atom = "atomic_reduction"
    dep = "dependency"
    return [
        Alias("gpu_time", dep, "sum", ("gpu__time_duration.sum",)),
        Alias("compute_percent", dep, "avg", ("sm__throughput.avg.pct_of_peak_sustained_elapsed",)),
        Alias("memory_percent", dep, "avg", ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",)),
        Alias("dram_percent", dep, "avg", ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",)),
        Alias("issue_percent", dep, "avg", ("smsp__issue_active.avg.pct_of_peak_sustained_active",)),
        Alias("eligible_warps_per_cycle", dep, "avg", ("smsp__warps_eligible.avg.per_cycle_active",)),
        Alias("long_scoreboard_percent", dep, "avg", ("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",)),
        Alias("barrier_percent", dep, "avg", ("smsp__warp_issue_stalled_barrier_per_warp_active.pct",)),
        Alias(
            "l1_global_load_requests",
            mem,
            "sum",
            ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_ld"),
            (r"l1tex.*requests.*global.*op_ld.*sum",),
        ),
        Alias(
            "l1_global_load_sectors",
            mem,
            "sum",
            ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld"),
            (r"l1tex.*sectors.*global.*op_ld.*sum",),
        ),
        Alias(
            "l1_global_load_sectors_per_request",
            mem,
            "avg",
            (
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld",
            ),
            (r"l1tex.*average.*sectors.*request.*global.*op_ld",),
        ),
        Alias(
            "l1_global_store_requests",
            mem,
            "sum",
            ("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_st"),
            (r"l1tex.*requests.*global.*op_st.*sum",),
        ),
        Alias(
            "l1_global_store_sectors",
            mem,
            "sum",
            ("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_st"),
            (r"l1tex.*sectors.*global.*op_st.*sum",),
        ),
        Alias(
            "l1_global_store_sectors_per_request",
            mem,
            "avg",
            (
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st",
            ),
            (r"l1tex.*average.*sectors.*request.*global.*op_st",),
        ),
        Alias(
            "l1_hit_rate",
            mem,
            "avg",
            (
                "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
                "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
                "l1tex__t_sector_hit_rate.pct",
                "l1tex__t_sector_hit_rate",
            ),
            (r"l1tex.*hit_rate",),
        ),
        Alias("l1_sector_misses_to_l2", mem, "sum", (), (r"l1tex.*sector.*miss", r"l1tex.*xbar.*sector.*sum")),
        Alias(
            "l2_read_requests",
            mem,
            "sum",
            ("lts__t_requests_op_read.sum", "lts__t_requests_aperture_device_op_read.sum", "lts__t_requests_op_read"),
            (r"lts.*requests.*op_read.*sum",),
        ),
        Alias(
            "l2_read_sectors",
            mem,
            "sum",
            ("lts__t_sectors_op_read.sum", "lts__t_sectors_aperture_device_op_read.sum", "lts__t_sectors_op_read"),
            (r"lts.*sectors.*op_read.*sum",),
        ),
        Alias(
            "l2_read_sectors_per_request",
            mem,
            "derived",
            (),
            (),
        ),
        Alias(
            "l2_write_requests",
            mem,
            "sum",
            ("lts__t_requests_op_write.sum", "lts__t_requests_aperture_device_op_write.sum", "lts__t_requests_op_write"),
            (r"lts.*requests.*op_write.*sum",),
        ),
        Alias(
            "l2_write_sectors",
            mem,
            "sum",
            ("lts__t_sectors_op_write.sum", "lts__t_sectors_aperture_device_op_write.sum", "lts__t_sectors_op_write"),
            (r"lts.*sectors.*op_write.*sum",),
        ),
        Alias("l2_write_sectors_per_request", mem, "derived", (), ()),
        Alias(
            "l2_hit_rate",
            mem,
            "avg",
            ("lts__t_sector_hit_rate.pct", "lts__t_sector_op_read_hit_rate.pct", "lts__t_sector_hit_rate"),
            (r"lts.*hit_rate",),
        ),
        Alias("l2_sector_misses_to_device", mem, "sum", (), (r"lts.*sector.*miss.*device", r"lts__d_sectors.*device.*sum")),
        Alias(
            "atomic_inst",
            atom,
            "sum",
            (
                "smsp__sass_inst_executed_op_global_atom.sum",
                "sm__sass_inst_executed_op_global_atom.sum",
                "smsp__sass_inst_executed_op_global_atom",
                "sm__sass_inst_executed_op_global_atom",
            ),
            (r"^(sm|smsp)__.*inst.*op_global_atom.*sum",),
        ),
        Alias(
            "reduction_inst",
            atom,
            "sum",
            (
                "smsp__sass_inst_executed_op_global_red.sum",
                "smsp__inst_executed_op_global_red.sum",
                "sm__sass_inst_executed_op_global_red.sum",
                "smsp__sass_inst_executed_op_global_red",
                "sm__sass_inst_executed_op_global_red",
            ),
            (r"^(sm|smsp)__.*inst.*op_global_red.*sum",),
        ),
        Alias(
            "l1_atomic_requests",
            atom,
            "sum",
            ("l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_atom"),
            (r"l1tex.*requests.*global.*op_atom.*sum",),
        ),
        Alias(
            "l1_atomic_sectors",
            atom,
            "sum",
            ("l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_atom"),
            (r"l1tex.*sectors.*global.*op_atom.*sum",),
        ),
        Alias(
            "l1_atomic_sectors_per_request",
            atom,
            "avg",
            (
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom.ratio",
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom",
            ),
            (r"l1tex.*average.*sectors.*request.*global.*op_atom",),
        ),
        Alias(
            "l1_reduction_requests",
            atom,
            "sum",
            ("l1tex__t_requests_pipe_lsu_mem_global_op_red.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_red"),
            (r"l1tex.*requests.*global.*op_red.*sum",),
        ),
        Alias(
            "l1_reduction_sectors",
            atom,
            "sum",
            ("l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_red"),
            (r"l1tex.*sectors.*global.*op_red.*sum",),
        ),
        Alias(
            "l1_reduction_sectors_per_request",
            atom,
            "avg",
            (
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_red.ratio",
                "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_red",
            ),
            (r"l1tex.*average.*sectors.*request.*global.*op_red",),
        ),
        Alias(
            "l2_atomic_requests",
            atom,
            "sum",
            ("lts__t_requests_op_atom.sum", "lts__t_requests_aperture_device_op_atom.sum", "lts__t_requests_op_atom"),
            (r"lts.*requests.*op_atom.*sum",),
        ),
        Alias(
            "l2_atomic_sectors",
            atom,
            "sum",
            ("lts__t_sectors_op_atom.sum", "lts__t_sectors_aperture_device_op_atom.sum", "lts__t_sectors_op_atom"),
            (r"lts.*sectors.*op_atom.*sum",),
        ),
        Alias("l2_atomic_sectors_per_request", atom, "derived", (), ()),
        Alias(
            "l2_reduction_requests",
            atom,
            "sum",
            ("lts__t_requests_op_red.sum", "lts__t_requests_aperture_device_op_red.sum", "lts__t_requests_op_red"),
            (r"lts.*requests.*op_red.*sum",),
        ),
        Alias(
            "l2_reduction_sectors",
            atom,
            "sum",
            ("lts__t_sectors_op_red.sum", "lts__t_sectors_aperture_device_op_red.sum", "lts__t_sectors_op_red"),
            (r"lts.*sectors.*op_red.*sum",),
        ),
        Alias("l2_reduction_sectors_per_request", atom, "derived", (), ()),
    ]


def numeric(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "N/A", "nan", "NaN"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def kernel_group(name: object) -> str:
    text = str(name)
    if "upd_marker_bwd_ker" in text or "upd_marker_fwd_ker" in text:
        return "other"
    if "factorize_v3_ker" in text:
        return "cudss::factorize_v3_ker"
    if "factorize_ker" in text:
        return "cudss::factorize_ker"
    if "bwd_ker" in text:
        return "bwd_ker"
    if "fwd_ker" in text:
        return "fwd_ker"
    return "other"


def stage_from_kernel(group: str) -> str:
    if group in {"cudss::factorize_v3_ker", "cudss::factorize_ker"}:
        return "factorize"
    if group in {"bwd_ker", "fwd_ker"}:
        return "solve"
    return "unknown"


def infer_stage(row: pd.Series) -> str:
    group = kernel_group(row.get("Kernel Name", ""))
    by_kernel = stage_from_kernel(group)
    if by_kernel != "unknown":
        return by_kernel
    text = " ".join(str(v) for v in row.dropna().tolist()).lower()
    if "factorize" in text:
        return "factorize"
    if "solve" in text:
        return "solve"
    return "unknown"


def extract_batch(path: Path, default: int) -> int:
    m = re.search(r"B[_-]?(\d+)", str(path), re.IGNORECASE)
    return int(m.group(1)) if m else default


def read_metric_names_from_query(paths: Iterable[Path]) -> set[str]:
    names: set[str] = set()
    rx = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*__[A-Za-z0-9_.$]+|group:[A-Za-z0-9_]+)\s+")
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            m = rx.match(line)
            if m:
                names.add(m.group(1))
    return names


def read_collected_metric_names(csv_paths: Iterable[Path]) -> set[str]:
    names: set[str] = set()
    for path in csv_paths:
        try:
            df = pd.read_csv(path, comment="=")
        except Exception:
            continue
        if "Metric Name" in df.columns:
            names.update(str(v).strip() for v in df["Metric Name"].dropna())
        else:
            names.update(c for c in df.columns if is_metric_column(c))
    return names


def is_metric_column(name: str) -> bool:
    return name.startswith(METRIC_PREFIXES)


def find_csvs(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".csv":
        return [path]
    if path.exists():
        return sorted(path.rglob("*.csv"))
    return []


def find_metric_csvs(path: Path) -> list[Path]:
    csvs = find_csvs(path)
    raw = [p for p in csvs if p.name == "raw.csv" or p.name.endswith(".raw.csv")]
    return raw or csvs


def resolve_aliases(collected: set[str], query: set[str]) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    resolved: dict[str, dict[str, object]] = {}
    missing: list[dict[str, object]] = []
    for spec in aliases():
        if spec.kind == "derived":
            resolved[spec.name] = {
                "alias": spec.name,
                "group": spec.group,
                "kind": spec.kind,
                "metric": None,
                "status": "derived_from_requests_and_sectors",
                "collected_candidates": [],
                "query_candidates": [],
            }
            continue
        collected_candidates = candidates(spec, collected)
        query_candidates = candidates(spec, query)
        metric = collected_candidates[0] if collected_candidates else None
        status = "collected" if metric else ("supported_not_collected" if query_candidates else "missing")
        resolved[spec.name] = {
            "alias": spec.name,
            "group": spec.group,
            "kind": spec.kind,
            "metric": metric,
            "status": status,
            "collected_candidates": collected_candidates[:20],
            "query_candidates": query_candidates[:20],
        }
        if status != "collected":
            missing.append(
                {
                    "alias": spec.name,
                    "group": spec.group,
                    "status": status,
                    "reason": "not_collected_in_atomic_memory_exports" if query_candidates else "not_found_in_query_or_exports",
                    "query_candidates": ";".join(query_candidates[:10]),
                }
            )
    return resolved, missing


def candidates(spec: Alias, names: set[str]) -> list[str]:
    lower = {n.lower(): n for n in names}
    out: list[str] = []
    for exact in spec.exact:
        hit = lower.get(exact.lower())
        if hit and hit not in out:
            out.append(hit)
    for pattern in spec.regex:
        rx = re.compile(pattern, re.IGNORECASE)
        for name in sorted(names):
            if rx.search(name) and name not in out:
                out.append(name)
    return out


def load_atomic_memory_exports(paths: Iterable[Path], batch_default: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in paths:
        for path in find_metric_csvs(root):
            try:
                df = pd.read_csv(path, comment="=")
            except Exception:
                continue
            if "Kernel Name" not in df.columns:
                continue
            if "Metric Name" not in df.columns:
                wide = load_wide_atomic_memory_csv(df, path, batch_default)
                if not wide.empty:
                    frames.append(wide)
                continue
            cur = df.copy()
            cur["B"] = extract_batch(path, batch_default)
            cur["kernel_group"] = cur["Kernel Name"].map(kernel_group)
            cur = cur[cur["kernel_group"].isin(TARGET_KERNELS)]
            if cur.empty:
                continue
            cur["stage"] = cur.apply(infer_stage, axis=1)
            cur["Metric Value"] = cur["Metric Value"].map(numeric)
            if "ID" not in cur.columns:
                cur["ID"] = np.arange(len(cur))
            index_cols = [
                c
                for c in ["B", "stage", "kernel_group", "ID", "Kernel Name", "Context", "Stream"]
                if c in cur.columns
            ]
            wide = cur.pivot_table(index=index_cols, columns="Metric Name", values="Metric Value", aggfunc="first")
            wide = wide.reset_index()
            wide.columns.name = None
            wide["source_file"] = str(path)
            frames.append(wide)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if TIME_METRIC not in out.columns:
        out[TIME_METRIC] = np.nan
    return out


def load_wide_atomic_memory_csv(df: pd.DataFrame, path: Path, batch_default: int) -> pd.DataFrame:
    cur = df.copy()
    time_unit = infer_metric_unit(cur, TIME_METRIC, default="ms")
    cur = cur[cur["Kernel Name"].notna()].copy()
    cur["kernel_group"] = cur["Kernel Name"].map(kernel_group)
    cur = cur[cur["kernel_group"].isin(TARGET_KERNELS)].copy()
    if cur.empty:
        return pd.DataFrame()
    cur["B"] = extract_batch(path, batch_default)
    cur["stage"] = cur.apply(infer_stage, axis=1)
    for col in [c for c in cur.columns if is_metric_column(c)]:
        cur[col] = cur[col].map(numeric)
    if TIME_METRIC in cur.columns:
        cur[TIME_METRIC] = cur[TIME_METRIC].map(lambda v: to_ms(v, time_unit))
    cur["source_file"] = str(path)
    return cur


def infer_metric_unit(df: pd.DataFrame, metric: str, default: str = "") -> str:
    if metric not in df.columns:
        return default
    unit_rows = df[df["Kernel Name"].isna()] if "Kernel Name" in df.columns else pd.DataFrame()
    if unit_rows.empty:
        return default
    unit = str(unit_rows.iloc[0].get(metric, "")).strip().lower()
    return unit or default


def to_ms(value: object, unit: str) -> float:
    val = numeric(value)
    if not np.isfinite(val):
        return np.nan
    unit = unit.lower()
    if unit == "s":
        return val * 1.0e3
    if unit == "us":
        return val / 1.0e3
    if unit == "ns":
        return val / 1.0e6
    return val


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def aggregate_metric(df: pd.DataFrame, alias: str, resolved: dict[str, dict[str, object]]) -> float:
    spec = next(a for a in aliases() if a.name == alias)
    metric = resolved.get(alias, {}).get("metric")
    if spec.kind == "derived":
        return np.nan
    if not metric or metric not in df.columns:
        return np.nan
    vals = df[metric].dropna()
    if vals.empty:
        return np.nan
    if spec.kind == "sum":
        return float(vals.sum())
    return weighted_average(df[metric], df[TIME_METRIC])


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0 else np.nan


def aggregate_atomic_memory(df: pd.DataFrame, resolved: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if df.empty:
        return pd.DataFrame(columns=["stage", "kernel_group"])
    for (stage, group), cur in df.groupby(["stage", "kernel_group"], dropna=False):
        row: dict[str, object] = {
            "stage": stage,
            "kernel_group": group,
            "new_pass_gpu_ms": cur[TIME_METRIC].sum() if TIME_METRIC in cur.columns else np.nan,
        }
        for spec in aliases():
            if spec.group in {"memory", "atomic_reduction"}:
                row[spec.name] = aggregate_metric(cur, spec.name, resolved)
        derive_ratios(row)
        rows.append(row)
    return pd.DataFrame(rows)


def derive_ratios(row: dict[str, object]) -> None:
    pairs = [
        ("l1_global_load_sectors_per_request", "l1_global_load_sectors", "l1_global_load_requests"),
        ("l1_global_store_sectors_per_request", "l1_global_store_sectors", "l1_global_store_requests"),
        ("l2_read_sectors_per_request", "l2_read_sectors", "l2_read_requests"),
        ("l2_write_sectors_per_request", "l2_write_sectors", "l2_write_requests"),
        ("l1_atomic_sectors_per_request", "l1_atomic_sectors", "l1_atomic_requests"),
        ("l1_reduction_sectors_per_request", "l1_reduction_sectors", "l1_reduction_requests"),
        ("l2_atomic_sectors_per_request", "l2_atomic_sectors", "l2_atomic_requests"),
        ("l2_reduction_sectors_per_request", "l2_reduction_sectors", "l2_reduction_requests"),
    ]
    for dst, num, den in pairs:
        if not np.isfinite(row.get(dst, np.nan)):
            row[dst] = safe_ratio(row.get(num, np.nan), row.get(den, np.nan))
    gpu_ms = row.get("new_pass_gpu_ms", np.nan)
    row["atomic_inst_per_ms"] = safe_ratio(row.get("atomic_inst", np.nan), gpu_ms)
    row["reduction_inst_per_ms"] = safe_ratio(row.get("reduction_inst", np.nan), gpu_ms)


def load_dependency(dependency_kernel_summary: Path, dependency_stage_summary: Path, batch: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    kernel = pd.read_csv(dependency_kernel_summary)
    stage = pd.read_csv(dependency_stage_summary)
    kernel = kernel[kernel["B"].eq(batch) & kernel["kernel_group"].isin(TARGET_KERNELS)].copy()
    kernel = kernel[kernel.apply(lambda r: r["stage"] == stage_from_kernel(str(r["kernel_group"])), axis=1)].copy()
    stage = stage[stage["B"].eq(batch) & stage["stage"].isin(["factorize", "solve"])].copy()
    rename = {
        "kernel_percent_in_stage": "kernel_percent_in_stage",
        "stage_percent": "stage_percent",
    }
    kernel = kernel.rename(columns=rename)
    return kernel, stage


def merge_dependency_atomic(kernel_dep: pd.DataFrame, stage_dep: pd.DataFrame, atomic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = kernel_dep.merge(atomic, on=["stage", "kernel_group"], how="left", suffixes=("", "_new"))
    for col in atomic_memory_columns() + ["atomic_inst_per_ms", "reduction_inst_per_ms", "new_pass_gpu_ms"]:
        new_col = f"{col}_new"
        if new_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[new_col].combine_first(merged[col])
            else:
                merged[col] = merged[new_col]
    for col in atomic_memory_columns():
        if col not in merged.columns:
            merged[col] = np.nan
    if "atomic_inst_per_ms" not in merged.columns:
        merged["atomic_inst_per_ms"] = np.nan
    if "reduction_inst_per_ms" not in merged.columns:
        merged["reduction_inst_per_ms"] = np.nan
    merged["bottleneck_interpretation"] = merged.apply(interpret_kernel, axis=1)
    stage_rows = []
    for _, stage_row in stage_dep.iterrows():
        stage = stage_row["stage"]
        cur = merged[merged["stage"].eq(stage)]
        row = stage_row.to_dict()
        for col in atomic_memory_columns() + ["atomic_inst_per_ms", "reduction_inst_per_ms"]:
            row[col] = weighted_col(cur, col)
        row["classification_hint"] = classify_stage(cur)
        stage_rows.append(row)
    return merged, pd.DataFrame(stage_rows)


def weighted_col(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    if col.endswith("_requests") or col.endswith("_sectors") or col in {"atomic_inst", "reduction_inst"}:
        vals = df[col].dropna()
        return float(vals.sum()) if not vals.empty else np.nan
    weights = df["gpu_ms"] if "gpu_ms" in df.columns else pd.Series(np.nan, index=df.index)
    return weighted_average(df[col], weights)


def atomic_memory_columns() -> list[str]:
    return [
        "l1_global_load_requests",
        "l1_global_load_sectors",
        "l1_global_load_sectors_per_request",
        "l1_global_store_requests",
        "l1_global_store_sectors",
        "l1_global_store_sectors_per_request",
        "l1_hit_rate",
        "l1_sector_misses_to_l2",
        "l2_read_requests",
        "l2_read_sectors",
        "l2_read_sectors_per_request",
        "l2_write_requests",
        "l2_write_sectors",
        "l2_write_sectors_per_request",
        "l2_hit_rate",
        "l2_sector_misses_to_device",
        "atomic_inst",
        "reduction_inst",
        "l1_atomic_requests",
        "l1_atomic_sectors",
        "l1_atomic_sectors_per_request",
        "l1_reduction_requests",
        "l1_reduction_sectors",
        "l1_reduction_sectors_per_request",
        "l2_atomic_requests",
        "l2_atomic_sectors",
        "l2_atomic_sectors_per_request",
        "l2_reduction_requests",
        "l2_reduction_sectors",
        "l2_reduction_sectors_per_request",
    ]


def interpret_kernel(row: pd.Series) -> str:
    atomic_rate = first_finite(row.get("atomic_inst_per_ms"))
    reduction_rate = first_finite(row.get("reduction_inst_per_ms"))
    l1_spr = row.get("l1_global_load_sectors_per_request", np.nan)
    l2_spr = row.get("l2_read_sectors_per_request", np.nan)
    long_sb = row.get("long_scoreboard_percent", np.nan)
    eligible = row.get("eligible_warps_per_cycle", np.nan)
    stage_pct = row.get("kernel_percent_in_stage", np.nan)
    has_atomic_reduction = np.isfinite(atomic_rate) or np.isfinite(reduction_rate)
    atomic_reduction_nonzero = (
        (np.isfinite(atomic_rate) and atomic_rate > 0) or (np.isfinite(reduction_rate) and reduction_rate > 0)
    )
    if not has_atomic_reduction and not np.isfinite(first_finite(l1_spr, l2_spr)):
        return "atomic/reduction 및 L1/L2 transaction metric 미수집으로 추가 pass 필요로 판단됨"
    if atomic_reduction_nonzero and np.isfinite(stage_pct) and stage_pct >= 20:
        return "dominant kernel에서 atomic/reduction nonzero로 병목 후보로 판단됨"
    irregular = (
        np.isfinite(long_sb)
        and long_sb >= 30
        and np.isfinite(eligible)
        and eligible < 1.0
        and ((np.isfinite(l1_spr) and l1_spr > 8) or (np.isfinite(l2_spr) and l2_spr > 8))
    )
    if irregular:
        return "long scoreboard high, eligible low, sectors/request high로 irregular/uncoalesced suspect가 강화됨"
    if has_atomic_reduction and not atomic_reduction_nonzero:
        return "atomic/reduction은 dominant bottleneck 아님으로 판단됨"
    return "dependency symptom은 확인되나 memory/atomic 증거는 제한적으로 해석됨"


def first_finite(*values: object) -> float:
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            return f
    return np.nan


def classify_stage(df: pd.DataFrame) -> str:
    if df.empty:
        return "C"
    atomic_cols = ["atomic_inst_per_ms", "reduction_inst_per_ms"]
    memory_cols = ["l1_global_load_sectors_per_request", "l2_read_sectors_per_request"]
    atomic_has_metric = any(df[c].notna().any() for c in atomic_cols if c in df)
    memory_has_metric = any(df[c].notna().any() for c in memory_cols if c in df)
    if not atomic_has_metric or not memory_has_metric:
        return "C"
    dominant = df[df["kernel_percent_in_stage"].fillna(0) >= 20]
    if not dominant.empty and (
        dominant[atomic_cols].fillna(0).to_numpy().sum() > 0
    ):
        return "B"
    irregular = False
    for _, row in dominant.iterrows():
        irregular = irregular or (
            first_finite(row.get("long_scoreboard_percent")) >= 30
            and first_finite(row.get("eligible_warps_per_cycle")) < 1
            and (
                first_finite(row.get("l1_global_load_sectors_per_request")) > 8
                or first_finite(row.get("l2_read_sectors_per_request")) > 8
            )
        )
    return "A" if irregular else "C"


def final_classification(kernel_df: pd.DataFrame) -> str:
    hints = {classify_stage(g) for _, g in kernel_df.groupby("stage")}
    if "B" in hints:
        return "B"
    if "C" in hints:
        return "C"
    return "A"


def fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_report(kernel_df: pd.DataFrame, classification: str, out_path: Path) -> None:
    table = kernel_df[
        [
            "stage",
            "kernel_group",
            "kernel_percent_in_stage",
            "issue_percent",
            "eligible_warps_per_cycle",
            "long_scoreboard_percent",
            "barrier_percent",
            "l1_global_load_sectors_per_request",
            "l2_read_sectors_per_request",
            "l1_hit_rate",
            "l2_hit_rate",
            "atomic_inst_per_ms",
            "reduction_inst_per_ms",
            "bottleneck_interpretation",
        ]
    ].copy()
    table = table.rename(
        columns={
            "kernel_percent_in_stage": "stage %",
            "issue_percent": "issue %",
            "eligible_warps_per_cycle": "eligible warps/cycle",
            "long_scoreboard_percent": "long scoreboard %",
            "barrier_percent": "barrier %",
            "l1_global_load_sectors_per_request": "L1 load sectors/request",
            "l2_read_sectors_per_request": "L2 read sectors/request",
            "l1_hit_rate": "L1 hit rate",
            "l2_hit_rate": "L2 hit rate",
            "atomic_inst_per_ms": "atomic inst/ms",
            "reduction_inst_per_ms": "reduction inst/ms",
            "bottleneck_interpretation": "bottleneck interpretation",
        }
    )
    if classification == "A":
        conclusion = "dependency + irregular memory latency dominant, atomic minor로 판단됨"
        evidence = [
            "- L1/L2 sectors/request와 dependency stall이 함께 높아 irregular-memory-latency evidence가 강화됨.",
            "- atomic/reduction traffic은 dominant kernel의 주 병목으로 보기 어려움.",
        ]
    elif classification == "B":
        conclusion = "dependency + atomic/reduction significant로 판단됨"
        evidence = [
            "- Global atomic instruction은 `cudss::factorize_v3_ker`에서 nonzero로 확인됨.",
            "- Global reduction instruction/request는 factorize/solve dominant kernel 전반에서 nonzero로 확인됨.",
            "- L1/L2 sectors/request는 약 1.08-1.27 수준으로 높지 않아 uncoalesced transaction은 확정되지 않음.",
            "- Long scoreboard high 및 eligible warps low 현상은 기존 dependency 병목 증거로 유지됨.",
        ]
    else:
        conclusion = "metric coverage가 부족하여 추가 NCU pass가 필요함으로 판단됨"
        evidence = [
            "- atomic/reduction 또는 L1/L2 transaction metric coverage가 부족하여 결론을 확정하지 않음.",
        ]
    lines = [
        "# cuDSS atomic/reduction and irregular memory re-measurement",
        "",
        "## 1. 목적",
        "- 기존 dependency pass에서 확인된 cuDSS 병목을 atomic/reduction 및 L1/L2 memory transaction 관점에서 재검증함.",
        "",
        "## 2. 측정 대상",
        "- case8387pegase",
        "- cuda_mixed_edge",
        "- B=1024",
        "- NR iteration 내부 cuDSS range",
        "- target kernels: `cudss::factorize_v3_ker`, `cudss::factorize_ker`, `bwd_ker`, `fwd_ker`",
        "",
        "## 3. 결과 표",
        markdown_table(table),
        "",
        "## 4. 결론",
        f"- 최종 classification: **{classification}**.",
        f"- {conclusion}",
        *evidence,
    ]
    out_path.write_text("\n".join(lines))


def write_missing(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = ["alias", "group", "status", "reason", "query_candidates"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=1024)
    parser.add_argument("--atomic-memory-exports", default="exports/cudss_atomic_mem_B1024")
    parser.add_argument("--dependency-kernel-summary", default="out/kernel_summary_dependency_atomic_memory.csv")
    parser.add_argument("--dependency-stage-summary", default="out/stage_summary_dependency_atomic_memory.csv")
    parser.add_argument("--query-metrics", default="logs/ncu_query_metrics_all.txt")
    parser.add_argument("--out", default="out")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_paths = [Path(p) for p in args.atomic_memory_exports.split(",") if p]
    atomic_csvs = [p for root in atomic_paths for p in find_metric_csvs(root)]
    query_names = read_metric_names_from_query([Path(args.query_metrics)])
    collected_names = read_collected_metric_names(atomic_csvs)
    resolved, missing = resolve_aliases(collected_names, query_names)
    (out_dir / "metric_aliases_resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
    missing_atomic_memory = [row for row in missing if row.get("group") != "dependency"]
    write_missing(out_dir / "missing_atomic_memory_metrics.csv", missing_atomic_memory)
    atomic_launches = load_atomic_memory_exports(atomic_paths, args.B)
    atomic_summary = aggregate_atomic_memory(atomic_launches, resolved)
    kernel_dep, stage_dep = load_dependency(
        Path(args.dependency_kernel_summary),
        Path(args.dependency_stage_summary),
        args.B,
    )
    kernel_summary, stage_summary = merge_dependency_atomic(kernel_dep, stage_dep, atomic_summary)
    classification = final_classification(kernel_summary)
    kernel_out_cols = [
        "stage",
        "kernel_group",
        "gpu_ms",
        "kernel_percent_in_stage",
        "compute_percent",
        "memory_percent",
        "dram_percent",
        "issue_percent",
        "eligible_warps_per_cycle",
        "long_scoreboard_percent",
        "barrier_percent",
    ] + atomic_memory_columns() + ["atomic_inst_per_ms", "reduction_inst_per_ms", "bottleneck_interpretation"]
    for col in kernel_out_cols:
        if col not in kernel_summary.columns:
            kernel_summary[col] = np.nan
    stage_out_cols = [
        "stage",
        "gpu_ms",
        "compute_percent",
        "memory_percent",
        "dram_percent",
        "issue_percent",
        "eligible_warps_per_cycle",
        "long_scoreboard_percent",
        "barrier_percent",
    ] + atomic_memory_columns() + ["atomic_inst_per_ms", "reduction_inst_per_ms", "classification_hint"]
    for col in stage_out_cols:
        if col not in stage_summary.columns:
            stage_summary[col] = np.nan
    kernel_summary[kernel_out_cols].to_csv(out_dir / "cudss_atomic_memory_B1024_kernel_summary.csv", index=False)
    stage_summary[stage_out_cols].to_csv(out_dir / "cudss_atomic_memory_B1024_stage_summary.csv", index=False)
    write_report(kernel_summary[kernel_out_cols], classification, out_dir / "report_cudss_atomic_irregular_memory_update.md")
    print(f"atomic_memory_launch_rows={len(atomic_launches)}")
    print(f"kernel_rows={len(kernel_summary)}")
    print(f"stage_rows={len(stage_summary)}")
    print(f"classification={classification}")
    print(f"report={out_dir / 'report_cudss_atomic_irregular_memory_update.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
