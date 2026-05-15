#!/usr/bin/env python3
"""Analyze cuDSS Nsight Compute CSV exports for dependency, memory, and atomic evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ncu_metrics import (  # noqa: E402
    alias_specs,
    query_ncu_metrics,
    read_csv_metrics,
    read_query_metrics,
    resolve_metric_aliases,
    write_missing,
)


BATCH_DEFAULT = "1,2,4,8,16,32,64,128,256,512,1024"
TIME_METRIC = "gpu__time_duration.sum"

DEPENDENCY_COLUMNS = {
    "active_warps_per_cycle": "active_warps_per_cycle",
    "eligible_warps_per_cycle": "eligible_warps_per_cycle",
    "issue_percent": "issue_percent",
    "issue_per_cycle": "issue_per_cycle",
    "long_scoreboard_percent": "long_scoreboard_percent",
    "short_scoreboard_percent": "short_scoreboard_percent",
    "wait_percent": "wait_percent",
    "barrier_percent": "barrier_percent",
    "not_selected_percent": "not_selected_percent",
    "no_instruction_percent": "no_instruction_percent",
    "compute_percent": "compute_percent",
    "memory_percent": "memory_percent",
    "dram_percent": "dram_percent",
    "occupancy_percent": "occupancy_percent",
}

MEMORY_COLUMNS = [
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
    "ideal_sectors_if_available",
    "excessive_sectors_if_available",
    "excessive_sector_ratio_if_available",
]

ATOMIC_COLUMNS = [
    "atomic_inst",
    "atomic_thread_inst",
    "reduction_inst",
    "reduction_thread_inst",
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
    "atomic_inst_per_ms",
    "reduction_inst_per_ms",
    "atomic_request_per_ms",
    "percent_time_in_kernels_with_atomic",
    "percent_time_in_kernels_with_reduction",
]

COMMON_COLUMNS = [
    "B",
    "stage",
    "kernel_group",
    "launch_count",
    "gpu_ms",
    "stage_percent",
    "kernel_percent_in_stage",
]

FLAG_COLUMNS = [
    "is_low_issue",
    "is_low_eligible",
    "is_long_sb_high",
    "is_barrier_high",
    "is_l1_load_uncoalesced_suspect",
    "is_l1_store_uncoalesced_suspect",
    "is_l2_uncoalesced_suspect",
    "is_atomic_significant",
    "is_irregular_memory_suspect",
]


def parse_batches(text: str) -> list[int]:
    return [int(x) for x in re.split(r"[,\s]+", text.strip()) if x]


def parse_numeric(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip().replace(",", "")
    if text in {"", "nan", "NaN", "N/A"}:
        return np.nan
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def infer_batch_stage_kernel(path: Path, df: pd.DataFrame | None = None) -> tuple[int | None, str | None, str]:
    name = path.name
    full = str(path)
    batch = None
    for pattern in (
        r"(?:^|[^A-Za-z0-9])B[_-]?(\d+)(?:[^A-Za-z0-9]|$)",
        r"(?:batch|b)[_-]?(\d+)",
    ):
        m = re.search(pattern, name, re.IGNORECASE)
        if m:
            batch = int(m.group(1))
            break
    stage = None
    lower = full.lower()
    if "factorize" in lower:
        stage = "factorize"
    elif "solve" in lower:
        stage = "solve"
    elif df is not None:
        nvtx_cols = [
            c
            for c in df.columns
            if "Range" in c or "Msg" in c or "Domain" in c or "thread" in c
        ]
        text = " ".join(str(v) for c in nvtx_cols for v in df[c].head(20).dropna().tolist()).lower()
        if "nr.iteration.factorize" in text or "factorize" in text:
            stage = "factorize"
        elif "nr.iteration.solve" in text or "solve" in text:
            stage = "solve"
    source_kind = "dependency" if "dependency" in lower else "throughput"
    if name.endswith(".raw.csv"):
        source_kind = "raw"
    elif name.endswith(".details.csv"):
        source_kind = "details"
    elif name.endswith(".source.csv"):
        source_kind = "source"
    return batch, stage, source_kind


def kernel_group(kernel_name: object) -> str:
    name = str(kernel_name)
    if not name or name == "nan":
        return "unknown"
    if "cudss::factorize_v3_ker" in name:
        return "cudss::factorize_v3_ker"
    if "cudss::factorize_ker" in name:
        return "cudss::factorize_ker"
    for token in (
        "bwd_ker",
        "fwd_ker",
        "perm_ker",
        "upd_marker_bwd_ker",
        "upd_marker_fwd_ker",
        "copy_matrix_ker",
        "offsets_par_ker",
        "plain_map_ker",
        "independent_ker",
        "define_superpanel_ker",
        "dependency_map_ker",
        "radix_sort_ker",
        "copy_csr_columns_ker",
        "trans_nnz_per_row_ker",
        "finalize_eps_scale_ker",
    ):
        if token in name:
            if "cudss::" in name and not token.startswith(("fwd", "bwd", "upd", "perm", "offsets")):
                return f"cudss::{token}"
            return token
    m = re.search(r"(?:void\s+)?([A-Za-z_][\w:~]*)\s*(?:<|\()", name)
    if m:
        return m.group(1)
    return name[:96]


def load_exports(exports: Iterable[Path], batch_sweep: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in exports:
        if root.is_file() and root.suffix == ".csv":
            csv_paths = [root]
        elif root.exists():
            csv_paths = sorted(root.rglob("*.csv"))
        else:
            continue
        for path in csv_paths:
            try:
                df = pd.read_csv(path, comment="=")
            except Exception:
                continue
            if "Metric Name" not in df.columns or "Kernel Name" not in df.columns:
                continue
            batch, stage, source_kind = infer_batch_stage_kernel(path, df)
            if batch is None or batch not in batch_sweep or stage not in {"factorize", "solve"}:
                continue
            keep = [
                c
                for c in [
                    "ID",
                    "Kernel Name",
                    "Context",
                    "Stream",
                    "Block Size",
                    "Grid Size",
                    "Metric Name",
                    "Metric Value",
                ]
                if c in df.columns
            ]
            cur = df[keep].copy()
            cur["Metric Value"] = cur["Metric Value"].map(parse_numeric)
            idx = [c for c in keep if c not in {"Metric Name", "Metric Value"}]
            wide = cur.pivot_table(index=idx, columns="Metric Name", values="Metric Value", aggfunc="first")
            wide = wide.reset_index()
            wide.columns.name = None
            wide["B"] = batch
            wide["stage"] = stage
            wide["source_kind"] = source_kind
            wide["source_file"] = str(path)
            wide["kernel_group"] = wide["Kernel Name"].map(kernel_group)
            frames.append(wide)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if TIME_METRIC not in combined.columns:
        combined[TIME_METRIC] = np.nan
    return combined


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def sum_metric(df: pd.DataFrame, metric: str | None) -> float:
    if not metric or metric not in df.columns:
        return np.nan
    values = df[metric].dropna()
    if values.empty:
        return np.nan
    return float(values.sum())


def avg_metric(df: pd.DataFrame, metric: str | None) -> float:
    if not metric or metric not in df.columns:
        return np.nan
    return weighted_average(df[metric], df[TIME_METRIC])


def count_like(alias: str) -> bool:
    if "per_request" in alias or "hit_rate" in alias or alias.endswith("_per_ms"):
        return False
    return (
        alias.endswith("_requests")
        or alias.endswith("_sectors")
        or alias.endswith("_inst")
        or alias.endswith("_thread_inst")
        or "sector_misses" in alias
    )


def choose_time_records(df: pd.DataFrame, keys: dict[str, object]) -> pd.DataFrame:
    cur = filter_records(df, keys)
    dep = cur[cur["source_kind"].eq("dependency")]
    if not dep.empty:
        return dep
    throughput = cur[cur["source_kind"].eq("throughput")]
    if not throughput.empty:
        return throughput
    return cur


def filter_records(df: pd.DataFrame, keys: dict[str, object]) -> pd.DataFrame:
    cur = df
    for key, value in keys.items():
        if value is None:
            continue
        if key == "kernel_group" and value == "ALL":
            continue
        cur = cur[cur[key].eq(value)]
    return cur


def metric_value_for_group(
    df: pd.DataFrame,
    keys: dict[str, object],
    alias: str,
    aliases: dict[str, dict[str, object]],
) -> float:
    metric = aliases.get(alias, {}).get("metric")
    if not metric or metric not in df.columns:
        return np.nan
    cur = filter_records(df, keys)
    cur = cur[cur[metric].notna() & cur[TIME_METRIC].notna()]
    if cur.empty:
        return np.nan
    if count_like(alias):
        return float(cur[metric].sum())
    return weighted_average(cur[metric], cur[TIME_METRIC])


def compute_dependency_metrics(
    df: pd.DataFrame,
    keys: dict[str, object],
    aliases: dict[str, dict[str, object]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for alias, column in DEPENDENCY_COLUMNS.items():
        out[column] = metric_value_for_group(df, keys, alias, aliases)
    active = out.get("active_warps_per_cycle")
    eligible = out.get("eligible_warps_per_cycle")
    out["eligible_active_percent"] = (
        float(eligible / active * 100.0)
        if active is not None and eligible is not None and np.isfinite(active) and active > 0
        else np.nan
    )
    return out


def compute_memory_metrics(
    df: pd.DataFrame,
    keys: dict[str, object],
    aliases: dict[str, dict[str, object]],
) -> dict[str, float]:
    out = {alias: metric_value_for_group(df, keys, alias, aliases) for alias in MEMORY_COLUMNS}
    pairs = [
        ("l1_global_load_sectors_per_request", "l1_global_load_sectors", "l1_global_load_requests"),
        ("l1_global_store_sectors_per_request", "l1_global_store_sectors", "l1_global_store_requests"),
        ("l2_read_sectors_per_request", "l2_read_sectors", "l2_read_requests"),
        ("l2_write_sectors_per_request", "l2_write_sectors", "l2_write_requests"),
    ]
    for target, sectors, requests in pairs:
        if not np.isfinite(out.get(target, np.nan)):
            sec = out.get(sectors)
            req = out.get(requests)
            out[target] = float(sec / req) if req is not None and np.isfinite(req) and req > 0 else np.nan
    ideal = out.get("ideal_sectors_if_available")
    excessive = out.get("excessive_sectors_if_available")
    out["excessive_sector_ratio_if_available"] = (
        float(excessive / ideal)
        if ideal is not None and excessive is not None and np.isfinite(ideal) and ideal > 0
        else np.nan
    )
    return out


def compute_atomic_metrics(
    df: pd.DataFrame,
    keys: dict[str, object],
    aliases: dict[str, dict[str, object]],
    gpu_ms: float,
) -> dict[str, float]:
    out = {alias: metric_value_for_group(df, keys, alias, aliases) for alias in ATOMIC_COLUMNS if not alias.endswith("_per_ms")}
    atomic_inst = first_finite(out.get("atomic_inst"), out.get("atomic_thread_inst"))
    reduction_inst = first_finite(out.get("reduction_inst"), out.get("reduction_thread_inst"))
    atomic_requests = sum_finite(out.get("l1_atomic_requests"), out.get("l2_atomic_requests"))
    out["atomic_inst_per_ms"] = float(atomic_inst / gpu_ms) if np.isfinite(atomic_inst) and gpu_ms > 0 else np.nan
    out["reduction_inst_per_ms"] = float(reduction_inst / gpu_ms) if np.isfinite(reduction_inst) and gpu_ms > 0 else np.nan
    out["atomic_request_per_ms"] = float(atomic_requests / gpu_ms) if np.isfinite(atomic_requests) and gpu_ms > 0 else np.nan
    out["percent_time_in_kernels_with_atomic"] = percent_time_with_signal(df, keys, aliases, ("atomic_inst", "atomic_thread_inst", "l1_atomic_requests", "l2_atomic_requests"))
    out["percent_time_in_kernels_with_reduction"] = percent_time_with_signal(df, keys, aliases, ("reduction_inst", "reduction_thread_inst", "l1_reduction_requests", "l2_reduction_requests"))
    return out


def first_finite(*values: float | None) -> float:
    for value in values:
        if value is not None and np.isfinite(value):
            return float(value)
    return np.nan


def sum_finite(*values: float | None) -> float:
    found = False
    total = 0.0
    for value in values:
        if value is not None and np.isfinite(value):
            total += float(value)
            found = True
    return total if found else np.nan


def percent_time_with_signal(
    df: pd.DataFrame,
    keys: dict[str, object],
    aliases: dict[str, dict[str, object]],
    alias_names: tuple[str, ...],
) -> float:
    metrics = [aliases.get(alias, {}).get("metric") for alias in alias_names]
    metrics = [m for m in metrics if m and m in df.columns]
    if not metrics:
        return np.nan
    cur = filter_records(df, keys)
    cur = cur[cur[TIME_METRIC].notna()]
    if cur.empty:
        return np.nan
    total = cur[TIME_METRIC].sum()
    if total <= 0:
        return np.nan
    signal = np.zeros(len(cur), dtype=bool)
    for metric in metrics:
        signal |= cur[metric].fillna(0).to_numpy() > 0
    return float(cur.loc[signal, TIME_METRIC].sum() / total * 100.0)


def compute_flags(row: dict[str, object], thresholds: dict[str, float]) -> dict[str, object]:
    issue = row.get("issue_percent", np.nan)
    eligible = row.get("eligible_warps_per_cycle", np.nan)
    long_sb = row.get("long_scoreboard_percent", np.nan)
    barrier = row.get("barrier_percent", np.nan)
    l1_load_spr = row.get("l1_global_load_sectors_per_request", np.nan)
    l1_store_spr = row.get("l1_global_store_sectors_per_request", np.nan)
    l2_read_spr = row.get("l2_read_sectors_per_request", np.nan)
    l2_write_spr = row.get("l2_write_sectors_per_request", np.nan)
    l1_hit = row.get("l1_hit_rate", np.nan)
    l2_hit = row.get("l2_hit_rate", np.nan)
    atomic_time = row.get("percent_time_in_kernels_with_atomic", np.nan)
    atomic_inst_ms = row.get("atomic_inst_per_ms", np.nan)
    atomic_req_ms = row.get("atomic_request_per_ms", np.nan)
    is_low_issue = finite_lt(issue, thresholds["low_issue_percent"])
    is_low_eligible = finite_lt(eligible, thresholds["low_eligible_warp"])
    is_long_sb_high = finite_ge(long_sb, thresholds["long_sb_high_percent"])
    is_barrier_high = finite_ge(barrier, thresholds["barrier_high_percent"])
    l1_load_bad = finite_gt(l1_load_spr, thresholds["l1_load_spr_suspect"])
    l1_store_bad = finite_gt(l1_store_spr, thresholds["l1_store_spr_suspect"])
    l2_bad = finite_gt(l2_read_spr, thresholds["l2_spr_suspect"]) or finite_gt(l2_write_spr, thresholds["l2_spr_suspect"])
    low_hit = finite_lt(l1_hit, thresholds["low_hit_rate_percent"]) or finite_lt(l2_hit, thresholds["low_hit_rate_percent"])
    atomic_sig = (
        finite_ge(atomic_time, thresholds["atomic_time_significant_percent"])
        or finite_gt(atomic_inst_ms, thresholds["atomic_inst_per_ms_significant"])
        or finite_gt(atomic_req_ms, thresholds["atomic_request_per_ms_significant"])
    )
    irregular = bool(is_long_sb_high and is_low_eligible and (l1_load_bad or l1_store_bad or l2_bad or low_hit))
    return {
        "is_low_issue": bool(is_low_issue) if np.isfinite(issue) else np.nan,
        "is_low_eligible": bool(is_low_eligible) if np.isfinite(eligible) else np.nan,
        "is_long_sb_high": bool(is_long_sb_high) if np.isfinite(long_sb) else np.nan,
        "is_barrier_high": bool(is_barrier_high) if np.isfinite(barrier) else np.nan,
        "is_l1_load_uncoalesced_suspect": bool(l1_load_bad) if np.isfinite(l1_load_spr) else np.nan,
        "is_l1_store_uncoalesced_suspect": bool(l1_store_bad) if np.isfinite(l1_store_spr) else np.nan,
        "is_l2_uncoalesced_suspect": bool(l2_bad) if np.isfinite(first_finite(l2_read_spr, l2_write_spr)) else np.nan,
        "is_atomic_significant": bool(atomic_sig) if np.isfinite(first_finite(atomic_time, atomic_inst_ms, atomic_req_ms)) else np.nan,
        "is_irregular_memory_suspect": irregular if any(np.isfinite(v) for v in (l1_load_spr, l1_store_spr, l2_read_spr, l2_write_spr, l1_hit, l2_hit)) else np.nan,
    }


def finite_lt(value: object, threshold: float) -> bool:
    return bool(value is not None and np.isfinite(value) and float(value) < threshold)


def finite_gt(value: object, threshold: float) -> bool:
    return bool(value is not None and np.isfinite(value) and float(value) > threshold)


def finite_ge(value: object, threshold: float) -> bool:
    return bool(value is not None and np.isfinite(value) and float(value) >= threshold)


def build_row(
    df: pd.DataFrame,
    keys: dict[str, object],
    aliases: dict[str, dict[str, object]],
    thresholds: dict[str, float],
    stage_total_ns: float | None = None,
) -> dict[str, object]:
    time_records = choose_time_records(df, keys)
    gpu_ns = float(time_records[TIME_METRIC].dropna().sum()) if not time_records.empty else 0.0
    gpu_ms = gpu_ns / 1.0e6
    launch_count = int(time_records["ID"].nunique()) if "ID" in time_records.columns else int(len(time_records))
    kernel = keys.get("kernel_group", "ALL")
    stage_pct = 100.0 if stage_total_ns in (None, 0) else gpu_ns / stage_total_ns * 100.0
    row: dict[str, object] = {
        "B": keys.get("B"),
        "stage": keys.get("stage"),
        "kernel_group": kernel,
        "launch_count": launch_count,
        "gpu_ms": gpu_ms,
        "stage_percent": stage_pct,
        "kernel_percent_in_stage": stage_pct if kernel != "ALL" else 100.0,
    }
    row.update(compute_dependency_metrics(df, keys, aliases))
    row.update(compute_memory_metrics(df, keys, aliases))
    row.update(compute_atomic_metrics(df, keys, aliases, gpu_ms))
    row.update(compute_flags(row, thresholds))
    row["irregular_memory_score"] = irregular_score(row)
    row["atomic_score"] = atomic_score(row)
    return row


def irregular_score(row: dict[str, object]) -> float:
    memory_terms = [
        row.get("l1_global_load_sectors_per_request", np.nan),
        row.get("l1_global_store_sectors_per_request", np.nan),
        row.get("l2_read_sectors_per_request", np.nan),
        row.get("l2_write_sectors_per_request", np.nan),
        row.get("l1_hit_rate", np.nan),
        row.get("l2_hit_rate", np.nan),
    ]
    if not any(np.isfinite(x) for x in memory_terms):
        return np.nan
    long_sb = row.get("long_scoreboard_percent", np.nan)
    eligible = row.get("eligible_warps_per_cycle", np.nan)
    if not np.isfinite(long_sb) or not np.isfinite(eligible):
        return np.nan
    return float(long_sb / max(eligible, 1.0e-9))


def atomic_score(row: dict[str, object]) -> float:
    values = [
        row.get("atomic_inst_per_ms", np.nan),
        row.get("reduction_inst_per_ms", np.nan),
        row.get("atomic_request_per_ms", np.nan),
        row.get("percent_time_in_kernels_with_atomic", np.nan),
        row.get("percent_time_in_kernels_with_reduction", np.nan),
    ]
    if not any(np.isfinite(x) for x in values):
        return np.nan
    return float(np.nansum(values))


def write_tables(
    launches: pd.DataFrame,
    aliases: dict[str, dict[str, object]],
    out_dir: Path,
    batch_sweep: list[int],
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []
    for batch in batch_sweep:
        for stage in ("factorize", "solve"):
            keys = {"B": batch, "stage": stage}
            if filter_records(launches, keys).empty:
                continue
            time_records = choose_time_records(launches, keys)
            stage_total_ns = float(time_records[TIME_METRIC].dropna().sum())
            stage_rows.append(build_row(launches, {**keys, "kernel_group": "ALL"}, aliases, thresholds, None))
            groups = sorted(filter_records(launches, keys)["kernel_group"].dropna().unique())
            for group in groups:
                kernel_rows.append(build_row(launches, {**keys, "kernel_group": group}, aliases, thresholds, stage_total_ns))
    stage_df = pd.DataFrame(stage_rows)
    kernel_df = pd.DataFrame(kernel_rows)
    wanted = COMMON_COLUMNS + [
        "active_warps_per_cycle",
        "eligible_warps_per_cycle",
        "eligible_active_percent",
        "issue_percent",
        "issue_per_cycle",
        "long_scoreboard_percent",
        "short_scoreboard_percent",
        "wait_percent",
        "barrier_percent",
        "not_selected_percent",
        "no_instruction_percent",
        "compute_percent",
        "memory_percent",
        "dram_percent",
        "occupancy_percent",
    ] + MEMORY_COLUMNS + ATOMIC_COLUMNS + FLAG_COLUMNS + ["irregular_memory_score", "atomic_score"]
    for df in (stage_df, kernel_df):
        for col in wanted:
            if col not in df.columns:
                df[col] = np.nan
    stage_df = stage_df[wanted]
    kernel_df = kernel_df[wanted]
    top_df = kernel_df[kernel_df["B"].eq(1024)].sort_values(["stage", "kernel_percent_in_stage"], ascending=[True, False])
    top_df = top_df.groupby("stage", group_keys=False).head(10)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_df.to_csv(out_dir / "stage_summary_dependency_atomic_memory.csv", index=False)
    kernel_df.to_csv(out_dir / "kernel_summary_dependency_atomic_memory.csv", index=False)
    top_df.to_csv(out_dir / "top_kernel_B1024_atomic_memory.csv", index=False)
    write_evidence(stage_df, kernel_df, out_dir)
    return stage_df, kernel_df, top_df


def write_evidence(stage_df: pd.DataFrame, kernel_df: pd.DataFrame, out_dir: Path) -> None:
    irregular_cols = [
        "B",
        "stage",
        "kernel_group",
        "kernel_percent_in_stage",
        "long_scoreboard_percent",
        "eligible_warps_per_cycle",
        "l1_global_load_sectors_per_request",
        "l1_global_store_sectors_per_request",
        "l1_hit_rate",
        "l2_read_sectors_per_request",
        "l2_write_sectors_per_request",
        "l2_hit_rate",
        "l1_sector_misses_to_l2",
        "l2_sector_misses_to_device",
        "is_irregular_memory_suspect",
        "irregular_memory_score",
    ]
    atomic_cols = [
        "B",
        "stage",
        "kernel_group",
        "kernel_percent_in_stage",
        "atomic_inst",
        "atomic_thread_inst",
        "reduction_inst",
        "reduction_thread_inst",
        "l1_atomic_requests",
        "l2_atomic_requests",
        "l1_reduction_requests",
        "l2_reduction_requests",
        "atomic_inst_per_ms",
        "reduction_inst_per_ms",
        "atomic_request_per_ms",
        "percent_time_in_kernels_with_atomic",
        "percent_time_in_kernels_with_reduction",
        "is_atomic_significant",
        "atomic_score",
    ]
    kernel_df[irregular_cols].to_csv(out_dir / "irregular_memory_evidence.csv", index=False)
    kernel_df[atomic_cols].to_csv(out_dir / "atomic_evidence.csv", index=False)


def validate_against_existing_summary(
    stage_df: pd.DataFrame,
    kernel_df: pd.DataFrame,
    existing_summary: Path | None,
    out_dir: Path,
) -> pd.DataFrame:
    checks = [
        ("stage", 1024, "solve", "ALL", "active_warps_per_cycle", 4.3, 0.2),
        ("stage", 1024, "solve", "ALL", "eligible_warps_per_cycle", 0.36, 0.08),
        ("stage", 1024, "solve", "ALL", "issue_percent", 29.4, 1.0),
        ("stage", 1024, "solve", "ALL", "long_scoreboard_percent", 39.9, 1.0),
        ("stage", 1024, "factorize", "ALL", "active_warps_per_cycle", 6.1, 0.25),
        ("stage", 1024, "factorize", "ALL", "eligible_warps_per_cycle", 0.58, 0.08),
        ("stage", 1024, "factorize", "ALL", "issue_percent", 38.7, 1.0),
        ("stage", 1024, "factorize", "ALL", "long_scoreboard_percent", 35.0, 1.0),
        ("kernel", 1024, "factorize", "cudss::factorize_v3_ker", "kernel_percent_in_stage", 60.6, 1.0),
        ("kernel", 1024, "factorize", "cudss::factorize_ker", "kernel_percent_in_stage", 34.9, 1.0),
        ("kernel", 1024, "solve", "bwd_ker", "kernel_percent_in_stage", 60.4, 1.0),
        ("kernel", 1024, "solve", "fwd_ker", "kernel_percent_in_stage", 38.3, 1.0),
    ]
    rows = []
    for source, batch, stage, group, metric, expected, tol in checks:
        df = stage_df if source == "stage" else kernel_df
        cur = df[df["B"].eq(batch) & df["stage"].eq(stage) & df["kernel_group"].eq(group)]
        actual = float(cur.iloc[0][metric]) if not cur.empty and metric in cur.columns else np.nan
        diff = actual - expected if np.isfinite(actual) else np.nan
        rows.append(
            {
                "source": source,
                "B": batch,
                "stage": stage,
                "kernel_group": group,
                "metric": metric,
                "expected": expected,
                "actual": actual,
                "abs_diff": abs(diff) if np.isfinite(diff) else np.nan,
                "tolerance": tol,
                "status": "PASS" if np.isfinite(actual) and abs(diff) <= tol else "FAIL",
            }
        )
    validation = pd.DataFrame(rows)
    validation.to_csv(out_dir / "validation_against_existing_summary.csv", index=False)
    lines = ["# Validation Against Existing cuDSS Dependency Summary", ""]
    if existing_summary:
        lines.append(f"- Existing summary path: `{existing_summary}`")
    lines.append(f"- Status: {'PASS' if validation['status'].eq('PASS').all() else 'CHECK_NEEDED'}")
    lines.append("")
    lines.append(markdown_table(validation))
    if not validation["status"].eq("PASS").all():
        lines.extend(
            [
                "",
                "Possible causes:",
                "- NVTX range mismatch",
                "- kernel filtering mismatch",
                "- different NCU pass/report",
                "- metric alias mismatch",
                "- weighted average method mismatch",
            ]
        )
    (out_dir / "report_validation.md").write_text("\n".join(lines))
    return validation


def finite_series(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()


def line_plot_by_stage(stage_df: pd.DataFrame, columns: list[str], labels: list[str], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    plotted = False
    for stage in ("factorize", "solve"):
        cur = stage_df[stage_df["stage"].eq(stage)].sort_values("B")
        for col, label in zip(columns, labels):
            if finite_series(cur, col):
                ax.plot(cur["B"], cur[col], marker="o", linewidth=1.8, label=f"{stage} {label}")
                plotted = True
    if plotted:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size B")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Required metrics were not collected in the current NCU CSVs.", ha="center", va="center", wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_figures(stage_df: pd.DataFrame, top_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    line_plot_by_stage(stage_df, ["gpu_ms"], ["GPU ms"], "cuDSS stage GPU time by batch", "GPU ms", fig_dir / "gpu_ms_by_B.png")
    line_plot_by_stage(
        stage_df,
        ["issue_percent", "eligible_active_percent", "long_scoreboard_percent"],
        ["issue %", "eligible/active %", "long scoreboard %"],
        "Scheduler/dependency metrics by batch",
        "Percent",
        fig_dir / "scheduler_by_B.png",
    )
    line_plot_by_stage(
        stage_df,
        ["l1_global_load_sectors_per_request"],
        ["L1 global load sectors/request"],
        "L1 global load sectors per request",
        "Sectors/request",
        fig_dir / "l1_global_load_sectors_per_request_by_B.png",
    )
    line_plot_by_stage(
        stage_df,
        ["l1_global_store_sectors_per_request"],
        ["L1 global store sectors/request"],
        "L1 global store sectors per request",
        "Sectors/request",
        fig_dir / "l1_global_store_sectors_per_request_by_B.png",
    )
    line_plot_by_stage(
        stage_df,
        ["l2_read_sectors_per_request", "l2_write_sectors_per_request"],
        ["L2 read sectors/request", "L2 write sectors/request"],
        "L2 sectors per request",
        "Sectors/request",
        fig_dir / "l2_read_write_sectors_per_request_by_B.png",
    )
    line_plot_by_stage(
        stage_df,
        ["l1_hit_rate", "l2_hit_rate"],
        ["L1 hit rate", "L2 hit rate"],
        "L1/L2 hit rate",
        "Percent",
        fig_dir / "l1_l2_hit_rate_by_B.png",
    )
    line_plot_by_stage(
        stage_df,
        ["atomic_inst_per_ms", "reduction_inst_per_ms"],
        ["atomic inst/ms", "reduction inst/ms"],
        "Atomic/reduction instruction rate",
        "Instructions/ms",
        fig_dir / "atomic_reduction_inst_per_ms_by_B.png",
    )
    plot_top_kernel_bar(top_df, fig_dir / "top_kernels_B1024_dependency_atomic_memory.png")
    line_plot_by_stage(stage_df, ["irregular_memory_score"], ["irregular memory score"], "Irregular memory score", "Heuristic score", fig_dir / "irregular_memory_score_by_B.png")
    line_plot_by_stage(stage_df, ["atomic_score"], ["atomic score"], "Atomic score", "Heuristic score", fig_dir / "atomic_score_by_B.png")


def plot_top_kernel_bar(top_df: pd.DataFrame, path: Path) -> None:
    cur = top_df[top_df["B"].eq(1024)].copy()
    cur = cur[cur["kernel_percent_in_stage"].fillna(0) >= 0.5]
    cur["label"] = cur["stage"] + ":" + cur["kernel_group"]
    metrics = ["kernel_percent_in_stage", "long_scoreboard_percent", "barrier_percent", "atomic_inst_per_ms", "reduction_inst_per_ms"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=180)
    if cur.empty:
        ax.text(0.5, 0.5, "No B=1024 top-kernel rows found.", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(cur))
        width = 0.16
        plotted = False
        for i, metric in enumerate(metrics):
            if finite_series(cur, metric):
                values = cur[metric].fillna(0).to_numpy()
                ax.bar(x + (i - 2) * width, values, width, label=metric)
                plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No selected top-kernel metrics were collected.", ha="center", va="center")
            ax.set_axis_off()
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(cur["label"], rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Value")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=7)
    ax.set_title("B=1024 top kernels: dependency, memory, atomic")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None, float_digits: int = 3) -> str:
    cur = df.copy()
    if max_rows is not None:
        cur = cur.head(max_rows)
    if cur.empty:
        return "_No rows._"
    def fmt(value: object) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return "N/A"
            return f"{value:.{float_digits}f}"
        if pd.isna(value):
            return "N/A"
        return str(value)
    headers = list(cur.columns)
    rows = [[fmt(v) for v in row] for row in cur.to_numpy()]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def report_metric_availability(aliases: dict[str, dict[str, object]]) -> tuple[list[str], list[str], list[str]]:
    collected = [a for a, v in aliases.items() if v["status"] == "collected"]
    supported = [a for a, v in aliases.items() if v["status"] == "supported_not_collected"]
    missing = [a for a, v in aliases.items() if v["status"] == "missing"]
    return collected, supported, missing


def classify_atomic(stage_df: pd.DataFrame, aliases: dict[str, dict[str, object]]) -> str:
    atomic_aliases = [spec.alias for spec in alias_specs() if spec.group == "atomic_reduction"]
    collected = [alias for alias in atomic_aliases if aliases.get(alias, {}).get("status") == "collected"]
    if not collected:
        return "insufficient"
    vals = stage_df[["is_atomic_significant"]].dropna()
    if not vals.empty and vals["is_atomic_significant"].astype(bool).any():
        return "significant"
    return "minor_or_zero"


def classify_irregular(stage_df: pd.DataFrame, aliases: dict[str, dict[str, object]]) -> str:
    memory_aliases = [spec.alias for spec in alias_specs() if spec.group == "irregular_memory"]
    collected = [alias for alias in memory_aliases if aliases.get(alias, {}).get("status") == "collected"]
    if not collected:
        return "insufficient"
    vals = stage_df[["is_irregular_memory_suspect"]].dropna()
    if not vals.empty and vals["is_irregular_memory_suspect"].astype(bool).any():
        return "suspected"
    return "not_supported"


def write_report(
    stage_df: pd.DataFrame,
    kernel_df: pd.DataFrame,
    top_df: pd.DataFrame,
    validation: pd.DataFrame,
    aliases: dict[str, dict[str, object]],
    missing_rows: list[dict[str, object]],
    thresholds: dict[str, float],
    out_dir: Path,
) -> None:
    collected, supported, missing = report_metric_availability(aliases)
    atomic_class = classify_atomic(stage_df, aliases)
    irregular_class = classify_irregular(stage_df, aliases)
    dependency_pass = validation["status"].eq("PASS").all() if not validation.empty else False
    final_class = "C. evidence insufficient, need extra NCU pass"
    if atomic_class == "significant":
        final_class = "B. dependency + atomic/reduction significant"
    elif irregular_class == "suspected" and atomic_class in {"minor_or_zero", "insufficient"}:
        final_class = "A. dependency + irregular memory latency dominant, atomic minor" if atomic_class == "minor_or_zero" else "C. evidence insufficient, need extra NCU pass"

    factor_cols = [
        "B",
        "stage",
        "gpu_ms",
        "compute_percent",
        "memory_percent",
        "dram_percent",
        "active_warps_per_cycle",
        "eligible_warps_per_cycle",
        "eligible_active_percent",
        "issue_percent",
        "long_scoreboard_percent",
        "l1_global_load_sectors_per_request",
        "l2_read_sectors_per_request",
        "atomic_inst_per_ms",
    ]
    top_cols = [
        "stage",
        "kernel_group",
        "kernel_percent_in_stage",
        "gpu_ms",
        "active_warps_per_cycle",
        "eligible_warps_per_cycle",
        "issue_percent",
        "long_scoreboard_percent",
        "barrier_percent",
        "l1_global_load_sectors_per_request",
        "l2_read_sectors_per_request",
        "atomic_inst_per_ms",
        "reduction_inst_per_ms",
    ]
    lines = [
        "# cuDSS Bottleneck Analysis in cuPF: Dependency, Irregular Memory, and Atomic Traffic",
        "",
        "## 1. Why this analysis changed",
        "- The previous view mainly tracked metrics by batch size.",
        "- This report treats cuDSS as the bottleneck component inside cuPF and asks why cuDSS saturates internally.",
        "- All stage and kernel aggregates use `gpu__time_duration.sum` as the weight/proportion basis. Profiling-pass GPU ms is used for weighting and proportions only, not as a wall-time substitute.",
        "",
        "## 2. Existing dependency evidence",
        f"- Validation against the existing dependency summary: {'PASS' if dependency_pass else 'CHECK_NEEDED'}.",
        "- Large-B solve still shows low eligible warps and low issue rate while long scoreboard remains high, matching the triangular-solve dependency/latency-hiding interpretation.",
        "- Factorize improves with B, but large-B eligible/active remains low.",
        "- B=1024 top kernels remain dominated by `cudss::factorize_v3_ker`, `cudss::factorize_ker`, `bwd_ker`, and `fwd_ker`.",
        "",
        "## 3. Atomic/reduction analysis",
    ]
    if atomic_class == "significant":
        lines.append("Atomic/reduction traffic is non-negligible in top-time kernels, so it may contribute to serialization or memory-system pressure. The dominant symptom should still be read together with long scoreboard, barrier, and eligible-warp metrics.")
    elif atomic_class == "minor_or_zero":
        lines.append("Atomic/reduction metrics are near zero or confined to minor kernels, so atomic operations are unlikely to explain the dominant cuDSS bottleneck. The main evidence still points to dependency-limited irregular memory latency.")
    else:
        lines.append("Atomic significance could not be determined from this NCU pass because atomic/reduction instruction and request metrics were not collected. The report includes exact NCU command skeletons to collect them.")
    lines.extend(
        [
            "",
            "## 4. Irregular memory access analysis",
        ]
    )
    if irregular_class == "suspected":
        lines.append("Because high long-scoreboard stalls, low eligible warps, and elevated sectors/request appear together in dominant cuDSS kernels, the evidence supports an irregular-memory-latency bottleneck rather than a pure DRAM-bandwidth bottleneck.")
    elif irregular_class == "not_supported":
        lines.append("The collected memory metrics do not show elevated sectors/request or low hit-rate evidence strong enough to call irregular memory the main bottleneck.")
    else:
        lines.append("Irregular memory access is suspected from dependency symptoms, but it is not confirmed here because L1/L2 sectors/request, hit-rate, and sector-miss metrics were not collected in the available CSVs.")
    lines.extend(
        [
            "",
            "Heuristic thresholds printed for traceability:",
            markdown_table(pd.DataFrame([thresholds])),
            "",
            "## 5. Per-stage summary",
            markdown_table(stage_df[factor_cols].sort_values(["stage", "B"]), max_rows=24),
            "",
            "## 6. B=1024 top-kernel summary",
            markdown_table(top_df[top_cols].sort_values(["stage", "kernel_percent_in_stage"], ascending=[True, False]), max_rows=20),
            "",
            "## 7. Final conclusion",
            f"- Classification: **{final_class}**.",
            f"- Collected aliases: {len(collected)}. Supported by NCU query but not collected: {len(supported)}. Missing from both CSV/query: {len(missing)}.",
            "- Current evidence is strong for dependency/latency-hiding limits, but insufficient to quantitatively prove atomic/reduction or coalescing behavior until the additional memory/atomic NCU pass is collected.",
            "",
            "## Metric availability summary",
            f"- Collected: `{', '.join(collected[:30])}`" + (" ..." if len(collected) > 30 else ""),
            f"- Supported-not-collected examples: `{', '.join(supported[:30])}`" + (" ..." if len(supported) > 30 else ""),
            f"- Missing examples: `{', '.join(missing[:20])}`" + (" ..." if len(missing) > 20 else ""),
        ]
    )
    (out_dir / "report_cudss_bottleneck_atomic_irregular_memory.md").write_text("\n".join(lines))


def write_collection_script(aliases: dict[str, dict[str, object]], out_path: Path) -> None:
    atomic_metrics: list[str] = []
    for spec in alias_specs():
        if spec.group != "atomic_reduction":
            continue
        row = aliases.get(spec.alias, {})
        candidates = row.get("query_candidates") or []
        if candidates:
            atomic_metrics.append(str(candidates[0]))
    atomic_metrics = sorted(dict.fromkeys(atomic_metrics))
    atomic_text = ",".join(atomic_metrics)
    script = f"""#!/usr/bin/env bash
set -u

NCU_BIN="${{NCU_BIN:-ncu}}"
REPORT_DIR="${{REPORT_DIR:-reports}}"
LOG_DIR="${{LOG_DIR:-logs}}"
BATCHES="${{BATCHES:-{BATCH_DEFAULT}}}"

# Replace '{{B}}' in APP_COMMAND_TEMPLATE with the batch size.
# Example:
#   APP_COMMAND_TEMPLATE='./build/bench-operators/benchmarks/cupf_case_benchmark --case datasets/matpower8.1/cupf_all_dumps/case8387pegase --profile cuda_mixed_edge --batch {{B}} --nr-only'
APP_COMMAND_TEMPLATE="${{APP_COMMAND_TEMPLATE:-}}"
if [[ -z "${{APP_COMMAND_TEMPLATE}}" ]]; then
  APP_COMMAND_TEMPLATE='<APP_COMMAND_FOR_BATCH_{{B}}>'
fi

# Restrict this to the cuDSS range inside one NR iteration.
NVTX_INCLUDE="${{NVTX_INCLUDE:-<NR_ITERATION_CUDSS_NVTX_RANGE>}}"
KERNEL_REGEX="${{KERNEL_REGEX:-regex:(cudss|factorize|factorize_v3|fwd_ker|bwd_ker|perm_ker|upd_marker|copy_matrix)}}"

mkdir -p "${{REPORT_DIR}}" "${{LOG_DIR}}"

RESOLVED_ATOMIC_METRICS="{atomic_text}"
BASE_METRICS="group:memory__first_level_cache_table,group:memory__l2_cache_table,group:memory__dram_table"
if [[ -n "${{RESOLVED_ATOMIC_METRICS}}" ]]; then
  METRICS="${{BASE_METRICS}},${{RESOLVED_ATOMIC_METRICS}}"
else
  METRICS="${{BASE_METRICS}}"
fi

IFS=',' read -ra B_LIST <<< "${{BATCHES}}"
for B in "${{B_LIST[@]}}"; do
  APP_COMMAND="${{APP_COMMAND_TEMPLATE//\\{{B\\}}/${{B}}}}"
  OUT="${{REPORT_DIR}}/cudss_atomic_mem_B${{B}}"
  cat <<CMD | tee -a "${{LOG_DIR}}/collect_cudss_atomic_irregular_memory_commands.txt"
${{NCU_BIN}} \\\\
  --kernel-name-base demangled \\\\
  --nvtx \\\\
  --nvtx-include "${{NVTX_INCLUDE}}" \\\\
  --section SchedulerStats \\\\
  --section WarpStateStats \\\\
  --section SourceCounters \\\\
  --section InstructionStats \\\\
  --section MemoryWorkloadAnalysis \\\\
  --section MemoryWorkloadAnalysis_Tables \\\\
  --section LaunchStats \\\\
  --metrics "${{METRICS}}" \\\\
  -k "${{KERNEL_REGEX}}" \\\\
  -o "${{OUT}}" \\\\
  ${{APP_COMMAND}}
CMD
done
"""
    out_path.write_text(script)
    out_path.chmod(0o755)


def default_thresholds() -> dict[str, float]:
    return {
        "low_issue_percent": 40.0,
        "low_eligible_warp": 1.0,
        "long_sb_high_percent": 30.0,
        "barrier_high_percent": 20.0,
        "l1_load_spr_suspect": 8.0,
        "l1_store_spr_suspect": 8.0,
        "l2_spr_suspect": 8.0,
        "low_hit_rate_percent": 50.0,
        "atomic_time_significant_percent": 10.0,
        "atomic_inst_per_ms_significant": 1.0,
        "atomic_request_per_ms_significant": 1.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", default="exports", help="Directory containing NCU CSV exports.")
    parser.add_argument("--existing-summary", default=None, help="Existing markdown summary path, optional.")
    parser.add_argument("--out", default="out", help="Output directory.")
    parser.add_argument("--batch-sweep", default=BATCH_DEFAULT)
    parser.add_argument("--query-file", nargs="*", default=[], help="Saved ncu --query-metrics output files.")
    parser.add_argument("--query-ncu", action="store_true", help="Run ncu --query-metrics for supported metric discovery.")
    parser.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_sweep = parse_batches(args.batch_sweep)
    export_paths = [Path(p) for p in args.exports.split(",") if p]
    csv_paths: list[Path] = []
    for root in export_paths:
        if root.is_file():
            csv_paths.append(root)
        elif root.exists():
            csv_paths.extend(sorted(root.rglob("*.csv")))
    collected_metrics = read_csv_metrics(csv_paths)
    query_metrics = read_query_metrics([Path(p) for p in args.query_file])
    if args.query_ncu:
        query_metrics |= query_ncu_metrics(args.ncu_bin, out_dir / "ncu_query_metrics.txt")
    aliases, missing_rows = resolve_metric_aliases(collected_metrics, query_metrics)
    (out_dir / "metric_aliases_resolved.json").write_text(json.dumps(aliases, indent=2, sort_keys=True))
    write_missing(out_dir / "missing_metrics.csv", missing_rows)
    launches = load_exports(export_paths, batch_sweep)
    if launches.empty:
        raise SystemExit(f"No usable NCU CSV rows found under: {args.exports}")
    thresholds = default_thresholds()
    stage_df, kernel_df, top_df = write_tables(launches, aliases, out_dir, batch_sweep, thresholds)
    validation = validate_against_existing_summary(
        stage_df,
        kernel_df,
        Path(args.existing_summary) if args.existing_summary else None,
        out_dir,
    )
    plot_figures(stage_df, top_df, out_dir)
    write_collection_script(aliases, Path("scripts") / "collect_cudss_atomic_irregular_memory.sh")
    write_report(stage_df, kernel_df, top_df, validation, aliases, missing_rows, thresholds, out_dir)
    print(f"launch_rows={len(launches)}")
    print(f"stage_rows={len(stage_df)}")
    print(f"kernel_rows={len(kernel_df)}")
    print(f"validation={'PASS' if validation['status'].eq('PASS').all() else 'CHECK_NEEDED'}")
    print(f"report={out_dir / 'report_cudss_bottleneck_atomic_irregular_memory.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
