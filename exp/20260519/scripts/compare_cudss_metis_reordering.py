#!/usr/bin/env python3
"""Compare cuDSS analysis reorder permutations against METIS NodeND on Jacobians."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

from plot_metis_ordering import (
    DEFAULT_CASES,
    DEFAULT_DATASET_ROOT,
    EXP_ROOT,
    REPO_ROOT,
    bandwidth,
    add_jacobian_pattern_entries,
    build_bus_maps,
    load_metis,
    load_int_values,
    load_ybus_pattern,
    metis_node_nd,
    profile,
    structural_graph,
)


DEFAULT_BENCH = EXP_ROOT / "build" / "cudss_pf_benchmark"
DEFAULT_OUT_DIR = EXP_ROOT / "figs" / "cudss_vs_metis"
DEFAULT_CUDSS_THREADING_LIB = Path(
    "/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so"
)

SUMMARY_FIELDS = [
    "case",
    "n",
    "nnz",
    "cudss_row_col_equal_fraction",
    "row_exact_match_fraction",
    "row_spearman",
    "row_mean_abs_delta_norm",
    "row_p95_abs_delta_norm",
    "col_exact_match_fraction",
    "col_spearman",
    "col_mean_abs_delta_norm",
    "col_p95_abs_delta_norm",
    "pattern_common_nnz",
    "pattern_metis_only_nnz",
    "pattern_cudss_only_nnz",
    "pattern_jaccard",
    "pattern_common_fraction",
    "relative_frobenius_diff",
    "common_value_relative_diff",
    "metis_bandwidth",
    "cudss_bandwidth",
    "metis_profile",
    "cudss_profile",
    "raw_vs_inverse_note",
    "dump_json",
    "figure",
    "matrix_figure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--precision", choices=["fp64", "fp32"], default="fp64")
    parser.add_argument("--rhs-mode", choices=["synthetic", "mismatch"], default="synthetic")
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--enable-mt", action="store_true", default=True)
    parser.add_argument("--disable-mt", action="store_false", dest="enable_mt")
    parser.add_argument("--cudss-threading-lib", type=Path, default=DEFAULT_CUDSS_THREADING_LIB)
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if args.enable_mt and args.cudss_threading_lib.exists():
        lib = str(args.cudss_threading_lib)
        env["CUDSS_THREADING_LIB"] = lib
        preload_items = [item for item in env.get("LD_PRELOAD", "").split() if item]
        if lib not in preload_items:
            preload_items.insert(0, lib)
        env["LD_PRELOAD"] = " ".join(preload_items)
    return env


def load_complex_pairs(path: Path) -> np.ndarray:
    values: List[complex] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue
            real, imag = stripped.split()[:2]
            values.append(complex(float(real), float(imag)))
    return np.asarray(values, dtype=np.complex128)


def load_ybus_values(case_dir: Path) -> csr_matrix:
    matrix = mmread(case_dir / "dump_Ybus.mtx").tocsr().astype(np.complex128)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def build_jacobian_matrix(case_dir: Path) -> csr_matrix:
    ybus = load_ybus_values(case_dir)
    v0 = load_complex_pairs(case_dir / "dump_V.txt")
    pv = load_int_values(case_dir / "dump_pv.txt")
    pq = load_int_values(case_dir / "dump_pq.txt")
    if v0.size != ybus.shape[0]:
        raise RuntimeError(f"V0/Ybus dimension mismatch for {case_dir}")

    row_pvpq, row_pq, n_pvpq, n_pq = build_bus_maps(ybus.shape[0], pv, pq)
    dim = n_pvpq + n_pq
    ibus = ybus @ v0
    vabs = np.maximum(np.abs(v0), 1.0e-8)
    vnorm = v0 / vabs

    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []

    coo = ybus.tocoo()
    for row_bus, col_bus, y in zip(coo.row, coo.col, coo.data):
        row_bus = int(row_bus)
        col_bus = int(col_bus)
        d_angle = -1j * v0[row_bus] * np.conj(y * v0[col_bus])
        d_magnitude = v0[row_bus] * np.conj(y * vnorm[col_bus])

        row_p = row_pvpq.get(row_bus)
        row_q = row_pq.get(row_bus)
        col_va = row_pvpq.get(col_bus)
        col_vm = row_pq.get(col_bus)
        if row_p is not None and col_va is not None:
            rows.append(row_p)
            cols.append(col_va)
            values.append(float(np.real(d_angle)))
        if row_q is not None and col_va is not None:
            rows.append(row_q)
            cols.append(col_va)
            values.append(float(np.imag(d_angle)))
        if row_p is not None and col_vm is not None:
            rows.append(row_p)
            cols.append(col_vm)
            values.append(float(np.real(d_magnitude)))
        if row_q is not None and col_vm is not None:
            rows.append(row_q)
            cols.append(col_vm)
            values.append(float(np.imag(d_magnitude)))

    for bus in range(ybus.shape[0]):
        d_angle_diag = 1j * (v0[bus] * np.conj(ibus[bus]))
        d_magnitude_diag = np.conj(ibus[bus]) * vnorm[bus]
        row_p = row_pvpq.get(bus)
        row_q = row_pq.get(bus)
        col_va = row_pvpq.get(bus)
        col_vm = row_pq.get(bus)
        if row_p is not None and col_va is not None:
            rows.append(row_p)
            cols.append(col_va)
            values.append(float(np.real(d_angle_diag)))
        if row_q is not None and col_va is not None:
            rows.append(row_q)
            cols.append(col_va)
            values.append(float(np.imag(d_angle_diag)))
        if row_p is not None and col_vm is not None:
            rows.append(row_p)
            cols.append(col_vm)
            values.append(float(np.real(d_magnitude_diag)))
        if row_q is not None and col_vm is not None:
            rows.append(row_q)
            cols.append(col_vm)
            values.append(float(np.imag(d_magnitude_diag)))

    matrix = csr_matrix((np.asarray(values), (rows, cols)), shape=(dim, dim), dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def run_cudss_dump(args: argparse.Namespace, case_name: str, env: Dict[str, str]) -> Path:
    out_path = args.out_dir / "dumps" / f"{case_name}_cudss_reorder.json"
    case_dir = args.dataset_root / case_name
    cmd = [
        str(args.bench),
        "--case-dir",
        str(case_dir),
        "--case",
        case_name,
        "--precision",
        args.precision,
        "--rhs-mode",
        args.rhs_mode,
        "--dump-reorder",
        str(out_path),
    ]
    if args.enable_mt:
        cmd.append("--enable-mt")
        if args.cudss_threading_lib.exists():
            cmd.extend(["--threading-lib", str(args.cudss_threading_lib)])
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=args.timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cuDSS reorder dump failed for {case_name}: returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return out_path


def inverse_perm(old_to_new: np.ndarray) -> np.ndarray:
    inv = np.empty_like(old_to_new)
    inv[old_to_new] = np.arange(old_to_new.size, dtype=old_to_new.dtype)
    return inv


def validate_perm(values: np.ndarray, name: str) -> None:
    n = values.size
    if values.min(initial=0) < 0 or values.max(initial=-1) >= n:
        raise RuntimeError(f"{name} has values outside 0..n-1")
    if np.unique(values).size != n:
        raise RuntimeError(f"{name} is not a permutation")


def compare_positions(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    validate_perm(reference, "reference")
    validate_perm(candidate, "candidate")
    n = reference.size
    denom = max(n - 1, 1)
    delta = np.abs(candidate.astype(np.int64) - reference.astype(np.int64)) / denom
    if n <= 1:
        spearman = 1.0
    else:
        spearman = float(np.corrcoef(reference.astype(float), candidate.astype(float))[0, 1])
    return {
        "exact_match_fraction": float(np.mean(reference == candidate)),
        "spearman": spearman,
        "mean_abs_delta_norm": float(delta.mean()),
        "p95_abs_delta_norm": float(np.quantile(delta, 0.95)),
    }


def downsample_coo(matrix: csr_matrix, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    coo = matrix.tocoo()
    rows = coo.row
    cols = coo.col
    if rows.size <= max_points:
        return rows, cols
    rng = np.random.default_rng(20260519)
    idx = rng.choice(rows.size, size=max_points, replace=False)
    return rows[idx], cols[idx]


def ordered_pattern(pattern: csr_matrix, row_pos: np.ndarray, col_pos: np.ndarray) -> csr_matrix:
    row_order = inverse_perm(row_pos)
    col_order = inverse_perm(col_pos)
    ordered = pattern[row_order, :][:, col_order].tocsr()
    ordered.sort_indices()
    return ordered


def sparse_pattern_keys(matrix: csr_matrix) -> set[Tuple[int, int]]:
    coo = matrix.tocoo()
    return set(zip(coo.row.astype(int).tolist(), coo.col.astype(int).tolist()))


def sparse_frobenius_norm(matrix: csr_matrix) -> float:
    return float(np.sqrt(np.sum(np.asarray(matrix.data, dtype=np.float64) ** 2)))


def matrix_comparison_metrics(metis_ordered: csr_matrix, cudss_ordered: csr_matrix) -> Dict[str, float]:
    metis_pattern = metis_ordered.copy()
    cudss_pattern = cudss_ordered.copy()
    metis_pattern.data = np.ones_like(metis_pattern.data)
    cudss_pattern.data = np.ones_like(cudss_pattern.data)
    metis_keys = sparse_pattern_keys(metis_pattern)
    cudss_keys = sparse_pattern_keys(cudss_pattern)
    common = metis_keys & cudss_keys
    union = metis_keys | cudss_keys
    diff = metis_ordered - cudss_ordered

    common_mask = metis_pattern.multiply(cudss_pattern)
    metis_common = metis_ordered.multiply(common_mask)
    cudss_common = cudss_ordered.multiply(common_mask)
    common_diff = metis_common - cudss_common
    common_denom = max(sparse_frobenius_norm(metis_common), np.finfo(float).tiny)

    return {
        "pattern_common_nnz": float(len(common)),
        "pattern_metis_only_nnz": float(len(metis_keys - cudss_keys)),
        "pattern_cudss_only_nnz": float(len(cudss_keys - metis_keys)),
        "pattern_jaccard": float(len(common) / len(union)) if union else 1.0,
        "pattern_common_fraction": float(len(common) / max(metis_ordered.nnz, 1)),
        "relative_frobenius_diff": sparse_frobenius_norm(diff) / max(sparse_frobenius_norm(metis_ordered), np.finfo(float).tiny),
        "common_value_relative_diff": sparse_frobenius_norm(common_diff) / common_denom,
    }


def diff_overlay_points(metis_ordered: csr_matrix, cudss_ordered: csr_matrix, max_points: int):
    metis_keys = sparse_pattern_keys(metis_ordered)
    cudss_keys = sparse_pattern_keys(cudss_ordered)
    sets = [
        ("common", metis_keys & cudss_keys, "#111827", 0.65, 0.16),
        ("METIS only", metis_keys - cudss_keys, "#2563eb", 0.75, 0.22),
        ("cuDSS only", cudss_keys - metis_keys, "#dc2626", 0.75, 0.22),
    ]
    total = sum(len(items) for _, items, _, _, _ in sets)
    rng = np.random.default_rng(20260519)
    out = []
    for label, items, color, alpha, size in sets:
        coords = np.asarray(list(items), dtype=np.int64)
        if coords.size == 0:
            out.append((label, np.array([], dtype=np.int64), np.array([], dtype=np.int64), color, alpha, size))
            continue
        budget = max(1, int(max_points * len(items) / max(total, 1)))
        if coords.shape[0] > budget:
            coords = coords[rng.choice(coords.shape[0], size=budget, replace=False)]
        out.append((label, coords[:, 0], coords[:, 1], color, alpha, size))
    return out


def plot_matrix_comparison(
    case_name: str,
    metis_ordered: csr_matrix,
    cudss_ordered: csr_matrix,
    out_path: Path,
    max_points: int,
    dpi: int,
) -> None:
    metrics = matrix_comparison_metrics(metis_ordered, cudss_ordered)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"{case_name} reordered Jacobian matrix comparison", fontsize=13)

    for ax, title, matrix in [
        (axes[0, 0], "METIS ordered", metis_ordered),
        (axes[0, 1], "cuDSS analysis ordered", cudss_ordered),
    ]:
        rows, cols = downsample_coo(matrix, max_points)
        ax.scatter(cols, rows, s=0.18, c="#1f2937", linewidths=0, alpha=0.9, rasterized=True)
        ax.set_title(f"{title}\nBW={bandwidth(matrix):,}, profile={profile(matrix):,}", fontsize=9)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
        ax.tick_params(labelsize=7)

    ax = axes[1, 0]
    for label, rows, cols, color, alpha, size in diff_overlay_points(metis_ordered, cudss_ordered, max_points):
        if rows.size:
            ax.scatter(cols, rows, s=size, c=color, linewidths=0, alpha=alpha, label=label, rasterized=True)
    ax.set_title(
        f"Pattern overlap\nJaccard={metrics['pattern_jaccard']:.3f}, common={metrics['pattern_common_fraction']:.3f}",
        fontsize=9,
    )
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, metis_ordered.shape[1] - 0.5)
    ax.set_ylim(metis_ordered.shape[0] - 0.5, -0.5)
    ax.legend(markerscale=8, loc="best", fontsize=8)
    ax.tick_params(labelsize=7)

    diff = metis_ordered - cudss_ordered
    axes[1, 1].hist(np.log10(np.abs(diff.data) + np.finfo(float).tiny), bins=80, color="#7c3aed", alpha=0.78)
    axes[1, 1].set_title(
        f"Coordinate-wise numeric diff\nrelative Frobenius={metrics['relative_frobenius_diff']:.3e}",
        fontsize=9,
    )
    axes[1, 1].set_xlabel("log10(|METIS ordered - cuDSS ordered|)")
    axes[1, 1].set_ylabel("count")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_case(
    case_name: str,
    pattern: csr_matrix,
    metis_pos: np.ndarray,
    cudss_row_pos: np.ndarray,
    cudss_col_pos: np.ndarray,
    out_path: Path,
    max_points: int,
    dpi: int,
) -> None:
    metis_ordered = ordered_pattern(pattern, metis_pos, metis_pos)
    cudss_ordered = ordered_pattern(pattern, cudss_row_pos, cudss_col_pos)
    denom = max(pattern.shape[0] - 1, 1)
    row_delta = np.abs(cudss_row_pos.astype(np.int64) - metis_pos.astype(np.int64)) / denom
    col_delta = np.abs(cudss_col_pos.astype(np.int64) - metis_pos.astype(np.int64)) / denom

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"{case_name} Jacobian: cuDSS analysis reorder vs METIS NodeND", fontsize=13)

    for ax, title, matrix in [
        (axes[0, 0], "METIS NodeND ordered pattern", metis_ordered),
        (axes[0, 1], "cuDSS analysis ordered pattern", cudss_ordered),
    ]:
        rows, cols = downsample_coo(matrix, max_points)
        ax.scatter(cols, rows, s=0.18, c="#1f2937", linewidths=0, alpha=0.9, rasterized=True)
        ax.set_title(f"{title}\nBW={bandwidth(matrix):,}, profile={profile(matrix):,}", fontsize=9)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
        ax.tick_params(labelsize=7)

    n = pattern.shape[0]
    sample = np.arange(n)
    if n > 30_000:
        sample = np.linspace(0, n - 1, 30_000, dtype=np.int64)
    axes[1, 0].scatter(metis_pos[sample], cudss_row_pos[sample], s=2.5, alpha=0.35, label="row", color="#2563eb")
    axes[1, 0].scatter(metis_pos[sample], cudss_col_pos[sample], s=2.0, alpha=0.25, label="col", color="#dc2626")
    axes[1, 0].plot([0, n - 1], [0, n - 1], color="#111827", linewidth=0.8, alpha=0.55)
    axes[1, 0].set_title("Position comparison")
    axes[1, 0].set_xlabel("METIS reordered position")
    axes[1, 0].set_ylabel("cuDSS reordered position")
    axes[1, 0].legend(loc="best")

    axes[1, 1].hist(row_delta, bins=60, alpha=0.65, label="row", color="#2563eb")
    axes[1, 1].hist(col_delta, bins=60, alpha=0.45, label="col", color="#dc2626")
    axes[1, 1].set_title("Normalized absolute position delta")
    axes[1, 1].set_xlabel("|cuDSS position - METIS position| / (n - 1)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run_case(args: argparse.Namespace, case_name: str, env: Dict[str, str], metis_lib) -> Dict[str, object]:
    dump_path = run_cudss_dump(args, case_name, env)
    dump = json.loads(dump_path.read_text(encoding="utf-8"))
    matrix = build_jacobian_matrix(args.dataset_root / case_name)
    pattern = matrix.copy()
    pattern.data = np.ones_like(pattern.data)
    pattern = pattern.astype(np.int8)
    graph = structural_graph(pattern)
    metis_pos, _ = metis_node_nd(graph, metis_lib)

    cudss_row = np.asarray(dump["perm_reorder_row"], dtype=np.int64)
    cudss_col = np.asarray(dump["perm_reorder_col"], dtype=np.int64)
    validate_perm(cudss_row, "cuDSS row permutation")
    validate_perm(cudss_col, "cuDSS col permutation")
    if cudss_row.size != pattern.shape[0] or cudss_col.size != pattern.shape[1]:
        raise RuntimeError(f"cuDSS permutation length mismatch for {case_name}")

    row_metrics = compare_positions(metis_pos.astype(np.int64), cudss_row)
    col_metrics = compare_positions(metis_pos.astype(np.int64), cudss_col)
    row_inverse_metrics = compare_positions(metis_pos.astype(np.int64), inverse_perm(cudss_row))
    col_inverse_metrics = compare_positions(metis_pos.astype(np.int64), inverse_perm(cudss_col))

    metis_ordered = ordered_pattern(pattern, metis_pos, metis_pos)
    cudss_ordered = ordered_pattern(pattern, cudss_row, cudss_col)
    fig_path = args.out_dir / f"{case_name}_jacobian_cudss_vs_metis.png"
    plot_case(case_name, pattern, metis_pos, cudss_row, cudss_col, fig_path, args.max_points, args.dpi)
    metis_matrix_ordered = ordered_pattern(matrix, metis_pos, metis_pos)
    cudss_matrix_ordered = ordered_pattern(matrix, cudss_row, cudss_col)
    matrix_metrics = matrix_comparison_metrics(metis_matrix_ordered, cudss_matrix_ordered)
    matrix_fig_path = args.out_dir / f"{case_name}_reordered_matrix_compare.png"
    plot_matrix_comparison(
        case_name,
        metis_matrix_ordered,
        cudss_matrix_ordered,
        matrix_fig_path,
        args.max_points,
        args.dpi,
    )

    return {
        "case": case_name,
        "n": matrix.shape[0],
        "nnz": matrix.nnz,
        "cudss_row_col_equal_fraction": float(np.mean(cudss_row == cudss_col)),
        "row_exact_match_fraction": row_metrics["exact_match_fraction"],
        "row_spearman": row_metrics["spearman"],
        "row_mean_abs_delta_norm": row_metrics["mean_abs_delta_norm"],
        "row_p95_abs_delta_norm": row_metrics["p95_abs_delta_norm"],
        "col_exact_match_fraction": col_metrics["exact_match_fraction"],
        "col_spearman": col_metrics["spearman"],
        "col_mean_abs_delta_norm": col_metrics["mean_abs_delta_norm"],
        "col_p95_abs_delta_norm": col_metrics["p95_abs_delta_norm"],
        "pattern_common_nnz": int(matrix_metrics["pattern_common_nnz"]),
        "pattern_metis_only_nnz": int(matrix_metrics["pattern_metis_only_nnz"]),
        "pattern_cudss_only_nnz": int(matrix_metrics["pattern_cudss_only_nnz"]),
        "pattern_jaccard": matrix_metrics["pattern_jaccard"],
        "pattern_common_fraction": matrix_metrics["pattern_common_fraction"],
        "relative_frobenius_diff": matrix_metrics["relative_frobenius_diff"],
        "common_value_relative_diff": matrix_metrics["common_value_relative_diff"],
        "metis_bandwidth": bandwidth(metis_matrix_ordered),
        "cudss_bandwidth": bandwidth(cudss_matrix_ordered),
        "metis_profile": profile(metis_matrix_ordered),
        "cudss_profile": profile(cudss_matrix_ordered),
        "raw_vs_inverse_note": (
            f"raw row mean={row_metrics['mean_abs_delta_norm']:.6f}, "
            f"inverse row mean={row_inverse_metrics['mean_abs_delta_norm']:.6f}; "
            f"raw col mean={col_metrics['mean_abs_delta_norm']:.6f}, "
            f"inverse col mean={col_inverse_metrics['mean_abs_delta_norm']:.6f}"
        ),
        "dump_json": str(dump_path.relative_to(EXP_ROOT)),
        "figure": str(fig_path.relative_to(EXP_ROOT)),
        "matrix_figure": str(matrix_fig_path.relative_to(EXP_ROOT)),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: object) -> str:
    return f"{100.0 * float(value):.2f}%"


def f3(value: object) -> str:
    return f"{float(value):.3f}"


def write_md(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# cuDSS Analysis Reorder vs METIS NodeND",
        "",
        f"- matrix: `newton_jacobian_at_v0`",
        f"- precision used for cuDSS analysis: `{args.precision}`",
        f"- cuDSS permutation interpretation in main metrics: raw `PERM_REORDER_ROW/COL` as original-index to reordered-position",
        "",
        "| case | n | nnz | row exact | row rho | row mean delta | row p95 delta | col exact | col rho | col mean delta | cuDSS row=col | METIS BW | cuDSS BW | METIS profile | cuDSS profile | figure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {n} | {nnz} | {rexact} | {rrho} | {rmean} | {rp95} | {cexact} | {crho} | {cmean} | {rceq} | {mbw} | {cbw} | {mprof} | {cprof} | {fig} |".format(
                case=row["case"],
                n=row["n"],
                nnz=row["nnz"],
                rexact=pct(row["row_exact_match_fraction"]),
                rrho=f3(row["row_spearman"]),
                rmean=f3(row["row_mean_abs_delta_norm"]),
                rp95=f3(row["row_p95_abs_delta_norm"]),
                cexact=pct(row["col_exact_match_fraction"]),
                crho=f3(row["col_spearman"]),
                cmean=f3(row["col_mean_abs_delta_norm"]),
                rceq=pct(row["cudss_row_col_equal_fraction"]),
                mbw=row["metis_bandwidth"],
                cbw=row["cudss_bandwidth"],
                mprof=row["metis_profile"],
                cprof=row["cudss_profile"],
                fig=row["figure"],
            )
        )
    lines.append("")
    lines.append("## Reordered Matrix Overlap")
    lines.append("")
    lines.append("| case | common nnz | METIS only | cuDSS only | Jaccard | common fraction | relative Frobenius diff | common-value relative diff | matrix figure |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {case} | {common} | {monly} | {conly} | {jac:.3f} | {frac:.3f} | {fdiff:.3e} | {cdiff:.3e} | {fig} |".format(
                case=row["case"],
                common=row["pattern_common_nnz"],
                monly=row["pattern_metis_only_nnz"],
                conly=row["pattern_cudss_only_nnz"],
                jac=float(row["pattern_jaccard"]),
                frac=float(row["pattern_common_fraction"]),
                fdiff=float(row["relative_frobenius_diff"]),
                cdiff=float(row["common_value_relative_diff"]),
                fig=row["matrix_figure"],
            )
        )
    lines.append("")
    lines.append("## Orientation Check")
    lines.append("")
    lines.append("These values compare raw cuDSS arrays and their inverses against METIS. Lower mean delta is closer.")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['case']}`: {row['raw_vs_inverse_note']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.bench.exists():
        raise FileNotFoundError(args.bench)
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    metis_lib = load_metis()
    rows: List[Dict[str, object]] = []
    for case_name in args.cases:
        row = run_case(args, case_name, env, metis_lib)
        rows.append(row)
        print(
            f"[OK] {case_name} row_delta={row['row_mean_abs_delta_norm']:.3f} "
            f"col_delta={row['col_mean_abs_delta_norm']:.3f} "
            f"row_rho={row['row_spearman']:.3f} col_rho={row['col_spearman']:.3f}"
        )
    write_csv(args.out_dir / "cudss_vs_metis_reorder_summary.csv", rows)
    write_md(args.out_dir / "cudss_vs_metis_reorder_summary.md", rows, args)
    print(f"[DONE] out={args.out_dir}")


if __name__ == "__main__":
    main()
