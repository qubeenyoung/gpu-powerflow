#!/usr/bin/env python3
"""Plot sparse patterns before and after METIS nested-dissection ordering."""

from __future__ import annotations

import argparse
import ctypes
import csv
from ctypes.util import find_library
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_FIG_DIR = EXP_ROOT / "figs" / "metis_ordering"
DEFAULT_CASES = [
    "case118",
    "case1197",
    "case3012wp",
    "case6468rte",
    "case8387pegase",
]

SUMMARY_FIELDS = [
    "matrix_kind",
    "case",
    "n",
    "nnz",
    "graph_edges",
    "bandwidth_original",
    "bandwidth_metis",
    "bandwidth_ratio",
    "profile_original",
    "profile_metis",
    "profile_ratio",
    "figure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--matrix-kind", choices=["ybus", "jacobian", "both"], default="both")
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_ybus_pattern(case_dir: Path) -> csr_matrix:
    matrix_path = case_dir / "dump_Ybus.mtx"
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    matrix = mmread(matrix_path).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    pattern = matrix.copy()
    pattern.data = np.ones_like(pattern.data, dtype=np.int8)
    return pattern.astype(np.int8).tocsr()


def load_int_values(path: Path) -> List[int]:
    values: List[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue
            values.append(int(stripped.split()[0]))
    return values


def build_bus_maps(n_bus: int, pv: List[int], pq: List[int]) -> Tuple[Dict[int, int], Dict[int, int], int, int]:
    n_pv = len(pv)
    n_pvpq = len(pv) + len(pq)
    row_pvpq = {bus: i for i, bus in enumerate(pv)}
    row_pvpq.update({bus: n_pv + i for i, bus in enumerate(pq)})
    row_pq = {bus: n_pvpq + i for i, bus in enumerate(pq)}
    if any(bus < 0 or bus >= n_bus for bus in row_pvpq) or any(bus < 0 or bus >= n_bus for bus in row_pq):
        raise RuntimeError("PV/PQ index out of range")
    return row_pvpq, row_pq, n_pvpq, len(pq)


def add_jacobian_pattern_entries(
    rows: List[int],
    cols: List[int],
    row_pvpq: Dict[int, int],
    row_pq: Dict[int, int],
    n_pvpq: int,
    row_bus: int,
    col_bus: int,
) -> None:
    row_p = row_pvpq.get(row_bus)
    row_q = row_pq.get(row_bus)
    col_va = row_pvpq.get(col_bus)
    col_vm = row_pq.get(col_bus)
    if row_p is not None and col_va is not None:
        rows.append(row_p)
        cols.append(col_va)
    if row_q is not None and col_va is not None:
        rows.append(row_q)
        cols.append(col_va)
    if row_p is not None and col_vm is not None:
        rows.append(row_p)
        cols.append(col_vm)
    if row_q is not None and col_vm is not None:
        rows.append(row_q)
        cols.append(col_vm)


def build_jacobian_pattern(case_dir: Path, ybus: csr_matrix) -> csr_matrix:
    pv = load_int_values(case_dir / "dump_pv.txt")
    pq = load_int_values(case_dir / "dump_pq.txt")
    row_pvpq, row_pq, n_pvpq, n_pq = build_bus_maps(ybus.shape[0], pv, pq)
    dim = n_pvpq + n_pq
    coo = ybus.tocoo()
    rows: List[int] = []
    cols: List[int] = []
    for row_bus, col_bus in zip(coo.row, coo.col):
        add_jacobian_pattern_entries(rows, cols, row_pvpq, row_pq, n_pvpq, int(row_bus), int(col_bus))
    for bus in range(ybus.shape[0]):
        add_jacobian_pattern_entries(rows, cols, row_pvpq, row_pq, n_pvpq, bus, bus)
    data = np.ones(len(rows), dtype=np.int8)
    pattern = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.int8)
    pattern.sum_duplicates()
    pattern.data[:] = 1
    pattern.eliminate_zeros()
    pattern.sort_indices()
    return pattern


def structural_graph(pattern: csr_matrix) -> csr_matrix:
    graph = ((pattern + pattern.T) != 0).astype(np.int8).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def load_metis() -> ctypes.CDLL:
    lib_name = find_library("metis")
    if not lib_name:
        raise RuntimeError("libmetis was not found")
    lib = ctypes.CDLL(lib_name)
    lib.METIS_NodeND.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.METIS_NodeND.restype = ctypes.c_int
    return lib


def metis_node_nd(graph: csr_matrix, lib: ctypes.CDLL) -> Tuple[np.ndarray, np.ndarray]:
    graph = graph.tocsr()
    graph.sort_indices()
    n = np.array([graph.shape[0]], dtype=np.int32)
    xadj = np.ascontiguousarray(graph.indptr, dtype=np.int32)
    adjncy = np.ascontiguousarray(graph.indices, dtype=np.int32)
    perm = np.empty(graph.shape[0], dtype=np.int32)
    inv_perm = np.empty(graph.shape[0], dtype=np.int32)
    status = lib.METIS_NodeND(
        n.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        xadj.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        adjncy.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        None,
        None,
        perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        inv_perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if status != 1:
        raise RuntimeError(f"METIS_NodeND failed with status {status}")
    return perm, inv_perm


def bandwidth(pattern: csr_matrix) -> int:
    coo = pattern.tocoo()
    if coo.nnz == 0:
        return 0
    return int(np.max(np.abs(coo.row - coo.col)))


def profile(pattern: csr_matrix) -> int:
    pattern = pattern.tocsr()
    total = 0
    for row in range(pattern.shape[0]):
        start = pattern.indptr[row]
        end = pattern.indptr[row + 1]
        if start == end:
            continue
        min_col = int(pattern.indices[start:end].min())
        if min_col < row:
            total += row - min_col
    return total


def maybe_downsample(row: np.ndarray, col: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if row.size <= max_points:
        return row, col
    rng = np.random.default_rng(20260519)
    idx = rng.choice(row.size, size=max_points, replace=False)
    return row[idx], col[idx]


def plot_patterns(
    case_name: str,
    matrix_kind: str,
    original: csr_matrix,
    ordered: csr_matrix,
    out_path: Path,
    max_points: int,
    dpi: int,
) -> None:
    matrices = [
        ("Original", original),
        ("METIS NodeND", ordered),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), constrained_layout=True)
    for ax, (title, matrix) in zip(axes, matrices):
        coo = matrix.tocoo()
        rows, cols = maybe_downsample(coo.row, coo.col, max_points)
        ax.scatter(cols, rows, s=0.18, c="#1f2937", linewidths=0, alpha=0.9, rasterized=True)
        ax.set_title(f"{title}\nBW={bandwidth(matrix):,}, profile={profile(matrix):,}", fontsize=9)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{case_name} {matrix_kind} sparsity pattern", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def ratio(after: float, before: float) -> float:
    return after / before if before else float("nan")


def run_matrix(case_name: str, matrix_kind: str, original: csr_matrix, args: argparse.Namespace, lib: ctypes.CDLL) -> Dict[str, object]:
    graph = structural_graph(original)
    _, inv_perm = metis_node_nd(graph, lib)
    ordered = original[inv_perm, :][:, inv_perm].tocsr()
    ordered.sort_indices()

    fig_path = args.fig_dir / f"{case_name}_{matrix_kind}_metis_before_after.png"
    plot_patterns(case_name, matrix_kind, original, ordered, fig_path, args.max_points, args.dpi)

    bw_original = bandwidth(original)
    bw_ordered = bandwidth(ordered)
    profile_original = profile(original)
    profile_ordered = profile(ordered)
    return {
        "matrix_kind": matrix_kind,
        "case": case_name,
        "n": original.shape[0],
        "nnz": original.nnz,
        "graph_edges": graph.nnz // 2,
        "bandwidth_original": bw_original,
        "bandwidth_metis": bw_ordered,
        "bandwidth_ratio": ratio(bw_ordered, bw_original),
        "profile_original": profile_original,
        "profile_metis": profile_ordered,
        "profile_ratio": ratio(profile_ordered, profile_original),
        "figure": str(fig_path.relative_to(EXP_ROOT)),
    }


def run_case(case_name: str, args: argparse.Namespace, lib: ctypes.CDLL) -> List[Dict[str, object]]:
    case_dir = args.dataset_root / case_name
    ybus = load_ybus_pattern(case_dir)
    rows: List[Dict[str, object]] = []
    if args.matrix_kind in ("ybus", "both"):
        rows.append(run_matrix(case_name, "ybus", ybus, args, lib))
    if args.matrix_kind in ("jacobian", "both"):
        rows.append(run_matrix(case_name, "jacobian", build_jacobian_pattern(case_dir, ybus), args, lib))
    return rows


def write_summary_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# METIS Ordering Sparsity Visuals",
        "",
        "| matrix | case | n | nnz | original BW | METIS BW | BW ratio | original profile | METIS profile | profile ratio | figure |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {kind} | {case} | {n} | {nnz} | {bw0} | {bw1} | {bwr:.3f} | {p0} | {p1} | {pr:.3f} | {fig} |".format(
                kind=row["matrix_kind"],
                case=row["case"],
                n=row["n"],
                nnz=row["nnz"],
                bw0=row["bandwidth_original"],
                bw1=row["bandwidth_metis"],
                bwr=float(row["bandwidth_ratio"]),
                p0=row["profile_original"],
                p1=row["profile_metis"],
                pr=float(row["profile_ratio"]),
                fig=row["figure"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    lib = load_metis()
    rows: List[Dict[str, object]] = []
    for case_name in args.cases:
        case_rows = run_case(case_name, args, lib)
        rows.extend(case_rows)
        for row in case_rows:
            print(
                f"[OK] {case_name} {row['matrix_kind']} "
                f"BW {row['bandwidth_original']} -> {row['bandwidth_metis']} "
                f"profile {row['profile_original']} -> {row['profile_metis']}"
            )
    write_summary_csv(args.fig_dir / "metis_ordering_summary.csv", rows)
    write_summary_md(args.fig_dir / "metis_ordering_summary.md", rows)
    print(f"[DONE] figures={args.fig_dir}")


if __name__ == "__main__":
    main()
