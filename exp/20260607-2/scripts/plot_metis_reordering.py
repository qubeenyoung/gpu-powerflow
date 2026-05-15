#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_MATRIX = EXP_ROOT / "raw" / "cupf_jf_dumps" / "case_ACTIVSg10k" / "J1.txt"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "figures" / "metis_reordering_case_ACTIVSg10k_J1_b64.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot original and METIS-reordered CSR sparsity patterns."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--case-name", default="case_ACTIVSg10k J1")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--metis-lib", default="libmetis.so.5")
    return parser.parse_args()


def expect(tokens: list[str], pos: int, expected: str, path: Path) -> int:
    if pos >= len(tokens) or tokens[pos] != expected:
        raise ValueError(f"expected token {expected!r} at token {pos} in {path}")
    return pos + 1


def load_cupf_csr(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    tokens = path.read_text(encoding="utf-8").split()
    pos = 0
    pos = expect(tokens, pos, "type", path)
    if tokens[pos] != "csr_matrix":
        raise ValueError(f"{path} is not a cuPF csr_matrix dump")
    pos += 1
    pos = expect(tokens, pos, "rows", path)
    rows = int(tokens[pos])
    pos += 1
    pos = expect(tokens, pos, "cols", path)
    cols = int(tokens[pos])
    pos += 1
    pos = expect(tokens, pos, "nnz", path)
    nnz = int(tokens[pos])
    pos += 1
    if rows != cols:
        raise ValueError("matrix must be square")
    pos = expect(tokens, pos, "row_ptr", path)
    row_ptr = np.asarray([int(value) for value in tokens[pos : pos + rows + 1]], dtype=np.int32)
    pos += rows + 1
    pos = expect(tokens, pos, "col_idx", path)
    col_idx = np.asarray([int(value) for value in tokens[pos : pos + nnz]], dtype=np.int32)
    return rows, row_ptr, col_idx


def build_symmetrized_graph(n: int, row_ptr: np.ndarray, col_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adjacency = [set() for _ in range(n)]
    for row in range(n):
        for pos in range(int(row_ptr[row]), int(row_ptr[row + 1])):
            col = int(col_idx[pos])
            if row == col:
                continue
            adjacency[row].add(col)
            adjacency[col].add(row)

    xadj = np.empty(n + 1, dtype=np.int32)
    neighbors: list[int] = []
    cursor = 0
    for row, row_neighbors in enumerate(adjacency):
        xadj[row] = cursor
        ordered = sorted(row_neighbors)
        neighbors.extend(ordered)
        cursor += len(ordered)
    xadj[n] = cursor
    adjncy = np.asarray(neighbors, dtype=np.int32)
    return xadj, adjncy


def metis_partition(n: int, xadj: np.ndarray, adjncy: np.ndarray, nparts: int, libname: str) -> np.ndarray:
    if nparts <= 1 or len(adjncy) == 0:
        return np.zeros(n, dtype=np.int32)

    metis = ctypes.CDLL(libname)
    part = np.zeros(n, dtype=np.int32)
    nvtxs = ctypes.c_int32(n)
    ncon = ctypes.c_int32(1)
    nparts_c = ctypes.c_int32(nparts)
    objval = ctypes.c_int32(0)
    status = metis.METIS_PartGraphKway(
        ctypes.byref(nvtxs),
        ctypes.byref(ncon),
        xadj.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        adjncy.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        None,
        None,
        None,
        ctypes.byref(nparts_c),
        None,
        None,
        None,
        ctypes.byref(objval),
        part.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if status != 1:
        raise RuntimeError(f"METIS_PartGraphKway failed with status={status}")
    return part


def build_permutation(part: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(part)
    blocks: list[list[int]] = []
    for part_id in sorted(set(int(value) for value in part)):
        group = np.flatnonzero(part == part_id).astype(np.int32).tolist()
        group.sort()
        if not group:
            continue
        chunks = max(1, (len(group) + block_size - 1) // block_size)
        begin = 0
        for chunk in range(chunks):
            remaining = len(group) - begin
            remaining_chunks = chunks - chunk
            chunk_size = (remaining + remaining_chunks - 1) // remaining_chunks
            blocks.append(group[begin : begin + chunk_size])
            begin += chunk_size

    new_to_old = np.asarray([node for block in blocks for node in block], dtype=np.int32)
    old_to_new = np.empty(n, dtype=np.int32)
    old_to_new[new_to_old] = np.arange(n, dtype=np.int32)
    block_starts = np.asarray(
        np.cumsum([0] + [len(block) for block in blocks[:-1]]), dtype=np.int32
    )
    return new_to_old, old_to_new, block_starts


def csr_points(row_ptr: np.ndarray, col_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_counts = np.diff(row_ptr)
    rows = np.repeat(np.arange(len(row_counts), dtype=np.int32), row_counts)
    return rows, col_idx.copy()


def compute_offblock_ratio(rows: np.ndarray, cols: np.ndarray, block_starts: np.ndarray, n: int) -> float:
    starts = np.append(block_starts, n)
    row_block = np.searchsorted(starts, rows, side="right") - 1
    col_block = np.searchsorted(starts, cols, side="right") - 1
    return float(np.mean(row_block != col_block))


def plot_patterns(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    reordered_rows: np.ndarray,
    reordered_cols: np.ndarray,
    block_starts: np.ndarray,
    offblock_ratio: float,
    args: argparse.Namespace,
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(
        f"{args.case_name}: CSR sparsity before/after METIS reorder "
        f"(target block={args.block_size}, offblock nnz={offblock_ratio:.1%})",
        fontsize=13,
    )

    axes[0].scatter(cols, rows, s=0.08, c="#111111", alpha=0.55, linewidths=0)
    axes[0].set_title("Original ordering")

    starts = np.append(block_starts, n)
    row_block = np.searchsorted(starts, reordered_rows, side="right") - 1
    col_block = np.searchsorted(starts, reordered_cols, side="right") - 1
    in_block = row_block == col_block
    axes[1].scatter(
        reordered_cols[~in_block],
        reordered_rows[~in_block],
        s=0.06,
        c="#9ca3af",
        alpha=0.35,
        linewidths=0,
        label="off-block",
    )
    axes[1].scatter(
        reordered_cols[in_block],
        reordered_rows[in_block],
        s=0.10,
        c="#2563eb",
        alpha=0.75,
        linewidths=0,
        label="within METIS block",
    )
    for start in block_starts:
        axes[1].axhline(int(start), color="#dc2626", linewidth=0.18, alpha=0.12)
        axes[1].axvline(int(start), color="#dc2626", linewidth=0.18, alpha=0.12)
    axes[1].set_title("METIS reordered")
    axes[1].legend(loc="upper right", frameon=False, markerscale=8)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1, n)
        ax.set_ylim(n, -1)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.grid(False)

    fig.savefig(args.output, dpi=args.dpi)
    fig.savefig(args.output.with_suffix(".pdf"))


def main() -> None:
    args = parse_args()
    n, row_ptr, col_idx = load_cupf_csr(args.matrix)
    xadj, adjncy = build_symmetrized_graph(n, row_ptr, col_idx)
    nparts = max(1, (n + args.block_size - 1) // args.block_size)
    part = metis_partition(n, xadj, adjncy, nparts, args.metis_lib)
    _, old_to_new, block_starts = build_permutation(part, args.block_size)

    rows, cols = csr_points(row_ptr, col_idx)
    reordered_rows = old_to_new[rows]
    reordered_cols = old_to_new[cols]
    offblock_ratio = compute_offblock_ratio(reordered_rows, reordered_cols, block_starts, n)
    plot_patterns(n, rows, cols, reordered_rows, reordered_cols, block_starts, offblock_ratio, args)
    print(f"[DONE] {args.output}")
    print(f"[DONE] {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
