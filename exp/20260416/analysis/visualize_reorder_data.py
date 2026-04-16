#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path("/workspace/exp/20260416/analysis")
DEFAULT_DUMP_ROOT = ANALYSIS_ROOT / "dumps"
DEFAULT_FIGURE_ROOT = ANALYSIS_ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cuDSS reordering permutation vectors and elimination trees."
    )
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--max-scatter-points", type=int, default=12000)
    return parser.parse_args()


def read_vector(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").split()
    return np.asarray([int(value) for value in text], dtype=np.int64)


def load_cases(dump_root: Path) -> list[dict]:
    summary_path = dump_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")

    summary = pd.read_csv(summary_path)
    cases: list[dict] = []
    for row in summary.to_dict(orient="records"):
        output_dir = Path(row["output_dir"])
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        cases.append(
            {
                "summary": row,
                "metadata": metadata,
                "output_dir": output_dir,
                "perm_row": read_vector(output_dir / "perm_reorder_row.txt"),
                "perm_col": read_vector(output_dir / "perm_reorder_col.txt"),
                "etree": read_vector(output_dir / "elimination_tree.txt"),
            }
        )
    return cases


def sample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_points, dtype=np.int64)


def split_bottom_to_top_etree(etree: np.ndarray) -> list[np.ndarray]:
    if etree.size == 0:
        return []
    levels = int(math.log2(etree.size + 1))
    if 2**levels - 1 != etree.size:
        raise ValueError(f"Elimination tree length is not 2^k - 1: {etree.size}")

    chunks: list[np.ndarray] = []
    offset = 0
    for level_from_bottom in range(levels):
        width = 2 ** (levels - level_from_bottom - 1)
        chunks.append(etree[offset : offset + width])
        offset += width
    return chunks


def etree_block_heatmap(etree: np.ndarray) -> np.ndarray:
    bottom_to_top = split_bottom_to_top_etree(etree)
    top_to_bottom = list(reversed(bottom_to_top))
    max_width = max(len(level) for level in top_to_bottom)
    heatmap = np.zeros((len(top_to_bottom), max_width), dtype=float)

    for row, level in enumerate(top_to_bottom):
        repeat = max_width // len(level)
        heatmap[row, :] = np.repeat(level.astype(float), repeat)
    return heatmap


def etree_level_stats(etree: np.ndarray) -> pd.DataFrame:
    bottom_to_top = split_bottom_to_top_etree(etree)
    rows = []
    n_levels = len(bottom_to_top)
    for level_from_bottom, values in enumerate(bottom_to_top):
        level_from_root = n_levels - level_from_bottom - 1
        rows.append(
            {
                "level_from_root": level_from_root,
                "level_from_bottom": level_from_bottom,
                "nodes": int(values.size),
                "nonzero_nodes": int(np.count_nonzero(values)),
                "sum_size": int(values.sum()),
                "max_size": int(values.max(initial=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("level_from_root")


def safe_case_slug(case: dict) -> str:
    target = int(case["summary"]["target_bus"])
    name = str(case["summary"]["case"])
    return f"target_{target}_{name}"


def plot_permutation(case: dict, figure_root: Path, max_scatter_points: int) -> Path:
    meta = case["metadata"]
    perm_row = case["perm_row"]
    perm_col = case["perm_col"]
    n = perm_row.size
    x = np.arange(n, dtype=np.int64)
    sample = sample_indices(n, max_scatter_points)
    row_disp = perm_row - x
    col_disp = perm_col - x

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        f"{meta['case']} permutation, target {meta['target_bus_label']} buses",
        fontsize=14,
    )

    axes[0, 0].scatter(sample, perm_row[sample], s=4, alpha=0.55, label="row", color="#1f77b4")
    axes[0, 0].scatter(sample, perm_col[sample], s=3, alpha=0.35, label="col", color="#d62728")
    axes[0, 0].plot([0, n - 1], [0, n - 1], color="#333333", linewidth=0.8, alpha=0.6)
    axes[0, 0].set_title("Permutation vector")
    axes[0, 0].set_xlabel("Original Jacobian index")
    axes[0, 0].set_ylabel("Reordered index")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(sample, row_disp[sample], linewidth=0.8, color="#2ca02c", label="row")
    axes[0, 1].plot(sample, col_disp[sample], linewidth=0.7, color="#9467bd", alpha=0.75, label="col")
    axes[0, 1].axhline(0, color="#333333", linewidth=0.8, alpha=0.6)
    axes[0, 1].set_title("Permutation displacement")
    axes[0, 1].set_xlabel("Original Jacobian index")
    axes[0, 1].set_ylabel("perm[i] - i")
    axes[0, 1].legend(loc="best")

    bins = min(80, max(20, int(np.sqrt(n))))
    axes[1, 0].hist(row_disp, bins=bins, alpha=0.6, label="row", color="#17becf")
    axes[1, 0].hist(col_disp, bins=bins, alpha=0.45, label="col", color="#ff7f0e")
    axes[1, 0].set_title("Displacement distribution")
    axes[1, 0].set_xlabel("perm[i] - i")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend(loc="best")

    axes[1, 1].hist(
        np.abs(row_disp) / max(n - 1, 1),
        bins=50,
        alpha=0.65,
        label="row",
        color="#8c564b",
    )
    axes[1, 1].hist(
        np.abs(col_disp) / max(n - 1, 1),
        bins=50,
        alpha=0.45,
        label="col",
        color="#e377c2",
    )
    axes[1, 1].set_title("Normalized absolute displacement")
    axes[1, 1].set_xlabel("|perm[i] - i| / (n - 1)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend(loc="best")

    path = figure_root / f"{safe_case_slug(case)}_permutation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_etree(case: dict, figure_root: Path) -> Path:
    meta = case["metadata"]
    etree = case["etree"]
    heatmap = etree_block_heatmap(etree)
    stats = etree_level_stats(etree)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        f"{meta['case']} elimination tree, target {meta['target_bus_label']} buses",
        fontsize=14,
    )

    image = axes[0].imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_title("Binary-tree node sizes, root at top")
    axes[0].set_xlabel("Tree position")
    axes[0].set_ylabel("Level from root")
    fig.colorbar(image, ax=axes[0], label="Node size")

    axes[1].bar(
        stats["level_from_root"],
        stats["sum_size"],
        color="#1f77b4",
        alpha=0.7,
        label="sum of node sizes",
    )
    axes[1].plot(
        stats["level_from_root"],
        stats["nonzero_nodes"],
        marker="o",
        color="#d62728",
        label="nonzero nodes",
    )
    axes[1].plot(
        stats["level_from_root"],
        stats["max_size"],
        marker="s",
        color="#2ca02c",
        label="max node size",
    )
    axes[1].set_title("Level profile")
    axes[1].set_xlabel("Level from root")
    axes[1].set_ylabel("Count / size")
    axes[1].legend(loc="best")
    axes[1].grid(True, axis="y", alpha=0.25)

    path = figure_root / f"{safe_case_slug(case)}_elimination_tree.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_overview(cases: list[dict], figure_root: Path) -> Path:
    labels = [str(case["summary"]["case"]) for case in cases]
    target = np.asarray([int(case["summary"]["target_bus"]) for case in cases])
    n_bus = np.asarray([int(case["summary"]["n_bus"]) for case in cases])
    dim = np.asarray([int(case["summary"]["jacobian_dim"]) for case in cases])
    nnz = np.asarray([int(case["summary"]["jacobian_nnz"]) for case in cases])

    mean_abs_disp = []
    p95_abs_disp = []
    etree_nonzero = []
    for case in cases:
        perm = case["perm_row"]
        n = max(perm.size - 1, 1)
        normalized = np.abs(perm - np.arange(perm.size)) / n
        mean_abs_disp.append(float(normalized.mean()))
        p95_abs_disp.append(float(np.quantile(normalized, 0.95)))
        etree_nonzero.append(int(np.count_nonzero(case["etree"])))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("cuDSS reorder data overview", fontsize=15)

    x = np.arange(len(cases))
    axes[0, 0].bar(x - 0.2, target, width=0.4, label="target bus", color="#7f7f7f")
    axes[0, 0].bar(x + 0.2, n_bus, width=0.4, label="selected n_bus", color="#1f77b4")
    axes[0, 0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0, 0].set_title("Target and selected cases")
    axes[0, 0].set_ylabel("Bus count")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(labels, dim, marker="o", label="Jacobian dim", color="#2ca02c")
    axes[0, 1].plot(labels, nnz, marker="s", label="Jacobian nnz", color="#d62728")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Jacobian size")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, axis="y", alpha=0.25)

    axes[1, 0].bar(x - 0.2, mean_abs_disp, width=0.4, label="mean", color="#17becf")
    axes[1, 0].bar(x + 0.2, p95_abs_disp, width=0.4, label="p95", color="#ff7f0e")
    axes[1, 0].set_xticks(x, labels, rotation=20, ha="right")
    axes[1, 0].set_title("Normalized permutation displacement")
    axes[1, 0].set_ylabel("|perm[i] - i| / (n - 1)")
    axes[1, 0].legend(loc="best")

    axes[1, 1].bar(labels, etree_nonzero, color="#9467bd", alpha=0.75)
    axes[1, 1].set_title("Nonzero elimination-tree nodes")
    axes[1, 1].set_ylabel("Node count")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(True, axis="y", alpha=0.25)

    path = figure_root / "overview.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_index(figure_root: Path, generated: list[Path]) -> None:
    lines = [
        "# cuDSS Reorder Visualizations",
        "",
        "Generated PNG figures:",
        "",
    ]
    for path in generated:
        lines.append(f"- [{path.name}]({path.name})")
    lines.append("")
    (figure_root / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.figure_root.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.dump_root)

    generated: list[Path] = []
    generated.append(plot_overview(cases, args.figure_root))
    for case in cases:
        generated.append(plot_permutation(case, args.figure_root, args.max_scatter_points))
        generated.append(plot_etree(case, args.figure_root))

    write_index(args.figure_root, generated)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
