#!/usr/bin/env python3
"""Generate HPL/LAPACK-style fp32 random dense general linear systems."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SIZES: tuple[int, ...] = (4096, 8192, 16384)
DEFAULT_SEED = 20260515


def experiment_root() -> Path:
    """Return the root directory of this experiment."""
    return Path(__file__).resolve().parents[1]


def parse_sizes(values: Iterable[int]) -> list[int]:
    """Validate dense matrix sizes from the command line."""
    sizes = list(values)
    if not sizes:
        raise ValueError("at least one matrix size must be provided")
    for n in sizes:
        if n <= 0:
            raise ValueError(f"dense matrix size must be positive, got {n}")
    return sizes


def complete_dataset(case_dir: Path) -> bool:
    """Return true when all dense dataset files already exist."""
    required = ("A.npy", "b.npy", "x_true.npy", "meta.json")
    return all((case_dir / name).exists() for name in required)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Save a NumPy array atomically without allowing pickle payloads."""
    tmp_path = path.with_suffix(path.suffix + ".part")
    with tmp_path.open("wb") as out:
        np.save(out, array, allow_pickle=False)
    tmp_path.replace(path)


def generate_dense_case(n: int, data_root: Path, seed: int, force: bool) -> Path:
    """Generate one dense random general Ax=b benchmark case."""
    case_dir = data_root / f"random_general_n{n}"
    if complete_dataset(case_dir) and not force:
        print(f"[skip] dense dataset already exists: {case_dir}")
        return case_dir

    case_dir.mkdir(parents=True, exist_ok=True)
    matrix_seed = seed + n
    rng = np.random.default_rng(matrix_seed)

    print(f"[generate] n={n} seed={matrix_seed}")
    scale = np.float32(1.0 / math.sqrt(float(n)))
    A = rng.standard_normal((n, n), dtype=np.float32)
    A = np.ascontiguousarray(A * scale, dtype=np.float32)
    x_true = np.ones(n, dtype=np.float32)
    b = np.asarray(A @ x_true, dtype=np.float32)

    save_npy_atomic(case_dir / "A.npy", A)
    save_npy_atomic(case_dir / "x_true.npy", x_true)
    save_npy_atomic(case_dir / "b.npy", b)

    meta = {
        "name": f"random_general_n{n}",
        "matrix_type": "dense_general",
        "rows": n,
        "cols": n,
        "dtype": "float32",
        "layout": "row-major",
        "npy_order": "C",
        "seed_base": seed,
        "matrix_seed": matrix_seed,
        "generation": "A = randn(n, n).astype(float32) / sqrt(n)",
        "x_true": "ones(n, dtype=float32)",
        "rhs": "b = A @ x_true",
    }
    with (case_dir / "meta.json").open("w", encoding="utf-8") as out:
        json.dump(meta, out, indent=2)
        out.write("\n")

    print(f"[done] {case_dir}")
    return case_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate fp32 HPL/LAPACK-style random dense general matrices."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIZES),
        help="Dense matrix sizes to generate.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=experiment_root() / "data" / "dense",
        help="Output root for random_general_n<n> directories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed. Each size uses seed + n for order-independent generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate datasets even when all output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all requested dense benchmark cases."""
    args = parse_args()
    sizes = parse_sizes(args.sizes)
    data_root = args.data_root.resolve()
    print(f"[root] {data_root}")
    for n in sizes:
        generate_dense_case(n=n, data_root=data_root, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
