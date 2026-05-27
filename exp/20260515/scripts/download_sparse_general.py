#!/usr/bin/env python3
"""Download nonsymmetric/general SuiteSparse matrices for fp32 benchmarks."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


SUITESPARSE_MM_BASE_URL = "https://sparse.tamu.edu/MM"


@dataclass(frozen=True)
class SparseMatrixSpec:
    """SuiteSparse Matrix Collection matrix identifier."""

    group: str
    name: str

    @property
    def url(self) -> str:
        """Return the Matrix Market tarball URL for this matrix."""
        return f"{SUITESPARSE_MM_BASE_URL}/{self.group}/{self.name}.tar.gz"


DEFAULT_MATRICES: tuple[SparseMatrixSpec, ...] = (
    SparseMatrixSpec("Hamm", "scircuit"),
    SparseMatrixSpec("Simon", "venkat01"),
    SparseMatrixSpec("Bova", "rma10"),
    SparseMatrixSpec("Rajat", "rajat31"),
)


def experiment_root() -> Path:
    """Return the root directory of this experiment."""
    return Path(__file__).resolve().parents[1]


def parse_matrix_filter(values: Iterable[str]) -> set[str]:
    """Normalize matrix filters to a set of accepted names or group/name keys."""
    return {value.strip() for value in values if value.strip()}


def selected_matrices(filters: set[str]) -> list[SparseMatrixSpec]:
    """Return the default matrix list filtered by optional names."""
    if not filters:
        return list(DEFAULT_MATRICES)

    selected: list[SparseMatrixSpec] = []
    for spec in DEFAULT_MATRICES:
        keys = {spec.name, f"{spec.group}/{spec.name}"}
        if keys & filters:
            selected.append(spec)

    missing = filters - {spec.name for spec in selected} - {
        f"{spec.group}/{spec.name}" for spec in selected
    }
    if missing:
        raise ValueError(f"unknown matrix filter(s): {', '.join(sorted(missing))}")
    return selected


def download_tarball(url: str, archive_path: Path, timeout: float) -> None:
    """Download a tarball with requests unless it already exists."""
    if archive_path.exists():
        print(f"[skip] archive already exists: {archive_path}")
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".part")
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as out:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
    tmp_path.replace(archive_path)
    print(f"[saved] {archive_path}")


def extract_matrix_market(archive_path: Path, matrix_name: str, destination: Path) -> Path:
    """Extract the requested .mtx member from a SuiteSparse tarball."""
    if destination.exists():
        print(f"[skip] Matrix Market file already exists: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_basename = f"{matrix_name}.mtx"
    with tarfile.open(archive_path, "r:gz") as tar:
        candidates = [
            member
            for member in tar.getmembers()
            if member.isfile() and Path(member.name).name == expected_basename
        ]
        if not candidates:
            raise RuntimeError(
                f"could not find {expected_basename} inside archive {archive_path}"
            )

        member = candidates[0]
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError(f"failed to extract {member.name} from {archive_path}")

        tmp_path = destination.with_suffix(destination.suffix + ".part")
        with source, tmp_path.open("wb") as out:
            shutil.copyfileobj(source, out)
        tmp_path.replace(destination)

    print(f"[mtx] {destination}")
    return destination


def prepare_matrix(spec: SparseMatrixSpec, output_root: Path, timeout: float) -> Path:
    """Download and extract one SuiteSparse Matrix Market tarball."""
    matrix_dir = output_root / spec.name
    destination = matrix_dir / f"{spec.name}.mtx"
    if destination.exists():
        print(f"[skip] prepared matrix already exists: {destination}")
        print(f"[mtx] {destination}")
        return destination

    archive_path = output_root / "_archives" / f"{spec.group}_{spec.name}.tar.gz"
    download_tarball(spec.url, archive_path, timeout)
    return extract_matrix_market(archive_path, spec.name, destination)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download SuiteSparse nonsymmetric/general Matrix Market benchmarks."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=experiment_root() / "data" / "sparse",
        help="Directory that will contain <matrix>/<matrix>.mtx.",
    )
    parser.add_argument(
        "--matrix",
        action="append",
        default=[],
        help="Optional matrix name or group/name filter. May be passed multiple times.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    """Download and extract all selected sparse benchmark matrices."""
    args = parse_args()
    filters = parse_matrix_filter(args.matrix)
    matrices = selected_matrices(filters)
    output_root = args.output_root.resolve()

    print(f"[root] {output_root}")
    prepared_paths = [
        prepare_matrix(spec, output_root=output_root, timeout=args.timeout) for spec in matrices
    ]

    print("\nPrepared Matrix Market files:")
    for path in prepared_paths:
        print(path)


if __name__ == "__main__":
    main()
