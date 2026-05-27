#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pypower.dSbus_dV import dSbus_dV
from scipy.io import mmread
from scipy.sparse import csr_matrix, hstack, vstack


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

DEFAULT_CASES = [
    "case_SyntheticUSA",
    "case_ACTIVSg70k",
    "case_ACTIVSg25k",
    "case13659pegase",
    "case_ACTIVSg10k",
    "case9241pegase",
    "case300",
]

DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_RESULTS_CSV = EXP_ROOT / "results" / "pypower_jacobian_sparsity.csv"
DEFAULT_RESULTS_MD = EXP_ROOT / "results" / "pypower_jacobian_sparsity.md"
DEFAULT_COMPARE_MD = EXP_ROOT / "results" / "pypower_vs_cupf_jacobian_sparsity.md"
DEFAULT_CUPF_CSV = EXP_ROOT / "results" / "jacobian_sparsity.csv"


@dataclass(frozen=True)
class CaseResult:
    case: str
    n_bus: int
    n_pv: int
    n_pq: int
    ybus_nnz: int
    rows: int
    cols: int
    stored_nnz: int
    numeric_nnz: int
    explicit_zero_count: int
    density: float
    sparsity: float
    numeric_density: float
    numeric_sparsity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure MATPOWER Newton Jacobian sparsity with PyPower dSbus_dV."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS_MD)
    parser.add_argument("--compare-md", type=Path, default=DEFAULT_COMPARE_MD)
    parser.add_argument("--cupf-csv", type=Path, default=DEFAULT_CUPF_CSV)
    parser.add_argument(
        "--keep-explicit-zeros",
        action="store_true",
        help="Do not eliminate explicit zero entries before reporting stored nnz.",
    )
    return parser.parse_args()


def read_complex_vector(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        if raw.size != 2:
            raise RuntimeError(f"expected two columns in {path}")
        raw = raw.reshape(1, 2)
    return raw[:, 0] + 1j * raw[:, 1]


def read_index_vector(path: Path) -> np.ndarray:
    if path.stat().st_size == 0:
        return np.array([], dtype=np.int64)
    raw = np.loadtxt(path, dtype=np.int64)
    return np.atleast_1d(raw).astype(np.int64, copy=False)


def load_metadata(case_dir: Path) -> dict[str, int]:
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        return {"n_bus": -1, "n_pv": -1, "n_pq": -1, "ybus_nnz": -1}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "n_bus": int(metadata.get("n_bus", -1)),
        "n_pv": int(metadata.get("n_pv", -1)),
        "n_pq": int(metadata.get("n_pq", -1)),
        "ybus_nnz": int(metadata.get("ybus_nnz", -1)),
    }


def build_pypower_jacobian(case_dir: Path) -> csr_matrix:
    ybus = mmread(case_dir / "dump_Ybus.mtx").tocsr()
    ybus.sort_indices()
    voltage = read_complex_vector(case_dir / "dump_V.txt")
    pv = read_index_vector(case_dir / "dump_pv.txt")
    pq = read_index_vector(case_dir / "dump_pq.txt")

    pvpq = np.r_[pv, pq]
    dS_dVm, dS_dVa = dSbus_dV(ybus, voltage)

    j11 = dS_dVa[np.array([pvpq]).T, pvpq].real
    j12 = dS_dVm[np.array([pvpq]).T, pq].real
    j21 = dS_dVa[np.array([pq]).T, pvpq].imag
    j22 = dS_dVm[np.array([pq]).T, pq].imag
    jacobian = vstack([hstack([j11, j12]), hstack([j21, j22])], format="csr")
    jacobian.sum_duplicates()
    jacobian.sort_indices()
    return jacobian


def measure_case(args: argparse.Namespace, case: str) -> CaseResult:
    case_dir = args.dataset_root / case
    if not case_dir.exists():
        raise FileNotFoundError(f"case directory not found: {case_dir}")
    metadata = load_metadata(case_dir)
    jacobian = build_pypower_jacobian(case_dir)

    numeric_nnz_before = int(np.count_nonzero(jacobian.data))
    explicit_zero_count = int(jacobian.nnz - numeric_nnz_before)
    if not args.keep_explicit_zeros:
        jacobian.eliminate_zeros()
        jacobian.sum_duplicates()
        jacobian.sort_indices()

    total_entries = float(jacobian.shape[0]) * float(jacobian.shape[1])
    stored_nnz = int(jacobian.nnz)
    numeric_nnz = int(np.count_nonzero(jacobian.data))
    density = float(stored_nnz) / total_entries
    sparsity = 1.0 - density
    numeric_density = float(numeric_nnz) / total_entries
    numeric_sparsity = 1.0 - numeric_density

    return CaseResult(
        case=case,
        n_bus=metadata["n_bus"],
        n_pv=metadata["n_pv"],
        n_pq=metadata["n_pq"],
        ybus_nnz=metadata["ybus_nnz"],
        rows=int(jacobian.shape[0]),
        cols=int(jacobian.shape[1]),
        stored_nnz=stored_nnz,
        numeric_nnz=numeric_nnz,
        explicit_zero_count=explicit_zero_count,
        density=density,
        sparsity=sparsity,
        numeric_density=numeric_density,
        numeric_sparsity=numeric_sparsity,
    )


def write_csv(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "n_bus",
        "n_pv",
        "n_pq",
        "ybus_nnz",
        "jacobian_rows",
        "jacobian_cols",
        "jacobian_stored_nnz",
        "jacobian_numeric_nnz",
        "jacobian_explicit_zero_count",
        "density",
        "sparsity",
        "numeric_density",
        "numeric_sparsity",
        "density_percent",
        "sparsity_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case": result.case,
                    "n_bus": result.n_bus,
                    "n_pv": result.n_pv,
                    "n_pq": result.n_pq,
                    "ybus_nnz": result.ybus_nnz,
                    "jacobian_rows": result.rows,
                    "jacobian_cols": result.cols,
                    "jacobian_stored_nnz": result.stored_nnz,
                    "jacobian_numeric_nnz": result.numeric_nnz,
                    "jacobian_explicit_zero_count": result.explicit_zero_count,
                    "density": f"{result.density:.17e}",
                    "sparsity": f"{result.sparsity:.17e}",
                    "numeric_density": f"{result.numeric_density:.17e}",
                    "numeric_sparsity": f"{result.numeric_sparsity:.17e}",
                    "density_percent": f"{100.0 * result.density:.12f}",
                    "sparsity_percent": f"{100.0 * result.sparsity:.12f}",
                }
            )


def write_markdown(path: Path, results: list[CaseResult], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PyPower Jacobian Sparsity",
        "",
        f"- Date: 2026-05-25",
        f"- Dataset root: `{args.dataset_root}`",
        "- Tool path: `pypower.dSbus_dV`",
        "- Matrix: MATPOWER/PyPower Newton four-block Jacobian `[J11 J12; J21 J22]`",
        "- Structural density = stored nnz / (rows * cols)",
        "- Structural sparsity = 1 - structural density",
        "",
        "| case | buses | J dim | stored nnz | density (%) | sparsity (%) | explicit zeros |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.case} | "
            f"{result.n_bus:,} | "
            f"{result.rows:,} | "
            f"{result.stored_nnz:,} | "
            f"{100.0 * result.density:.9f} | "
            f"{100.0 * result.sparsity:.9f} | "
            f"{result.explicit_zero_count:,} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def read_cupf_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["case"]: row for row in csv.DictReader(fh)}


def write_compare_markdown(
    path: Path,
    results: list[CaseResult],
    pypower_csv: Path,
    cupf_csv: Path,
) -> None:
    cupf_rows = read_cupf_rows(cupf_csv)
    if not cupf_rows:
        return

    lines = [
        "# PyPower vs cuPF Jacobian Sparsity",
        "",
        f"- PyPower CSV: `{pypower_csv}`",
        f"- cuPF CSV: `{cupf_csv}`",
        "",
        "| case | PyPower nnz | cuPF nnz | nnz diff | PyPower sparsity (%) | cuPF sparsity (%) | sparsity diff (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        cupf = cupf_rows.get(result.case)
        if cupf is None:
            continue
        cupf_nnz = int(cupf["jacobian_stored_nnz"])
        cupf_sparsity_pct = 100.0 * float(cupf["sparsity"])
        pypower_sparsity_pct = 100.0 * result.sparsity
        lines.append(
            "| "
            f"{result.case} | "
            f"{result.stored_nnz:,} | "
            f"{cupf_nnz:,} | "
            f"{result.stored_nnz - cupf_nnz:,} | "
            f"{pypower_sparsity_pct:.9f} | "
            f"{cupf_sparsity_pct:.9f} | "
            f"{pypower_sparsity_pct - cupf_sparsity_pct:.9f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

    results: list[CaseResult] = []
    for case in args.cases:
        result = measure_case(args, case)
        results.append(result)
        print(
            f"[OK] {case} dim={result.rows} nnz={result.stored_nnz} "
            f"density={100.0 * result.density:.9f}% "
            f"sparsity={100.0 * result.sparsity:.9f}%",
            flush=True,
        )

    write_csv(args.results_csv, results)
    write_markdown(args.results_md, results, args)
    write_compare_markdown(args.compare_md, results, args.results_csv, args.cupf_csv)
    print(f"[DONE] csv={args.results_csv}", flush=True)
    print(f"[DONE] markdown={args.results_md}", flush=True)
    if args.cupf_csv.exists():
        print(f"[DONE] compare={args.compare_md}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
