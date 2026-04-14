#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp


DEFAULT_DATASET_ROOT = Path("/workspace/datasets/nr_dataset")
DEFAULT_DUMP_ROOT = Path("/workspace/exp/20260413/iterative/pf_dumps")
DEFAULT_OUTPUT = Path("/workspace/exp/20260413/j_spd_exp/results/j_spd_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check cuPF Jacobian structural symmetry and SPD diagnostics."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Short case names, e.g. case14_ieee case118_ieee. Defaults to a smoke set.",
    )
    parser.add_argument("--all", action="store_true", help="Process all dump cases.")
    parser.add_argument(
        "--max-eig-dim",
        type=int,
        default=300,
        help="Only run dense eigenvalue diagnostics up to this matrix dimension.",
    )
    parser.add_argument(
        "--symmetry-tol",
        type=float,
        default=1e-9,
        help="Tolerance for declaring numeric symmetry.",
    )
    return parser.parse_args()


def short_case_name(dataset_name: str) -> str:
    if dataset_name.startswith("pglib_opf_"):
        return dataset_name.removeprefix("pglib_opf_")
    return dataset_name


def dataset_dir_for_case(dataset_root: Path, case_name: str) -> Path:
    direct = dataset_root / case_name
    if direct.exists():
        return direct

    pglib = dataset_root / f"pglib_opf_{case_name}"
    if pglib.exists():
        return pglib

    raise FileNotFoundError(f"Cannot find dataset directory for {case_name}")


def case_names(args: argparse.Namespace) -> list[str]:
    if args.all:
        return sorted(p.name for p in args.dump_root.iterdir() if p.is_dir())
    if args.cases:
        return args.cases
    return ["case14_ieee", "case60_c", "case118_ieee", "case9241_pegase"]


def load_dump_csr(path: Path) -> sp.csr_matrix:
    rows = cols = nnz = None
    row_ptr = col_idx = values = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            key = parts[0]
            if key == "rows":
                rows = int(parts[1])
            elif key == "cols":
                cols = int(parts[1])
            elif key == "nnz":
                nnz = int(parts[1])
            elif key == "row_ptr":
                row_ptr = np.fromiter((int(x) for x in parts[1:]), dtype=np.int32)
            elif key == "col_idx":
                col_idx = np.fromiter((int(x) for x in parts[1:]), dtype=np.int32)
            elif key == "values":
                values = np.fromiter((float(x) for x in parts[1:]), dtype=np.float64)

    if rows is None or cols is None or nnz is None:
        raise ValueError(f"{path}: missing shape metadata")
    if row_ptr is None or len(row_ptr) != rows + 1:
        raise ValueError(f"{path}: invalid row_ptr")
    if col_idx is None or len(col_idx) != nnz:
        raise ValueError(f"{path}: invalid col_idx")
    if values is None or len(values) != nnz:
        raise ValueError(f"{path}: invalid values")

    return sp.csr_matrix((values, col_idx, row_ptr), shape=(rows, cols))


def pattern_matrix(A: sp.spmatrix) -> sp.csr_matrix:
    B = A.tocsr(copy=True)
    B.data = np.ones(B.nnz, dtype=np.int8)
    return B


def structural_asym_nnz(A: sp.spmatrix) -> int:
    P = pattern_matrix(A)
    D = (P - P.T).tocsr()
    D.eliminate_zeros()
    return int(D.nnz)


def pattern_xor_nnz(A: sp.spmatrix, B: sp.spmatrix) -> int | None:
    if A.shape != B.shape:
        return None
    PA = pattern_matrix(A)
    PB = pattern_matrix(B)
    D = (PA - PB).tocsr()
    D.eliminate_zeros()
    return int(D.nnz)


def max_abs_data(A: sp.spmatrix) -> float:
    B = A.tocsr()
    if B.nnz == 0:
        return 0.0
    return float(np.max(np.abs(B.data)))


def cupf_jacobian_pattern(
    ybus: sp.csr_matrix,
    pv: np.ndarray,
    pq: np.ndarray,
    force_symmetric_y: bool = False,
) -> sp.csr_matrix:
    n_bus = ybus.shape[0]
    pv = np.asarray(pv, dtype=np.int32)
    pq = np.asarray(pq, dtype=np.int32)
    pvpq = np.r_[pv, pq].astype(np.int32)
    n_pvpq = int(pvpq.size)
    n_pq = int(pq.size)
    dim = n_pvpq + n_pq

    if force_symmetric_y:
        ypat = pattern_matrix(ybus)
        ypat = ((ypat + ypat.T) > 0).astype(np.int8).tocsr()
        ypat = (ypat + sp.eye(n_bus, dtype=np.int8, format="csr")).astype(np.int8)
        ypat.data = np.ones(ypat.nnz, dtype=np.int8)
        ypat.eliminate_zeros()
        y_for_pattern = ypat.tocsr()
    else:
        y_for_pattern = ybus.tocsr()

    rmap_pvpq = np.full(n_bus, -1, dtype=np.int32)
    rmap_pq = np.full(n_bus, -1, dtype=np.int32)
    cmap_pvpq = np.full(n_bus, -1, dtype=np.int32)
    cmap_pq = np.full(n_bus, -1, dtype=np.int32)

    for slot, bus in enumerate(pvpq):
        rmap_pvpq[bus] = slot
        cmap_pvpq[bus] = slot
    for slot, bus in enumerate(pq):
        mapped = n_pvpq + slot
        rmap_pq[bus] = mapped
        cmap_pq[bus] = mapped

    rows: list[int] = []
    cols: list[int] = []

    def add_four(i_bus: int, j_bus: int) -> None:
        ji_pvpq = int(rmap_pvpq[i_bus])
        ji_pq = int(rmap_pq[i_bus])
        jj_pvpq = int(cmap_pvpq[j_bus])
        jj_pq = int(cmap_pq[j_bus])

        if ji_pvpq >= 0 and jj_pvpq >= 0:
            rows.append(ji_pvpq)
            cols.append(jj_pvpq)
        if ji_pq >= 0 and jj_pvpq >= 0:
            rows.append(ji_pq)
            cols.append(jj_pvpq)
        if ji_pvpq >= 0 and jj_pq >= 0:
            rows.append(ji_pvpq)
            cols.append(jj_pq)
        if ji_pq >= 0 and jj_pq >= 0:
            rows.append(ji_pq)
            cols.append(jj_pq)

    for i_bus in range(n_bus):
        row_begin = y_for_pattern.indptr[i_bus]
        row_end = y_for_pattern.indptr[i_bus + 1]
        for k in range(row_begin, row_end):
            j_bus = int(y_for_pattern.indices[k])
            if i_bus == j_bus:
                continue
            add_four(i_bus, j_bus)

    for bus in range(n_bus):
        add_four(bus, bus)

    if not rows:
        return sp.csr_matrix((dim, dim), dtype=np.int8)

    data = np.ones(len(rows), dtype=np.int8)
    J = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    J.data = np.ones(J.nnz, dtype=np.int8)
    J.sort_indices()
    return J


def dense_eig_diagnostics(
    J: sp.csr_matrix,
    max_dim: int,
    symmetry_tol: float,
) -> dict[str, float | bool | str]:
    dim = J.shape[0]
    if dim > max_dim:
        return {
            "eig_status": "skipped_dim",
            "raw_spd": False,
            "sympart_min_eig": math.nan,
            "sympart_max_eig": math.nan,
            "sympart_spd": False,
            "normal_min_eig": math.nan,
            "normal_max_eig": math.nan,
            "normal_spd": False,
            "normal_cond": math.nan,
        }

    value_asym = max_abs_data(J - J.T)
    symmetric = value_asym <= symmetry_tol

    sympart = ((J + J.T) * 0.5).toarray()
    sympart_eigs = np.linalg.eigvalsh(sympart)
    sympart_min = float(sympart_eigs[0])
    sympart_max = float(sympart_eigs[-1])

    normal = (J.T @ J).toarray()
    normal_eigs = np.linalg.eigvalsh(normal)
    normal_min = float(normal_eigs[0])
    normal_max = float(normal_eigs[-1])
    normal_cond = normal_max / normal_min if normal_min > 0 else math.inf

    raw_spd = bool(symmetric and sympart_min > symmetry_tol)
    sympart_spd = bool(sympart_min > symmetry_tol)
    normal_spd = bool(normal_min > symmetry_tol)

    return {
        "eig_status": "computed",
        "raw_spd": raw_spd,
        "sympart_min_eig": sympart_min,
        "sympart_max_eig": sympart_max,
        "sympart_spd": sympart_spd,
        "normal_min_eig": normal_min,
        "normal_max_eig": normal_max,
        "normal_spd": normal_spd,
        "normal_cond": float(normal_cond),
    }


def analyze_case(args: argparse.Namespace, case_name: str) -> dict[str, object]:
    dataset_dir = dataset_dir_for_case(args.dataset_root, case_name)
    dump_path = args.dump_root / short_case_name(case_name) / "cuda_mixed_edge" / "iter_000" / "J.csr"

    ybus = sp.load_npz(dataset_dir / "Ybus.npz").tocsr()
    ybus.sort_indices()
    pv = np.load(dataset_dir / "pv.npy").astype(np.int32, copy=False)
    pq = np.load(dataset_dir / "pq.npy").astype(np.int32, copy=False)

    cupf_pattern = cupf_jacobian_pattern(ybus, pv, pq, force_symmetric_y=False)
    lift_pattern = cupf_jacobian_pattern(ybus, pv, pq, force_symmetric_y=True)

    row: dict[str, object] = {
        "case": short_case_name(case_name),
        "dataset_case": dataset_dir.name,
        "n_bus": int(ybus.shape[0]),
        "n_pv": int(pv.size),
        "n_pq": int(pq.size),
        "dimF": int(cupf_pattern.shape[0]),
        "ybus_nnz": int(ybus.nnz),
        "ybus_struct_asym_nnz": structural_asym_nnz(ybus),
        "cupf_pattern_nnz": int(cupf_pattern.nnz),
        "cupf_pattern_struct_asym_nnz": structural_asym_nnz(cupf_pattern),
        "lift_pattern_nnz": int(lift_pattern.nnz),
        "lift_pattern_struct_asym_nnz": structural_asym_nnz(lift_pattern),
        "lift_vs_cupf_xor_nnz": pattern_xor_nnz(lift_pattern, cupf_pattern),
        "dump_exists": dump_path.exists(),
        "dump_dim": "",
        "dump_nnz": "",
        "dump_pattern_struct_asym_nnz": "",
        "dump_pattern_vs_cupf_xor_nnz": "",
        "dump_value_asym_inf": "",
        "eig_status": "no_dump",
        "raw_spd": "",
        "sympart_min_eig": "",
        "sympart_max_eig": "",
        "sympart_spd": "",
        "normal_min_eig": "",
        "normal_max_eig": "",
        "normal_spd": "",
        "normal_cond": "",
    }

    if dump_path.exists():
        J = load_dump_csr(dump_path)
        row.update(
            {
                "dump_dim": int(J.shape[0]),
                "dump_nnz": int(J.nnz),
                "dump_pattern_struct_asym_nnz": structural_asym_nnz(J),
                "dump_pattern_vs_cupf_xor_nnz": pattern_xor_nnz(J, cupf_pattern),
                "dump_value_asym_inf": max_abs_data(J - J.T),
            }
        )
        row.update(dense_eig_diagnostics(J, args.max_eig_dim, args.symmetry_tol))

    return row


def main() -> None:
    args = parse_args()
    names = case_names(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [analyze_case(args, name) for name in names]
    fieldnames = list(rows[0].keys()) if rows else []

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dump_rows = [row for row in rows if row["dump_exists"]]
    structurally_asym = [
        row for row in dump_rows if row["dump_pattern_struct_asym_nnz"] not in ("", 0)
    ]
    numerically_symmetric = [
        row
        for row in dump_rows
        if row["dump_value_asym_inf"] != "" and float(row["dump_value_asym_inf"]) <= args.symmetry_tol
    ]

    print(f"wrote {args.output}")
    print(f"cases={len(rows)} dump_cases={len(dump_rows)}")
    print(f"dump_structurally_asymmetric_cases={len(structurally_asym)}")
    print(f"dump_numerically_symmetric_cases={len(numerically_symmetric)}")

    eig_rows = [row for row in dump_rows if row["eig_status"] == "computed"]
    if eig_rows:
        raw_spd = sum(1 for row in eig_rows if row["raw_spd"])
        sympart_spd = sum(1 for row in eig_rows if row["sympart_spd"])
        normal_spd = sum(1 for row in eig_rows if row["normal_spd"])
        print(f"eig_cases={len(eig_rows)} raw_spd={raw_spd}")
        print(f"eig_cases={len(eig_rows)} sympart_spd={sympart_spd}")
        print(f"eig_cases={len(eig_rows)} normal_spd={normal_spd}")


if __name__ == "__main__":
    main()
