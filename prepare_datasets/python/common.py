from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path
import json

import numpy as np
import scipy.sparse as sp
from numpy import c_, exp, ones, pi, zeros
from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.idx_brch import QT
from pypower.idx_bus import VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, VG
from pypower.loadcase import loadcase
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from scipy.io import mmwrite


DATASETS_ROOT = Path(environ.get("CUPF_DATASETS_ROOT", "/workspace/datasets"))
PGLIB_OPF_DATASET_ROOT = Path(environ.get("PGLIB_OPF_DATASET_ROOT", str(DATASETS_ROOT / "pglib-opf")))
MAT_DATASET_ROOT = Path(environ.get("MAT_DATASET_ROOT", str(PGLIB_OPF_DATASET_ROOT / "pf_dataset")))
NR_DATASET_ROOT = Path(environ.get("NR_DATASET_ROOT", str(PGLIB_OPF_DATASET_ROOT / "nr_dataset")))
CUPF_DUMP_ROOT = Path(environ.get("CUPF_DUMP_ROOT", str(PGLIB_OPF_DATASET_ROOT / "cuPF_datasets")))
TARGET_CASES = (
    "118_ieee",
    "793_goc",
    "1354_pegase",
    "2746wop_k",
    "4601_goc",
    "8387_pegase",
    "9241_pegase",
)


@dataclass
class PreprocessedCase:
    case_stem: str
    short_case_name: str
    source_path: Path
    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    ref: np.ndarray
    pv: np.ndarray
    pq: np.ndarray
    V0: np.ndarray
    Ybus: sp.csr_matrix
    Sbus: np.ndarray


def case_stem(case_name: str) -> str:
    stem = Path(case_name).stem
    if stem.startswith("pglib_opf_case"):
        return stem
    if stem.startswith("pglib_opf_"):
        return stem
    if stem.startswith("case"):
        return f"pglib_opf_{stem}"
    return f"pglib_opf_case{stem}"


def short_case_name(case_name: str) -> str:
    stem = Path(case_name).stem
    if stem.startswith("pglib_opf_"):
        return stem.removeprefix("pglib_opf_")
    return stem


def mat_case_path(case_name: str, mat_root: str | Path = MAT_DATASET_ROOT) -> Path:
    root = Path(mat_root)
    raw_path = Path(case_name)
    candidates = []

    if raw_path.exists():
        return raw_path
    if raw_path.suffix == ".mat":
        candidates.append(root / raw_path.name)
    else:
        candidates.append(root / f"{raw_path.stem}.mat")
    candidates.append(root / f"{case_stem(case_name)}.mat")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"MAT case not found: {candidates[-1]}")


def ensure_branch_result_columns(ppc: dict) -> dict:
    ppc = dict(ppc)
    branch = np.asarray(ppc["branch"])
    needed_cols = QT + 1
    if branch.shape[1] < needed_cols:
        branch = c_[
            branch,
            zeros((branch.shape[0], needed_cols - branch.shape[1])),
        ]
    ppc["branch"] = branch
    return ppc


def preprocess_case(
    case_name: str,
    mat_root: str | Path = MAT_DATASET_ROOT,
) -> PreprocessedCase:
    case_path = mat_case_path(case_name, mat_root=mat_root)
    stem = case_path.stem
    short_name = short_case_name(case_name)

    ppc = loadcase(str(case_path))
    ppc = ensure_branch_result_columns(ppc)
    ppc = ext2int(ppc)

    base_mva = float(np.asarray(ppc["baseMVA"]).squeeze())
    bus = np.asarray(ppc["bus"])
    gen = np.asarray(ppc["gen"])
    branch = np.asarray(ppc["branch"])

    ref, pv, pq = bustypes(bus, gen)

    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    gbus = gen[on, GEN_BUS].astype(int)

    V0 = bus[:, VM] * exp(1j * pi / 180.0 * bus[:, VA])
    vcb = ones(V0.shape)
    vcb[pq] = 0
    k = np.flatnonzero(vcb[gbus])
    if k.size:
        V0[gbus[k]] = gen[on[k], VG] / np.abs(V0[gbus[k]]) * V0[gbus[k]]

    Ybus, _, _ = makeYbus(base_mva, bus, branch)
    Ybus = Ybus.tocsr()
    Ybus.sort_indices()
    Ybus.data = Ybus.data.astype(np.complex128, copy=False)
    Ybus.indices = Ybus.indices.astype(np.int32, copy=False)
    Ybus.indptr = Ybus.indptr.astype(np.int32, copy=False)

    Sbus = np.asarray(makeSbus(base_mva, bus, gen), dtype=np.complex128).reshape(-1)

    return PreprocessedCase(
        case_stem=stem,
        short_case_name=short_name,
        source_path=case_path,
        base_mva=base_mva,
        bus=bus,
        gen=gen,
        branch=branch,
        ref=np.asarray(ref, dtype=np.int32),
        pv=np.asarray(pv, dtype=np.int32),
        pq=np.asarray(pq, dtype=np.int32),
        V0=np.asarray(V0, dtype=np.complex128).reshape(-1),
        Ybus=Ybus,
        Sbus=Sbus,
    )


def case_metadata(case_data: PreprocessedCase) -> dict:
    return {
        "case_stem": case_data.case_stem,
        "short_case_name": case_data.short_case_name,
        "source_path": str(case_data.source_path),
        "base_mva": case_data.base_mva,
        "n_bus": int(case_data.Ybus.shape[0]),
        "ybus_nnz": int(case_data.Ybus.nnz),
        "n_ref": int(case_data.ref.size),
        "n_pv": int(case_data.pv.size),
        "n_pq": int(case_data.pq.size),
    }


def write_json(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def save_nr_data(
    case_data: PreprocessedCase,
    output_root: str | Path = NR_DATASET_ROOT,
) -> Path:
    case_dir = Path(output_root) / case_data.case_stem
    case_dir.mkdir(parents=True, exist_ok=True)

    np.save(case_dir / "pv.npy", case_data.pv.astype(np.int32, copy=False))
    np.save(case_dir / "pq.npy", case_data.pq.astype(np.int32, copy=False))
    np.save(case_dir / "Sbus.npy", case_data.Sbus.astype(np.complex128, copy=False))
    np.save(case_dir / "V0.npy", case_data.V0.astype(np.complex128, copy=False))
    sp.save_npz(case_dir / "Ybus.npz", case_data.Ybus)

    write_json(case_dir / "metadata.json", case_metadata(case_data))
    return case_dir


def _write_complex_txt(path: Path, values: np.ndarray) -> None:
    matrix = np.column_stack((values.real, values.imag))
    np.savetxt(path, matrix, fmt="%.18e")


def _write_int_txt(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values.astype(np.int32, copy=False), fmt="%d")


def save_cupf_dump(
    case_data: PreprocessedCase,
    output_root: str | Path = CUPF_DUMP_ROOT,
) -> Path:
    case_dir = Path(output_root) / case_data.short_case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    mmwrite(case_dir / "dump_Ybus.mtx", case_data.Ybus)
    _write_complex_txt(case_dir / "dump_Sbus.txt", case_data.Sbus)
    _write_complex_txt(case_dir / "dump_V.txt", case_data.V0)
    _write_int_txt(case_dir / "dump_pv.txt", case_data.pv)
    _write_int_txt(case_dir / "dump_pq.txt", case_data.pq)

    write_json(case_dir / "metadata.json", case_metadata(case_data))
    return case_dir


def all_mat_case_names(mat_root: str | Path = MAT_DATASET_ROOT) -> list[str]:
    return sorted(path.stem for path in Path(mat_root).glob("*.mat"))
