#!/usr/bin/env python3
"""Dump independent Newton-Raphson power-flow Jacobian systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix, hstack, identity, vstack
from scipy.sparse.linalg import spsolve


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
DUMP_ROOT = EXP_ROOT / "datasets" / "dumped_systems"
THIRD_PARTY_PYTHON = EXP_ROOT / "third_party" / "python"

PREFERRED_CASES = [
    "case14",
    "case118",
    "case300",
    "case1354pegase",
    "case2869pegase",
    "case9241pegase",
    "case13659pegase",
    "case_ACTIVSg70k",
]


def ensure_imports() -> None:
    if THIRD_PARTY_PYTHON.exists():
        sys.path.insert(0, str(THIRD_PARTY_PYTHON))
    missing = []
    for module in ["pypower", "matpowercaseframes"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        THIRD_PARTY_PYTHON.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(THIRD_PARTY_PYTHON),
                *missing,
            ]
        )
        sys.path.insert(0, str(THIRD_PARTY_PYTHON))


ensure_imports()

from matpowercaseframes import CaseFrames  # noqa: E402
from pypower.bustypes import bustypes  # noqa: E402
from pypower.dSbus_dV import dSbus_dV  # noqa: E402
from pypower.ext2int import ext2int  # noqa: E402
from pypower.idx_bus import VA, VM  # noqa: E402
from pypower.makeSbus import makeSbus  # noqa: E402
from pypower.makeYbus import makeYbus  # noqa: E402


@dataclass
class Snapshot:
    iteration: int
    voltage: np.ndarray
    mismatch: np.ndarray
    jacobian: csr_matrix
    rhs: np.ndarray
    norm_inf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(PREFERRED_CASES))
    parser.add_argument("--iterations", default="0,1,last")
    parser.add_argument("--max-it", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--output-root", default=str(DUMP_ROOT))
    parser.add_argument("--include-synthetic-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def case_key(path: Path) -> str:
    return path.stem


def locate_case_files() -> Dict[str, Path]:
    case_files: Dict[str, Path] = {}
    for path in REPO_ROOT.rglob("*.m"):
        rel = path.relative_to(REPO_ROOT)
        if "exp/20260510/lin_sol" in str(rel) or "__MACOSX" in rel.parts:
            continue
        name = case_key(path)
        lower = name.lower()
        if not (lower.startswith("case") or "korea" in lower or "korean" in lower):
            continue
        existing = case_files.get(name)
        if existing is None:
            case_files[name] = path
        elif "matpower" in str(path).lower() and "matpower" not in str(existing).lower():
            case_files[name] = path
    return case_files


def requested_cases(raw: str, available: Dict[str, Path]) -> List[str]:
    if raw.strip().lower() in {"auto", "preferred"}:
        names = [name for name in PREFERRED_CASES if name in available]
    else:
        names = [item.strip() for item in raw.split(",") if item.strip()]
    for name in sorted(available):
        lower = name.lower()
        if ("korea" in lower or "korean" in lower) and name not in names:
            names.append(name)
    return names


def dataframe_array(cf: CaseFrames, key: str) -> np.ndarray:
    data = cf.to_dict()[key]
    return np.array(data, dtype=float)


def parse_number(token: str) -> float:
    cleaned = token.strip().strip(";")
    lower = cleaned.lower()
    if lower in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if lower in {"-inf", "-infinity"}:
        return float("-inf")
    return float(cleaned)


def parse_matpower_array(text: str, name: str) -> np.ndarray:
    match = re.search(rf"mpc\.{re.escape(name)}\s*=\s*\[(.*?)\];", text, flags=re.S)
    if not match:
        raise ValueError(f"mpc.{name} array not found")
    rows = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("%", 1)[0].replace(";", " ").strip()
        if not line:
            continue
        rows.append([parse_number(tok) for tok in line.split()])
    if not rows:
        raise ValueError(f"mpc.{name} array is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"mpc.{name} has ragged rows")
    return np.array(rows, dtype=float)


def parse_matpower_case_fallback(path: Path) -> Dict[str, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([^;]+);", text)
    if not base_match:
        raise ValueError("mpc.baseMVA not found")
    version_match = re.search(r"mpc\.version\s*=\s*'([^']+)'", text)
    ppc = {
        "version": version_match.group(1) if version_match else "2",
        "baseMVA": parse_number(base_match.group(1)),
        "bus": parse_matpower_array(text, "bus"),
        "gen": parse_matpower_array(text, "gen"),
        "branch": parse_matpower_array(text, "branch"),
    }
    try:
        ppc["gencost"] = parse_matpower_array(text, "gencost")
    except ValueError:
        pass
    return ext2int(ppc)


def load_matpower_case(path: Path) -> Dict[str, np.ndarray]:
    try:
        cf = CaseFrames(str(path))
        case_dict = cf.to_dict()
        ppc = {
            "version": str(case_dict.get("version", cf.version)),
            "baseMVA": float(case_dict.get("baseMVA", cf.baseMVA)),
            "bus": dataframe_array(cf, "bus"),
            "gen": dataframe_array(cf, "gen"),
            "branch": dataframe_array(cf, "branch"),
        }
        if "gencost" in case_dict:
            ppc["gencost"] = np.array(case_dict["gencost"], dtype=float)
        return ext2int(ppc)
    except Exception:
        return parse_matpower_case_fallback(path)


def build_system(Ybus: csr_matrix, Sbus: np.ndarray, V: np.ndarray, pv: np.ndarray, pq: np.ndarray) -> Tuple[np.ndarray, csr_matrix, np.ndarray, float]:
    pvpq = np.r_[pv, pq]
    mis = V * np.conj(Ybus * V) - Sbus
    F = np.r_[mis[pv].real, mis[pq].real, mis[pq].imag]
    dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
    J11 = dS_dVa[np.array([pvpq]).T, pvpq].real
    J12 = dS_dVm[np.array([pvpq]).T, pq].real
    J21 = dS_dVa[np.array([pq]).T, pvpq].imag
    J22 = dS_dVm[np.array([pq]).T, pq].imag
    J = vstack([hstack([J11, J12]), hstack([J21, J22])], format="csr")
    rhs = -F
    return F, J, rhs, float(np.linalg.norm(F, ord=np.inf))


def update_voltage(V: np.ndarray, dx: np.ndarray, pv: np.ndarray, pq: np.ndarray) -> np.ndarray:
    Va = np.angle(V).copy()
    Vm = np.abs(V).copy()
    npv = len(pv)
    npq = len(pq)
    if npv:
        Va[pv] = Va[pv] + dx[0:npv]
    if npq:
        Va[pq] = Va[pq] + dx[npv : npv + npq]
        Vm[pq] = Vm[pq] + dx[npv + npq : npv + 2 * npq]
    V = Vm * np.exp(1j * Va)
    return np.abs(V) * np.exp(1j * np.angle(V))


def pattern_hash(matrix: csr_matrix) -> str:
    mat = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(mat.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(mat.indices, dtype=np.int64).tobytes())
    digest.update(str(mat.shape).encode("ascii"))
    return digest.hexdigest()


def parse_iteration_selection(raw: str) -> Tuple[set[int], bool]:
    explicit: set[int] = set()
    include_last = False
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "last":
            include_last = True
        else:
            explicit.add(int(item))
    return explicit, include_last


def run_cpu_newton(ppc: Dict[str, np.ndarray], iterations: str, max_it: int, tol: float) -> Tuple[List[Snapshot], bool, int, float]:
    base_mva = ppc["baseMVA"]
    bus = ppc["bus"]
    gen = ppc["gen"]
    branch = ppc["branch"]
    Ybus, _, _ = makeYbus(base_mva, bus, branch)
    Sbus = makeSbus(base_mva, bus, gen)
    ref, pv, pq = bustypes(bus, gen)
    del ref
    V = bus[:, VM] * np.exp(1j * np.pi / 180.0 * bus[:, VA])
    explicit, include_last = parse_iteration_selection(iterations)
    snapshots: Dict[int, Snapshot] = {}
    converged = False
    final_iteration = 0
    final_norm_inf = math.inf
    last_snapshot: Optional[Snapshot] = None

    for iteration in range(max_it + 1):
        F, J, rhs, norm_inf = build_system(Ybus, Sbus, V, pv, pq)
        snap = Snapshot(iteration, V.copy(), F.copy(), J.copy(), rhs.copy(), norm_inf)
        last_snapshot = snap
        final_iteration = iteration
        final_norm_inf = norm_inf
        if iteration in explicit:
            snapshots[iteration] = snap
        if norm_inf < tol:
            converged = True
            break
        if iteration == max_it:
            break
        dx = spsolve(J, rhs)
        V = update_voltage(V, dx, pv, pq)

    if include_last and last_snapshot is not None:
        snapshots[last_snapshot.iteration] = last_snapshot
    if not snapshots and last_snapshot is not None:
        snapshots[last_snapshot.iteration] = last_snapshot
    return [snapshots[k] for k in sorted(snapshots)], converged, final_iteration, final_norm_inf


def write_vector(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, fmt="%.17e")


def dump_snapshot(
    out_root: Path,
    case_name: str,
    source_path: str,
    ppc: Dict[str, np.ndarray],
    snap: Snapshot,
    converged: bool,
    final_iteration: int,
    final_norm_inf: float,
    force: bool,
) -> Dict[str, object]:
    out_dir = out_root / case_name / f"iter_{snap.iteration:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "J.mtx"
    rhs_path = out_dir / "rhs.txt"
    xref_path = out_dir / "x_ref.txt"
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    J = snap.jacobian.tocsr()
    rhs = snap.rhs.astype(np.float64)
    x_ref = spsolve(J, rhs)
    rhs_norm_2 = float(np.linalg.norm(rhs))
    denom = max(rhs_norm_2, np.finfo(float).tiny)
    cpu_res = float(np.linalg.norm(J @ x_ref - rhs) / denom)
    mmwrite(matrix_path, J)
    write_vector(rhs_path, rhs)
    write_vector(xref_path, x_ref)
    bus = ppc["bus"]
    branch = ppc["branch"]
    gen = ppc["gen"]
    _, pv, pq = bustypes(bus, gen)
    meta = {
        "case_name": case_name,
        "iteration": snap.iteration,
        "num_bus": int(bus.shape[0]),
        "num_branch": int(branch.shape[0]),
        "num_pv": int(len(pv)),
        "num_pq": int(len(pq)),
        "matrix_rows": int(J.shape[0]),
        "matrix_cols": int(J.shape[1]),
        "nnz": int(J.nnz),
        "rhs_norm_2": rhs_norm_2,
        "rhs_norm_inf": float(np.linalg.norm(rhs, ord=np.inf)),
        "pattern_hash": pattern_hash(J),
        "source_case_path": source_path,
        "converged_by_cpu_nr": bool(converged),
        "cpu_reference_residual": cpu_res,
        "cpu_nr_final_iteration": int(final_iteration),
        "cpu_nr_final_mismatch_inf": float(final_norm_inf),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def dump_synthetic(out_root: Path, force: bool) -> Dict[str, object]:
    case_name = "synthetic_validation"
    out_dir = out_root / case_name / "iter_000"
    out_dir.mkdir(parents=True, exist_ok=True)
    J = csr_matrix(
        np.array(
            [
                [4.0, -1.0, 0.0, 0.0, 0.5],
                [-1.0, 4.5, -1.0, 0.0, 0.0],
                [0.2, -1.0, 3.5, -0.5, 0.0],
                [0.0, 0.0, -0.5, 2.5, -0.4],
                [0.0, 0.1, 0.0, -0.4, 2.0],
            ],
            dtype=np.float64,
        )
    )
    rhs = np.array([1.0, -2.0, 0.5, 0.25, -0.75], dtype=np.float64)
    x_ref = spsolve(J, rhs)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    mmwrite(out_dir / "J.mtx", J)
    write_vector(out_dir / "rhs.txt", rhs)
    write_vector(out_dir / "x_ref.txt", x_ref)
    meta = {
        "case_name": case_name,
        "iteration": 0,
        "num_bus": 0,
        "num_branch": 0,
        "num_pv": 0,
        "num_pq": 0,
        "matrix_rows": int(J.shape[0]),
        "matrix_cols": int(J.shape[1]),
        "nnz": int(J.nnz),
        "rhs_norm_2": float(np.linalg.norm(rhs)),
        "rhs_norm_inf": float(np.linalg.norm(rhs, ord=np.inf)),
        "pattern_hash": pattern_hash(J),
        "source_case_path": "synthetic_validation",
        "converged_by_cpu_nr": True,
        "cpu_reference_residual": float(np.linalg.norm(J @ x_ref - rhs) / np.linalg.norm(rhs)),
        "cpu_nr_final_iteration": 0,
        "cpu_nr_final_mismatch_inf": 0.0,
        "validation_only": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    available = locate_case_files()
    selected = requested_cases(args.cases, available)
    log = {
        "repo_root": str(REPO_ROOT),
        "output_root": str(out_root),
        "selected_cases": selected,
        "dumped": [],
        "skipped": [],
    }
    if args.include_synthetic_validation:
        meta = dump_synthetic(out_root, args.force)
        log["dumped"].append({"case_name": "synthetic_validation", "iteration": 0, "meta": meta})

    for name in selected:
        path = available.get(name)
        if path is None:
            log["skipped"].append({"case_name": name, "reason": "case file not found in repository"})
            continue
        try:
            ppc = load_matpower_case(path)
            snapshots, converged, final_iteration, final_norm_inf = run_cpu_newton(
                ppc, args.iterations, args.max_it, args.tol
            )
            for snap in snapshots:
                meta = dump_snapshot(
                    out_root,
                    name,
                    str(path.relative_to(REPO_ROOT)),
                    ppc,
                    snap,
                    converged,
                    final_iteration,
                    final_norm_inf,
                    args.force,
                )
                log["dumped"].append({"case_name": name, "iteration": snap.iteration, "meta": meta})
        except Exception as exc:
            log["skipped"].append({"case_name": name, "source_case_path": str(path), "reason": repr(exc)})
    (out_root / "dump_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_root / "dump_log.json")


if __name__ == "__main__":
    main()
