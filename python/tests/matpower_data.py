from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import re
import tempfile
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

import matpowercaseframes.reader as mpc_reader
import matpowercaseframes.utils as mpc_utils
from matpowercaseframes import CaseFrames
from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.idx_brch import F_BUS, T_BUS, branch_cols
from pandapower.pypower.idx_bus import BUS_I, VA, VM, bus_cols
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, VG, gen_cols
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.makeYbus import makeYbus

DEFAULT_DATASET_ROOT = Path("/datasets/matpower")


def _robust_number_or_string(value: str) -> int | float | str:
    try:
        float_value = float(value)
    except ValueError:
        expression = value.replace("^", "**")
        if not re.fullmatch(r"[0-9eE+\-*/(). _sqrtnanifINFNaN]+", expression):
            return value
        try:
            float_value = float(
                eval(
                    expression,
                    {"__builtins__": {}},
                    {
                        "sqrt": math.sqrt,
                        "Inf": math.inf,
                        "inf": math.inf,
                        "NaN": math.nan,
                        "nan": math.nan,
                    },
                )
            )
        except Exception:
            return value
    if not np.isfinite(float_value):
        return float_value
    int_value = int(float_value)
    return int_value if int_value == float_value else float_value


mpc_reader.int_else_float_except_string = _robust_number_or_string
mpc_utils.int_else_float_except_string = _robust_number_or_string


@dataclass
class PreprocessedCase:
    case_name: str
    source_path: Path
    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    ref: np.ndarray
    pv: np.ndarray
    pq: np.ndarray
    ybus: sp.csr_matrix
    sbus: np.ndarray
    v0: np.ndarray


@dataclass
class ReferenceResult:
    converged: bool
    iterations: int
    final_mismatch: float
    voltage: np.ndarray


def dataset_raw_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    raw = root / "raw"
    return raw if raw.is_dir() else root


def list_case_paths(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> list[Path]:
    root = dataset_raw_root(dataset_root)
    return sorted(root.glob("case*.m"))


def resolve_case_paths(dataset_root: str | Path, cases: Iterable[str] | None) -> list[Path]:
    root = dataset_raw_root(dataset_root)
    if not cases:
        return list_case_paths(dataset_root)

    out: list[Path] = []
    for case in cases:
        raw = Path(case)
        candidates: list[Path] = []
        if raw.exists():
            candidates.append(raw)
        name = raw.name if raw.suffix == ".m" else f"{raw.name}.m"
        candidates.extend([root / name, root / raw.name, Path(dataset_root) / name])
        for candidate in candidates:
            if candidate.exists():
                out.append(candidate)
                break
        else:
            raise FileNotFoundError(f"MATPOWER case not found: {case} under {dataset_root}")
    return out


def _normalize_mpc(ppc: dict) -> dict:
    normalized = {}
    for key, value in ppc.items():
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)
        normalized[key] = value
    return normalized


def _caseframes_to_mpc(path: Path) -> dict:
    try:
        case_frames = CaseFrames(str(path), update_index=False)
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")
        with tempfile.NamedTemporaryFile("w", suffix=".m", encoding="utf-8", delete=False) as tmp_file:
            tmp_file.write(text)
            tmp_path = Path(tmp_file.name)
        try:
            case_frames = CaseFrames(str(tmp_path), update_index=False)
        finally:
            tmp_path.unlink(missing_ok=True)
    return _normalize_mpc(case_frames.to_mpc())


def _apply_matpower_postprocessing(ppc: dict, input_path: Path) -> dict:
    text = input_path.read_text(encoding="cp1252", errors="replace")
    ppc = dict(ppc)

    for key in ("bus", "branch", "gen", "gencost"):
        if key in ppc and ppc[key] is not None:
            try:
                ppc[key] = np.asarray(ppc[key], dtype=float)
            except (TypeError, ValueError):
                pass

    if "mpc.branch(:, [BR_R BR_X])" in text:
        bus = np.asarray(ppc["bus"], dtype=float)
        branch = np.asarray(ppc["branch"], dtype=float)
        vbase = bus[0, 9] * 1e3
        sbase = float(ppc["baseMVA"]) * 1e6
        branch[:, [2, 3]] = branch[:, [2, 3]] / (vbase**2 / sbase)
        ppc["branch"] = branch

    if "mpc.bus(:, [PD, QD])" in text and "/ 1e3" in text:
        bus = np.asarray(ppc["bus"], dtype=float)
        bus[:, [2, 3]] = bus[:, [2, 3]] / 1e3
        ppc["bus"] = bus

    pf_match = re.search(r"pf\s*=\s*([0-9.]+)\s*;", text)
    if pf_match and "sin(acos(pf))" in text:
        pf = float(pf_match.group(1))
        bus = np.asarray(ppc["bus"], dtype=float)
        bus[:, 3] = bus[:, 2] * math.sin(math.acos(pf))
        bus[:, 2] = bus[:, 2] * pf
        ppc["bus"] = bus

    return ppc


def _as_2d_float(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return arr


def _pad_cols(arr: np.ndarray, cols: int) -> np.ndarray:
    if arr.shape[1] >= cols:
        return arr.astype(float, copy=True)
    out = np.zeros((arr.shape[0], cols), dtype=float)
    out[:, : arr.shape[1]] = arr
    return out


def _internalize_bus_numbers(bus: np.ndarray, gen: np.ndarray, branch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bus = bus.copy()
    gen = gen.copy()
    branch = branch.copy()
    bus_numbers = [int(round(x)) for x in bus[:, BUS_I]]
    mapping = {bus_number: idx for idx, bus_number in enumerate(bus_numbers)}
    if len(mapping) != len(bus_numbers):
        raise ValueError("duplicate MATPOWER BUS_I values")

    bus[:, BUS_I] = np.arange(bus.shape[0], dtype=float)
    for row in range(gen.shape[0]):
        ext = int(round(gen[row, GEN_BUS]))
        if ext not in mapping:
            raise ValueError(f"generator references unknown bus {ext}")
        gen[row, GEN_BUS] = mapping[ext]
    for col in (F_BUS, T_BUS):
        for row in range(branch.shape[0]):
            ext = int(round(branch[row, col]))
            if ext not in mapping:
                raise ValueError(f"branch references unknown bus {ext}")
            branch[row, col] = mapping[ext]
    return bus, gen, branch


def load_case(path: str | Path) -> PreprocessedCase:
    source_path = Path(path)
    ppc = _apply_matpower_postprocessing(_caseframes_to_mpc(source_path), source_path)

    bus = _pad_cols(_as_2d_float(ppc["bus"], "bus"), bus_cols)
    gen = _pad_cols(_as_2d_float(ppc["gen"], "gen"), gen_cols)
    branch = _pad_cols(_as_2d_float(ppc["branch"], "branch"), branch_cols)
    bus, gen, branch = _internalize_bus_numbers(bus, gen, branch)

    base_mva = float(np.asarray(ppc["baseMVA"]).squeeze())
    ref, pv, pq = bustypes(bus, gen)

    v0 = bus[:, VM] * np.exp(1j * np.pi / 180.0 * bus[:, VA])
    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    gbus = gen[on, GEN_BUS].astype(int)
    vcb = np.ones(v0.shape)
    vcb[pq] = 0
    k = np.flatnonzero(vcb[gbus]) if gbus.size else np.array([], dtype=int)
    if k.size:
        v0[gbus[k]] = gen[on[k], VG] / np.abs(v0[gbus[k]]) * v0[gbus[k]]

    ybus, _, _ = makeYbus(base_mva, bus, branch)
    ybus = ybus.tocsr()
    ybus.sum_duplicates()
    ybus.sort_indices()
    ybus.data = ybus.data.astype(np.complex128, copy=False)
    ybus.indices = ybus.indices.astype(np.int32, copy=False)
    ybus.indptr = ybus.indptr.astype(np.int32, copy=False)

    sbus = np.asarray(makeSbus(base_mva, bus, gen), dtype=np.complex128).reshape(-1)
    return PreprocessedCase(
        case_name=source_path.stem,
        source_path=source_path,
        base_mva=base_mva,
        bus=bus,
        gen=gen,
        branch=branch,
        ref=np.asarray(ref, dtype=np.int32),
        pv=np.asarray(pv, dtype=np.int32),
        pq=np.asarray(pq, dtype=np.int32),
        ybus=ybus,
        sbus=sbus,
        v0=np.asarray(v0, dtype=np.complex128).reshape(-1),
    )


def mismatch_vector(ybus: sp.csr_matrix, sbus: np.ndarray, voltage: np.ndarray, pv: np.ndarray, pq: np.ndarray) -> np.ndarray:
    mis = voltage * np.conjugate(ybus @ voltage) - sbus
    return np.r_[mis[pv].real, mis[pq].real, mis[pq].imag]


def mismatch_norm(ybus: sp.csr_matrix, sbus: np.ndarray, voltage: np.ndarray, pv: np.ndarray, pq: np.ndarray) -> float:
    f = mismatch_vector(ybus, sbus, voltage, pv, pq)
    if f.size == 0:
        return 0.0
    return float(np.linalg.norm(f, np.inf))


def solve_reference(case: PreprocessedCase, tolerance: float = 1e-10, max_iter: int = 50) -> ReferenceResult:
    voltage = case.v0.astype(np.complex128, copy=True)
    va = np.angle(voltage)
    vm = np.abs(voltage)
    pvpq = np.r_[case.pv, case.pq]
    npv = len(case.pv)
    npq = len(case.pq)
    converged = False
    final_norm = mismatch_norm(case.ybus, case.sbus, voltage, case.pv, case.pq)
    if final_norm < tolerance:
        return ReferenceResult(True, 0, final_norm, voltage)

    for iteration in range(1, max_iter + 1):
        dS_dVm, dS_dVa = dSbus_dV(case.ybus, voltage)
        j11 = dS_dVa[np.ix_(pvpq, pvpq)].real
        j12 = dS_dVm[np.ix_(pvpq, case.pq)].real
        j21 = dS_dVa[np.ix_(case.pq, pvpq)].imag
        j22 = dS_dVm[np.ix_(case.pq, case.pq)].imag
        jacobian = vstack([hstack([j11, j12]), hstack([j21, j22])], format="csr")
        f = mismatch_vector(case.ybus, case.sbus, voltage, case.pv, case.pq)
        dx = -np.asarray(spsolve(jacobian, f)).reshape(-1)

        if npv:
            va[case.pv] += dx[0:npv]
        if npq:
            va[case.pq] += dx[npv : npv + npq]
            vm[case.pq] += dx[npv + npq : npv + 2 * npq]
        voltage = vm * np.exp(1j * va)
        vm = np.abs(voltage)
        va = np.angle(voltage)
        final_norm = mismatch_norm(case.ybus, case.sbus, voltage, case.pv, case.pq)
        if final_norm < tolerance:
            converged = True
            return ReferenceResult(converged, iteration, final_norm, voltage)

    return ReferenceResult(False, max_iter, final_norm, voltage)


def voltage_error(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.complex128).reshape(-1)
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    if actual.shape != reference.shape:
        return {
            "max_abs_v_error": math.nan,
            "rms_abs_v_error": math.nan,
            "max_abs_vm_error": math.nan,
            "max_abs_va_error": math.nan,
        }
    delta = actual - reference
    va_delta = np.angle(actual) - np.angle(reference)
    va_delta = (va_delta + np.pi) % (2 * np.pi) - np.pi
    return {
        "max_abs_v_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rms_abs_v_error": float(np.sqrt(np.mean(np.abs(delta) ** 2))) if delta.size else 0.0,
        "max_abs_vm_error": float(np.max(np.abs(np.abs(actual) - np.abs(reference)))) if delta.size else 0.0,
        "max_abs_va_error": float(np.max(np.abs(va_delta))) if delta.size else 0.0,
    }


def _write_complex_pairs(path: Path, values: np.ndarray) -> None:
    matrix = np.column_stack((values.real, values.imag))
    np.savetxt(path, matrix, fmt="%.18e")


def _write_int_values(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values.astype(np.int32, copy=False), fmt="%d")


def save_dump_case(case: PreprocessedCase, output_root: str | Path, reference: ReferenceResult | None = None) -> Path:
    case_dir = Path(output_root) / case.case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    mmwrite(case_dir / "dump_Ybus.mtx", case.ybus)
    _write_complex_pairs(case_dir / "dump_Sbus.txt", case.sbus)
    _write_complex_pairs(case_dir / "dump_V.txt", case.v0)
    _write_int_values(case_dir / "dump_pv.txt", case.pv)
    _write_int_values(case_dir / "dump_pq.txt", case.pq)
    if reference is not None:
        _write_complex_pairs(case_dir / "dump_Vref.txt", reference.voltage)
    metadata = {
        "case_name": case.case_name,
        "source_path": str(case.source_path),
        "base_mva": case.base_mva,
        "n_bus": int(case.ybus.shape[0]),
        "ybus_nnz": int(case.ybus.nnz),
        "n_ref": int(case.ref.size),
        "n_pv": int(case.pv.size),
        "n_pq": int(case.pq.size),
    }
    if reference is not None:
        metadata.update(
            {
                "reference_converged": reference.converged,
                "reference_iterations": reference.iterations,
                "reference_final_mismatch": reference.final_mismatch,
            }
        )
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return case_dir


def write_manifest_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
