"""Build Newton-Raphson Jacobian linear systems (`J.mtx` / `F.mtx`) from MATPOWER
`.m` cases, for consumption by the custom_linear_solver. Reuses the pandapower
case loader in :mod:`benchmark.common.matpower_data` (no standalone `pypower`).

    python3 -m prepare_datasets.convert_linear_system --dataset-root /datasets/matpower --cases case9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy import angle, conj, exp, r_
from pandapower.pypower.dSbus_dV import dSbus_dV
from scipy.io import mmwrite
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

from benchmark.common.matpower_data import DEFAULT_DATASET_ROOT, PreprocessedCase, load_case, resolve_case_paths

DEFAULT_OUTPUT_ROOT = Path("/datasets/matpower_linear_systems")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Newton-Raphson Jacobian linear systems from MATPOWER .m cases."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases", nargs="*", help="Case names or .m paths. Defaults to all case*.m cases.")
    parser.add_argument("--dump-iteration", type=int, default=2)
    return parser.parse_args()


def newton_jacobian_and_mismatch(
    case: PreprocessedCase,
    dump_iteration: int = 2,
):
    if dump_iteration < 1:
        raise ValueError("dump_iteration must be >= 1")

    pv = case.pv
    pq = case.pq
    pvpq = r_[pv, pq]
    npv = len(pv)
    npq = len(pq)

    j1 = 0
    j2 = npv
    j3 = j2
    j4 = j2 + npq
    j5 = j4
    j6 = j4 + npq

    V = case.v0.copy()
    Va = angle(V)
    Vm = np.abs(V)

    mis = V * conj(case.ybus * V) - case.sbus
    mismatch = r_[mis[pv].real, mis[pq].real, mis[pq].imag]

    for iteration in range(1, dump_iteration + 1):
        dS_dVm, dS_dVa = dSbus_dV(case.ybus, V)

        J11 = dS_dVa[np.array([pvpq]).T, pvpq].real
        J12 = dS_dVm[np.array([pvpq]).T, pq].real
        J21 = dS_dVa[np.array([pq]).T, pvpq].imag
        J22 = dS_dVm[np.array([pq]).T, pq].imag

        jacobian = vstack(
            [
                hstack([J11, J12]),
                hstack([J21, J22]),
            ],
            format="csr",
        )
        jacobian.sort_indices()

        if iteration == dump_iteration:
            return jacobian, mismatch.astype(np.float64, copy=False)

        dx = -spsolve(jacobian, mismatch)

        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]

        V = Vm * exp(1j * Va)
        Vm = np.abs(V)
        Va = angle(V)

        mis = V * conj(case.ybus * V) - case.sbus
        mismatch = r_[mis[pv].real, mis[pq].real, mis[pq].imag]

    raise RuntimeError("unreachable")


def save_linear_system(
    case_path: Path,
    output_root: Path,
    dump_iteration: int,
) -> Path:
    case = load_case(case_path)
    jacobian, mismatch = newton_jacobian_and_mismatch(case, dump_iteration=dump_iteration)

    case_dir = output_root / case.case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    mmwrite(case_dir / "J.mtx", jacobian)
    mmwrite(case_dir / "F.mtx", mismatch.reshape((-1, 1)))

    metadata = {
        "case_name": case.case_name,
        "source_path": str(case.source_path),
        "base_mva": case.base_mva,
        "n_bus": int(case.ybus.shape[0]),
        "ybus_nnz": int(case.ybus.nnz),
        "n_ref": int(case.ref.size),
        "n_pv": int(case.pv.size),
        "n_pq": int(case.pq.size),
        "dump_iteration": dump_iteration,
        "jacobian_shape": [int(jacobian.shape[0]), int(jacobian.shape[1])],
        "jacobian_nnz": int(jacobian.nnz),
        "mismatch_inf_norm": float(np.linalg.norm(mismatch, np.inf)),
        "matrix_market_files": {"jacobian": "J.mtx", "mismatch": "F.mtx"},
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return case_dir


def main() -> None:
    args = parse_args()
    case_paths = resolve_case_paths(args.dataset_root, args.cases)

    success = 0
    failures = 0
    for case_path in case_paths:
        try:
            output_dir = save_linear_system(
                case_path=case_path,
                output_root=args.output_root,
                dump_iteration=args.dump_iteration,
            )
            success += 1
            print(f"[OK] {case_path.stem} -> {output_dir}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {case_path}: {exc}")

    print(f"Processed={success + failures} Success={success} Failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
