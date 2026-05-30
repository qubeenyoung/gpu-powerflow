from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy import angle, conj, exp, r_
from pypower.dSbus_dV import dSbus_dV
from scipy.io import mmwrite
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

from .common import DATASETS_ROOT, PreprocessedCase, all_mat_case_names, case_metadata, preprocess_case, write_json


DEFAULT_OUTPUT_ROOT = DATASETS_ROOT / "matpower_linear_systems"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Newton-Raphson Jacobian linear systems from MATPOWER .mat cases."
    )
    parser.add_argument("--input-root", type=Path, default=DATASETS_ROOT / "matpower_mat")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases", nargs="*", help="Case names or .mat stems. Defaults to all .mat cases.")
    parser.add_argument("--dump-iteration", type=int, default=2)
    return parser.parse_args()


def newton_jacobian_and_mismatch(
    case_data: PreprocessedCase,
    dump_iteration: int = 2,
):
    if dump_iteration < 1:
        raise ValueError("dump_iteration must be >= 1")

    pv = case_data.pv
    pq = case_data.pq
    pvpq = r_[pv, pq]
    npv = len(pv)
    npq = len(pq)

    j1 = 0
    j2 = npv
    j3 = j2
    j4 = j2 + npq
    j5 = j4
    j6 = j4 + npq

    V = case_data.V0.copy()
    Va = angle(V)
    Vm = np.abs(V)

    mis = V * conj(case_data.Ybus * V) - case_data.Sbus
    mismatch = r_[mis[pv].real, mis[pq].real, mis[pq].imag]

    for iteration in range(1, dump_iteration + 1):
        dS_dVm, dS_dVa = dSbus_dV(case_data.Ybus, V)

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

        mis = V * conj(case_data.Ybus * V) - case_data.Sbus
        mismatch = r_[mis[pv].real, mis[pq].real, mis[pq].imag]

    raise RuntimeError("unreachable")


def save_linear_system(
    case_name: str,
    input_root: Path,
    output_root: Path,
    dump_iteration: int,
) -> Path:
    case_data = preprocess_case(case_name, mat_root=input_root)
    jacobian, mismatch = newton_jacobian_and_mismatch(case_data, dump_iteration=dump_iteration)

    case_dir = output_root / case_data.case_stem
    case_dir.mkdir(parents=True, exist_ok=True)

    mmwrite(case_dir / "J.mtx", jacobian)
    mmwrite(case_dir / "F.mtx", mismatch.reshape((-1, 1)))

    metadata = case_metadata(case_data)
    metadata.update(
        {
            "dump_iteration": dump_iteration,
            "jacobian_shape": [int(jacobian.shape[0]), int(jacobian.shape[1])],
            "jacobian_nnz": int(jacobian.nnz),
            "mismatch_inf_norm": float(np.linalg.norm(mismatch, np.inf)),
            "matrix_market_files": {
                "jacobian": "J.mtx",
                "mismatch": "F.mtx",
            },
        }
    )
    write_json(case_dir / "metadata.json", metadata)
    return case_dir


def main() -> None:
    args = parse_args()
    case_names = args.cases if args.cases else all_mat_case_names(args.input_root)

    success = 0
    failures = 0
    for case_name in case_names:
        try:
            output_dir = save_linear_system(
                case_name=case_name,
                input_root=args.input_root,
                output_root=args.output_root,
                dump_iteration=args.dump_iteration,
            )
            success += 1
            print(f"[OK] {case_name} -> {output_dir}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {case_name}: {exc}")

    print(f"Processed={success + failures} Success={success} Failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
