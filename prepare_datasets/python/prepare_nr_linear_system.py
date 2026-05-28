from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy import angle, conj, exp, r_
from pypower.dSbus_dV import dSbus_dV
from scipy.io import mmwrite
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve

from common import PreprocessedCase, case_metadata, preprocess_case, write_json


DEFAULT_MAT_ROOT = Path("/datasets/power_system/matpower_mat")
DEFAULT_OUTPUT_ROOT = Path("/datasets/power_system/nr_linear_systems")
TARGET_CASES = (
    "case30",
    "case118",
    "case1197",
    "case_ACTIVSg2000",
    "case3012wp",
    "case6468rte",
    "case8387pegase",
    "case_ACTIVSg25k",
    "case_SyntheticUSA",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump Newton-Raphson Jacobian linear systems from MATPOWER .mat cases."
    )
    parser.add_argument("--mat-root", type=Path, default=DEFAULT_MAT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases", nargs="*", default=list(TARGET_CASES))
    parser.add_argument("--dump-iteration", type=int, default=2)
    return parser.parse_args()


def newton_jacobian_and_mismatch(
    case_data: PreprocessedCase,
    dump_iteration: int = 2,
) -> tuple[object, np.ndarray]:
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
    F = r_[mis[pv].real, mis[pq].real, mis[pq].imag]

    for iteration in range(1, dump_iteration + 1):
        dS_dVm, dS_dVa = dSbus_dV(case_data.Ybus, V)

        J11 = dS_dVa[np.array([pvpq]).T, pvpq].real
        J12 = dS_dVm[np.array([pvpq]).T, pq].real
        J21 = dS_dVa[np.array([pq]).T, pvpq].imag
        J22 = dS_dVm[np.array([pq]).T, pq].imag

        J = vstack(
            [
                hstack([J11, J12]),
                hstack([J21, J22]),
            ],
            format="csr",
        )
        J.sort_indices()

        if iteration == dump_iteration:
            return J, F.astype(np.float64, copy=False)

        dx = -spsolve(J, F)

        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]

        V = Vm * exp(1j * Va)
        Vm = np.abs(V)
        Va = angle(V)

        mis = V * conj(case_data.Ybus * V) - case_data.Sbus
        F = r_[mis[pv].real, mis[pq].real, mis[pq].imag]

    raise RuntimeError("unreachable")


def save_linear_system(
    case_name: str,
    mat_root: Path,
    output_root: Path,
    dump_iteration: int,
) -> Path:
    case_data = preprocess_case(case_name, mat_root=mat_root)
    J, mismatch = newton_jacobian_and_mismatch(case_data, dump_iteration=dump_iteration)

    case_dir = output_root / case_data.case_stem
    case_dir.mkdir(parents=True, exist_ok=True)

    mmwrite(case_dir / "J.mtx", J)
    mmwrite(case_dir / "F.mtx", mismatch.reshape((-1, 1)))

    metadata = case_metadata(case_data)
    metadata.update(
        {
            "dump_iteration": dump_iteration,
            "jacobian_shape": [int(J.shape[0]), int(J.shape[1])],
            "jacobian_nnz": int(J.nnz),
            "mismatch_inf_norm": float(np.linalg.norm(mismatch, np.inf)),
            "linear_system": "J * x_true = rhs",
            "standard_rhs": "rhs = F",
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

    success = 0
    failures = 0
    for case_name in args.cases:
        try:
            output_path = save_linear_system(
                case_name=case_name,
                mat_root=args.mat_root,
                output_root=args.output_root,
                dump_iteration=args.dump_iteration,
            )
            success += 1
            print(f"[OK] {case_name} -> {output_path}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {case_name}: {exc}")

    print(f"Processed={success + failures} Success={success} Failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
