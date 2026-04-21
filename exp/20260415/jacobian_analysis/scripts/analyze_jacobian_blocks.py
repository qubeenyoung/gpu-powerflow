#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = REPO_ROOT / "exp" / "20260414" / "amgx" / "cupf_dumps"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "exp" / "20260415" / "jacobian_analysis" / "results"


@dataclass
class CsrComplex:
    rows: int
    cols: int
    row_ptr: list[int]
    col_idx: list[int]
    values: list[complex]


@dataclass
class CaseData:
    name: str
    ybus: CsrComplex
    voltage: list[complex]
    pv: list[int]
    pq: list[int]


@dataclass
class BlockNorm:
    case: str
    block: str
    rows: int
    cols: int
    nnz: int
    frobenius: float
    inf_norm: float
    one_norm: float
    max_abs: float


def is_payload(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("%") and not stripped.startswith("#")


def payload_lines(path: Path):
    with path.open() as handle:
        for line in handle:
            if is_payload(line):
                yield line


def load_complex_vector(path: Path) -> list[complex]:
    values: list[complex] = []
    for line in payload_lines(path):
        real, imag = line.split()[:2]
        values.append(complex(float(real), float(imag)))
    return values


def load_int_vector(path: Path) -> list[int]:
    return [int(line.split()[0]) for line in payload_lines(path)]


def load_matrix_market_complex_csr(path: Path) -> CsrComplex:
    with path.open() as handle:
        header = handle.readline().strip().lower()
        if "matrixmarket" not in header or "coordinate" not in header or "complex" not in header:
            raise ValueError(f"unsupported MatrixMarket format: {path}")
        symmetric = "symmetric" in header

        dims_line = ""
        for line in handle:
            if is_payload(line):
                dims_line = line
                break
        if not dims_line:
            raise ValueError(f"missing MatrixMarket dimensions: {path}")

        rows, cols, nnz = (int(token) for token in dims_line.split()[:3])
        entries: DefaultDict[tuple[int, int], complex] = defaultdict(complex)

        seen = 0
        for line in handle:
            if not is_payload(line):
                continue
            row_1, col_1, real, imag = line.split()[:4]
            row = int(row_1) - 1
            col = int(col_1) - 1
            value = complex(float(real), float(imag))
            entries[(row, col)] += value
            if symmetric and row != col:
                entries[(col, row)] += value
            seen += 1
            if seen == nnz:
                break

    row_entries: list[list[tuple[int, complex]]] = [[] for _ in range(rows)]
    for (row, col), value in entries.items():
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(f"matrix entry out of bounds: {path}")
        row_entries[row].append((col, value))

    row_ptr = [0]
    col_idx: list[int] = []
    values: list[complex] = []
    for row in row_entries:
        row.sort(key=lambda item: item[0])
        for col, value in row:
            col_idx.append(col)
            values.append(value)
        row_ptr.append(len(col_idx))

    return CsrComplex(rows=rows, cols=cols, row_ptr=row_ptr, col_idx=col_idx, values=values)


def load_case(case_dir: Path) -> CaseData:
    ybus = load_matrix_market_complex_csr(case_dir / "dump_Ybus.mtx")
    voltage = load_complex_vector(case_dir / "dump_V.txt")
    pv = load_int_vector(case_dir / "dump_pv.txt")
    pq = load_int_vector(case_dir / "dump_pq.txt")
    if ybus.rows != ybus.cols:
        raise ValueError(f"Ybus is not square: {case_dir}")
    if len(voltage) != ybus.rows:
        raise ValueError(f"voltage size does not match Ybus: {case_dir}")
    return CaseData(name=case_dir.name, ybus=ybus, voltage=voltage, pv=pv, pq=pq)


def ybus_matvec(ybus: CsrComplex, voltage: list[complex]) -> list[complex]:
    current = [0j for _ in range(ybus.rows)]
    for row in range(ybus.rows):
        acc = 0j
        for pos in range(ybus.row_ptr[row], ybus.row_ptr[row + 1]):
            acc += ybus.values[pos] * voltage[ybus.col_idx[pos]]
        current[row] = acc
    return current


def add_value(block: DefaultDict[tuple[int, int], float], row: int | None, col: int | None, value: float) -> None:
    if row is not None and col is not None:
        block[(row, col)] += value


def summarize_block(case_name: str, block_name: str, rows: int, cols: int, values: dict[tuple[int, int], float]) -> BlockNorm:
    row_sums = [0.0 for _ in range(rows)]
    col_sums = [0.0 for _ in range(cols)]
    sum_sq = 0.0
    max_abs = 0.0
    numeric_nnz = 0

    for (row, col), value in values.items():
        abs_value = abs(value)
        if abs_value == 0.0:
            continue
        numeric_nnz += 1
        sum_sq += value * value
        max_abs = max(max_abs, abs_value)
        row_sums[row] += abs_value
        col_sums[col] += abs_value

    return BlockNorm(
        case=case_name,
        block=block_name,
        rows=rows,
        cols=cols,
        nnz=numeric_nnz,
        frobenius=math.sqrt(sum_sq),
        inf_norm=max(row_sums, default=0.0),
        one_norm=max(col_sums, default=0.0),
        max_abs=max_abs,
    )


def analyze_case(case: CaseData) -> list[BlockNorm]:
    ybus = case.ybus
    voltage = case.voltage
    current = ybus_matvec(ybus, voltage)
    voltage_norm = [v / max(abs(v), 1e-8) for v in voltage]

    pvpq = case.pv + case.pq
    theta_pos = {bus: pos for pos, bus in enumerate(pvpq)}
    vm_pos = {bus: pos for pos, bus in enumerate(case.pq)}

    j11: DefaultDict[tuple[int, int], float] = defaultdict(float)
    j12: DefaultDict[tuple[int, int], float] = defaultdict(float)
    j21: DefaultDict[tuple[int, int], float] = defaultdict(float)
    j22: DefaultDict[tuple[int, int], float] = defaultdict(float)

    for row in range(ybus.rows):
        row_theta = theta_pos.get(row)
        row_vm = vm_pos.get(row)
        vi = voltage[row]

        for pos in range(ybus.row_ptr[row], ybus.row_ptr[row + 1]):
            col = ybus.col_idx[pos]
            y = ybus.values[pos]
            col_theta = theta_pos.get(col)
            col_vm = vm_pos.get(col)

            term_va = -1j * vi * (y * voltage[col]).conjugate()
            term_vm = vi * (y * voltage_norm[col]).conjugate()

            add_value(j11, row_theta, col_theta, term_va.real)
            add_value(j12, row_theta, col_vm, term_vm.real)
            add_value(j21, row_vm, col_theta, term_va.imag)
            add_value(j22, row_vm, col_vm, term_vm.imag)

    for bus in range(ybus.rows):
        row_theta = theta_pos.get(bus)
        row_vm = vm_pos.get(bus)
        term_va = 1j * voltage[bus] * current[bus].conjugate()
        term_vm = current[bus].conjugate() * voltage_norm[bus]

        add_value(j11, row_theta, row_theta, term_va.real)
        add_value(j12, row_theta, row_vm, term_vm.real)
        add_value(j21, row_vm, row_theta, term_va.imag)
        add_value(j22, row_vm, row_vm, term_vm.imag)

    n_pvpq = len(pvpq)
    n_pq = len(case.pq)
    return [
        summarize_block(case.name, "J11_dP_dtheta", n_pvpq, n_pvpq, j11),
        summarize_block(case.name, "J12_dP_dVm", n_pvpq, n_pq, j12),
        summarize_block(case.name, "J21_dQ_dtheta", n_pq, n_pvpq, j21),
        summarize_block(case.name, "J22_dQ_dVm", n_pq, n_pq, j22),
    ]


def find_case_dirs(dataset_root: Path, selected_cases: list[str]) -> list[Path]:
    if selected_cases:
        return [dataset_root / case for case in selected_cases]
    return sorted(
        path for path in dataset_root.iterdir()
        if path.is_dir() and (path / "dump_Ybus.mtx").exists()
    )


def write_block_norms(path: Path, rows: list[BlockNorm]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "block", "rows", "cols", "nnz", "frobenius", "inf_norm", "one_norm", "max_abs"])
        for row in rows:
            writer.writerow([
                row.case,
                row.block,
                row.rows,
                row.cols,
                row.nnz,
                f"{row.frobenius:.17e}",
                f"{row.inf_norm:.17e}",
                f"{row.one_norm:.17e}",
                f"{row.max_abs:.17e}",
            ])


def write_summary(path: Path, rows: list[BlockNorm]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_case: dict[str, dict[str, BlockNorm]] = {}
    for row in rows:
        by_case.setdefault(row.case, {})[row.block] = row

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case",
            "n_pvpq",
            "n_pq",
            "j11_fro",
            "j12_fro",
            "j21_fro",
            "j22_fro",
            "diag_fro",
            "offdiag_fro",
            "offdiag_to_diag_fro",
            "j12_to_j11_fro",
            "j21_to_j22_fro",
        ])
        for case_name in sorted(by_case):
            blocks = by_case[case_name]
            j11 = blocks["J11_dP_dtheta"]
            j12 = blocks["J12_dP_dVm"]
            j21 = blocks["J21_dQ_dtheta"]
            j22 = blocks["J22_dQ_dVm"]
            diag_fro = math.hypot(j11.frobenius, j22.frobenius)
            offdiag_fro = math.hypot(j12.frobenius, j21.frobenius)
            writer.writerow([
                case_name,
                j11.rows,
                j22.rows,
                f"{j11.frobenius:.17e}",
                f"{j12.frobenius:.17e}",
                f"{j21.frobenius:.17e}",
                f"{j22.frobenius:.17e}",
                f"{diag_fro:.17e}",
                f"{offdiag_fro:.17e}",
                f"{offdiag_fro / diag_fro if diag_fro else 0.0:.17e}",
                f"{j12.frobenius / j11.frobenius if j11.frobenius else 0.0:.17e}",
                f"{j21.frobenius / j22.frobenius if j22.frobenius else 0.0:.17e}",
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure cuPF Jacobian block norms for dump cases.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", default=[], help="Case name under dataset root. Repeatable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dirs = find_case_dirs(args.dataset_root, args.case)
    if not case_dirs:
        raise SystemExit(f"no cases found under {args.dataset_root}")

    all_norms: list[BlockNorm] = []
    for case_dir in case_dirs:
        print(f"analyzing {case_dir.name}")
        case = load_case(case_dir)
        all_norms.extend(analyze_case(case))

    write_block_norms(args.output_dir / "block_norms.csv", all_norms)
    write_summary(args.output_dir / "block_norm_summary.csv", all_norms)
    print(f"wrote {args.output_dir / 'block_norms.csv'}")
    print(f"wrote {args.output_dir / 'block_norm_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
