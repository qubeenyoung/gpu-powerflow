#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_JF_ROOT = EXP_ROOT / "raw" / "cupf_jf_dumps"
DEFAULT_CASE_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_OUTPUT = EXP_ROOT / "results" / "jf_component_scale_j0_j1_j2_linear5.csv"
DEFAULT_SUMMARY = EXP_ROOT / "results" / "jf_component_scale_j0_j1_j2_linear5.md"
DEFAULT_CASES = [
    "case1197",
    "case2736sp",
    "case3375wp",
    "case6468rte",
    "case_ACTIVSg10k",
]


@dataclass
class CsrMatrix:
    rows: int
    cols: int
    row_ptr: list[int]
    col_idx: list[int]
    values: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze J/F component scales for dumped NR linear systems.")
    parser.add_argument("--jf-root", type=Path, default=DEFAULT_JF_ROOT)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--iterations", default="0,1,2")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def split_int_list(value: str) -> list[int]:
    return [int(item) for item in split_list(value)]


def read_int_vector(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_csr_dump(path: Path) -> CsrMatrix:
    tokens = path.read_text(encoding="utf-8").split()
    pos = 0

    def expect(token: str) -> None:
        nonlocal pos
        if pos >= len(tokens) or tokens[pos] != token:
            got = tokens[pos] if pos < len(tokens) else "<eof>"
            raise ValueError(f"{path}: expected {token}, got {got}")
        pos += 1

    expect("type")
    matrix_type = tokens[pos]
    pos += 1
    if matrix_type != "csr_matrix":
        raise ValueError(f"{path}: expected csr_matrix, got {matrix_type}")
    expect("rows")
    rows = int(tokens[pos])
    pos += 1
    expect("cols")
    cols = int(tokens[pos])
    pos += 1
    expect("nnz")
    nnz = int(tokens[pos])
    pos += 1
    expect("row_ptr")
    row_ptr = [int(tokens[pos + i]) for i in range(rows + 1)]
    pos += rows + 1
    expect("col_idx")
    col_idx = [int(tokens[pos + i]) for i in range(nnz)]
    pos += nnz
    expect("values")
    values = [float(tokens[pos + i]) for i in range(nnz)]
    if row_ptr[0] != 0 or row_ptr[-1] != nnz:
        raise ValueError(f"{path}: malformed row_ptr")
    return CsrMatrix(rows=rows, cols=cols, row_ptr=row_ptr, col_idx=col_idx, values=values)


def load_vector_dump(path: Path) -> list[float]:
    tokens = path.read_text(encoding="utf-8").split()
    pos = 0
    if tokens[pos] != "type":
        raise ValueError(f"{path}: missing type")
    pos += 1
    vector_type = tokens[pos]
    pos += 1
    if vector_type != "vector":
        raise ValueError(f"{path}: expected vector, got {vector_type}")
    if tokens[pos] != "size":
        raise ValueError(f"{path}: missing size")
    pos += 1
    size = int(tokens[pos])
    pos += 1
    if tokens[pos] != "values":
        raise ValueError(f"{path}: missing values")
    pos += 1
    values = [0.0] * size
    # The dump format stores "index value" pairs.
    for _ in range(size):
        index = int(tokens[pos])
        value = float(tokens[pos + 1])
        pos += 2
        values[index] = value
    return values


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return sorted_values[lo]
    weight = position - lo
    return (1.0 - weight) * sorted_values[lo] + weight * sorted_values[hi]


def stats_for_values(values: list[float]) -> dict[str, float | int]:
    finite = [value for value in values if math.isfinite(value)]
    abs_values = sorted(abs(value) for value in finite)
    count = len(finite)
    if count == 0:
        return {
            "count": 0,
            "mean_abs": 0.0,
            "median_abs": 0.0,
            "p95_abs": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "l2_norm": 0.0,
        }
    sum_abs = sum(abs_values)
    sum_sq = sum(value * value for value in finite)
    l2_norm = math.sqrt(sum_sq)
    return {
        "count": count,
        "mean_abs": sum_abs / count,
        "median_abs": percentile(abs_values, 0.50),
        "p95_abs": percentile(abs_values, 0.95),
        "max_abs": abs_values[-1],
        "rms": l2_norm / math.sqrt(count),
        "l2_norm": l2_norm,
    }


def quadrant_name(row: int, col: int, n_pvpq: int) -> str:
    row_p = row < n_pvpq
    col_theta = col < n_pvpq
    if row_p and col_theta:
        return "J11_P_theta"
    if row_p and not col_theta:
        return "J12_P_Vm"
    if not row_p and col_theta:
        return "J21_Q_theta"
    return "J22_Q_Vm"


def analyze_matrix(case: str, iteration: int, matrix: CsrMatrix, n_pvpq: int) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = {
        "J11_P_theta": [],
        "J12_P_Vm": [],
        "J21_Q_theta": [],
        "J22_Q_Vm": [],
    }
    for row in range(matrix.rows):
        for pos in range(matrix.row_ptr[row], matrix.row_ptr[row + 1]):
            col = matrix.col_idx[pos]
            grouped[quadrant_name(row, col, n_pvpq)].append(matrix.values[pos])

    rows: list[dict[str, object]] = []
    for component, values in grouped.items():
        item = {
            "case": case,
            "iteration": iteration,
            "system": f"J{iteration}",
            "kind": "jacobian",
            "component": component,
        }
        item.update(stats_for_values(values))
        rows.append(item)
    return rows


def analyze_vector(case: str, iteration: int, vector: list[float], n_pvpq: int) -> list[dict[str, object]]:
    components = {
        "F_P": vector[:n_pvpq],
        "F_Q": vector[n_pvpq:],
    }
    rows: list[dict[str, object]] = []
    for component, values in components.items():
        item = {
            "case": case,
            "iteration": iteration,
            "system": f"F{iteration}",
            "kind": "rhs",
            "component": component,
        }
        item.update(stats_for_values(values))
        rows.append(item)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "iteration",
        "system",
        "kind",
        "component",
        "count",
        "mean_abs",
        "median_abs",
        "p95_abs",
        "max_abs",
        "rms",
        "l2_norm",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by = {(row["case"], int(row["iteration"]), row["component"]): row for row in rows}
    cases = sorted({str(row["case"]) for row in rows})
    iterations = sorted({int(row["iteration"]) for row in rows})
    lines: list[str] = []
    lines.append("# J/F Component Scale Summary")
    lines.append("")
    lines.append("- Scale metric shown below is RMS of values in each component.")
    lines.append("- Jacobian components: `J11=P-theta`, `J12=P-Vm`, `J21=Q-theta`, `J22=Q-Vm`.")
    lines.append("- RHS components: `F_P` and `F_Q` split by the same NR ordering.")
    lines.append("")
    lines.append("## Jacobian RMS by Quadrant")
    lines.append("")
    lines.append("| case | iter | J11 P-theta | J12 P-Vm | J21 Q-theta | J22 Q-Vm | largest |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for case in cases:
        for iteration in iterations:
            comps = ["J11_P_theta", "J12_P_Vm", "J21_Q_theta", "J22_Q_Vm"]
            values = {component: float(by[(case, iteration, component)]["rms"]) for component in comps}
            largest = max(values.items(), key=lambda item: item[1])
            lines.append(
                f"| {case} | J{iteration} | "
                f"{values['J11_P_theta']:.3e} | {values['J12_P_Vm']:.3e} | "
                f"{values['J21_Q_theta']:.3e} | {values['J22_Q_Vm']:.3e} | "
                f"{largest[0]} |"
            )
    lines.append("")
    lines.append("## RHS RMS by P/Q")
    lines.append("")
    lines.append("| case | iter | F_P | F_Q | F_Q / F_P |")
    lines.append("|---|---:|---:|---:|---:|")
    for case in cases:
        for iteration in iterations:
            f_p = float(by[(case, iteration, "F_P")]["rms"])
            f_q = float(by[(case, iteration, "F_Q")]["rms"])
            ratio = f_q / f_p if f_p > 0.0 else 0.0
            lines.append(f"| {case} | F{iteration} | {f_p:.3e} | {f_q:.3e} | {ratio:.3f} |")
    lines.append("")
    lines.append("## Aggregate RMS Across Cases")
    lines.append("")
    lines.append("| iter | J11 | J12 | J21 | J22 | F_P | F_Q |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for iteration in iterations:
        def avg(component: str) -> float:
            return mean(float(by[(case, iteration, component)]["rms"]) for case in cases)

        lines.append(
            f"| {iteration} | {avg('J11_P_theta'):.3e} | {avg('J12_P_Vm'):.3e} | "
            f"{avg('J21_Q_theta'):.3e} | {avg('J22_Q_Vm'):.3e} | "
            f"{avg('F_P'):.3e} | {avg('F_Q'):.3e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cases = split_list(args.cases)
    iterations = split_int_list(args.iterations)
    rows: list[dict[str, object]] = []
    for case in cases:
        pv = read_int_vector(args.case_root / case / "dump_pv.txt")
        pq = read_int_vector(args.case_root / case / "dump_pq.txt")
        n_pvpq = len(pv) + len(pq)
        repeat_dir = args.jf_root / case / "repeat_00"
        for iteration in iterations:
            matrix_path = repeat_dir / f"jacobian_iter{iteration}.txt"
            vector_path = repeat_dir / f"residual_before_update_iter{iteration}.txt"
            matrix = load_csr_dump(matrix_path)
            vector = load_vector_dump(vector_path)
            if matrix.rows != n_pvpq + len(pq) or len(vector) != matrix.rows:
                raise ValueError(f"{case} iter {iteration}: dimension mismatch")
            rows.extend(analyze_matrix(case, iteration, matrix, n_pvpq))
            rows.extend(analyze_vector(case, iteration, vector, n_pvpq))
            print(f"[OK] case={case} iter={iteration} n={matrix.rows} nnz={len(matrix.values)}")
    write_csv(rows, args.output)
    write_summary(rows, args.summary)
    print(f"[DONE] output={args.output}")
    print(f"[DONE] summary={args.summary}")


if __name__ == "__main__":
    main()
