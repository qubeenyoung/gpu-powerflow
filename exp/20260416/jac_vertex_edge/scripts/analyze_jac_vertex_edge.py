#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = REPO_ROOT / "exp" / "20260414" / "amgx" / "cupf_dumps"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "exp" / "20260416" / "jac_vertex_edge" / "results"
BLOCKS = ("J11", "J21", "J12", "J22")


@dataclass
class CsrPattern:
    rows: int
    cols: int
    row_ptr: list[int]
    col_idx: list[int]

    @property
    def nnz(self) -> int:
        return len(self.col_idx)


@dataclass
class CaseData:
    name: str
    ybus: CsrPattern
    pv: list[int]
    pq: list[int]


@dataclass
class JacobianPattern:
    dim: int
    row_ptr: list[int]
    col_idx: list[int]
    pvpq: list[int]
    maps: dict[str, list[int]]
    diag: dict[str, list[int]]
    target_diag_info: dict[int, tuple[str, int]]

    @property
    def nnz(self) -> int:
        return len(self.col_idx)


def is_payload(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("%") and not stripped.startswith("#")


def payload_lines(path: Path) -> Iterable[str]:
    with path.open() as handle:
        for line in handle:
            if is_payload(line):
                yield line


def load_int_vector(path: Path) -> list[int]:
    return [int(line.split()[0]) for line in payload_lines(path)]


def load_matrix_market_pattern(path: Path) -> CsrPattern:
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
        entries: list[tuple[int, int]] = []

        seen = 0
        for line in handle:
            if not is_payload(line):
                continue
            row_1, col_1 = line.split()[:2]
            row = int(row_1) - 1
            col = int(col_1) - 1
            entries.append((row, col))
            if symmetric and row != col:
                entries.append((col, row))
            seen += 1
            if seen == nnz:
                break

    entries.sort()
    merged: list[tuple[int, int]] = []
    for entry in entries:
        if merged and merged[-1] == entry:
            continue
        row, col = entry
        if row < 0 or row >= rows or col < 0 or col >= cols:
            raise ValueError(f"matrix entry out of bounds: {path}")
        merged.append(entry)

    row_ptr = [0 for _ in range(rows + 1)]
    for row, _ in merged:
        row_ptr[row + 1] += 1
    for row in range(rows):
        row_ptr[row + 1] += row_ptr[row]

    col_idx = [0 for _ in range(len(merged))]
    cursor = row_ptr.copy()
    for row, col in merged:
        pos = cursor[row]
        col_idx[pos] = col
        cursor[row] += 1

    return CsrPattern(rows=rows, cols=cols, row_ptr=row_ptr, col_idx=col_idx)


def load_case(case_dir: Path) -> CaseData:
    ybus = load_matrix_market_pattern(case_dir / "dump_Ybus.mtx")
    if ybus.rows != ybus.cols:
        raise ValueError(f"Ybus is not square: {case_dir}")
    return CaseData(
        name=case_dir.name,
        ybus=ybus,
        pv=load_int_vector(case_dir / "dump_pv.txt"),
        pq=load_int_vector(case_dir / "dump_pq.txt"),
    )


def find_case_dirs(dataset_root: Path, selected_cases: list[str]) -> list[Path]:
    if selected_cases:
        return [dataset_root / case for case in selected_cases]
    return sorted(
        path for path in dataset_root.iterdir()
        if path.is_dir() and (path / "dump_Ybus.mtx").exists()
    )


def build_jacobian_pattern(case: CaseData) -> JacobianPattern:
    ybus = case.ybus
    n_bus = ybus.rows
    pvpq = case.pv + case.pq
    n_pvpq = len(pvpq)
    n_pq = len(case.pq)
    dim = n_pvpq + n_pq

    idx_pvpq = [-1 for _ in range(n_bus)]
    idx_pq = [-1 for _ in range(n_bus)]
    for pos, bus in enumerate(pvpq):
        idx_pvpq[bus] = pos
    for pos, bus in enumerate(case.pq):
        idx_pq[bus] = n_pvpq + pos

    row_cols: list[set[int]] = [set() for _ in range(dim)]

    def add(row: int, col: int) -> None:
        if row >= 0 and col >= 0:
            row_cols[row].add(col)

    for row_bus in range(n_bus):
        for pos in range(ybus.row_ptr[row_bus], ybus.row_ptr[row_bus + 1]):
            col_bus = ybus.col_idx[pos]
            if row_bus == col_bus:
                continue
            add(idx_pvpq[row_bus], idx_pvpq[col_bus])
            add(idx_pq[row_bus], idx_pvpq[col_bus])
            add(idx_pvpq[row_bus], idx_pq[col_bus])
            add(idx_pq[row_bus], idx_pq[col_bus])

    for bus in range(n_bus):
        add(idx_pvpq[bus], idx_pvpq[bus])
        add(idx_pq[bus], idx_pvpq[bus])
        add(idx_pvpq[bus], idx_pq[bus])
        add(idx_pq[bus], idx_pq[bus])

    row_ptr = [0]
    col_idx: list[int] = []
    coord_to_pos: dict[tuple[int, int], int] = {}
    for row, cols in enumerate(row_cols):
        sorted_cols = sorted(cols)
        for col in sorted_cols:
            coord_to_pos[(row, col)] = len(col_idx)
            col_idx.append(col)
        row_ptr.append(len(col_idx))

    def target(row: int, col: int) -> int:
        if row < 0 or col < 0:
            return -1
        return coord_to_pos.get((row, col), -1)

    maps = {block: [-1 for _ in range(ybus.nnz)] for block in BLOCKS}
    for row_bus in range(n_bus):
        for pos in range(ybus.row_ptr[row_bus], ybus.row_ptr[row_bus + 1]):
            col_bus = ybus.col_idx[pos]
            maps["J11"][pos] = target(idx_pvpq[row_bus], idx_pvpq[col_bus])
            maps["J21"][pos] = target(idx_pq[row_bus], idx_pvpq[col_bus])
            maps["J12"][pos] = target(idx_pvpq[row_bus], idx_pq[col_bus])
            maps["J22"][pos] = target(idx_pq[row_bus], idx_pq[col_bus])

    diag = {block: [-1 for _ in range(n_bus)] for block in BLOCKS}
    target_diag_info: dict[int, tuple[str, int]] = {}
    for bus in range(n_bus):
        diag["J11"][bus] = target(idx_pvpq[bus], idx_pvpq[bus])
        diag["J21"][bus] = target(idx_pq[bus], idx_pvpq[bus])
        diag["J12"][bus] = target(idx_pvpq[bus], idx_pq[bus])
        diag["J22"][bus] = target(idx_pq[bus], idx_pq[bus])
        for block in BLOCKS:
            pos = diag[block][bus]
            if pos >= 0:
                target_diag_info[pos] = (block, bus)

    return JacobianPattern(
        dim=dim,
        row_ptr=row_ptr,
        col_idx=col_idx,
        pvpq=pvpq,
        maps=maps,
        diag=diag,
        target_diag_info=target_diag_info,
    )


def percentile_nearest(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize_values(values: list[int]) -> dict[str, str | int | float]:
    count = len(values)
    total = sum(values)
    mean = total / count if count else 0.0
    p50 = percentile_nearest(values, 0.50)
    p90 = percentile_nearest(values, 0.90)
    p95 = percentile_nearest(values, 0.95)
    p99 = percentile_nearest(values, 0.99)
    max_value = max(values, default=0)
    return {
        "count": count,
        "total": total,
        "min": min(values, default=0),
        "mean": mean,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": max_value,
        "max_over_mean": max_value / mean if mean else 0.0,
        "p95_over_mean": p95 / mean if mean else 0.0,
    }


def count_atomic_targets(case: CaseData, jac: JacobianPattern) -> dict[str, Counter[int]]:
    ybus = case.ybus
    map_counts: Counter[int] = Counter()
    diag_counts: Counter[int] = Counter()
    all_counts: Counter[int] = Counter()

    for row_bus in range(ybus.rows):
        diag_targets = [jac.diag[block][row_bus] for block in BLOCKS]
        for pos in range(ybus.row_ptr[row_bus], ybus.row_ptr[row_bus + 1]):
            for block in BLOCKS:
                target = jac.maps[block][pos]
                if target >= 0:
                    map_counts[target] += 1
                    all_counts[target] += 1
            for target in diag_targets:
                if target >= 0:
                    diag_counts[target] += 1
                    all_counts[target] += 1

    diag_target_set = set(jac.target_diag_info)
    all_diag_counts = Counter({
        target: count
        for target, count in all_counts.items()
        if target in diag_target_set
    })
    all_offdiag_counts = Counter({
        target: count
        for target, count in all_counts.items()
        if target not in diag_target_set
    })

    return {
        "all_atomic_writes": all_counts,
        "map_writes_only": map_counts,
        "diag_writes_only": diag_counts,
        "all_writes_to_diag_targets": all_diag_counts,
        "all_writes_to_offdiag_targets": all_offdiag_counts,
    }


def row_length_metrics(case: CaseData, jac: JacobianPattern) -> dict[str, list[tuple[int, int | None, int]]]:
    ybus = case.ybus
    all_ybus = [
        (bus, bus, ybus.row_ptr[bus + 1] - ybus.row_ptr[bus])
        for bus in range(ybus.rows)
    ]
    active_ybus = [
        (slot, bus, ybus.row_ptr[bus + 1] - ybus.row_ptr[bus])
        for slot, bus in enumerate(jac.pvpq)
    ]
    jacobian_rows = [
        (row, None, jac.row_ptr[row + 1] - jac.row_ptr[row])
        for row in range(jac.dim)
    ]
    return {
        "ybus_all_rows": all_ybus,
        "ybus_active_pvpq_rows": active_ybus,
        "jacobian_csr_rows": jacobian_rows,
    }


def fmt_float(value: float) -> str:
    return f"{value:.10g}"


def write_csv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze cuPF Jacobian vertex load balance and edge atomic fan-in."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", default=[], help="Case name under dataset root. Repeatable.")
    args = parser.parse_args()

    case_dirs = find_case_dirs(args.dataset_root, args.case)
    if not case_dirs:
        raise SystemExit(f"no cases found under {args.dataset_root}")

    dimension_rows: list[list[object]] = []
    row_length_rows: list[list[object]] = []
    row_length_hist_rows: list[list[object]] = []
    load_balance_summary_rows: list[list[object]] = []
    fanin_hist_rows: list[list[object]] = []
    atomic_summary_rows: list[list[object]] = []
    markdown_key_rows: list[dict[str, object]] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_fanin_path = args.output_dir / "target_fanin.csv"
    diag_target_fanin_path = args.output_dir / "diag_target_fanin.csv"
    with target_fanin_path.open("w", newline="") as target_handle, \
            diag_target_fanin_path.open("w", newline="") as diag_target_handle:
        target_writer = csv.writer(target_handle)
        diag_target_writer = csv.writer(diag_target_handle)
        target_writer.writerow([
            "case",
            "target_index",
            "fan_in",
            "is_diagonal_target",
            "jacobian_row",
            "jacobian_col",
            "diag_block",
            "diag_bus",
        ])
        diag_target_writer.writerow([
            "case",
            "metric",
            "target_index",
            "fan_in",
            "jacobian_row",
            "jacobian_col",
            "diag_block",
            "diag_bus",
        ])

        for case_dir in case_dirs:
            print(f"analyzing {case_dir.name}")
            case = load_case(case_dir)
            jac = build_jacobian_pattern(case)
            atomic_counts = count_atomic_targets(case, jac)
            row_metrics = row_length_metrics(case, jac)
            target_rows = jac_target_rows(jac)

            dimension_rows.append([
                case.name,
                case.ybus.rows,
                len(case.pv),
                len(case.pq),
                len(jac.pvpq),
                jac.dim,
                case.ybus.nnz,
                jac.nnz,
            ])

            lb_summaries: dict[str, dict[str, object]] = {}
            for metric, triples in row_metrics.items():
                values = [length for _, _, length in triples]
                summary = summarize_values(values)
                lb_summaries[metric] = summary
                load_balance_summary_rows.append([
                    case.name,
                    metric,
                    summary["count"],
                    summary["total"],
                    summary["min"],
                    fmt_float(float(summary["mean"])),
                    summary["p50"],
                    summary["p90"],
                    summary["p95"],
                    summary["p99"],
                    summary["max"],
                    fmt_float(float(summary["max_over_mean"])),
                    fmt_float(float(summary["p95_over_mean"])),
                ])
                for row_index, bus_index, length in triples:
                    row_length_rows.append([
                        case.name,
                        metric,
                        row_index,
                        "" if bus_index is None else bus_index,
                        length,
                    ])
                for length, count in sorted(Counter(values).items()):
                    row_length_hist_rows.append([case.name, metric, length, count])

            atomic_summaries: dict[str, dict[str, object]] = {}
            for metric, counts in atomic_counts.items():
                values = list(counts.values())
                summary = summarize_values(values)
                atomic_summaries[metric] = summary
                atomic_summary_rows.append([
                    case.name,
                    metric,
                    summary["count"],
                    summary["total"],
                    summary["min"],
                    fmt_float(float(summary["mean"])),
                    summary["p50"],
                    summary["p90"],
                    summary["p95"],
                    summary["p99"],
                    summary["max"],
                    fmt_float(float(summary["max_over_mean"])),
                    fmt_float(float(summary["p95_over_mean"])),
                ])
                for fan_in, count in sorted(Counter(values).items()):
                    fanin_hist_rows.append([case.name, metric, fan_in, count])

            for target, fan_in in sorted(atomic_counts["all_atomic_writes"].items()):
                diag_info = jac.target_diag_info.get(target)
                target_writer.writerow([
                    case.name,
                    target,
                    fan_in,
                    target in jac.target_diag_info,
                    target_rows[target],
                    jac.col_idx[target],
                    "" if diag_info is None else diag_info[0],
                    "" if diag_info is None else diag_info[1],
                ])

            for metric in ("diag_writes_only", "all_writes_to_diag_targets"):
                for target, fan_in in sorted(atomic_counts[metric].items()):
                    diag_info = jac.target_diag_info[target]
                    diag_target_writer.writerow([
                        case.name,
                        metric,
                        target,
                        fan_in,
                        target_rows[target],
                        jac.col_idx[target],
                        diag_info[0],
                        diag_info[1],
                    ])

            active_lb = lb_summaries["ybus_active_pvpq_rows"]
            diag_only = atomic_summaries["diag_writes_only"]
            diag_all = atomic_summaries["all_writes_to_diag_targets"]
            markdown_key_rows.append({
                "case": case.name,
                "n_bus": case.ybus.rows,
                "ybus_nnz": case.ybus.nnz,
                "jac_nnz": jac.nnz,
                "active_mean": active_lb["mean"],
                "active_p95": active_lb["p95"],
                "active_max": active_lb["max"],
                "active_p95_over_mean": active_lb["p95_over_mean"],
                "active_max_over_mean": active_lb["max_over_mean"],
                "diag_p95": diag_only["p95"],
                "diag_max": diag_only["max"],
                "diag_all_p95": diag_all["p95"],
                "diag_all_max": diag_all["max"],
            })

    write_csv(
        args.output_dir / "case_dimensions.csv",
        ["case", "n_bus", "n_pv", "n_pq", "n_pvpq", "jacobian_dim", "ybus_nnz", "jacobian_nnz"],
        dimension_rows,
    )
    write_csv(
        args.output_dir / "load_balance_summary.csv",
        [
            "case",
            "metric",
            "row_count",
            "total_work",
            "min",
            "mean",
            "p50",
            "p90",
            "p95",
            "p99",
            "max",
            "max_over_mean",
            "p95_over_mean",
        ],
        load_balance_summary_rows,
    )
    write_csv(
        args.output_dir / "row_lengths.csv",
        ["case", "metric", "row_index", "bus_index", "row_length"],
        row_length_rows,
    )
    write_csv(
        args.output_dir / "row_length_histogram.csv",
        ["case", "metric", "row_length", "count"],
        row_length_hist_rows,
    )
    write_csv(
        args.output_dir / "atomic_fanin_summary.csv",
        [
            "case",
            "metric",
            "target_count",
            "total_writes",
            "min",
            "mean",
            "p50",
            "p90",
            "p95",
            "p99",
            "max",
            "max_over_mean",
            "p95_over_mean",
        ],
        atomic_summary_rows,
    )
    write_csv(
        args.output_dir / "fanin_histogram.csv",
        ["case", "metric", "fan_in", "count"],
        fanin_hist_rows,
    )
    write_summary_markdown(args.output_dir / "SUMMARY.md", markdown_key_rows)

    print(f"wrote {args.output_dir}")
    return 0


def jac_target_rows(jac: JacobianPattern) -> list[int]:
    rows = [0 for _ in range(jac.nnz)]
    for row in range(jac.dim):
        for target in range(jac.row_ptr[row], jac.row_ptr[row + 1]):
            rows[target] = row
    return rows


def write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: int(row["n_bus"]))
    lines = [
        "# Jacobian Vertex/Edge Structural Metrics",
        "",
        "This summary is generated by `scripts/analyze_jac_vertex_edge.py`.",
        "",
        "- Load balance uses `Ybus.row_ptr[i+1] - Ybus.row_ptr[i]` over the active `pvpq` buses used by the vertex Jacobian kernel.",
        "- Atomic fan-in counts structural edge-kernel writes into cuPF reduced-Jacobian `J.values` targets.",
        "- `diag_only` counts only `diagJ**[i]` writes from each edge in row `i`; `diag_all` also includes the self-edge `mapJ**` write to the same diagonal target.",
        "- Percentiles use nearest-rank p95.",
        "",
        "## Key Table",
        "",
        "| case | buses | Ybus nnz | J nnz | active row mean | active p95 | active max | p95/mean | max/mean | diag_only p95 | diag_only max | diag_all p95 | diag_all max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {n_bus} | {ybus_nnz} | {jac_nnz} | {active_mean:.3f} | {active_p95} | {active_max} | {active_p95_over_mean:.3f} | {active_max_over_mean:.3f} | {diag_p95} | {diag_max} | {diag_all_p95} | {diag_all_max} |".format(
                **row
            )
        )
    lines.extend([
        "",
        "## Output Files",
        "",
        "- `case_dimensions.csv`: case sizes and reconstructed reduced-Jacobian dimensions.",
        "- `load_balance_summary.csv`: max/mean, p95/mean and related row-length statistics.",
        "- `row_lengths.csv`: every extracted row length.",
        "- `row_length_histogram.csv`: exact histograms for row lengths.",
        "- `atomic_fanin_summary.csv`: target fan-in max/mean, p95/mean and related statistics.",
        "- `target_fanin.csv`: per target-index fan-in counts for all edge-kernel atomic writes.",
        "- `diag_target_fanin.csv`: diagonal target-index fan-in counts for diag-only and diag-all views.",
        "- `fanin_histogram.csv`: exact fan-in histograms.",
    ])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
