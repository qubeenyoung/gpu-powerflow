#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "tmp" / "tutorial_cpp_ref" / "raw_b1.csv"
OUT_DIR = ROOT / "data"

INIT_SOLVE_OUT = OUT_DIR / "tutorial_cpp_reference_b1_init_solve.csv"
OPS_OUT = OUT_DIR / "tutorial_cpp_reference_b1_ops_ms.csv"

INIT_SOLVE_FIELDS = [
    "case_name",
    "n_bus",
    "ybus_nnz",
    "B",
    "init_ms",
    "solve_ms",
    "solve_ms_per_system",
    "iterations",
    "max_final_mismatch",
    "source",
]

OPS_FIELDS = [
    "case_name",
    "n_bus",
    "B",
    "solve_total_ms",
    "factorize_ms",
    "triangular_solve_ms",
    "upload_ms",
    "download_ms",
    "ibus_ms",
    "mismatch_ms",
    "mnorm_ms",
    "jacobian_ms",
    "prepare_rhs_ms",
    "voltage_update_ms",
    "iterations",
    "source",
]

SOURCE = "tutorial_cpp_reference_b1_median"


def fmedian(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with RAW.open(newline="", encoding="utf-8") as handle:
        raw_rows = [row for row in csv.DictReader(handle) if row["converged"] == "1"]

    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        by_case[row["case_name"]].append(row)

    init_rows: list[dict[str, object]] = []
    ops_rows: list[dict[str, object]] = []

    for case_name in sorted(by_case, key=lambda name: int(by_case[name][0]["n_bus"])):
        rows = by_case[case_name]
        n_bus = int(rows[0]["n_bus"])
        ybus_nnz = int(rows[0]["ybus_nnz"])
        iterations = fmedian([float(row["iterations"]) for row in rows])
        init_ms = fmedian([float(row["init_ms"]) for row in rows])
        solve_ms = fmedian([float(row["solve_total_ms"]) for row in rows])
        max_mismatch = max(float(row["final_mismatch"]) for row in rows)

        init_rows.append(
            {
                "case_name": case_name,
                "n_bus": n_bus,
                "ybus_nnz": ybus_nnz,
                "B": 1,
                "init_ms": init_ms,
                "solve_ms": solve_ms,
                "solve_ms_per_system": solve_ms,
                "iterations": iterations,
                "max_final_mismatch": max_mismatch,
                "source": SOURCE,
            }
        )

        ops_rows.append(
            {
                "case_name": case_name,
                "n_bus": n_bus,
                "B": 1,
                "solve_total_ms": solve_ms,
                "factorize_ms": fmedian([float(row["factorize_ms"]) for row in rows]),
                "triangular_solve_ms": fmedian([float(row["triangular_solve_ms"]) for row in rows]),
                "upload_ms": 0.0,
                "download_ms": 0.0,
                "ibus_ms": fmedian([float(row["ibus_ms"]) for row in rows]),
                "mismatch_ms": fmedian([float(row["mismatch_ms"]) for row in rows]),
                "mnorm_ms": fmedian([float(row["mnorm_ms"]) for row in rows]),
                "jacobian_ms": fmedian([float(row["jacobian_ms"]) for row in rows]),
                "prepare_rhs_ms": fmedian([float(row["prepare_rhs_ms"]) for row in rows]),
                "voltage_update_ms": fmedian([float(row["voltage_update_ms"]) for row in rows]),
                "iterations": iterations,
                "source": SOURCE,
            }
        )

    write_csv(INIT_SOLVE_OUT, INIT_SOLVE_FIELDS, init_rows)
    write_csv(OPS_OUT, OPS_FIELDS, ops_rows)
    print(INIT_SOLVE_OUT)
    print(OPS_OUT)


if __name__ == "__main__":
    main()
