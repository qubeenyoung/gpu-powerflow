#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = REPO_ROOT / "exp" / "20260414" / "amgx" / "cupf_dumps"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "exp" / "20260415" / "jacobian_analysis" / "results"
QUANTILES = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]


@dataclass
class CaseVoltageData:
    name: str
    voltage: np.ndarray
    pv: np.ndarray
    pq: np.ndarray


def is_payload(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("%") and not stripped.startswith("#")


def payload_lines(path: Path):
    with path.open() as handle:
        for line in handle:
            if is_payload(line):
                yield line


def load_complex_vector(path: Path) -> np.ndarray:
    values = []
    for line in payload_lines(path):
        real, imag = line.split()[:2]
        values.append(complex(float(real), float(imag)))
    return np.asarray(values, dtype=np.complex128)


def load_int_vector(path: Path) -> np.ndarray:
    return np.asarray([int(line.split()[0]) for line in payload_lines(path)], dtype=np.int64)


def load_case(case_dir: Path) -> CaseVoltageData:
    voltage = load_complex_vector(case_dir / "dump_V.txt")
    pv = load_int_vector(case_dir / "dump_pv.txt")
    pq = load_int_vector(case_dir / "dump_pq.txt")
    max_index = max(int(pv.max(initial=-1)), int(pq.max(initial=-1)))
    if max_index >= voltage.size:
        raise ValueError(f"PV/PQ bus index exceeds voltage length: {case_dir}")
    return CaseVoltageData(name=case_dir.name, voltage=voltage, pv=pv, pq=pq)


def find_case_dirs(dataset_root: Path, selected_cases: list[str]) -> list[Path]:
    if selected_cases:
        return [dataset_root / case for case in selected_cases]
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "dump_V.txt").exists()
    )


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def slack_buses(n_bus: int, pv: np.ndarray, pq: np.ndarray) -> np.ndarray:
    active = np.zeros(n_bus, dtype=bool)
    active[pv] = True
    active[pq] = True
    return np.flatnonzero(~active)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def slack_relative_theta(case: CaseVoltageData, theta: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    slack = slack_buses(theta.size, case.pv, case.pq)
    reference = float(theta[slack[0]]) if slack.size else 0.0
    return wrap_to_pi(theta - reference), reference, slack


def stats_row(case: str, variable: str, subset: str, values: np.ndarray) -> dict[str, str]:
    clean = finite_values(values)
    if clean.size == 0:
        raise ValueError(f"empty finite value set for {case}/{variable}/{subset}")

    quantile_values = np.quantile(clean, QUANTILES)
    row = {
        "case": case,
        "variable": variable,
        "subset": subset,
        "n": str(clean.size),
        "min": f"{float(clean.min()):.17e}",
        "max": f"{float(clean.max()):.17e}",
        "mean": f"{float(clean.mean()):.17e}",
        "std": f"{float(clean.std(ddof=0)):.17e}",
        "abs_max": f"{float(np.max(np.abs(clean))):.17e}",
        "rms": f"{float(math.sqrt(np.mean(clean * clean))):.17e}",
    }
    for quantile, value in zip(QUANTILES, quantile_values, strict=True):
        name = f"p{quantile * 100:g}".replace(".", "_")
        row[name] = f"{float(value):.17e}"
    return row


def case_stats_rows(case: CaseVoltageData) -> list[dict[str, str]]:
    voltage = case.voltage
    vm = np.abs(voltage)
    theta = np.angle(voltage)
    theta_deg = np.rad2deg(theta)
    theta_rel, _, _ = slack_relative_theta(case, theta)
    theta_rel_deg = np.rad2deg(theta_rel)

    pvpq = np.concatenate([case.pv, case.pq])
    rows = [
        stats_row(case.name, "Vm_pu", "all_buses", vm),
        stats_row(case.name, "theta_rad", "all_buses", theta),
        stats_row(case.name, "theta_deg", "all_buses", theta_deg),
        stats_row(case.name, "theta_rel_slack_rad", "all_buses", theta_rel),
        stats_row(case.name, "theta_rel_slack_deg", "all_buses", theta_rel_deg),
        stats_row(case.name, "Vm_pu", "newton_pq", vm[case.pq]),
        stats_row(case.name, "theta_rad", "newton_pvpq", theta[pvpq]),
        stats_row(case.name, "theta_deg", "newton_pvpq", theta_deg[pvpq]),
        stats_row(case.name, "theta_rel_slack_rad", "newton_pvpq", theta_rel[pvpq]),
        stats_row(case.name, "theta_rel_slack_deg", "newton_pvpq", theta_rel_deg[pvpq]),
    ]
    return rows


def overview_row(
    case: str,
    n_bus: int,
    n_theta_pvpq: int,
    n_vm_pq: int,
    n_slack: int,
    vm_pq: np.ndarray,
    theta_pvpq: np.ndarray,
    theta_rel_pvpq: np.ndarray,
    theta_ref_slack_deg: float,
) -> dict[str, str]:
    theta_raw_abs_deg = np.abs(np.rad2deg(theta_pvpq))
    theta_rel_abs_deg = np.abs(np.rad2deg(theta_rel_pvpq))

    theta_raw_q = np.quantile(theta_raw_abs_deg, [0.5, 0.95, 0.99])
    theta_rel_q = np.quantile(theta_rel_abs_deg, [0.5, 0.95, 0.99])
    vm_q = np.quantile(vm_pq, [0.01, 0.5, 0.99])
    vm_outside_095_105 = np.mean((vm_pq < 0.95) | (vm_pq > 1.05)) * 100.0
    vm_outside_090_110 = np.mean((vm_pq < 0.90) | (vm_pq > 1.10)) * 100.0
    return {
        "case": case,
        "n_bus": str(n_bus),
        "n_theta_pvpq": str(n_theta_pvpq),
        "n_vm_pq": str(n_vm_pq),
        "n_slack": str(n_slack),
        "vm_pq_min": f"{float(vm_pq.min()):.8f}",
        "vm_pq_p01": f"{float(vm_q[0]):.8f}",
        "vm_pq_p50": f"{float(vm_q[1]):.8f}",
        "vm_pq_p99": f"{float(vm_q[2]):.8f}",
        "vm_pq_max": f"{float(vm_pq.max()):.8f}",
        "vm_pq_mean": f"{float(vm_pq.mean()):.8f}",
        "vm_pq_std": f"{float(vm_pq.std(ddof=0)):.8f}",
        "vm_pq_outside_095_105_pct": f"{float(vm_outside_095_105):.8f}",
        "vm_pq_outside_090_110_pct": f"{float(vm_outside_090_110):.8f}",
        "theta_ref_slack_deg": f"{float(theta_ref_slack_deg):.8f}",
        "theta_raw_abs_deg_p50": f"{float(theta_raw_q[0]):.8f}",
        "theta_raw_abs_deg_p95": f"{float(theta_raw_q[1]):.8f}",
        "theta_raw_abs_deg_p99": f"{float(theta_raw_q[2]):.8f}",
        "theta_raw_abs_deg_max": f"{float(theta_raw_abs_deg.max()):.8f}",
        "theta_rel_slack_abs_deg_p50": f"{float(theta_rel_q[0]):.8f}",
        "theta_rel_slack_abs_deg_p95": f"{float(theta_rel_q[1]):.8f}",
        "theta_rel_slack_abs_deg_p99": f"{float(theta_rel_q[2]):.8f}",
        "theta_rel_slack_abs_deg_max": f"{float(theta_rel_abs_deg.max()):.8f}",
        "theta_rel_abs_gt_030_deg_pct": f"{float(np.mean(theta_rel_abs_deg > 30.0) * 100.0):.8f}",
        "theta_rel_abs_gt_060_deg_pct": f"{float(np.mean(theta_rel_abs_deg > 60.0) * 100.0):.8f}",
        "theta_rel_abs_gt_090_deg_pct": f"{float(np.mean(theta_rel_abs_deg > 90.0) * 100.0):.8f}",
        "theta_rel_abs_gt_120_deg_pct": f"{float(np.mean(theta_rel_abs_deg > 120.0) * 100.0):.8f}",
        "theta_rel_abs_gt_150_deg_pct": f"{float(np.mean(theta_rel_abs_deg > 150.0) * 100.0):.8f}",
        "theta_raw_rad_min": f"{float(theta_pvpq.min()):.8f}",
        "theta_raw_rad_max": f"{float(theta_pvpq.max()):.8f}",
        "theta_raw_rad_std": f"{float(theta_pvpq.std(ddof=0)):.8f}",
        "theta_rel_slack_rad_min": f"{float(theta_rel_pvpq.min()):.8f}",
        "theta_rel_slack_rad_max": f"{float(theta_rel_pvpq.max()):.8f}",
        "theta_rel_slack_rad_std": f"{float(theta_rel_pvpq.std(ddof=0)):.8f}",
    }


def build_overview(case: CaseVoltageData) -> dict[str, str]:
    voltage = case.voltage
    vm_pq = np.abs(voltage[case.pq])
    theta = np.angle(voltage)
    pvpq = np.concatenate([case.pv, case.pq])
    theta_pvpq = theta[pvpq]
    theta_rel, theta_ref, slack = slack_relative_theta(case, theta)
    theta_rel_pvpq = theta_rel[pvpq]
    return overview_row(
        case=case.name,
        n_bus=voltage.size,
        n_theta_pvpq=case.pv.size + case.pq.size,
        n_vm_pq=case.pq.size,
        n_slack=slack.size,
        vm_pq=vm_pq,
        theta_pvpq=theta_pvpq,
        theta_rel_pvpq=theta_rel_pvpq,
        theta_ref_slack_deg=float(np.rad2deg(theta_ref)),
    )


def global_stats_rows(cases: list[CaseVoltageData]) -> list[dict[str, str]]:
    vm_all_parts = []
    theta_all_parts = []
    theta_rel_all_parts = []
    vm_pq_parts = []
    theta_pvpq_parts = []
    theta_rel_pvpq_parts = []

    for case in cases:
        voltage = case.voltage
        vm = np.abs(voltage)
        theta = np.angle(voltage)
        theta_rel, _, _ = slack_relative_theta(case, theta)
        pvpq = np.concatenate([case.pv, case.pq])

        vm_all_parts.append(vm)
        theta_all_parts.append(theta)
        theta_rel_all_parts.append(theta_rel)
        vm_pq_parts.append(vm[case.pq])
        theta_pvpq_parts.append(theta[pvpq])
        theta_rel_pvpq_parts.append(theta_rel[pvpq])

    vm_all = np.concatenate(vm_all_parts)
    theta_all = np.concatenate(theta_all_parts)
    theta_rel_all = np.concatenate(theta_rel_all_parts)
    vm_pq = np.concatenate(vm_pq_parts)
    theta_pvpq = np.concatenate(theta_pvpq_parts)
    theta_rel_pvpq = np.concatenate(theta_rel_pvpq_parts)

    return [
        stats_row("GLOBAL", "Vm_pu", "all_buses", vm_all),
        stats_row("GLOBAL", "theta_rad", "all_buses", theta_all),
        stats_row("GLOBAL", "theta_deg", "all_buses", np.rad2deg(theta_all)),
        stats_row("GLOBAL", "theta_rel_slack_rad", "all_buses", theta_rel_all),
        stats_row("GLOBAL", "theta_rel_slack_deg", "all_buses", np.rad2deg(theta_rel_all)),
        stats_row("GLOBAL", "Vm_pu", "newton_pq", vm_pq),
        stats_row("GLOBAL", "theta_rad", "newton_pvpq", theta_pvpq),
        stats_row("GLOBAL", "theta_deg", "newton_pvpq", np.rad2deg(theta_pvpq)),
        stats_row("GLOBAL", "theta_rel_slack_rad", "newton_pvpq", theta_rel_pvpq),
        stats_row("GLOBAL", "theta_rel_slack_deg", "newton_pvpq", np.rad2deg(theta_rel_pvpq)),
    ]


def build_global_overview(cases: list[CaseVoltageData]) -> dict[str, str]:
    vm_pq_parts = []
    theta_pvpq_parts = []
    theta_rel_pvpq_parts = []
    n_bus = 0
    n_theta = 0
    n_vm = 0
    n_slack = 0

    for case in cases:
        voltage = case.voltage
        theta = np.angle(voltage)
        theta_rel, _, slack = slack_relative_theta(case, theta)
        pvpq = np.concatenate([case.pv, case.pq])

        vm_pq_parts.append(np.abs(voltage[case.pq]))
        theta_pvpq_parts.append(theta[pvpq])
        theta_rel_pvpq_parts.append(theta_rel[pvpq])
        n_bus += voltage.size
        n_theta += pvpq.size
        n_vm += case.pq.size
        n_slack += slack.size

    return overview_row(
        case="GLOBAL",
        n_bus=n_bus,
        n_theta_pvpq=n_theta,
        n_vm_pq=n_vm,
        n_slack=n_slack,
        vm_pq=np.concatenate(vm_pq_parts),
        theta_pvpq=np.concatenate(theta_pvpq_parts),
        theta_rel_pvpq=np.concatenate(theta_rel_pvpq_parts),
        theta_ref_slack_deg=float("nan"),
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure voltage magnitude and theta scales in cuPF dumps.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", default=[], help="Case name under dataset root. Repeatable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dirs = find_case_dirs(args.dataset_root, args.case)
    if not case_dirs:
        raise SystemExit(f"no cases found under {args.dataset_root}")

    all_cases: list[CaseVoltageData] = []
    stats_rows: list[dict[str, str]] = []
    overview_rows: list[dict[str, str]] = []
    for case_dir in case_dirs:
        print(f"analyzing {case_dir.name}")
        case = load_case(case_dir)
        all_cases.append(case)
        stats_rows.extend(case_stats_rows(case))
        overview_rows.append(build_overview(case))

    if len(all_cases) > 1:
        stats_rows.extend(global_stats_rows(all_cases))
        overview_rows.append(build_global_overview(all_cases))

    write_csv(args.output_dir / "voltage_scale_stats.csv", stats_rows)
    write_csv(args.output_dir / "voltage_scale_overview.csv", overview_rows)
    print(f"wrote {args.output_dir / 'voltage_scale_stats.csv'}")
    print(f"wrote {args.output_dir / 'voltage_scale_overview.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
