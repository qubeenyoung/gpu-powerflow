#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess


ROOT = Path("/workspace")
EXP_ROOT = ROOT / "exp/20260416/refactor"
DATASET_ROOT = ROOT / "datasets/texas_univ_cases/cuPF_datasets"
DEFAULT_BUILD_DIR = EXP_ROOT / "build/lu_nnz"
DEFAULT_BINARY = DEFAULT_BUILD_DIR / "probe_cudss_lu_nnz"
DEFAULT_CASE_LIST = EXP_ROOT / "cases.txt"
DEFAULT_OUTPUT = EXP_ROOT / "lu_nnz_by_case_reorder.csv"
DEFAULT_ALGORITHMS = ("DEFAULT", "ALG_1", "ALG_2", "ALG_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe cuDSS LU nnz without running Newton iterations.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--case-list", type=Path, default=DEFAULT_CASE_LIST)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def read_cases(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def run(cmd: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=True)


def parse_line(line: str) -> dict[str, str]:
    if not line.startswith("LU_NNZ "):
        raise RuntimeError(f"Unexpected probe output: {line}")
    row: dict[str, str] = {}
    for token in line[len("LU_NNZ "):].split():
        key, value = token.split("=", 1)
        row[key] = value
    return row


def build(args: argparse.Namespace) -> None:
    if args.skip_build:
        return
    run([
        "cmake",
        "-S", str(EXP_ROOT),
        "-B", str(args.build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ])
    run([
        "cmake",
        "--build", str(args.build_dir),
        "--target", "probe_cudss_lu_nnz",
        "-j",
    ])


def main() -> None:
    args = parse_args()
    build(args)

    rows: list[dict[str, str]] = []
    for case in read_cases(args.case_list):
        case_dir = args.dataset_root / case
        for alg in args.algorithms:
            proc = run([
                str(args.binary),
                "--case-dir", str(case_dir),
                "--reordering-alg", alg,
            ])
            row = parse_line(proc.stdout.strip())
            rows.append({
                "case_name": row["case"],
                "cudss_reordering_alg": row["alg"],
                "n_bus": row["n_bus"],
                "jacobian_dim": row["jacobian_dim"],
                "jacobian_nnz": row["jacobian_nnz"],
                "lu_nnz_total": row["lu_nnz_total"],
                "lu_nnz_values": row["lu_nnz_values"],
            })
            print(f"{case} {alg} lu_nnz={row['lu_nnz_total']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
