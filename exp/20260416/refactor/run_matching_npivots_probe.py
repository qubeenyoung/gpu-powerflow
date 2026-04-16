#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import subprocess


ROOT = Path("/workspace")
EXP_ROOT = ROOT / "exp/20260416/refactor"
DATASET_ROOT = ROOT / "datasets/texas_univ_cases/cuPF_datasets"
DEFAULT_BUILD_DIR = EXP_ROOT / "build/lu_nnz"
DEFAULT_BINARY = DEFAULT_BUILD_DIR / "probe_cudss_lu_nnz"
DEFAULT_CASE_LIST = EXP_ROOT / "cases.txt"
DEFAULT_OUTPUT = EXP_ROOT / "matching_npivots_by_case.csv"
DEFAULT_SUMMARY = EXP_ROOT / "MATCHING_NPIVOTS_SUMMARY.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe cuDSS NPIVOTS with DEFAULT reordering and matching off/on.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--case-list", type=Path, default=DEFAULT_CASE_LIST)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--matching-alg", default="DEFAULT")
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


def fmt_int(value: str | int) -> str:
    return f"{int(value):,}"


def fmt_x(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return "-"
    return f"{value:.2f}x"


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    cases: list[str] = []
    by: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        case = row["case_name"]
        label = row["matching_label"]
        if case not in cases:
            cases.append(case)
        by[(case, label)] = row

    lines: list[str] = []
    lines += ["# cuDSS Matching NPIVOTS Probe", ""]
    lines += ["- Reordering algorithm: `DEFAULT`"]
    lines += ["- Matrix: edge-based Newton Jacobian at `V0`"]
    lines += ["- cuDSS phases: `ANALYSIS` + one `FACTORIZATION` only"]
    lines += ["- No Newton iteration loop or solve loop was run."]
    lines += ["- Output CSV: `matching_npivots_by_case.csv`", ""]
    lines += ["## NPIVOTS", ""]
    headers = ["case", "J dim", "no matching", "matching", "delta", "ratio"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    ratios: list[float] = []
    for case in cases:
        off = by[(case, "off")]
        on = by[(case, "on")]
        off_value = int(off["npivots_total"])
        on_value = int(on["npivots_total"])
        ratio = on_value / off_value if off_value else math.nan
        if off_value:
            ratios.append(ratio)
        lines.append("| " + " | ".join([
            case,
            fmt_int(off["jacobian_dim"]),
            fmt_int(off_value),
            fmt_int(on_value),
            fmt_int(on_value - off_value),
            fmt_x(ratio),
        ]) + " |")

    lines += ["", "## LU NNZ Side Check", ""]
    headers = ["case", "LU no matching", "LU matching", "delta", "ratio"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lu_ratios: list[float] = []
    for case in cases:
        off = by[(case, "off")]
        on = by[(case, "on")]
        off_value = int(off["lu_nnz_total"])
        on_value = int(on["lu_nnz_total"])
        ratio = on_value / off_value if off_value else math.nan
        if off_value:
            lu_ratios.append(ratio)
        lines.append("| " + " | ".join([
            case,
            fmt_int(off_value),
            fmt_int(on_value),
            fmt_int(on_value - off_value),
            fmt_x(ratio),
        ]) + " |")

    lines += ["", "## Geomean Ratios", ""]
    if ratios:
        lines.append(f"- NPIVOTS matching/on over off: {fmt_x(math.exp(statistics.mean(math.log(v) for v in ratios)))}")
    else:
        lines.append("- NPIVOTS matching/on over off: not defined because every off value is 0.")
    if lu_ratios:
        lines.append(f"- LU_NNZ matching/on over off: {fmt_x(math.exp(statistics.mean(math.log(v) for v in lu_ratios)))}")

    lines += ["", "## Notes", ""]
    matching_algs = sorted({row["matching_alg"] for row in rows if row["matching_label"] == "on"})
    if all(int(row["npivots_total"]) == 0 for row in rows):
        lines.append("- `CUDSS_DATA_NPIVOTS` returned 0 for every case, both with matching disabled and enabled.")
    lines.append(f"- Matching was tested with `CUDSS_CONFIG_MATCHING_ALG={', '.join('CUDSS_ALG_' + alg.split('_', 1)[1] if alg.startswith('ALG_') else 'CUDSS_ALG_DEFAULT' for alg in matching_algs)}`.")
    lines.append("- Reordering was fixed to `CUDSS_ALG_DEFAULT`.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    build(args)

    rows: list[dict[str, str]] = []
    for case in read_cases(args.case_list):
        case_dir = args.dataset_root / case
        for label, use_matching in (("off", "0"), ("on", "1")):
            proc = run([
                str(args.binary),
                "--case-dir", str(case_dir),
                "--reordering-alg", "DEFAULT",
                "--use-matching", use_matching,
                "--matching-alg", args.matching_alg,
            ])
            row = parse_line(proc.stdout.strip())
            rows.append({
                "case_name": row["case"],
                "matching_label": label,
                "use_matching": row["use_matching"],
                "matching_alg": row["matching_alg"],
                "reordering_alg": row["alg"],
                "n_bus": row["n_bus"],
                "jacobian_dim": row["jacobian_dim"],
                "jacobian_nnz": row["jacobian_nnz"],
                "lu_nnz_total": row["lu_nnz_total"],
                "lu_nnz_values": row["lu_nnz_values"],
                "npivots_total": row["npivots_total"],
                "npivots_values": row["npivots_values"],
            })
            print(
                f"{case} matching={label} npivots={row['npivots_total']} lu_nnz={row['lu_nnz_total']}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(args.summary, rows)
    print(f"[OK] wrote {args.output}")
    print(f"[OK] wrote {args.summary}")


if __name__ == "__main__":
    main()
