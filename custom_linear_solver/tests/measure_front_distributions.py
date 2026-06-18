#!/usr/bin/env python3
"""Download STRUMPACK SuiteSparse cases and summarize multifrontal front sizes."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


STRUMPACK_SUITESPARSE = [
    "Serena",
    "Geo_1438",
    "Hook_1498",
    "ML_Geer",
    "Transport",
    "Flan_1565",
    "Cube_Coup_dt0",
]


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def download(url: str, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"download {url} -> {out}", flush=True)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=8 * 1024 * 1024)
    tmp.replace(out)


def extract_matrix(tar_path: Path, out_dir: Path, name: str) -> Path:
    mtx = out_dir / name / f"{name}.mtx"
    if mtx.exists():
        return mtx
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    if not mtx.exists():
        matches = list(out_dir.glob(f"**/{name}.mtx"))
        if not matches:
            raise FileNotFoundError(f"could not find {name}.mtx after extracting {tar_path}")
        return matches[0]
    return mtx


def matrix_rows(mtx: Path) -> int:
    with mtx.open() as f:
        for line in f:
            if line.startswith("%"):
                continue
            return int(line.split()[0])
    raise ValueError(f"no MatrixMarket size line in {mtx}")


def ensure_rhs(case_dir: Path, mtx: Path) -> Path:
    rhs = case_dir / "F.mtx"
    if rhs.exists():
        return rhs
    n = matrix_rows(mtx)
    with rhs.open("w") as f:
        f.write("%%MatrixMarket matrix array real general\n")
        f.write(f"{n} 1\n")
        for _ in range(n):
            f.write("1\n")
    return rhs


def percentile(xs: list[int], p: float) -> float:
    if not xs:
        return math.nan
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize_fronts(front_csv: Path, matrix_set: str, matrix: str) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    with front_csv.open() as f:
        rows = list(csv.DictReader(f))
    fsz = sorted(int(r["fsz"]) for r in rows)
    nc = [int(r["nc"]) for r in rows]
    uc = [int(r["uc"]) for r in rows]
    extend = [int(r["extend_elems"]) for r in rows] if rows and "extend_elems" in rows[0] else []
    bins = {
        "fronts_le_32": sum(x <= 32 for x in fsz),
        "fronts_33_96": sum(33 <= x <= 96 for x in fsz),
        "fronts_97_159": sum(97 <= x <= 159 for x in fsz),
        "fronts_160_512": sum(160 <= x <= 512 for x in fsz),
        "fronts_gt_512": sum(x > 512 for x in fsz),
    }
    total = max(1, len(fsz))
    out: dict[str, object] = {
        "set": matrix_set,
        "matrix": matrix,
        "fronts": len(fsz),
        "fsz_min": fsz[0] if fsz else "",
        "fsz_p50": percentile(fsz, 0.50),
        "fsz_p90": percentile(fsz, 0.90),
        "fsz_p99": percentile(fsz, 0.99),
        "fsz_max": fsz[-1] if fsz else "",
        "nc_p50": percentile(sorted(nc), 0.50),
        "nc_max": max(nc) if nc else "",
        "uc_p50": percentile(sorted(uc), 0.50),
        "uc_max": max(uc) if uc else "",
        "extend_elems_sum": sum(extend) if extend else "",
    }
    out.update(bins)
    for key, val in bins.items():
        out[key + "_pct"] = 100.0 * int(val) / total
    return out


def analyze_case(repo: Path, exe: Path, matrix: Path, rhs: Path, out_csv: Path, args: argparse.Namespace) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not args.force:
        return
    cmd = [
        str(exe),
        "--matrix", str(matrix),
        "--rhs", str(rhs),
        "--precision", args.precision,
        "--analyze-only",
        "--dump-fronts", str(out_csv),
        "--metis-seed", str(args.metis_seed),
    ]
    if args.serial_nd:
        cmd.append("--serial-nd")
    run(cmd, repo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--exe", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("custom_linear_solver/results/front_distributions"))
    ap.add_argument("--precision", default="fp32", choices=["fp64", "fp32", "tf32"])
    ap.add_argument("--metis-seed", type=int, default=42)
    ap.add_argument("--serial-nd", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--only-power", action="store_true")
    ap.add_argument("--only-suitesparse", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    exe = args.exe or (repo / "custom_linear_solver/build-nograph/custom_linear_solver_run")
    out = (repo / args.out).resolve()
    data = out / "data"
    fronts = out / "fronts"
    summaries: list[dict[str, object]] = []

    if not args.only_power:
        for name in STRUMPACK_SUITESPARSE:
            case_dir = data / "suitesparse" / "Janna" / name
            tar_path = data / "downloads" / f"{name}.tar.gz"
            if not args.skip_download:
                download(f"https://sparse.tamu.edu/MM/Janna/{name}.tar.gz", tar_path)
            mtx = extract_matrix(tar_path, data / "suitesparse" / "Janna", name)
            rhs = ensure_rhs(case_dir, mtx)
            csv_path = fronts / "suitesparse" / f"{name}.csv"
            analyze_case(repo, exe, mtx, rhs, csv_path, args)
            summaries.append(summarize_fronts(csv_path, "suitesparse", name))

    if not args.only_suitesparse:
        for case in sorted((repo / "exp/cases").glob("case*/J.mtx")):
            case_dir = case.parent
            name = case_dir.name
            csv_path = fronts / "power" / f"{name}.csv"
            analyze_case(repo, exe, case, case_dir / "F.mtx", csv_path, args)
            summaries.append(summarize_fronts(csv_path, "power", name))

    summary_csv = out / "front_distribution_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if summaries:
        fields = list(summaries[0].keys())
        with summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summaries)
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
