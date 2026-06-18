#!/usr/bin/env python3
"""Measure front-size distributions with STRUMPACK's own symbolic tree."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


TABLE2_JANNA = [
    "Serena",
    "Geo_1438",
    "Hook_1498",
    "ML_Geer",
    "Transport",
    "Flan_1565",
    "Cube_Coup_dt0",
]


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
    print(f"extract {tar_path}", flush=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    if mtx.exists():
        return mtx
    matches = list(out_dir.glob(f"**/{name}.mtx"))
    if not matches:
        raise FileNotFoundError(f"could not find {name}.mtx after extracting {tar_path}")
    return matches[0]


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


def summarize_fronts(front_csv: Path, matrix_set: str, matrix: str, reorder_ms: str) -> dict[str, object]:
    with front_csv.open() as f:
        rows = list(csv.DictReader(f))
    fsz = sorted(int(r["dim_blk"]) for r in rows)
    nc = sorted(int(r["dim_sep"]) for r in rows)
    uc = sorted(int(r["dim_upd"]) for r in rows)
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
        "front_csv": str(front_csv),
        "reorder_ms": reorder_ms,
        "fronts": len(fsz),
        "fsz_min": fsz[0] if fsz else "",
        "fsz_p50": percentile(fsz, 0.50),
        "fsz_p90": percentile(fsz, 0.90),
        "fsz_p99": percentile(fsz, 0.99),
        "fsz_max": fsz[-1] if fsz else "",
        "nc_p50": percentile(nc, 0.50),
        "nc_max": nc[-1] if nc else "",
        "uc_p50": percentile(uc, 0.50),
        "uc_max": uc[-1] if uc else "",
    }
    out.update(bins)
    for key, val in bins.items():
        out[key + "_pct"] = 100.0 * int(val) / total
    return out


def run_front_dump(exe: Path, matrix: Path, out_csv: Path, args: argparse.Namespace) -> str:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not args.force:
        print(f"skip existing {out_csv}", flush=True)
        return ""
    cmd = [
        str(exe),
        str(matrix),
        str(out_csv),
        "--sp_compression",
        "none",
        "--sp_reordering_method",
        args.reordering,
    ]
    print("+", " ".join(cmd), flush=True)
    cp = subprocess.run(
        cmd,
        cwd=args.repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(cp.stdout, end="", flush=True)
    match = re.search(r"reorder_ms=([0-9.eE+-]+)", cp.stdout)
    return match.group(1) if match else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument(
        "--exe",
        type=Path,
        default=Path("/root/baselines/STRUMPACK/build_nomagma/examples/sparse/front_dump"),
    )
    ap.add_argument("--out", type=Path, default=Path("custom_linear_solver/results/front_distributions/strumpack"))
    ap.add_argument("--power-root", type=Path, default=Path("/workspace/cls_linsys"))
    ap.add_argument("--reordering", default="metis")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--only-power", action="store_true")
    ap.add_argument("--only-suitesparse", action="store_true")
    args = ap.parse_args()

    args.repo = args.repo.resolve()
    args.power_root = args.power_root.resolve()
    out = (args.repo / args.out).resolve()
    data = out / "data"
    fronts = out / "fronts"
    summaries: list[dict[str, object]] = []

    if not args.only_power:
        for name in TABLE2_JANNA:
            tar_path = data / "downloads" / f"{name}.tar.gz"
            if not args.skip_download:
                download(f"https://sparse.tamu.edu/MM/Janna/{name}.tar.gz", tar_path)
            mtx = extract_matrix(tar_path, data / "suitesparse" / "Janna", name)
            csv_path = fronts / "suitesparse" / f"{name}.csv"
            reorder_ms = run_front_dump(args.exe, mtx, csv_path, args)
            summaries.append(summarize_fronts(csv_path, "suitesparse", name, reorder_ms))

    if not args.only_suitesparse:
        for matrix in sorted(args.power_root.glob("case*/J.mtx")):
            name = matrix.parent.name
            csv_path = fronts / "power" / f"{name}.csv"
            reorder_ms = run_front_dump(args.exe, matrix, csv_path, args)
            summaries.append(summarize_fronts(csv_path, "power", name, reorder_ms))

    summary_csv = out / "front_distribution_summary.csv"
    if summaries:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
