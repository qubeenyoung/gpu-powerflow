from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy import c_, exp, ones, pi, zeros
from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.idx_brch import QT
from pypower.idx_bus import VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, VG
from pypower.loadcase import loadcase
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from scipy.io import mmwrite


DEFAULT_PF_ROOT = Path("/workspace/datasets/pf_dataset")
DEFAULT_BINARY = Path("/workspace/exp/20260413/iterative/build/dump_linear_systems")
DEFAULT_CONVERTED_ROOT = Path("/workspace/exp/20260413/iterative/pf_cupf_dumps")
DEFAULT_OUTPUT_ROOT = Path("/workspace/exp/20260413/iterative/pf_dumps")
DEFAULT_SUMMARY_CSV = Path("/workspace/exp/20260413/iterative/results/pf_dataset_snapshot_summary.csv")


@dataclass
class ConvertedCase:
    source_mat: Path
    case_name: str
    output_dir: Path
    n_bus: int
    ybus_nnz: int
    n_ref: int
    n_pv: int
    n_pq: int


SUMMARY_RE = re.compile(
    r"SUMMARY case=(?P<case>\S+) profile=(?P<profile>\S+) "
    r"converged=(?P<converged>true|false) "
    r"iterations=(?P<iterations>\d+) "
    r"final_mismatch=(?P<final_mismatch>\S+) "
    r"snapshots=(?P<snapshots>\d+) "
    r"total_sec=(?P<total_sec>\S+)"
)
DUMP_RE = re.compile(r"DUMP .* path=\"?(?P<path>[^\"\n]+)\"?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all pf_dataset .mat cases and dump one Newton linear-system snapshot per case."
    )
    parser.add_argument("--pf-root", type=Path, default=DEFAULT_PF_ROOT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--converted-root", type=Path, default=DEFAULT_CONVERTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--profile", default="cuda_mixed_edge")
    parser.add_argument("--tolerance", default="1e-8")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=1, help="1-based index in sorted pf_dataset order.")
    parser.add_argument("--append", action="store_true", help="Append to an existing summary CSV.")
    parser.add_argument("--case-list", type=Path, help="Text file of .mat paths or filenames to run.")
    parser.add_argument("--limit", type=int, help="Only run the first N cases, for smoke testing.")
    return parser.parse_args()


def short_case_name(stem: str) -> str:
    if stem.startswith("pglib_opf_"):
        return stem.removeprefix("pglib_opf_")
    return stem


def ensure_branch_result_columns(ppc: dict) -> dict:
    ppc = dict(ppc)
    branch = np.asarray(ppc["branch"])
    needed_cols = QT + 1
    if branch.shape[1] < needed_cols:
        branch = c_[branch, zeros((branch.shape[0], needed_cols - branch.shape[1]))]
    ppc["branch"] = branch
    return ppc


def write_complex_txt(path: Path, values: np.ndarray) -> None:
    matrix = np.column_stack((values.real, values.imag))
    np.savetxt(path, matrix, fmt="%.18e")


def write_int_txt(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values.astype(np.int32, copy=False), fmt="%d")


def convert_mat_to_cupf_dump(mat_path: Path, output_root: Path) -> ConvertedCase:
    ppc = loadcase(str(mat_path))
    ppc = ensure_branch_result_columns(ppc)
    ppc = ext2int(ppc)

    base_mva = float(np.asarray(ppc["baseMVA"]).squeeze())
    bus = np.asarray(ppc["bus"])
    gen = np.asarray(ppc["gen"])
    branch = np.asarray(ppc["branch"])

    ref, pv, pq = bustypes(bus, gen)

    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    gbus = gen[on, GEN_BUS].astype(int)

    v0 = bus[:, VM] * exp(1j * pi / 180.0 * bus[:, VA])
    vcb = ones(v0.shape)
    vcb[pq] = 0
    k = np.flatnonzero(vcb[gbus])
    if k.size:
        v0[gbus[k]] = gen[on[k], VG] / np.abs(v0[gbus[k]]) * v0[gbus[k]]

    ybus, _, _ = makeYbus(base_mva, bus, branch)
    ybus = ybus.tocsr()
    ybus.sort_indices()
    ybus.data = ybus.data.astype(np.complex128, copy=False)
    ybus.indices = ybus.indices.astype(np.int32, copy=False)
    ybus.indptr = ybus.indptr.astype(np.int32, copy=False)

    sbus = np.asarray(makeSbus(base_mva, bus, gen), dtype=np.complex128).reshape(-1)

    case_name = short_case_name(mat_path.stem)
    case_dir = output_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    mmwrite(case_dir / "dump_Ybus.mtx", ybus)
    write_complex_txt(case_dir / "dump_Sbus.txt", sbus)
    write_complex_txt(case_dir / "dump_V.txt", np.asarray(v0, dtype=np.complex128).reshape(-1))
    write_int_txt(case_dir / "dump_pv.txt", np.asarray(pv, dtype=np.int32))
    write_int_txt(case_dir / "dump_pq.txt", np.asarray(pq, dtype=np.int32))

    return ConvertedCase(
        source_mat=mat_path,
        case_name=case_name,
        output_dir=case_dir,
        n_bus=int(ybus.shape[0]),
        ybus_nnz=int(ybus.nnz),
        n_ref=int(np.asarray(ref).size),
        n_pv=int(np.asarray(pv).size),
        n_pq=int(np.asarray(pq).size),
    )


def parse_dump_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        summary = SUMMARY_RE.search(line)
        if summary:
            parsed.update(summary.groupdict())
        dump = DUMP_RE.search(line)
        if dump:
            parsed["snapshot_dir"] = dump.group("path")
    return parsed


def run_dump_binary(args: argparse.Namespace, converted: ConvertedCase) -> tuple[dict[str, str], str, str, int]:
    cmd = [
        str(args.binary),
        "--profile",
        args.profile,
        "--case-dir",
        str(converted.output_dir),
        "--max-dump-iters",
        "1",
        "--max-iter",
        str(args.max_iter),
        "--tolerance",
        str(args.tolerance),
        "--output-root",
        str(args.output_root),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    parsed = parse_dump_output(proc.stdout)
    return parsed, proc.stdout, proc.stderr, proc.returncode


def open_summary_writer(path: Path, append: bool) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = True
    if append and path.exists() and path.stat().st_size > 0:
        needs_header = False
    handle = path.open("a" if append else "w", newline="", encoding="utf-8")
    fieldnames = [
        "source_mat",
        "case",
        "converted_dir",
        "snapshot_dir",
        "profile",
        "status",
        "converged",
        "iterations",
        "final_mismatch",
        "snapshots",
        "total_sec",
        "n_bus",
        "ybus_nnz",
        "n_ref",
        "n_pv",
        "n_pq",
        "returncode",
        "error",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if needs_header:
        writer.writeheader()
    return handle, writer


def main() -> int:
    args = parse_args()
    if args.case_list is not None:
        mat_paths = []
        for line in args.case_list.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            path = Path(item)
            if not path.is_absolute():
                path = args.pf_root / item
            mat_paths.append(path)
    else:
        mat_paths = sorted(args.pf_root.glob("*.mat"))
        if args.start_index < 1:
            raise RuntimeError("--start-index must be >= 1")
        mat_paths = mat_paths[args.start_index - 1 :]
    if args.limit is not None:
        mat_paths = mat_paths[: args.limit]
    if not mat_paths:
        raise RuntimeError(f"No .mat cases found under {args.pf_root}")
    if not args.binary.exists():
        raise RuntimeError(f"dump binary not found: {args.binary}")

    handle, writer = open_summary_writer(args.summary_csv, args.append)
    last_index = args.start_index + len(mat_paths) - 1
    started = time.time()
    try:
        for offset, mat_path in enumerate(mat_paths, start=0):
            idx = args.start_index + offset
            row = {
                "source_mat": str(mat_path),
                "case": short_case_name(mat_path.stem),
                "converted_dir": "",
                "snapshot_dir": "",
                "profile": args.profile,
                "status": "failed",
                "converged": "",
                "iterations": "",
                "final_mismatch": "",
                "snapshots": "",
                "total_sec": "",
                "n_bus": "",
                "ybus_nnz": "",
                "n_ref": "",
                "n_pv": "",
                "n_pq": "",
                "returncode": "",
                "error": "",
            }
            print(f"[{idx}/{last_index}] {mat_path.name}", flush=True)
            try:
                converted = convert_mat_to_cupf_dump(mat_path, args.converted_root)
                row.update(
                    {
                        "case": converted.case_name,
                        "converted_dir": str(converted.output_dir),
                        "n_bus": converted.n_bus,
                        "ybus_nnz": converted.ybus_nnz,
                        "n_ref": converted.n_ref,
                        "n_pv": converted.n_pv,
                        "n_pq": converted.n_pq,
                    }
                )

                parsed, stdout, stderr, returncode = run_dump_binary(args, converted)
                row["returncode"] = returncode
                row["snapshot_dir"] = parsed.get("snapshot_dir", "")
                row["converged"] = parsed.get("converged", "")
                row["iterations"] = parsed.get("iterations", "")
                row["final_mismatch"] = parsed.get("final_mismatch", "")
                row["snapshots"] = parsed.get("snapshots", "")
                row["total_sec"] = parsed.get("total_sec", "")
                row["status"] = "ok" if returncode == 0 else "failed"
                if returncode != 0:
                    row["error"] = (stderr or stdout).strip().replace("\n", " | ")[:1000]
            except Exception as exc:
                row["error"] = str(exc)

            writer.writerow(row)
            handle.flush()
            print(
                f"  status={row['status']} converged={row['converged']} "
                f"snapshots={row['snapshots']} error={row['error'][:120]}",
                flush=True,
            )
    finally:
        handle.close()

    elapsed = time.time() - started
    print(f"[done] wrote {args.summary_csv} in {elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
