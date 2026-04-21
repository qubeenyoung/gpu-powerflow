from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from numpy import c_, exp, ones, pi, zeros
from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.idx_brch import QT
from pypower.idx_bus import VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, VG
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from scipy.io import mmwrite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Texas MATPOWER .m cases into cuPF dump directories."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("datasets/texas_univ_cases"),
        help="Directory containing MATPOWER .m cases.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/texas_univ_cases/cuPF_datasets"),
        help="Directory where cuPF dump case directories are written.",
    )
    parser.add_argument(
        "--case",
        default="all",
        help="Case stem to convert, for example case_ACTIVSg200. Use all for every .m file.",
    )
    return parser.parse_args()


def parse_matrix(text: str, name: str) -> np.ndarray:
    match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S)
    if not match:
        raise RuntimeError(f"missing mpc.{name}")

    rows: list[list[float]] = []
    for line in match.group(1).splitlines():
        line = line.split("%", 1)[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(value) for value in line.split()])

    if not rows:
        raise RuntimeError(f"mpc.{name} is empty")

    width = max(len(row) for row in rows)
    return np.array([row + [0.0] * (width - len(row)) for row in rows], dtype=float)


def parse_matpower_case(path: Path) -> dict[str, object]:
    text = path.read_text(errors="ignore")
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([^;]+);", text)
    if not base_match:
        raise RuntimeError(f"missing mpc.baseMVA in {path}")

    return {
        "version": "2",
        "baseMVA": float(base_match.group(1)),
        "bus": parse_matrix(text, "bus"),
        "gen": parse_matrix(text, "gen"),
        "branch": parse_matrix(text, "branch"),
        "gencost": parse_matrix(text, "gencost"),
    }


def write_complex_txt(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values).reshape(-1)
    matrix = np.column_stack((values.real, values.imag))
    np.savetxt(path, matrix, fmt="%.18e")


def write_int_txt(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, np.asarray(values, dtype=np.int32), fmt="%d")


def convert_case(mat_path: Path, output_root: Path) -> None:
    ppc = parse_matpower_case(mat_path)

    branch = np.asarray(ppc["branch"])
    needed_cols = QT + 1
    if branch.shape[1] < needed_cols:
        ppc["branch"] = c_[branch, zeros((branch.shape[0], needed_cols - branch.shape[1]))]

    ppc = ext2int(ppc)
    base_mva = float(np.asarray(ppc["baseMVA"]).squeeze())
    bus = np.asarray(ppc["bus"])
    gen = np.asarray(ppc["gen"])
    branch = np.asarray(ppc["branch"])

    _, pv, pq = bustypes(bus, gen)
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
    sbus = np.asarray(makeSbus(base_mva, bus, gen), dtype=np.complex128).reshape(-1)

    case_dir = output_root / mat_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    mmwrite(case_dir / "dump_Ybus.mtx", ybus)
    write_complex_txt(case_dir / "dump_Sbus.txt", sbus)
    write_complex_txt(case_dir / "dump_V.txt", v0)
    write_int_txt(case_dir / "dump_pv.txt", pv)
    write_int_txt(case_dir / "dump_pq.txt", pq)

    print(
        f"{mat_path.stem}: n_bus={ybus.shape[0]} nnz={ybus.nnz} "
        f"n_pv={len(pv)} n_pq={len(pq)}"
    )


def main() -> None:
    args = parse_args()
    if args.case == "all":
        cases = sorted(args.input_root.glob("*.m"))
    else:
        cases = [args.input_root / f"{args.case}.m"]

    if not cases:
        raise RuntimeError(f"no .m cases found under {args.input_root}")

    for mat_path in cases:
        if not mat_path.exists():
            raise RuntimeError(f"case file not found: {mat_path}")
        convert_case(mat_path, args.output_root)


if __name__ == "__main__":
    main()
