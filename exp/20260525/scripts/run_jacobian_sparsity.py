#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

DEFAULT_CASES = [
    "case_SyntheticUSA",
    "case_ACTIVSg70k",
    "case_ACTIVSg25k",
    "case13659pegase",
    "case_ACTIVSg10k",
    "case9241pegase",
    "case300",
]

DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "20260525-jacobian-sparsity"
DEFAULT_DUMP_ROOT = EXP_ROOT / "raw" / "jacobian_dumps"
DEFAULT_RESULTS_CSV = EXP_ROOT / "results" / "jacobian_sparsity.csv"
DEFAULT_RESULTS_MD = EXP_ROOT / "results" / "jacobian_sparsity.md"
DEFAULT_LOG_DIR = EXP_ROOT / "logs"
DEFAULT_CUDSS_LIB_DIR = Path(
    "/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib"
)
DEFAULT_CUDSS_THREADING_LIB = DEFAULT_CUDSS_LIB_DIR / "libcudss_mtlayer_gomp.so"


@dataclass(frozen=True)
class CsrStats:
    rows: int
    cols: int
    stored_nnz: int
    numeric_nnz: int | None
    numeric_zero_count: int | None


@dataclass(frozen=True)
class CaseResult:
    case: str
    n_bus: int
    n_pv: int
    n_pq: int
    ybus_nnz: int
    rows: int
    cols: int
    stored_nnz: int
    numeric_nnz: int | None
    numeric_zero_count: int | None
    density: float
    sparsity: float
    numeric_density: float | None
    numeric_sparsity: float | None
    dump_path: Path
    returncode: int
    success: str
    iterations: str
    final_mismatch: str


RUN_RE = re.compile(
    r"RUN .*?success=(?P<success>\S+) .*?iterations=(?P<iterations>\S+) "
    r".*?final_mismatch=(?P<final_mismatch>\S+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump cuPF MATPOWER Jacobians and measure structural sparsity."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS_MD)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--profile", default="cuda_mixed_edge")
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-numeric-count", action="store_true")
    parser.add_argument("--numeric-zero-tol", type=float, default=0.0)
    parser.add_argument("--cudss-lib-dir", type=Path, default=DEFAULT_CUDSS_LIB_DIR)
    parser.add_argument("--cudss-threading-lib", type=Path, default=DEFAULT_CUDSS_THREADING_LIB)
    parser.add_argument("--no-cudss-threading-env", action="store_true")
    parser.add_argument("--no-cudss-matching", action="store_true")
    return parser.parse_args()


def benchmark_binary(args: argparse.Namespace) -> Path:
    if args.binary is not None:
        return args.binary
    return args.build_dir / "benchmarks" / "cupf_case_benchmark"


def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def build_benchmark(args: argparse.Namespace) -> Path:
    binary = benchmark_binary(args)
    if args.skip_build:
        if not binary.exists():
            raise FileNotFoundError(f"benchmark binary not found: {binary}")
        return binary

    configure_cmd = [
        "cmake",
        "-S",
        str(REPO_ROOT / "cuPF"),
        "-B",
        str(args.build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_BENCHMARKS=ON",
        "-DENABLE_DUMP=ON",
        "-DENABLE_LOG=OFF",
        "-DENABLE_TIMING=OFF",
        "-DENABLE_NVTX=OFF",
        "-DWITH_CUDA=ON",
        "-DCUPF_CUDSS_REORDERING_ALG=DEFAULT",
    ]
    build_cmd = [
        "cmake",
        "--build",
        str(args.build_dir),
        "--target",
        "cupf_case_benchmark",
        "-j",
        str(max(1, os.cpu_count() or 1)),
    ]
    run_checked(configure_cmd)
    run_checked(build_cmd)
    if not binary.exists():
        raise FileNotFoundError(f"benchmark build finished but binary is missing: {binary}")
    return binary


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.cudss_lib_dir.exists():
        old_ld = env.get("LD_LIBRARY_PATH", "")
        entries = [str(args.cudss_lib_dir)]
        entries.extend(item for item in old_ld.split(":") if item)
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(entries))

    if not args.no_cudss_threading_env and args.cudss_threading_lib.exists():
        lib = str(args.cudss_threading_lib)
        env["CUDSS_THREADING_LIB"] = lib
        preload_entries = [item for item in env.get("LD_PRELOAD", "").split() if item]
        if lib not in preload_entries:
            preload_entries.insert(0, lib)
        env["LD_PRELOAD"] = " ".join(preload_entries)
    return env


def load_metadata(dataset_root: Path, case: str) -> dict[str, int]:
    metadata_path = dataset_root / case / "metadata.json"
    if not metadata_path.exists():
        return {"n_bus": -1, "n_pv": -1, "n_pq": -1, "ybus_nnz": -1}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "n_bus": int(metadata.get("n_bus", -1)),
        "n_pv": int(metadata.get("n_pv", -1)),
        "n_pq": int(metadata.get("n_pq", -1)),
        "ybus_nnz": int(metadata.get("ybus_nnz", -1)),
    }


def dump_path_for(args: argparse.Namespace, case: str) -> Path:
    return args.dump_root / case / "repeat_00" / "jacobian_iter0.txt"


def run_case(args: argparse.Namespace, binary: Path, case: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    case_dir = args.dataset_root / case
    if not case_dir.exists():
        raise FileNotFoundError(f"case directory not found: {case_dir}")

    case_dump_root = args.dump_root / case
    case_dump_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--case-dir",
        str(case_dir),
        "--profile",
        args.profile,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--max-iter",
        str(args.max_iter),
        "--tolerance",
        str(args.tolerance),
        "--batch-size",
        str(args.batch_size),
        "--dump-residuals",
        "--dump-dir",
        str(case_dump_root),
    ]
    if not args.no_cudss_matching:
        cmd.extend(
            [
                "--cudss-use-matching",
                "--cudss-matching-alg",
                "ALG_5",
                "--cudss-pivot-epsilon",
                "1e-12",
            ]
        )

    print(f"[RUN] {case}", flush=True)
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / f"{case}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.log_dir / f"{case}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    return completed


def parse_run_line(stdout: str) -> tuple[str, str, str]:
    for line in stdout.splitlines():
        match = RUN_RE.search(line)
        if match:
            return (
                match.group("success"),
                match.group("iterations"),
                match.group("final_mismatch"),
            )
    return "", "", ""


def parse_csr_stats(path: Path, count_numeric: bool, zero_tol: float) -> CsrStats:
    rows = cols = stored_nnz = -1
    numeric_nnz: int | None = None
    numeric_zero_count: int | None = None

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("rows "):
                rows = int(line.split()[1])
            elif line.startswith("cols "):
                cols = int(line.split()[1])
            elif line.startswith("nnz "):
                stored_nnz = int(line.split()[1])
            elif count_numeric and line.startswith("values"):
                raw = line.partition(" ")[2]
                values = np.fromstring(raw, sep=" ", dtype=np.float64)
                if values.size != stored_nnz:
                    raise RuntimeError(
                        f"{path}: values count {values.size} does not match nnz {stored_nnz}"
                    )
                if zero_tol == 0.0:
                    numeric_nnz = int(np.count_nonzero(values))
                else:
                    numeric_nnz = int(np.count_nonzero(np.abs(values) > zero_tol))
                numeric_zero_count = int(stored_nnz - numeric_nnz)

    if rows <= 0 or cols <= 0 or stored_nnz < 0:
        raise RuntimeError(f"incomplete CSR header: {path}")
    return CsrStats(rows, cols, stored_nnz, numeric_nnz, numeric_zero_count)


def measure_case(args: argparse.Namespace, binary: Path, case: str, env: dict[str, str]) -> CaseResult:
    jacobian_path = dump_path_for(args, case)
    completed: subprocess.CompletedProcess[str] | None = None
    if not (args.skip_existing and jacobian_path.exists()):
        completed = run_case(args, binary, case, env)
    else:
        print(f"[SKIP] {case} existing dump={jacobian_path}", flush=True)

    if not jacobian_path.exists():
        stdout = completed.stdout if completed is not None else ""
        stderr = completed.stderr if completed is not None else ""
        raise FileNotFoundError(
            f"jacobian dump was not produced for {case}: {jacobian_path}\n"
            f"stdout tail={stdout[-1000:]}\nstderr tail={stderr[-1000:]}"
        )

    metadata = load_metadata(args.dataset_root, case)
    stats = parse_csr_stats(
        jacobian_path,
        count_numeric=not args.no_numeric_count,
        zero_tol=args.numeric_zero_tol,
    )
    total_entries = float(stats.rows) * float(stats.cols)
    density = float(stats.stored_nnz) / total_entries
    sparsity = 1.0 - density
    numeric_density = None
    numeric_sparsity = None
    if stats.numeric_nnz is not None:
        numeric_density = float(stats.numeric_nnz) / total_entries
        numeric_sparsity = 1.0 - numeric_density

    returncode = completed.returncode if completed is not None else 0
    success, iterations, final_mismatch = parse_run_line(completed.stdout if completed is not None else "")
    return CaseResult(
        case=case,
        n_bus=metadata["n_bus"],
        n_pv=metadata["n_pv"],
        n_pq=metadata["n_pq"],
        ybus_nnz=metadata["ybus_nnz"],
        rows=stats.rows,
        cols=stats.cols,
        stored_nnz=stats.stored_nnz,
        numeric_nnz=stats.numeric_nnz,
        numeric_zero_count=stats.numeric_zero_count,
        density=density,
        sparsity=sparsity,
        numeric_density=numeric_density,
        numeric_sparsity=numeric_sparsity,
        dump_path=jacobian_path,
        returncode=returncode,
        success=success,
        iterations=iterations,
        final_mismatch=final_mismatch,
    )


def write_csv(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "n_bus",
        "n_pv",
        "n_pq",
        "ybus_nnz",
        "jacobian_rows",
        "jacobian_cols",
        "jacobian_stored_nnz",
        "jacobian_numeric_nnz",
        "jacobian_numeric_zero_count",
        "density",
        "sparsity",
        "numeric_density",
        "numeric_sparsity",
        "density_percent",
        "sparsity_percent",
        "dump_path",
        "returncode",
        "success",
        "iterations",
        "final_mismatch",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case": result.case,
                    "n_bus": result.n_bus,
                    "n_pv": result.n_pv,
                    "n_pq": result.n_pq,
                    "ybus_nnz": result.ybus_nnz,
                    "jacobian_rows": result.rows,
                    "jacobian_cols": result.cols,
                    "jacobian_stored_nnz": result.stored_nnz,
                    "jacobian_numeric_nnz": result.numeric_nnz if result.numeric_nnz is not None else "",
                    "jacobian_numeric_zero_count": (
                        result.numeric_zero_count
                        if result.numeric_zero_count is not None
                        else ""
                    ),
                    "density": f"{result.density:.17e}",
                    "sparsity": f"{result.sparsity:.17e}",
                    "numeric_density": (
                        f"{result.numeric_density:.17e}"
                        if result.numeric_density is not None
                        else ""
                    ),
                    "numeric_sparsity": (
                        f"{result.numeric_sparsity:.17e}"
                        if result.numeric_sparsity is not None
                        else ""
                    ),
                    "density_percent": f"{100.0 * result.density:.12f}",
                    "sparsity_percent": f"{100.0 * result.sparsity:.12f}",
                    "dump_path": str(result.dump_path),
                    "returncode": result.returncode,
                    "success": result.success,
                    "iterations": result.iterations,
                    "final_mismatch": result.final_mismatch,
                }
            )


def fmt_int(value: int | None) -> str:
    if value is None:
        return ""
    return f"{value:,}"


def write_markdown(path: Path, results: list[CaseResult], args: argparse.Namespace, binary: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MATPOWER cuPF Jacobian Sparsity",
        "",
        f"- Date: 2026-05-25",
        f"- Dataset root: `{args.dataset_root}`",
        f"- cuPF binary: `{binary}`",
        f"- Profile: `{args.profile}`",
        f"- Dumped matrix: `jacobian_iter0.txt` from repeat 00",
        "- Structural density = stored nnz / (rows * cols)",
        "- Structural sparsity = 1 - structural density",
        "",
        "| case | buses | J dim | stored nnz | density (%) | sparsity (%) | numeric nnz | returncode |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.case} | "
            f"{result.n_bus:,} | "
            f"{result.rows:,} | "
            f"{result.stored_nnz:,} | "
            f"{100.0 * result.density:.9f} | "
            f"{100.0 * result.sparsity:.9f} | "
            f"{fmt_int(result.numeric_nnz)} | "
            f"{result.returncode} |"
        )
    lines.extend(
        [
            "",
            "## Raw Dumps",
            "",
        ]
    )
    for result in results:
        lines.append(f"- `{result.case}`: `{result.dump_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")
    args.dump_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    binary = build_benchmark(args)
    env = build_env(args)

    results: list[CaseResult] = []
    for case in args.cases:
        result = measure_case(args, binary, case, env)
        results.append(result)
        print(
            f"[OK] {case} dim={result.rows} nnz={result.stored_nnz} "
            f"density={100.0 * result.density:.9f}% "
            f"sparsity={100.0 * result.sparsity:.9f}%",
            flush=True,
        )

    write_csv(args.results_csv, results)
    write_markdown(args.results_md, results, args, binary)
    print(f"[DONE] csv={args.results_csv}", flush=True)
    print(f"[DONE] markdown={args.results_md}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
