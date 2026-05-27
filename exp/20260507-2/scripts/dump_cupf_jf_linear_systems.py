#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_DUMP_ROOT = EXP_ROOT / "raw" / "cupf_jf_dumps"
DEFAULT_BENCHMARK_BINARY = (
    REPO_ROOT.parent
    / "gpu-powerflow-master"
    / "cuPF"
    / "build"
    / "bench-end2end-superlu-cudss-mt-auto-dump"
    / "benchmarks"
    / "cupf_case_benchmark"
)
DEFAULT_CUDSS_THREADING_LIB = Path(
    "/root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so"
)


@dataclass(frozen=True)
class CaseInfo:
    name: str
    n_bus: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump cuPF Newton-Raphson linear systems Jk/Fk for MATPOWER cases. "
            "cuPF iter1 corresponds to the second Newton iteration, so J1/F1 are "
            "jacobian_iter1.txt and residual_iter1.txt."
        )
    )
    parser.add_argument("--benchmark-binary", type=Path, default=DEFAULT_BENCHMARK_BINARY)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--cases", nargs="*", help="Explicit case names to dump.")
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--profile", default="cuda_mixed_edge")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Default 0.0 forces cuPF to run to max_iter so later Jk/Fk files exist.",
    )
    parser.add_argument("--min-buses", type=int, default=1000)
    parser.add_argument("--max-buses", type=int, default=10000)
    parser.add_argument("--limit", type=int, help="Dump only the first N selected cases.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-aliases", action="store_true")
    parser.add_argument("--cudss-device-count", type=int)
    parser.add_argument("--cudss-threading-lib", type=Path, default=DEFAULT_CUDSS_THREADING_LIB)
    parser.add_argument("--no-cudss-threading-env", action="store_true")
    return parser.parse_args()


def metadata_for_case(dataset_root: Path, case_name: str) -> dict[str, object]:
    metadata_path = dataset_root / case_name / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def case_info(dataset_root: Path, case_name: str) -> CaseInfo:
    metadata = metadata_for_case(dataset_root, case_name)
    n_bus = int(metadata.get("n_bus", -1))
    return CaseInfo(name=case_name, n_bus=n_bus)


def load_case_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_cases(dataset_root: Path, min_buses: int, max_buses: int) -> list[CaseInfo]:
    cases: list[CaseInfo] = []
    for metadata_path in sorted(dataset_root.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        n_bus = int(metadata.get("n_bus", -1))
        if min_buses <= n_bus <= max_buses:
            cases.append(CaseInfo(name=metadata_path.parent.name, n_bus=n_bus))
    return sorted(cases, key=lambda item: (item.n_bus, item.name))


def select_cases(args: argparse.Namespace) -> list[CaseInfo]:
    if args.cases:
        cases = [case_info(args.dataset_root, name) for name in args.cases]
    elif args.case_list:
        cases = [case_info(args.dataset_root, name) for name in load_case_list(args.case_list)]
        cases = [
            item
            for item in cases
            if item.n_bus < 0 or args.min_buses <= item.n_bus <= args.max_buses
        ]
    else:
        cases = discover_cases(args.dataset_root, args.min_buses, args.max_buses)
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def binary_supports_arg(binary: Path, arg: str) -> bool:
    completed = subprocess.run(
        [str(binary), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    return arg in completed.stdout or arg in completed.stderr


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.no_cudss_threading_env:
        return env
    if args.cudss_threading_lib.exists():
        lib = str(args.cudss_threading_lib)
        env["CUDSS_THREADING_LIB"] = lib
        preload = env.get("LD_PRELOAD", "")
        preload_items = [item for item in preload.split() if item]
        if lib not in preload_items:
            preload_items.insert(0, lib)
        env["LD_PRELOAD"] = " ".join(preload_items)
    return env


def expected_files_exist(case_dump_dir: Path, max_iter: int) -> bool:
    repeat_dir = case_dump_dir / "repeat_00"
    last_iter = max_iter - 1
    return (
        (repeat_dir / f"jacobian_iter{last_iter}.txt").exists()
        and (repeat_dir / f"residual_iter{last_iter}.txt").exists()
        and (repeat_dir / f"dx_iter{last_iter}.txt").exists()
    )


def run_case(
    args: argparse.Namespace,
    case: CaseInfo,
    supports_cudss_device_count: bool,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    case_dir = args.dataset_root / case.name
    dump_dir = args.dump_root / case.name
    dump_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.benchmark_binary),
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
        "--dump-residuals",
        "--dump-dir",
        str(dump_dir),
    ]
    if args.cudss_device_count is not None and supports_cudss_device_count:
        cmd.extend(["--cudss-device-count", str(args.cudss_device_count)])

    if args.dry_run:
        print(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)


def available_iterations(case_dump_dir: Path) -> list[int]:
    repeat_dir = case_dump_dir / "repeat_00"
    iterations: list[int] = []
    pattern = re.compile(r"jacobian_iter(\d+)\.txt$")
    for jacobian_path in sorted(repeat_dir.glob("jacobian_iter*.txt")):
        match = pattern.search(jacobian_path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        if (repeat_dir / f"residual_iter{iteration}.txt").exists():
            iterations.append(iteration)
    return iterations


def available_named_iterations(case_dump_dir: Path, prefix: str) -> list[int]:
    repeat_dir = case_dump_dir / "repeat_00"
    iterations: list[int] = []
    pattern = re.compile(rf"{re.escape(prefix)}_iter(\d+)\.txt$")
    for path in sorted(repeat_dir.glob(f"{prefix}_iter*.txt")):
        match = pattern.search(path.name)
        if match:
            iterations.append(int(match.group(1)))
    return iterations


def make_aliases(
    case_dump_dir: Path,
    j_iterations: list[int],
    f_iterations: list[int],
    dx_iterations: list[int],
) -> None:
    repeat_dir = case_dump_dir / "repeat_00"
    pairs: list[tuple[Path, Path]] = []
    for iteration in j_iterations:
        pairs.append((repeat_dir / f"jacobian_iter{iteration}.txt", case_dump_dir / f"J{iteration}.txt"))
    for iteration in f_iterations:
        pairs.append((repeat_dir / f"residual_iter{iteration}.txt", case_dump_dir / f"F{iteration}.txt"))
    for iteration in dx_iterations:
        pairs.append((repeat_dir / f"dx_iter{iteration}.txt", case_dump_dir / f"dx{iteration}.txt"))
    for target, link in pairs:
        if not target.exists():
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        relative_target = os.path.relpath(target, link.parent)
        os.symlink(relative_target, link)


def read_csr_header(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return (-1, -1, -1)
    tokens: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            tokens.extend(line.split())
            if len(tokens) >= 8:
                break
    if len(tokens) < 8 or tokens[0] != "type" or tokens[1] != "csr_matrix":
        return (-1, -1, -1)
    return (int(tokens[3]), int(tokens[5]), int(tokens[7]))


def path_or_empty(path: Path) -> str:
    return str(path) if path.exists() or path.is_symlink() else ""


def summary_row(args: argparse.Namespace, case: CaseInfo, returncode: int) -> dict[str, object]:
    case_dump_dir = args.dump_root / case.name
    iterations = available_iterations(case_dump_dir)
    f_iterations = available_named_iterations(case_dump_dir, "residual")
    dx_iterations = available_named_iterations(case_dump_dir, "dx")
    if not args.no_aliases:
        make_aliases(case_dump_dir, iterations, f_iterations, dx_iterations)

    j1_path = case_dump_dir / "J1.txt"
    f1_path = case_dump_dir / "F1.txt"
    j2_path = case_dump_dir / "J2.txt"
    f2_path = case_dump_dir / "F2.txt"
    dx1_path = case_dump_dir / "dx1.txt"
    dx2_path = case_dump_dir / "dx2.txt"
    rows, cols, nnz = read_csr_header(case_dump_dir / "repeat_00" / "jacobian_iter1.txt")
    return {
        "case_name": case.name,
        "n_bus": case.n_bus,
        "linear_dim": rows,
        "linear_nnz": nnz,
        "returncode": returncode,
        "dump_dir": str(case_dump_dir),
        "available_iterations": " ".join(str(item) for item in iterations),
        "available_F_iterations": " ".join(str(item) for item in f_iterations),
        "available_dx_iterations": " ".join(str(item) for item in dx_iterations),
        "J1": path_or_empty(j1_path),
        "F1": path_or_empty(f1_path),
        "dx1": path_or_empty(dx1_path),
        "J2": path_or_empty(j2_path),
        "F2": path_or_empty(f2_path),
        "dx2": path_or_empty(dx2_path),
    }


def main() -> None:
    args = parse_args()
    if not args.benchmark_binary.exists():
        raise FileNotFoundError(f"benchmark binary not found: {args.benchmark_binary}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

    cases = select_cases(args)
    args.dump_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.dump_root / "linear_system_dump_summary.csv"
    supports_cudss_device_count = binary_supports_arg(args.benchmark_binary, "--cudss-device-count")
    env = build_env(args)

    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        case_dump_dir = args.dump_root / case.name
        if args.skip_existing and expected_files_exist(case_dump_dir, args.max_iter):
            print(f"[SKIP] {index}/{len(cases)} {case.name} n_bus={case.n_bus}")
            rows.append(summary_row(args, case, 0))
            continue

        completed = run_case(args, case, supports_cudss_device_count, env)
        rows.append(summary_row(args, case, completed.returncode))
        status = "OK" if completed.returncode == 0 else "FAIL"
        print(f"[{status}] {index}/{len(cases)} {case.name} n_bus={case.n_bus}")
        if completed.stdout.strip():
            print(completed.stdout.strip())
        if completed.returncode != 0 and completed.stderr.strip():
            print(completed.stderr.strip())

    fieldnames = [
        "case_name",
        "n_bus",
        "linear_dim",
        "linear_nnz",
        "returncode",
        "dump_dir",
        "available_iterations",
        "available_F_iterations",
        "available_dx_iterations",
        "J1",
        "F1",
        "dx1",
        "J2",
        "F2",
        "dx2",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
