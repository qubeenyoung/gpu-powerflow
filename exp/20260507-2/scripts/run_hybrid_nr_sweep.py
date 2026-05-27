#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import tempfile


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_BINARY = EXP_ROOT / "build" / "hybrid_nr_bench"
DEFAULT_CASE_ROOT = EXP_ROOT.parents[1] / "datasets" / "matpower8.1" / "cupf_all_dumps"
DEFAULT_CASES = [
    "case1197",
    "case2736sp",
    "case3375wp",
    "case6468rte",
    "case_ACTIVSg10k",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid NR GMRES block-Jacobi sweeps on representative cuPF dump cases."
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--output", type=Path, default=EXP_ROOT / "results" / "hybrid_nr_sweep.csv")
    parser.add_argument(
        "--iter-output",
        type=Path,
        default=EXP_ROOT / "results" / "hybrid_nr_sweep_iters.csv",
    )
    parser.add_argument("--mode", choices=["fixed", "rtol", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-nr-iters", type=int, default=20)
    parser.add_argument("--cudss-polish-threshold", type=float, default=1e-4)
    parser.add_argument("--no-pure-cudss-baseline", action="store_true")
    return parser.parse_args()


def append_csv(src: Path, dst: Path, write_header: bool) -> bool:
    with src.open("r", encoding="utf-8", newline="") as in_fh:
        reader = csv.reader(in_fh)
        rows = list(reader)
    if not rows:
        return write_header
    with dst.open("a", encoding="utf-8", newline="") as out_fh:
        writer = csv.writer(out_fh)
        if write_header:
            writer.writerow(rows[0])
        writer.writerows(rows[1:])
    return False


def run_one(args: argparse.Namespace, tmp_dir: Path, combo: dict[str, object]) -> tuple[Path, Path]:
    summary = tmp_dir / "summary.csv"
    iters = tmp_dir / "iters.csv"
    cmd = [
        str(args.binary),
        "--case-root",
        str(args.case_root),
        "--case",
        args.cases,
        "--solver",
        "hybrid",
        "--max-nr-iters",
        str(args.max_nr_iters),
        "--warmup",
        str(args.warmup),
        "--cudss-bootstrap-iters",
        "1",
        "--cudss-polish-threshold",
        str(args.cudss_polish_threshold),
        "--block-size",
        str(combo["block_size"]),
        "--gmres-restart",
        str(combo["restart"]),
        "--gmres-max-iters",
        str(combo["max_iters"]),
        "--gmres-rtol",
        str(combo["rtol"]),
        "--gmres-fixed-iter-mode",
        "true" if combo["fixed"] else "false",
        "--enable-cudss-fallback",
        "true",
        "--accept-iterative-by-mismatch",
        "true",
        "--output",
        str(summary),
        "--iter-output",
        str(iters),
    ]
    if args.no_pure_cudss_baseline:
        cmd.append("--no-pure-cudss-baseline")
    completed = subprocess.run(cmd, cwd=EXP_ROOT, text=True, capture_output=True, check=False)
    print(completed.stdout, end="")
    if completed.returncode != 0:
        print(completed.stderr, end="")
        raise RuntimeError(f"hybrid_nr_bench failed for combo={combo}")
    return summary, iters


def fixed_combos() -> list[dict[str, object]]:
    return [
        {
            "block_size": block_size,
            "restart": restart,
            "max_iters": max_iters,
            "rtol": 0.2,
            "fixed": True,
        }
        for block_size in (32, 64)
        for restart in (8, 16)
        for max_iters in (4, 8, 16, 32)
    ]


def rtol_combos() -> list[dict[str, object]]:
    return [
        {
            "block_size": 64,
            "restart": 16,
            "max_iters": 32,
            "rtol": rtol,
            "fixed": False,
        }
        for rtol in (1e-1, 2e-1, 3e-1, 4e-1)
    ]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.iter_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    args.iter_output.unlink(missing_ok=True)

    combos: list[dict[str, object]] = []
    if args.mode in ("fixed", "both"):
        combos.extend(fixed_combos())
    if args.mode in ("rtol", "both"):
        combos.extend(rtol_combos())

    write_summary_header = True
    write_iter_header = True
    with tempfile.TemporaryDirectory(prefix="hybrid_nr_sweep_", dir=args.output.parent) as tmp:
        tmp_dir = Path(tmp)
        for idx, combo in enumerate(combos, start=1):
            print(f"[{idx}/{len(combos)}] combo={combo}")
            summary, iters = run_one(args, tmp_dir, combo)
            write_summary_header = append_csv(summary, args.output, write_summary_header)
            write_iter_header = append_csv(iters, args.iter_output, write_iter_header)

    print(f"[DONE] summary={args.output}")
    print(f"[DONE] iters={args.iter_output}")


if __name__ == "__main__":
    main()
