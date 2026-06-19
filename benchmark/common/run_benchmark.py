"""Cross-tool baseline comparison matrix (pypower + MATPOWER).

cuPF itself is run via ``cuPF/tests/run_cupf.py`` (the single cuPF entry point);
this harness runs the CPU/MATLAB baselines and aggregates them into runs.csv.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from . import eval_common

PYPPOWER_VARIANT = "pypower-pandapower"
MATPOWER_VARIANTS = ["matpower-default", "matpower-lu5"]


def default_variants() -> list[str]:
    return [PYPPOWER_VARIANT, *MATPOWER_VARIANTS]


def _base_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        "--dataset-root",
        str(args.dataset_root),
        "--run-name",
        args.run_name,
        "--output-root",
        str(args.output_root),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--tolerance",
        str(args.tolerance),
        "--max-iter",
        str(args.max_iter),
        "--reference-tolerance",
        str(args.reference_tolerance),
        "--reference-max-iter",
        str(args.reference_max_iter),
        "--no-aggregate",
    ]
    if args.cases:
        cmd += ["--cases", *args.cases]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    return cmd


def _run_module(module: str, extra: list[str], args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", module, *_base_args(args), *extra]
    print("[benchmark][RUN] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_benchmark(args: argparse.Namespace) -> None:
    run_dir = eval_common.run_root(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.variants or default_variants())
    eval_common.write_json(
        run_dir / "run.json",
        {
            "run_name": args.run_name,
            "dataset_root": str(args.dataset_root),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "tolerance": args.tolerance,
            "max_iter": args.max_iter,
            "variants": sorted(requested),
        },
    )

    if not args.skip_pypower and PYPPOWER_VARIANT in requested:
        _run_module("benchmark.backends.pypower", [], args)

    matpower = sorted(v for v in requested if v in MATPOWER_VARIANTS)
    if matpower and not args.skip_matlab:
        extra = ["--variants", *matpower, "--matlab-timeout-sec", str(args.matlab_timeout_sec)]
        if args.matlab_bin:
            extra += ["--matlab-bin", args.matlab_bin]
        if args.matpower_home:
            extra += ["--matpower-home", str(args.matpower_home)]
        _run_module("benchmark.backends.matpower", extra, args)
    elif matpower and args.skip_matlab:
        for variant in matpower:
            eval_common.write_skip(run_dir / variant, "skipped by --skip-matlab")

    if not args.no_aggregate:
        from .aggregate_results import aggregate

        aggregate(run_dir)
    print(f"[benchmark] done -> {run_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run representative benchmark variants and aggregate results.")
    eval_common.add_common_args(parser)
    parser.add_argument("--variants", nargs="*", help="Variant IDs. Defaults to the baseline matrix.")
    parser.add_argument("--skip-pypower", action="store_true")
    parser.add_argument("--skip-matlab", action="store_true")
    parser.add_argument("--matlab-bin", default=None)
    parser.add_argument("--matpower-home", default=None)
    parser.add_argument("--matlab-timeout-sec", type=int, default=7200)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
