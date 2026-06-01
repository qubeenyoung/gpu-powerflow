"""Run the pandapower.pypower Newton-Raphson baseline benchmark."""
from __future__ import annotations

import argparse
import time

from . import eval_common
from .matpower_data import load_case, solve_reference

VARIANT = "pypower-pandapower"


def run_pypower(args: argparse.Namespace) -> None:
    paths = eval_common.selected_case_paths(args)
    out = eval_common.variant_dir(args, VARIANT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SKIPPED.txt").unlink(missing_ok=True)
    eval_common.write_json(
        out / "run.json",
        eval_common.manifest(
            args,
            mode="pypower",
            variant=VARIANT,
            cases=paths,
            backend="cpu",
            compute="fp64",
            linear_solver="scipy-spsolve",
            entrypoint="pandapower.pypower",
        ),
    )

    with eval_common.CsvSink(out / "runs.csv") as sink:
        for path in paths:
            try:
                for repeat_idx in range(args.warmup + args.repeats):
                    warmup = repeat_idx < args.warmup
                    init_start = time.perf_counter()
                    case = load_case(path)
                    initialize_ms = (time.perf_counter() - init_start) * 1000.0

                    solve_start = time.perf_counter()
                    reference = solve_reference(case, args.tolerance, args.max_iter)
                    solve_ms = (time.perf_counter() - solve_start) * 1000.0
                    row = {
                        "mode": "pypower",
                        "variant": VARIANT,
                        "case_name": case.case_name,
                        "case_path": str(path),
                        "backend": "cpu",
                        "compute": "fp64",
                        "linear_solver": "scipy-spsolve",
                        "entrypoint": "pandapower.pypower",
                        "repeat_idx": repeat_idx - args.warmup if not warmup else repeat_idx,
                        "warmup": eval_common.bool_for_csv(warmup),
                        "success": "1",
                        "converged": eval_common.bool_for_csv(reference.converged),
                        "iterations": int(reference.iterations),
                        "error_message": "",
                        **eval_common.dimensions(case),
                        "initialize_ms": initialize_ms,
                        "solve_ms": solve_ms,
                        "total_ms": initialize_ms + solve_ms,
                        "output_mismatch": float(reference.final_mismatch),
                        **eval_common.reference_fields(reference),
                    }
                    sink.write(row)
                    if not warmup:
                        print(
                            f"[{VARIANT}][OK] {case.case_name} repeat={row['repeat_idx']} "
                            f"init_ms={initialize_ms:.3f} solve_ms={solve_ms:.3f} "
                            f"iters={reference.iterations} resid={reference.final_mismatch:.3e}",
                            flush=True,
                        )
            except Exception as exc:
                sink.write(
                    {
                        "mode": "pypower",
                        "variant": VARIANT,
                        "case_name": path.stem,
                        "case_path": str(path),
                        "backend": "cpu",
                        "compute": "fp64",
                        "linear_solver": "scipy-spsolve",
                        "entrypoint": "pandapower.pypower",
                        "success": "0",
                        "converged": "0",
                        "error_message": repr(exc),
                    }
                )
                print(f"[{VARIANT}][FAIL] {path}: {exc}", flush=True)
                if not args.continue_on_error:
                    raise
    print(f"[{VARIANT}] wrote {out / 'runs.csv'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pandapower.pypower baseline benchmark.")
    eval_common.add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    eval_common.run_root(args).mkdir(parents=True, exist_ok=True)
    run_pypower(args)
    if not args.no_aggregate:
        from .aggregate_results import aggregate

        aggregate(eval_common.run_root(args))


if __name__ == "__main__":
    main()
