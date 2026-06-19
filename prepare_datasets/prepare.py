"""`prepare` CLI: parse MATPOWER `.m` cases, build Ybus/Sbus/V0/PV/PQ, solve a
SciPy reference Newton PF, and write cuPF dump directories. Pure pandapower side;
no cuPF extension is imported.

    python3 -m prepare_datasets.prepare --dataset-root /datasets/matpower --cases case9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.common.eval_common import (
    add_common_args,
    load_case_and_reference,
    manifest,
    selected_case_paths,
    variant_dir,
)
from benchmark.common.matpower_data import save_dump_case, write_manifest_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "prepare_datasets" / "results"


def prepare_command(args: argparse.Namespace) -> None:
    cases = selected_case_paths(args)
    out = variant_dir(args, "prepare")
    dump_root = args.dump_root if args.dump_root is not None else out / "dumps"
    rows: list[dict[str, object]] = []
    for path in cases:
        try:
            case, reference = load_case_and_reference(path, args)
            case_dir = save_dump_case(case, dump_root, reference)
            rows.append(
                {
                    "case_name": case.case_name,
                    "case_path": str(path),
                    "dump_dir": str(case_dir),
                    "success": True,
                    "error_message": "",
                    "n_bus": int(case.ybus.shape[0]),
                    "ybus_nnz": int(case.ybus.nnz),
                    "n_ref": int(case.ref.size),
                    "n_pv": int(case.pv.size),
                    "n_pq": int(case.pq.size),
                    "reference_converged": reference.converged,
                    "reference_iterations": reference.iterations,
                    "reference_final_mismatch": reference.final_mismatch,
                }
            )
            print(f"[prepare][OK] {case.case_name} -> {case_dir}", flush=True)
        except Exception as exc:
            rows.append({"case_name": path.stem, "case_path": str(path), "success": False, "error_message": repr(exc)})
            print(f"[prepare][FAIL] {path}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
    write_manifest_csv(out / "manifest.csv", rows)
    (out / "run.json").write_text(json.dumps(manifest(args, "prepare", cases), indent=2, sort_keys=True), encoding="utf-8")
    print(f"[prepare] wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert MATPOWER .m cases to cuPF dump directories.")
    add_common_args(parser)
    parser.add_argument("--dump-root", type=Path, default=None, help="Override the dump output directory.")
    parser.set_defaults(func=prepare_command, output_root=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
