from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from ..converters.common import TARGET_CASES, case_stem, write_json
from .runpf import my_runpf
from .timer import summarize_entries


DEFAULT_RESULTS_ROOT = Path("/workspace/exp/pypower_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PYPOWER on workspace MATPOWER .mat cases.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--cases", nargs="*", default=list(TARGET_CASES))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--emit-timing-log", action="store_true")
    return parser.parse_args()


def flatten_summary(summary: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for key, values in summary.items():
        for metric_name, metric_value in values.items():
            flattened[f"{key}.{metric_name}"] = metric_value
    return flattened


def main() -> None:
    args = parse_args()
    run_root = args.results_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(run_root),
        "cases": list(args.cases),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "emit_timing_log": args.emit_timing_log,
    }
    write_json(run_root / "manifest.json", manifest)

    rows: list[dict[str, object]] = []

    for case_name in args.cases:
        resolved_case_stem = case_stem(case_name)

        for _ in range(args.warmup):
            my_runpf(
                casedata=case_name,
                log_pf=True,
                log_newtonpf=True,
                print_results=False,
                emit_timing_log=False,
                emit_status=False,
            )

        for repeat_idx in range(args.repeats):
            result = my_runpf(
                casedata=case_name,
                log_pf=True,
                log_newtonpf=True,
                print_results=False,
                emit_timing_log=args.emit_timing_log,
                emit_status=False,
            )
            summary = summarize_entries(result.timing_entries)
            flat_summary = flatten_summary(summary)

            row = {
                "case_stem": resolved_case_stem,
                "case_path": str(result.case_path),
                "repeat_idx": repeat_idx,
                "success": result.success,
                "iterations": result.iterations,
                "elapsed_sec": result.elapsed_sec,
                "final_mismatch": result.final_mismatch,
                **flat_summary,
            }
            rows.append(row)

            case_dir = run_root / resolved_case_stem
            case_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                case_dir / f"run_{repeat_idx:02d}.json",
                {
                    "summary": row,
                    "timing_entries": [
                        {
                            "tag": entry.tag,
                            "op_name": entry.op_name,
                            "iter_idx": entry.iter_idx,
                            "elapsed_sec": entry.elapsed_sec,
                        }
                        for entry in result.timing_entries
                    ],
                },
            )
            print(
                f"[OK] case={resolved_case_stem} repeat={repeat_idx} success={result.success} "
                f"iterations={result.iterations} elapsed_sec={result.elapsed_sec:.6f}"
            )

    all_keys = sorted({key for row in rows for key in row.keys()})
    with (run_root / "runs.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
