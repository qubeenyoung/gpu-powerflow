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


METADATA_FIELDS = [
    "experiment",
    "polish_threshold",
    "force_gmres_min_steps",
    "gmres_iters",
    "accept_mismatch_ratio",
    "reject_mismatch_ratio",
    "damping_enabled",
    "damping_factors",
    "fallback_policy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the requested hybrid NR policy sweep.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument(
        "--output",
        type=Path,
        default=EXP_ROOT / "results" / "hybrid_nr_policy_sweep.csv",
    )
    parser.add_argument(
        "--iter-output",
        type=Path,
        default=EXP_ROOT / "results" / "hybrid_nr_policy_sweep_iters.csv",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-nr-iters", type=int, default=20)
    return parser.parse_args()


def base_combo(experiment: str) -> dict[str, object]:
    return {
        "experiment": experiment,
        "polish_threshold": "1e-4",
        "force_gmres_min_steps": 0,
        "gmres_iters": 8,
        "accept_mismatch_ratio": 0.95,
        "reject_mismatch_ratio": 1.05,
        "damping_enabled": False,
        "damping_factors": "1.0,0.5,0.25",
        "fallback_policy": "immediate",
    }


def build_combos() -> list[dict[str, object]]:
    combos: list[dict[str, object]] = []

    for threshold in ("disabled", "1e-8", "1e-6", "1e-5", "1e-4"):
        combo = base_combo("A_polish_threshold")
        combo["polish_threshold"] = threshold
        combos.append(combo)

    for min_steps in (0, 1, 2, 3):
        combo = base_combo("B_forced_middle")
        combo["force_gmres_min_steps"] = min_steps
        combo["polish_threshold"] = "1e-4"
        combos.append(combo)

    for threshold in ("disabled", "1e-6"):
        for gmres_iters in (1, 2, 4, 8, 16):
            combo = base_combo("C_gmres_fixed_iterations")
            combo["polish_threshold"] = threshold
            combo["gmres_iters"] = gmres_iters
            combos.append(combo)

    for accept in (0.99, 0.97, 0.95, 0.90):
        for reject in (1.01, 1.05, 1.10):
            combo = base_combo("D_accept_reject_policy")
            combo["accept_mismatch_ratio"] = accept
            combo["reject_mismatch_ratio"] = reject
            combos.append(combo)

    combo = base_combo("E_damping_before_fallback")
    combo["damping_enabled"] = True
    combo["damping_factors"] = "1.0,0.5,0.25"
    combos.append(combo)

    for policy in ("off", "immediate", "after_two_failures"):
        combo = base_combo("F_fallback_policy")
        combo["fallback_policy"] = policy
        combos.append(combo)

    return combos


def run_binary(args: argparse.Namespace,
               tmp_dir: Path,
               combo: dict[str, object],
               output_name: str,
               solver: str = "hybrid") -> tuple[Path, Path, str]:
    summary = tmp_dir / f"{output_name}.csv"
    iters = tmp_dir / f"{output_name}_iters.csv"
    cmd = [
        str(args.binary),
        "--case-root",
        str(args.case_root),
        "--case",
        args.cases,
        "--solver",
        solver,
        "--warmup",
        str(args.warmup),
        "--max-nr-iters",
        str(args.max_nr_iters),
        "--cudss-bootstrap-iters",
        "1",
        "--cudss-polish-threshold",
        str(combo["polish_threshold"]),
        "--force-gmres-min-steps",
        str(combo["force_gmres_min_steps"]),
        "--block-size",
        "64",
        "--gmres-restart",
        "16",
        "--gmres-max-iters",
        str(combo["gmres_iters"]),
        "--gmres-fixed-iter-mode",
        "true",
        "--fallback-policy",
        str(combo["fallback_policy"]),
        "--accept-mismatch-ratio",
        str(combo["accept_mismatch_ratio"]),
        "--reject-mismatch-ratio",
        str(combo["reject_mismatch_ratio"]),
        "--enable-damped-iterative-step",
        "true" if combo["damping_enabled"] else "false",
        "--damping-factors",
        str(combo["damping_factors"]),
        "--no-pure-cudss-baseline",
        "--output",
        str(summary),
        "--iter-output",
        str(iters),
    ]
    completed = subprocess.run(cmd, cwd=EXP_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="")
        raise RuntimeError(f"hybrid_nr_bench failed for {combo}")
    return summary, iters, completed.stdout


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path,
               metadata_fields: list[str],
               data_fields: list[str],
               rows: list[dict[str, str]],
               append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=metadata_fields + data_fields)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def decorate_rows(rows: list[dict[str, str]],
                  combo: dict[str, object],
                  pure_times: dict[str, float],
                  update_pure_columns: bool) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        case_name = row["case_name"]
        if update_pure_columns:
            pure_time = pure_times.get(case_name, 0.0)
            total_time = float(row.get("total_seconds", "0") or 0.0)
            row["pure_cudss_total_seconds"] = f"{pure_time:.12g}"
            row["speedup_vs_pure_cudss"] = (
                f"{(pure_time / total_time):.12g}" if total_time > 0 else "0"
            )
        decorated = {field: str(combo[field]) for field in METADATA_FIELDS}
        decorated.update(row)
        out.append(decorated)
    return out


def summarize_stdout(stdout: str) -> None:
    for line in stdout.splitlines():
        if line.startswith("[result]"):
            print(line)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.iter_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    args.iter_output.unlink(missing_ok=True)

    combos = build_combos()
    with tempfile.TemporaryDirectory(prefix="hybrid_nr_policy_", dir=args.output.parent) as tmp:
        tmp_dir = Path(tmp)
        pure_combo = base_combo("pure_cudss_baseline")
        pure_summary, _, pure_stdout = run_binary(args, tmp_dir, pure_combo, "pure", solver="pure_cudss")
        _, pure_rows = read_rows(pure_summary)
        pure_times = {row["case_name"]: float(row["total_seconds"]) for row in pure_rows}
        print("[baseline] pure cuDSS")
        summarize_stdout(pure_stdout)

        summary_fields: list[str] | None = None
        iter_fields: list[str] | None = None

        for idx, combo in enumerate(combos, start=1):
            print(f"[{idx}/{len(combos)}] {combo}")
            summary, iters, stdout = run_binary(args, tmp_dir, combo, f"run_{idx:03d}")
            summarize_stdout(stdout)

            fields, rows = read_rows(summary)
            decorated = decorate_rows(rows, combo, pure_times, update_pure_columns=True)
            if summary_fields is None:
                summary_fields = fields
            write_rows(args.output, METADATA_FIELDS, summary_fields, decorated, append=idx > 1)

            iter_read_fields, iter_rows = read_rows(iters)
            decorated_iters = decorate_rows(iter_rows, combo, pure_times, update_pure_columns=False)
            if iter_fields is None:
                iter_fields = iter_read_fields
            write_rows(args.iter_output,
                       METADATA_FIELDS,
                       iter_fields,
                       decorated_iters,
                       append=idx > 1)

    print(f"[DONE] summary={args.output}")
    print(f"[DONE] iters={args.iter_output}")


if __name__ == "__main__":
    main()
