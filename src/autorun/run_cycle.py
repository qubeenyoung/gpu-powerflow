#!/usr/bin/env python3
"""Mechanical executor for a single autorun cycle (PLAN §3.2).

This runs the deterministic part of a cycle: (optional) build, benchmark the
active matrix set, parse the CSV, update STATE.json, and render the
benchmark_report. The *planning* and *editing* steps (§3.2 step 2-3) and the
relaying of reports to Discord are performed by the Claude Code agent, not here.

Exit code is non-zero on accuracy violation so a caller can gate a commit.
"""

import argparse
import math
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_csv  # noqa: E402
import discord_report as dr  # noqa: E402
import state as state_mod  # noqa: E402

BUILD_DIR = "/tmp/profile-build"
DEFAULT_BENCH = os.path.join(BUILD_DIR, "benchmark")


def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="PLAN.md")
    ap.add_argument("--state", required=True)
    ap.add_argument("--bench", default=DEFAULT_BENCH)
    ap.add_argument("--solvers", default="klu,umfpack,cudss-gpu,mysolver")
    ap.add_argument("--matrix-set", default=None,
                    help="override the active_matrix_set from STATE")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--warmup-gpu", action="store_true", default=True)
    args = ap.parse_args()

    st = state_mod.load(args.state)
    cycle = st["cycle_count"] + 1
    milestone = st["current_milestone"]
    status = st["milestone_status"]
    mset = args.matrix_set or st["active_matrix_set"]
    solvers = args.solvers.split(",")

    if not args.no_build:
        run(["cmake", "--build", BUILD_DIR, "--target", "benchmark", "-j"])

    out_csv = f"report/benchmark/cycle_{cycle:04d}.csv"
    if not args.no_bench:
        # active_matrix_set may be a known set name (smoke/suitesparse/...) or an
        # explicit comma-separated matrix list (overrides --matrix-set).
        selector = ["--matrices", mset] if "," in mset else ["--matrix-set", mset]
        cmd = [args.bench, *selector, "--solver", ",".join(solvers), "--output", out_csv]
        if args.warmup_gpu:
            cmd.append("--warmup-gpu")
        run(cmd)

    prev_best = dict(st.get("best_metrics_per_case", {}))
    res = analyze_csv.analyze(out_csv, target="mysolver", best=prev_best, milestone=milestone)

    # Improve-only update of best metrics for the target solver.
    best = st.get("best_metrics_per_case", {})
    for m, mm in res["target_metrics"].items():
        if not mm["success"]:
            continue
        cur = best.get(m, {})
        for k in ("factor_ms", "solve_ms"):
            v = mm[k]
            if not math.isnan(v) and (k not in cur or v < cur[k]):
                cur[k] = v
        cur["berr"] = mm["berr"]
        best[m] = cur
    st["best_metrics_per_case"] = best

    st["cycle_count"] = cycle
    ts = state_mod.utc_now()
    st["last_cycle_at"] = ts

    report = dr.benchmark_report(cycle, milestone, status, ts, mset, solvers, res,
                                 out_csv, have_prev_best=bool(prev_best))
    st["last_report_at"] = ts
    state_mod.save(args.state, st)

    print("\n===BENCHMARK_REPORT===")
    print(report)
    print("===END===")

    if not res["accuracy_ok"]:
        print("ACCURACY_VIOLATION", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
