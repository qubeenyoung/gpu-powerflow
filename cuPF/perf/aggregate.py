#!/usr/bin/env python3
"""Aggregate timing.csv from cupf_cpp_evaluate into a per-stage breakdown."""
import csv
import sys
from collections import defaultdict

STAGES = [
    "NR.iteration.ibus", "NR.iteration.mismatch", "NR.iteration.mismatch_norm",
    "NR.iteration.jacobian", "NR.iteration.prepare_rhs", "NR.iteration.factorize",
    "NR.iteration.solve", "NR.iteration.voltage_update",
    "NR.solve.upload", "NR.solve.download", "NR.solve.total",
]


def load(path):
    # (case, stage) -> list of total_us across repeats (solve phase only)
    acc = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["phase"] != "solve":
                continue
            acc[(row["case_name"], row["timer_name"])].append(float(row["total_us"]))
    return acc


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    path = sys.argv[1]
    acc = load(path)
    cases = sorted({c for (c, _) in acc}, key=lambda c: avg(acc.get((c, "NR.solve.total"), [0])))
    print(f"\n=== {path} : per-stage solve time (us, avg over repeats) ===")
    hdr = ["stage"] + cases
    print("{:<28}".format(hdr[0]) + "".join(f"{c:>16}" for c in cases))
    for st in STAGES:
        label = st.replace("NR.iteration.", "").replace("NR.solve.", "*")
        cells = []
        for c in cases:
            t = avg(acc.get((c, st), [0]))
            tot = avg(acc.get((c, "NR.solve.total"), [1]))
            pct = 100.0 * t / tot if tot else 0.0
            cells.append(f"{t:8.0f}({pct:4.1f}%)")
        print("{:<28}".format(label) + "".join(f"{x:>16}" for x in cells))
    # linear solve share
    print("-" * 28)
    for c in cases:
        fac = avg(acc.get((c, "NR.iteration.factorize"), [0]))
        sol = avg(acc.get((c, "NR.iteration.solve"), [0]))
        tot = avg(acc.get((c, "NR.solve.total"), [1]))
        print(f"  {c:<24} linear_solve(factorize+solve) = {fac+sol:8.0f} us / {tot:8.0f} = {100*(fac+sol)/tot:5.1f}%")


if __name__ == "__main__":
    main()
