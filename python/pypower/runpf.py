from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from sys import stdout
from time import perf_counter

import numpy as np
from numpy import c_, exp, ones, pi, zeros
from numpy import flatnonzero as find
from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.int2ext import int2ext
from pypower.loadcase import loadcase
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from pypower.pfsoln import pfsoln
from pypower.ppoption import ppoption
from pypower.printpf import printpf
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_bus import VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG

from ..converters.common import case_stem, mat_case_path
from .newtonpf import NewtonPfResult, my_newtonpf
from .timer import BlockTimer, TimingEntry, TimingLog


@dataclass
class RunPfResult:
    case_stem: str
    case_path: Path
    success: bool
    iterations: int
    elapsed_sec: float
    final_mismatch: float
    timing_entries: list[TimingEntry]
    results_ppc: dict | None = None


def my_runpf(
    casedata: str | None = None,
    log_pf: bool = False,
    log_newtonpf: bool = False,
    ppopt=None,
    print_results: bool = False,
    emit_timing_log: bool = True,
    emit_status: bool = True,
    timing_log: TimingLog | None = None,
) -> RunPfResult:
    if casedata is None:
        casedata = "118_ieee"

    case_path = mat_case_path(casedata)
    resolved_case_stem = case_stem(casedata)

    ppopt = ppoption(ppopt)
    ppopt["TIMING"] = log_newtonpf

    if timing_log is None:
        timing_log = TimingLog(enabled=(log_pf or log_newtonpf), emit_log=emit_timing_log)

    with BlockTimer(timing_log if log_pf else None, "runpf", "load_case_data", 0):
        ppc = loadcase(str(case_path))

    if ppc["branch"].shape[1] < QT + 1:
        ppc["branch"] = c_[
            ppc["branch"],
            zeros((ppc["branch"].shape[0], QT - ppc["branch"].shape[1] + 1)),
        ]

    with BlockTimer(timing_log if log_pf else None, "runpf", "bus_indexing", 0):
        ppc = ext2int(ppc)
        baseMVA, bus, gen, branch = (
            ppc["baseMVA"],
            ppc["bus"],
            ppc["gen"],
            ppc["branch"],
        )
        ref, pv, pq = bustypes(bus, gen)
        on = find(gen[:, GEN_STATUS] > 0)
        gbus = gen[on, GEN_BUS].astype(int)

    wall_start = perf_counter()

    V0 = bus[:, VM] * exp(1j * pi / 180.0 * bus[:, VA])
    vcb = ones(V0.shape)
    vcb[pq] = 0
    k = find(vcb[gbus])
    if len(k):
        V0[gbus[k]] = gen[on[k], VG] / np.abs(V0[gbus[k]]) * V0[gbus[k]]

    with BlockTimer(timing_log if log_pf else None, "runpf", "build_Ybus", 0):
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    with BlockTimer(timing_log if log_pf else None, "runpf", "build_Sbus", 0):
        Sbus = makeSbus(baseMVA, bus, gen)

    with BlockTimer(timing_log if log_pf else None, "runpf", "newtonpf", 0):
        newton_result: NewtonPfResult = my_newtonpf(
            Ybus,
            Sbus,
            V0,
            ref,
            pv,
            pq,
            ppopt=ppopt,
            timing_log=timing_log if log_newtonpf else None,
            emit_status=emit_status,
        )

    bus, gen, branch = pfsoln(
        baseMVA,
        bus,
        gen,
        branch,
        Ybus,
        Yf,
        Yt,
        newton_result.V,
        ref,
        pv,
        pq,
    )

    ppc["et"] = perf_counter() - wall_start
    ppc["success"] = int(newton_result.converged)
    ppc["bus"], ppc["gen"], ppc["branch"] = bus, gen, branch
    results = int2ext(ppc)

    if len(results["order"]["gen"]["status"]["off"]) > 0:
        results["gen"][np.ix_(results["order"]["gen"]["status"]["off"], [PG, QG])] = 0

    if len(results["order"]["branch"]["status"]["off"]) > 0:
        results["branch"][np.ix_(results["order"]["branch"]["status"]["off"], [PF, QF, PT, QT])] = 0

    if print_results:
        printpf(results, stdout, ppopt)

    return RunPfResult(
        case_stem=resolved_case_stem,
        case_path=case_path,
        success=bool(newton_result.converged),
        iterations=newton_result.iterations,
        elapsed_sec=float(ppc["et"]),
        final_mismatch=newton_result.final_mismatch,
        timing_entries=list(timing_log.entries),
        results_ppc=results if print_results else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PYPOWER Newton-Raphson on a workspace MATPOWER .mat case.")
    parser.add_argument("case", nargs="?", default="118_ieee")
    parser.add_argument("--timing", action="store_true", help="Emit runpf and newtonpf timing logs.")
    parser.add_argument("--print-results", action="store_true", help="Print the full power-flow report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = my_runpf(
        casedata=args.case,
        log_pf=args.timing,
        log_newtonpf=args.timing,
        print_results=args.print_results,
        emit_timing_log=args.timing,
        emit_status=True,
    )
    print(
        f"case={result.case_stem} success={result.success} "
        f"iterations={result.iterations} elapsed_sec={result.elapsed_sec:.6f} "
        f"mismatch={result.final_mismatch:.6e}"
    )


if __name__ == "__main__":
    main()
