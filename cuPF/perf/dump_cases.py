#!/usr/bin/env python3
"""Dump pandapower cases into cuPF's dump-case format for benchmarking.

Output per case under <out>/<name>/:
  dump_Ybus.mtx  (MatrixMarket coordinate complex general, 1-based)
  dump_Sbus.txt  ("re im" per bus)
  dump_V.txt     (flat-start initial voltage)
  dump_Vref.txt  (converged reference voltage)
  dump_pv.txt / dump_pq.txt (0-based internal-order indices)
"""
import sys
import os
import numpy as np

import pandapower as pp
import pandapower.networks as nw
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.idx_bus import VM, VA


def dump_case(name, net, out_root):
    pp.runpp(net, max_iteration=50, tolerance_mva=1e-10)
    ppc = net["_ppc"]
    baseMVA, bus, branch, gen = ppc["baseMVA"], ppc["bus"], ppc["branch"], ppc["gen"]

    Ybus, _, _ = makeYbus(baseMVA, bus, branch)
    Ybus = Ybus.tocoo()
    ref, pv, pq = bustypes(bus, gen)
    Sbus = makeSbus(baseMVA, bus, gen)

    # Converged solution (internal ordering).
    Vref = ppc["internal"]["V"]
    n = bus.shape[0]

    # Flat start: |V|=1 (keep ref/pv setpoint magnitudes), angle=0.
    Vm0 = np.ones(n)
    Vm0[ref] = bus[ref, VM]
    Vm0[pv] = bus[pv, VM]
    V0 = Vm0 * np.exp(1j * 0.0)

    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "dump_Ybus.mtx"), "w") as f:
        f.write("%%MatrixMarket matrix coordinate complex general\n")
        f.write(f"{n} {n} {Ybus.nnz}\n")
        for r, c, v in zip(Ybus.row, Ybus.col, Ybus.data):
            f.write(f"{int(r) + 1} {int(c) + 1} {float(v.real)!r} {float(v.imag)!r}\n")

    def dump_complex(fn, arr):
        with open(os.path.join(d, fn), "w") as f:
            for v in arr:
                f.write(f"{float(v.real)!r} {float(v.imag)!r}\n")

    def dump_int(fn, arr):
        with open(os.path.join(d, fn), "w") as f:
            for v in arr:
                f.write(f"{int(v)}\n")

    dump_complex("dump_Sbus.txt", Sbus)
    dump_complex("dump_V.txt", V0)
    dump_complex("dump_Vref.txt", Vref)
    dump_int("dump_pv.txt", pv)
    dump_int("dump_pq.txt", pq)

    print(f"[dump] {name}: n_bus={n} nnz={Ybus.nnz} n_pv={len(pv)} n_pq={len(pq)} ref={len(ref)}")


CASES = {
    "case118": nw.case118,
    "case1354pegase": nw.case1354pegase,
    "case2869pegase": nw.case2869pegase,
    "case9241pegase": nw.case9241pegase,
}


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else "perf/cases"
    want = sys.argv[2:] if len(sys.argv) > 2 else list(CASES)
    for name in want:
        try:
            dump_case(name, CASES[name](), out_root)
        except Exception as exc:  # noqa: BLE001
            print(f"[dump] {name} FAILED: {exc}")


if __name__ == "__main__":
    main()
