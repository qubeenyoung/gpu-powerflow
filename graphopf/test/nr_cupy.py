from __future__ import annotations

import os
import sys
from time import time

import numpy as np
from numpy import c_, exp, ix_, ones, pi, zeros
from numpy import flatnonzero as find

from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.int2ext import int2ext
from pypower.loadcase import loadcase
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from pypower.pfsoln import pfsoln
from pypower.ppoption import ppoption

from pypower.idx_bus import VA, VM
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG

sys.path.append("/workspace/graphopf")
from newton_method.cupy.newtonpf import NewtonSolver


CONVERGED_CASES: set[str] = {
    "pglib_opf_case5_pjm",
    "pglib_opf_case14_ieee",
    "pglib_opf_case24_ieee_rts",
    "pglib_opf_case30_as",
    "pglib_opf_case30_ieee",
    "pglib_opf_case57_ieee",
    "pglib_opf_case60_c",
    "pglib_opf_case73_ieee_rts",
    "pglib_opf_case89_pegase",
    "pglib_opf_case118_ieee",
    "pglib_opf_case197_snem",
    "pglib_opf_case200_activ",
    "pglib_opf_case588_sdet",
    "pglib_opf_case793_goc",
    "pglib_opf_case1354_pegase",
    "pglib_opf_case2312_goc",
    "pglib_opf_case2383wp_k",
    "pglib_opf_case2736sp_k",
    "pglib_opf_case2737sop_k",
    "pglib_opf_case2746wop_k",
    "pglib_opf_case2746wp_k",
    "pglib_opf_case2869_pegase",
    "pglib_opf_case3012wp_k",
    "pglib_opf_case3120sp_k",
    "pglib_opf_case3375wp_k",
    "pglib_opf_case3970_goc",
    "pglib_opf_case4601_goc",
    "pglib_opf_case4619_goc",
    "pglib_opf_case5658_epigrids",
    "pglib_opf_case7336_epigrids",
    "pglib_opf_case8387_pegase",
    "pglib_opf_case9241_pegase",
}


def _case_basename(path: str) -> str:
    name = os.path.basename(path)
    base, _ext = os.path.splitext(name)
    return base


def my_newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt, batch_size: int = 1):
    import cupy as cp

    tol = float(ppopt.get("PF_TOL", 1e-6))
    max_it = int(ppopt.get("PF_MAX_IT", 10))

    Ybus = Ybus.tocsr()
    solver = NewtonSolver(Ybus, pv, pq, batch_size=batch_size)
    try:
        # Multi-batch experiment: replicate same inputs
        sbus_cp = cp.asarray(np.tile(Sbus[None, :], (batch_size, 1)), dtype=cp.complex128)
        v0_cp = cp.asarray(np.tile(V0[None, :], (batch_size, 1)), dtype=cp.complex128)

        v_cp = solver.solve(
            sbus=sbus_cp,
            v0=v0_cp,
            tolerance=tol,
            max_iter=max_it,
        )

        # Use batch 0 as representative for pfsoln
        V = cp.asnumpy(v_cp[0])

        # Check final mismatch (CPU) for success on batch 0
        Ibus = Ybus.dot(V)
        mis = V * np.conj(Ibus) - Sbus
        f = np.r_[mis[pv].real, mis[pq].real, mis[pq].imag]
        normF = float(np.max(np.abs(f)))
        success = bool(normF < tol)

        
        return V, success, normF
    finally:
        solver.close()


def my_runpf(casedata: str, batch_size: int = 64):
    if not casedata:
        raise ValueError("casedata must be provided.")

    case_name = _case_basename(casedata)
    if case_name not in CONVERGED_CASES:
        print(f"[SKIP] {case_name}")
        return None, False

    ppopt = ppoption()

    ppc = loadcase(casedata)

    # Ensure branch matrix has columns through QT
    if ppc["branch"].shape[1] < QT:
        pad_cols = QT - ppc["branch"].shape[1] + 1
        ppc["branch"] = c_[ppc["branch"], zeros((ppc["branch"].shape[0], pad_cols))]

    # Convert to internal indexing
    ppc = ext2int(ppc)
    baseMVA = ppc["baseMVA"]
    bus = ppc["bus"]
    gen = ppc["gen"]
    branch = ppc["branch"]

    ref, pv, pq = bustypes(bus, gen)

    on = find(gen[:, GEN_STATUS] > 0)
    gbus = gen[on, GEN_BUS].astype(int)

    t0 = time()

    # Non-flat start using Vm/Va in case file
    V0 = bus[:, VM] * exp(1j * pi / 180.0 * bus[:, VA])

    # Enforce generator voltage setpoints at PV/REF buses
    vcb = ones(V0.shape)
    vcb[pq] = 0
    k = find(vcb[gbus])
    V0[gbus[k]] = gen[on[k], VG] / np.abs(V0[gbus[k]]) * V0[gbus[k]]

    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Sbus = makeSbus(baseMVA, bus, gen)

    V, success, normF = my_newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt, batch_size=batch_size)
    print(f"[{case_name}] {'success' if success else 'fail'}: normF={normF:e}")

    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq)

    ppc["et"] = time() - t0
    ppc["success"] = success
    ppc["bus"], ppc["gen"], ppc["branch"] = bus, gen, branch

    results = int2ext(ppc)

    # Zero out results for out-of-service gens/branches
    off_gens = results["order"]["gen"]["status"]["off"]
    if len(off_gens) > 0:
        results["gen"][ix_(off_gens, [PG, QG])] = 0

    off_branches = results["order"]["branch"]["status"]["off"]
    if len(off_branches) > 0:
        results["branch"][ix_(off_branches, [PF, QF, PT, QT])] = 0

    return results, success


if __name__ == "__main__":
    dataset_dir = "datasets/pf_dataset"
    batch_size = 64

    for fname in sorted(os.listdir(dataset_dir)):
        case_path = os.path.join(dataset_dir, fname)
        my_runpf(casedata=case_path, batch_size=batch_size)
