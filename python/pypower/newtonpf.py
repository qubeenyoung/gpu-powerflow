from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np
from numpy import angle, array, conj, exp, linalg, r_
from pypower.dSbus_dV import dSbus_dV
from pypower.pplinsolve import pplinsolve
from pypower.ppoption import ppoption
from scipy.sparse import hstack, vstack

from .timer import BlockTimer, TimingLog


@dataclass
class NewtonPfResult:
    V: np.ndarray
    converged: bool
    iterations: int
    final_mismatch: float


def my_newtonpf(
    Ybus,
    Sbus,
    V0,
    ref,
    pv,
    pq,
    ppopt=None,
    timing_log: TimingLog | None = None,
    emit_status: bool = True,
) -> NewtonPfResult:
    del ref

    if ppopt is None:
        ppopt = ppoption()

    tol = ppopt["PF_TOL"]
    max_it = ppopt["PF_MAX_IT"]
    lin_solver = ppopt["PF_LIN_SOLVER_NR"]

    converged = False
    iteration = 0
    V = np.asarray(V0, dtype=np.complex128).copy()
    Va = angle(V)
    Vm = np.abs(V)

    with BlockTimer(timing_log, "newtonpf", "init_index", 0):
        pvpq = r_[pv, pq]
        npv = len(pv)
        npq = len(pq)
        j1 = 0
        j2 = npv
        j3 = j2
        j4 = j2 + npq
        j5 = j4
        j6 = j4 + npq

    with BlockTimer(timing_log, "newtonpf", "mismatch", 0):
        mis = V * conj(Ybus * V) - Sbus
        F = r_[mis[pv].real, mis[pq].real, mis[pq].imag]
        normF = float(linalg.norm(F, np.inf))

    if normF < tol:
        converged = True

    while (not converged) and iteration < max_it:
        iteration += 1

        with BlockTimer(timing_log, "newtonpf", "jacobian", iteration):
            dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

            J11 = dS_dVa[array([pvpq]).T, pvpq].real
            J12 = dS_dVm[array([pvpq]).T, pq].real
            J21 = dS_dVa[array([pq]).T, pvpq].imag
            J22 = dS_dVm[array([pq]).T, pq].imag

            J = vstack(
                [
                    hstack([J11, J12]),
                    hstack([J21, J22]),
                ],
                format="csr",
            )

        with BlockTimer(timing_log, "newtonpf", "solve", iteration):
            dx = -1.0 * pplinsolve(J, F, lin_solver)
            dx = np.asarray(dx).reshape(-1)

        with BlockTimer(timing_log, "newtonpf", "update_voltage", iteration):
            if npv:
                Va[pv] = Va[pv] + dx[j1:j2]
            if npq:
                Va[pq] = Va[pq] + dx[j3:j4]
                Vm[pq] = Vm[pq] + dx[j5:j6]
            V = Vm * exp(1j * Va)
            Vm = np.abs(V)
            Va = angle(V)

        with BlockTimer(timing_log, "newtonpf", "mismatch", iteration):
            mis = V * conj(Ybus * V) - Sbus
            F = r_[mis[pv].real, mis[pq].real, mis[pq].imag]
            normF = float(linalg.norm(F, np.inf))

        if normF < tol:
            converged = True

    if emit_status:
        if converged:
            sys.stdout.write(f"\nNewton's method power flow converged in {iteration} iterations.\n")
        else:
            sys.stdout.write(f"\nNewton's method power flow did not converge in {iteration} iterations.\n")

    return NewtonPfResult(
        V=V,
        converged=converged,
        iterations=iteration,
        final_mismatch=normF,
    )
