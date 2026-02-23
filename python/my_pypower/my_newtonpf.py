# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Solves the power flow using a full Newton's method.
"""
#소영 dump code
from scipy.io import mmwrite
import numpy as np

import sys
from math import inf
from numpy import array, angle, exp, linalg, conj, r_

from scipy.sparse import hstack, vstack

from pypower.dSbus_dV import dSbus_dV
from pypower.ppoption import ppoption
from pypower.pplinsolve import pplinsolve
from timer import BlockTimer


def my_newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt=None):

    ## default arguments
    if ppopt is None:
        ppopt = ppoption()

    ## options
    tol     = ppopt['PF_TOL']
    max_it  = ppopt['PF_MAX_IT']
    lin_solver = ppopt['PF_LIN_SOLVER_NR']
    timing = ppopt.get('TIMING', False)
    
    timing_data = {}
    
    ## initialize
    converged = 0
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)

    with BlockTimer(timing, "newtonpf", "init_index", 0):
        ## set up indexing for updating V
        pvpq = r_[pv, pq]
        npv = len(pv)
        npq = len(pq)
        j1 = 0;         j2 = npv           ## j1:j2 - V angle of pv buses
        j3 = j2;        j4 = j2 + npq      ## j3:j4 - V angle of pq buses
        j5 = j4;        j6 = j4 + npq      ## j5:j6 - V mag of pq buses
    
    with BlockTimer(timing, "newtonpf", "mismatch", 0):
        ## evaluate F(x0)
        mis = V * conj(Ybus * V) - Sbus
        F = r_[  mis[pv].real,
                mis[pq].real,
                mis[pq].imag  ]
    
        ## check tolerance
        normF = linalg.norm(F, inf)
    if normF < tol:
        converged = 1
        
    ## do Newton iterations
    while (not converged and i < max_it):
        ## update iteration counter
        i = i + 1

        with BlockTimer(timing, "newtonpf", "jacobian", i):
            ## evaluate Jacobian
            dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

            J11 = dS_dVa[array([pvpq]).T, pvpq].real
            J12 = dS_dVm[array([pvpq]).T, pq].real
            J21 = dS_dVa[array([pq]).T, pvpq].imag
            J22 = dS_dVm[array([pq]).T, pq].imag

            J = vstack([
                    hstack([J11, J12]),
                    hstack([J21, J22])
                ], format="csr")
            # ==========================================================
            # [추가할 코드] 1회차 반복(Iter 1)에서 데이터 덤프 후 종료
            # ==========================================================
            # BENCHMARK MODE: Dump code disabled
            # if i == 1:
            #     print(f"\n>>> [DEBUG] Dumping Data for C++/CUDA Validation (Iter {i})")
            #
            #     # 1. 희소 행렬 저장 (Matrix Market Format -> C++에서 읽기 편함)
            #     # Ybus와 J 행렬을 저장합니다.
            #     mmwrite("dump_Ybus.mtx", Ybus)
            #     mmwrite("dump_J.mtx", J)
            #
            #     # 2. 벡터 데이터 저장 (TXT Format -> C++ fscanf로 읽기 편함)
            #     # 복소수(V, Sbus)는 실수부/허수부 두 컬럼으로 저장
            #     # V는 업데이트 되기 전(V0 상태)이어야 비교가 정확하므로 루프 시작 시점 기준인 V를 저장
            #     np.savetxt("dump_V.txt", V.view(float).reshape(-1, 2), header="Real Imag")
            #     np.savetxt("dump_Sbus.txt", Sbus.view(float).reshape(-1, 2), header="Real Imag")
            #
            #     # 정수 인덱스 (PV, PQ)
            #     np.savetxt("dump_pv.txt", pv, fmt='%d')
            #     np.savetxt("dump_pq.txt", pq, fmt='%d')
            #
            #     print(">>> [DEBUG] Dump saved: dump_*.mtx, dump_*.txt")
            #     print(">>> [DEBUG] Stopping execution for validation.")
            #     sys.exit(0) # 데이터만 뽑고 강제 종료
            # ==========================================================

        with BlockTimer(timing, "newtonpf", "solve", i):
            ## compute update step
            dx = -1 * pplinsolve(J, F, lin_solver)

        with BlockTimer(timing, "newtonpf", "update_voltage", i):
            ## update voltage
            if npv:
                Va[pv] = Va[pv] + dx[j1:j2]
            if npq:
                Va[pq] = Va[pq] + dx[j3:j4]
                Vm[pq] = Vm[pq] + dx[j5:j6]
            V = Vm * exp(1j * Va)
            Vm = abs(V)            ## update Vm and Va again in case
            Va = angle(V)          ## we wrapped around with a negative Vm

        with BlockTimer(timing, "newtonpf", "mismatch", i):
            ## evalute F(x)
            mis = V * conj(Ybus * V) - Sbus
            F = r_[  mis[pv].real,
                    mis[pq].real,
                    mis[pq].imag  ]

            ## check for convergence
            normF = linalg.norm(F, inf)
        
        if normF < tol:
            converged = 1
    
    if not converged:
        sys.stdout.write("\nNewton's method power did not converge in %d "
                            "iterations.\n" % i)
    else:
        sys.stdout.write("\nNewton's method power flow converged in "
                                 "%d iterations.\n" % i)
        
    return V, converged, i
