# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
Runs a power flow. (PYPOWER runpf의 간소화 버전)
"""

# --- 표준 라이브러리 임포트 ---
import os
from sys import stdout
from os.path import dirname, join
from time import time

# --- Third-Party 임포트 (Numpy) ---
from numpy import c_, ix_, zeros, pi, ones, exp
from numpy import flatnonzero as find

# --- PYPOWER 함수 임포트 ---
from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.loadcase import loadcase
from pypower.ppoption import ppoption
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from pypower.newtonpf import newtonpf
from pypower.pfsoln import pfsoln
from pypower.printpf import printpf
from pypower.int2ext import int2ext

from my_newtonpf import my_newtonpf
from timer import BlockTimer

# --- PYPOWER 상수 임포트 ---
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS


'''
간소화된 버전
1. dcpf, fdpf, gausspf 알고리즘 호출 부분 제거
2. 알고리즘 뉴턴 메소드(newtonpf)로 고정
3. Q 한계 강제 적용 부분 제거
4. 불필요한 주석 제거
5. verbose 옵션 관련 코드 제거
'''

def my_runpf(casedata=None, log_pf=False, log_newtonpf=False, ppopt=None, fname='', solvedcase=''):

    '''
    ----------------------------------------
    1. 설정 및 데이터 로드
    ----------------------------------------
    '''
    ## default arguments
    if casedata is None:
        casedata = join(dirname(__file__), 'case9')
    ppopt = ppoption(ppopt)
    ppopt['TIMING'] = log_newtonpf
    
    with BlockTimer(log_pf, "runpf", "load_case_data", 0):
        ## read data
        ppc = loadcase(casedata)

    ## branch 행렬이 결과(PF, QF, PT, QT)를 저장할 만큼 충분히 넓은지 확인
    ## Pypower는 QT (인덱스 16)까지의 열을 기대함
    if ppc["branch"].shape[1] < QT:
        ppc["branch"] = c_[ppc["branch"],
                           zeros((ppc["branch"].shape[0],
                                  QT - ppc["branch"].shape[1] + 1))]

    ''' 
    ----------------------------------------
    2. 데이터 내부 인덱싱 변환 및 준비
    ----------------------------------------
    '''
    with BlockTimer(log_pf, "runpf", "bus_indexing", 0):
        ## convert to internal indexing
        ppc = ext2int(ppc)
        baseMVA, bus, gen, branch = \
            ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

        ## get bus index lists of each type of bus
        ref, pv, pq = bustypes(bus, gen)

        ## generator info
        on = find(gen[:, GEN_STATUS] > 0)      ## which generators are on?
        gbus = gen[on, GEN_BUS].astype(int)    ## what buses are they at?


    '''
    ----------------------------------------
    3. 조류 계산 실행 (Newton-Raphson)
    ----------------------------------------
    '''
    t0 = time()

    ## ----- initial state -----    
    # 케이스 파일에 명시된 Vm, Va 값을 초기값으로 사용 (non-flat start)
    V0  = bus[:, VM] * exp(1j * pi/180 * bus[:, VA])
    
    # 전압 제어 버스(PV, REF) 마스크 생성
    vcb = ones(V0.shape)    # create mask of voltage-controlled buses
    vcb[pq] = 0             # exclude PQ buses
    
    # 전압 제어 버스에 연결된 'online' 발전기 인덱스 탐색
    k = find(vcb[gbus])     # in-service gens at v-c buses
    
    # PV 버스의 전압 크기(magnitude)를 발전기 설정값(VG)으로 강제 설정
    # (각도는 초기값 VA를 그대로 유지)
    V0[gbus[k]] = gen[on[k], VG] / abs(V0[gbus[k]]) * V0[gbus[k]]

    with BlockTimer(log_pf, "runpf", "build_Ybus", 0):
        ## ----- build admittance matrices -----
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    with BlockTimer(log_pf, "runpf", "build_Sbus", 0):
        ## ----- compute complex bus power injections -----
        # Sbus = Sgen - Sload
        Sbus = makeSbus(baseMVA, bus, gen)

    with BlockTimer(log_pf, "runpf", "newtonpf", 0):
        ## ----- run the power flow -----
        V, success, _ = my_newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
    
    ## update data matrices with solution
    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq)

    ppc["et"] = time() - t0
    ppc["success"] = success

    '''
    ----------------------------------------
    4. 결과 처리 및 출력
    ----------------------------------------
    '''
    ## convert back to original bus numbering
    ppc["bus"], ppc["gen"], ppc["branch"] = bus, gen, branch
    results = int2ext(ppc)

    ## zero out result fields of out-of-service gens & branches
    if len(results["order"]["gen"]["status"]["off"]) > 0:
        results["gen"][ix_(results["order"]["gen"]["status"]["off"], [PG, QG])] = 0

    if len(results["order"]["branch"]["status"]["off"]) > 0:
        results["branch"][ix_(results["order"]["branch"]["status"]["off"], [PF, QF, PT, QT])] = 0

    ## print results
    printpf(results, stdout, ppopt)

    return results, success

#소영 dump
if __name__ == '__main__':
    import sys
    import os
    
    dataset_dir = "/workspace/datasets/pf_dataset"

    if len(sys.argv) > 1:
        target_case_name = sys.argv[1]
    else:
        target_case_name = "pglib_opf_case14_ieee.mat"
    
    case_data_path = os.path.join(dataset_dir, target_case_name)
    
    print(f"========================================")
    print(f"Target Case: {target_case_name}")
    print(f"File Path  : {case_data_path}")
    print(f"========================================")
    
    if not os.path.exists(case_data_path):
        print(f"[Error] File not found: {case_data_path}")
        sys.exit(1)

    my_runpf(casedata=case_data_path)