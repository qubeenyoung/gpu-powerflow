import sys
import time
import numpy as np
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import splu

# Pypower Imports
from pypower.api import loadcase, makeYbus, makeSbus, ext2int, bustypes
from pypower.idx_bus import VM, VA, PD, QD
from pypower.idx_gen import GEN_BUS, PG, QG, GEN_STATUS
from pypower.dSbus_dV import dSbus_dV

def profile_python_case(case_path):
    print(f"========================================")
    print(f"      Python Pypower Profiling          ")
    print(f"========================================")
    print(f"Target: {case_path}")

    # ---------------------------------------------------------
    # 1. Setup (데이터 로드 및 초기화)
    # ---------------------------------------------------------
    t0 = time.time()
    
    ppc = loadcase(case_path)
    ppc = ext2int(ppc) # 내부 인덱싱 변환
    baseMVA, bus, gen, branch = ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    # Ybus 만들기 (Analyze 포함됨)
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Sbus = makeSbus(baseMVA, bus, gen)
    
    # 초기 전압 설정
    V0  = bus[:, VM] * np.exp(1j * np.pi/180 * bus[:, VA])
    vcb = np.ones(V0.shape) 
    
    # 버스 타입 분류
    ref, pv, pq = bustypes(bus, gen)
    pvpq = np.r_[pv, pq]
    
    t1 = time.time()
    time_setup = (t1 - t0) * 1000.0
    
    print(f"[Setup] Done. Ybus size: {Ybus.shape}")

    # ---------------------------------------------------------
    # 2. Iteration Simulation (1회 수행)
    # ---------------------------------------------------------
    print("\n>>> Measuring Execution Time (One Loop) <<<")
    
    V = V0
    
    # (A) Jacobian Update (우리가 C++로 가속한 부분)
    # dSbus_dV 계산 + 행렬 조립(Assembly) 시간을 합쳐야 함
    t_start_jac = time.time()
    
    dS_dVm, dS_dVa = dSbus_dV(Ybus, V) # 편미분 계산
    
    # 행렬 조립 (Assembly)
    J11 = dS_dVa[np.array([pvpq]).T, pvpq].real
    J12 = dS_dVm[np.array([pvpq]).T, pq].real
    J21 = dS_dVa[np.array([pq]).T, pvpq].imag
    J22 = dS_dVm[np.array([pq]).T, pq].imag

    J = vstack([
            hstack([J11, J12]),
            hstack([J21, J22])
        ], format="csr")
    
    t_end_jac = time.time()
    time_jac = (t_end_jac - t_start_jac) * 1000.0
    
    
    # (B) Linear Solver
    # Scipy의 splu (SuperLU) 사용 - Pypower의 기본 Solver
    # J * dx = F (F는 더미로 만듦, Solver 성능 측정엔 영향 없음)
    dim = J.shape[0]
    F = np.random.rand(dim)
    
    t_start_solve = time.time()
    
    # Pypower는 보통 scipy.sparse.linalg.spsolve 또는 splu를 씀
    # 여기선 Factorization 시간을 명확히 보기 위해 splu 사용
    lu = splu(J)      # Factorize (가장 느림)
    dx = lu.solve(F)  # Solve
    
    t_end_solve = time.time()
    time_solve = (t_end_solve - t_start_solve) * 1000.0
    
    
    # ---------------------------------------------------------
    # 결과 출력
    # ---------------------------------------------------------
    print("\n[Python Breakdown]")
    print("-" * 50)
    print(f"1. Setup (Load+Ybus) : {time_setup:10.4f} ms")
    print("-" * 50)
    print("2. Iteration (One Loop):")
    print(f"   - Jacobian Update : {time_jac:10.4f} ms")
    print(f"   - Linear Solver   : {time_solve:10.4f} ms")
    print("-" * 50)
    
    total_iter = time_jac + time_solve
    print(f"TOTAL (1 Iteration)    : {total_iter:10.4f} ms")

    print("\n[Bottleneck Analysis]")
    print(f"Jacobian Share : {time_jac / total_iter * 100.0:.2f} %")
    print(f"Solver Share   : {time_solve / total_iter * 100.0:.2f} %")

if __name__ == "__main__":
    # 실행 시 파일명을 인자로 받음 (기본값 case30000)
    case_name = sys.argv[1] if len(sys.argv) > 1 else "pglib_opf_case30000_goc.mat"
    # 경로 설정 (사용자 환경에 맞게)
    dataset_dir = "/workspace/datasets/pf_dataset"
    full_path = f"{dataset_dir}/{case_name}"
    
    profile_python_case(full_path)