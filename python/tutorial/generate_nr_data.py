from pypower.loadcase import loadcase
from pypower.ext2int import ext2int  # <-- 내부 인덱싱 변환 함수
from pypower.bustypes import bustypes
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus

from numpy import r_, c_, ix_, zeros, pi, ones, exp, argmax, union1d
import numpy as np
import scipy.sparse as sp
import os

DATASET_ROOT = "/workspace/datasets/pf_dataset"
SAVE_DIR = "/workspace/datasets/nr_dataset"

def preprocess(case_file):
    ppc = loadcase(case_file)

    ppc["branch"] = c_[ppc["branch"],
                            zeros((ppc["branch"].shape[0],
                                    QT - ppc["branch"].shape[1] + 1))]

    ## convert to internal indexing
    ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    print("Calculating Y-bus (Admittance Matrix)...")

    ref, pv, pq = bustypes(bus, gen)

    ## generator info
    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)      ## which generators are on?
    gbus = gen[on, GEN_BUS].astype(int)    ## what buses are they at?

    ## initial state
    # V0    = ones(bus.shape[0])            ## flat start
    V0  = bus[:, VM] * exp(1j * pi/180 * bus[:, VA])
    vcb = ones(V0.shape)    # create mask of voltage-controlled buses
    vcb[pq] = 0     # exclude PQ buses
    k = np.flatnonzero(vcb[gbus])     # in-service gens at v-c buses
    V0[gbus[k]] = gen[on[k], VG] / abs(V0[gbus[k]]) * V0[gbus[k]]

    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    Sbus = makeSbus(baseMVA, bus, gen)

    return Ybus, Sbus, V0, pv, pq

files = os.listdir(DATASET_ROOT)
for file in files:
    if not file.endswith(".mat"):
        continue
    
    try:

        file_path = os.path.join(DATASET_ROOT, file)
        Ybus, Sbus, V0, pv, pq = preprocess(file_path)

        base_filename = os.path.splitext(file)[0]
        save_dir = os.path.join(SAVE_DIR, f"{base_filename}")

        os.makedirs(save_dir, exist_ok=True) # <--- 수정됨 (2)

        # int32로 통일
        pv = pv.astype(np.int32)
        pq = pq.astype(np.int32)
        
        Ybus.data = Ybus.data.astype(np.complex128)
        Ybus.indices = Ybus.indices.astype(np.int32)
        Ybus.indptr = Ybus.indptr.astype(np.int32)
        
        np.save(os.path.join(save_dir, "pv.npy"), pv)
        np.save(os.path.join(save_dir, "pq.npy"), pq)
        np.save(os.path.join(save_dir, "Sbus.npy"), Sbus)
        np.save(os.path.join(save_dir, "V0.npy"), V0)
        sp.save_npz(os.path.join(save_dir, "Ybus.npz"), Ybus)

    except Exception as e:
        print(f"!!! Failed to process {file}. Error: {e}")
