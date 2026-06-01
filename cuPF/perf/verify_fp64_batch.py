#!/usr/bin/env python3
"""Correctness check for FP64 batched CUDA solve.

Builds a batch of DISTINCT scenarios (per-case Sbus scaling), solves them as one
batch, and compares each case against an independent single-case FP64 solve. If
batching contaminated cases (wrong offsets, shared buffers), the per-case error
would be large. Also checks the nominal case against the dumped reference V.
"""
import sys, os
import numpy as np
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "build-cuda"))
import _cupf as cupf


def load_case(d):
    # Ybus.mtx: MatrixMarket coordinate complex general, 1-based.
    rows = cols = nnz = None
    R, C, V = [], [], []
    with open(os.path.join(d, "dump_Ybus.mtx")) as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.split()
            if rows is None:
                rows, cols, nnz = int(parts[0]), int(parts[1]), int(parts[2])
                continue
            R.append(int(parts[0]) - 1); C.append(int(parts[1]) - 1)
            V.append(float(parts[2]) + 1j * float(parts[3]))
    Y = coo_matrix((np.array(V), (np.array(R), np.array(C))), shape=(rows, cols)).tocsr()

    def rc(fn):
        out = []
        with open(os.path.join(d, fn)) as f:
            for ln in f:
                a = ln.split(); out.append(float(a[0]) + 1j * float(a[1]))
        return np.array(out, dtype=np.complex128)

    def ri(fn):
        with open(os.path.join(d, fn)) as f:
            return np.array([int(x) for x in f.read().split()], dtype=np.int32)

    sbus = rc("dump_Sbus.txt"); v0 = rc("dump_V.txt")
    vref = rc("dump_Vref.txt") if os.path.exists(os.path.join(d, "dump_Vref.txt")) else None
    pv = ri("dump_pv.txt"); pq = ri("dump_pq.txt")
    return Y, rows, sbus, v0, vref, pv, pq


def fp64_opts():
    o = cupf.NewtonOptions()
    o.backend = cupf.BackendKind.CUDA
    o.compute = cupf.ComputePolicy.FP64
    return o


def main():
    case_dir = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    Y, n, sbus, v0, vref, pv, pq = load_case(case_dir)
    indptr = Y.indptr.astype(np.int32); indices = Y.indices.astype(np.int32)
    data = Y.data.astype(np.complex128)
    cfg = cupf.NRConfig(); cfg.tolerance = 1e-10; cfg.max_iter = 50

    # Distinct scenarios: scale each case's Sbus by a slightly different factor.
    scales = 1.0 + 0.02 * np.arange(B) / max(B - 1, 1)   # 1.00 .. 1.02
    sbus_batch = np.stack([sbus * s for s in scales]).astype(np.complex128)
    v0_batch = np.stack([v0 for _ in range(B)]).astype(np.complex128)

    # Batched FP64 solve.
    s = cupf.NewtonSolver(fp64_opts())
    s.initialize(indptr, indices, data, n, n, pv, pq)
    rb = s.solve_batch(indptr, indices, data, n, n, sbus_batch, v0_batch, pv, pq, cfg)
    Vb = rb.V_numpy   # [B, n]

    # Independent single-case solves of the same scenarios.
    max_err = 0.0
    for b in range(B):
        ss = cupf.NewtonSolver(fp64_opts())
        ss.initialize(indptr, indices, data, n, n, pv, pq)
        r1 = ss.solve(indptr, indices, data, n, n,
                      (sbus * scales[b]).astype(np.complex128), v0, pv, pq, cfg)
        v1 = np.array(r1.V, dtype=np.complex128)
        e = np.max(np.abs(Vb[b] - v1))
        max_err = max(max_err, e)
    print(f"case={os.path.basename(case_dir)} n_bus={n} B={B}")
    print(f"  iterations per case: {list(rb.iterations_numpy)}")
    print(f"  max |V_batch[b] - V_single(scenario_b)|  = {max_err:.3e}")
    if vref is not None:
        # Nominal case is scale=1.0 -> batch index 0.
        eref = np.max(np.abs(Vb[0] - vref))
        print(f"  nominal case[0] vs dumped Vref            = {eref:.3e}")
    ok = max_err < 1e-8
    print("  RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
