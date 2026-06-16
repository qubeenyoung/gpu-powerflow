#!/usr/bin/env python3
# Convert a complex MatrixMarket Ybus dump into a REAL, diagonally-dominant J.mtx + ones F.mtx
# case dir, for PROFILING the solver (pattern/size are what matter, not the numeric values).
import sys, os

def convert(src, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(src) as f:
        lines = f.readlines()
    # skip comment/header lines starting with %
    i = 0
    assert lines[0].startswith('%%MatrixMarket'), lines[0]
    is_complex = 'complex' in lines[0]
    i = 1
    while lines[i].startswith('%'):
        i += 1
    n, m, nnz = (int(x) for x in lines[i].split())
    i += 1
    assert n == m, f"not square: {n}x{m}"
    # accumulate real part per (r,c); also build pattern for symmetry + row abs-sum
    vals = {}
    for k in range(nnz):
        parts = lines[i + k].split()
        r = int(parts[0]); c = int(parts[1])
        re = float(parts[2]) if len(parts) >= 3 else 1.0
        vals[(r, c)] = vals.get((r, c), 0.0) + re
    # symmetrize pattern (ensure (c,r) present) so METIS-ND graph is symmetric
    for (r, c) in list(vals.keys()):
        if (c, r) not in vals:
            vals[(c, r)] = vals[(r, c)]
    # diagonal-dominance: diag[r] = sum_{c!=r} |vals[r,c]| + 1  (no-pivot LU safe)
    rowabs = {}
    for (r, c), v in vals.items():
        if r != c:
            rowabs[r] = rowabs.get(r, 0.0) + abs(v)
    for r in range(1, n + 1):
        vals[(r, r)] = rowabs.get(r, 0.0) + 1.0
    # write J.mtx (real coordinate general)
    Jpath = os.path.join(outdir, 'J.mtx')
    with open(Jpath, 'w') as f:
        f.write('%%MatrixMarket matrix coordinate real general\n')
        f.write(f'{n} {n} {len(vals)}\n')
        for (r, c) in sorted(vals.keys(), key=lambda x: (x[1], x[0])):  # column-major
            f.write(f'{r} {c} {vals[(r,c)]:.10g}\n')
    # write F.mtx (dense real vector of ones)
    Fpath = os.path.join(outdir, 'F.mtx')
    with open(Fpath, 'w') as f:
        f.write('%%MatrixMarket matrix array real general\n')
        f.write(f'{n} 1\n')
        for _ in range(n):
            f.write('1.0\n')
    print(f'{outdir}: n={n} nnz_real={len(vals)} (complex_src={is_complex})')

if __name__ == '__main__':
    for case in sys.argv[1:]:
        src = f'/workspace/gpu-powerflow-master/paper/exp/results/cpu-reference/cupf-cpu-klu-native/dumps/{case}/dump_Ybus.mtx'
        convert(src, f'/tmp/cls_cases/{case}')
