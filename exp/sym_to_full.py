#!/usr/bin/env python3
# Expand a SuiteSparse symmetric coordinate .mtx (lower triangle only) into a FULL general .mtx
# (our solver's reader does not expand symmetric → triangular structure breaks the ordering).
# Also emits a ones RHS F.mtx (array real general), matching the power-flow case layout.
import sys, os

def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(inp) as f:
        header = f.readline()
        low = header.lower()
        pattern = "pattern" in low
        sym = ("symmetric" in low) or ("hermitian" in low)
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()
        nr, nc, nz = map(int, line.split())
        rows = []  # (i, j, val)
        for _ in range(nz):
            parts = f.readline().split()
            i = int(parts[0]); j = int(parts[1])
            val = 1.0 if pattern else float(parts[2])
            rows.append((i, j, val))
            if sym and i != j:
                rows.append((j, i, val))
    out_mtx = os.path.join(outdir, "J.mtx")
    with open(out_mtx, "w") as g:
        g.write("%%MatrixMarket matrix coordinate real general\n")
        g.write(f"{nr} {nc} {len(rows)}\n")
        for (i, j, v) in rows:
            g.write(f"{i} {j} {v!r}\n")
    out_rhs = os.path.join(outdir, "F.mtx")
    with open(out_rhs, "w") as g:
        g.write("%%MatrixMarket matrix array real general\n")
        g.write(f"{nr} 1\n")
        for _ in range(nr):
            g.write("1.0\n")
    print(f"wrote {out_mtx}  n={nr} nnz_full={len(rows)} (sym={sym})")
    print(f"wrote {out_rhs}  n={nr}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
