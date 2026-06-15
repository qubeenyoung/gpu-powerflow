#!/usr/bin/env python3
"""
Symbolic-structure depth analysis for power-grid Jacobians.

Loads a matrix, runs METIS NodeND, builds the etree, and reports panel-etree depth
under several amalgamation strategies — to determine whether the current 30+ panel
levels (vs cuDSS-reported depth=10) is reachable by amalgamation alone.

Usage: python depth_analysis.py /path/to/J.mtx
"""
import sys
import time
import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix
from collections import defaultdict


def read_mtx(path):
    t = time.time()
    m = mmread(path)
    print(f"  read mtx: {time.time()-t:.2f}s  shape={m.shape}  nnz={m.nnz}")
    return m


def metis_nd(A):
    try:
        import nxmetis
        import networkx as nx
    except ImportError:
        # Try pymetis
        try:
            import pymetis
        except ImportError:
            raise ImportError("Need pymetis or nxmetis")
        # Build adjacency
        n = A.shape[0]
        S = (A + A.T).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        adjlist = [list(S.indices[S.indptr[i]:S.indptr[i+1]]) for i in range(n)]
        t = time.time()
        perm, iperm = pymetis.nested_dissection(adjacency=adjlist)
        print(f"  metis_nd: {time.time()-t:.2f}s")
        return np.array(perm, dtype=np.int32)
    # nxmetis path (preferred when available)
    G = nx.from_scipy_sparse_array((A + A.T))
    t = time.time()
    perm = nxmetis.node_nested_dissection(G)
    print(f"  metis_nd (nxmetis): {time.time()-t:.2f}s")
    return np.array(perm, dtype=np.int32)


def permute_pattern(A, perm):
    """Return symmetric pattern permuted by perm (newpos[oldpos] = newpos)."""
    n = A.shape[0]
    iperm = np.empty(n, dtype=np.int32)
    iperm[perm] = np.arange(n)
    S = (A + A.T).tocsr().astype(bool).astype(np.int8)
    S.setdiag(0)
    S.eliminate_zeros()
    # Row/col permute
    Sp = S[perm, :][:, perm].tocsc()
    return Sp


def etree(Sp):
    """Liu (1986) elimination tree from a SYMMETRIC pattern (CSC)."""
    n = Sp.shape[0]
    parent = np.full(n, -1, dtype=np.int32)
    ancestor = np.full(n, -1, dtype=np.int32)
    indptr, indices = Sp.indptr, Sp.indices
    for k in range(n):
        for p in range(indptr[k], indptr[k+1]):
            i = indices[p]
            while i != -1 and i < k:
                inext = ancestor[i]
                ancestor[i] = k
                if inext == -1:
                    parent[i] = k
                i = inext
    return parent


def column_counts(Sp, parent):
    """Approximate column counts (Lp[j+1] - Lp[j]) using subtree-summed structure."""
    n = Sp.shape[0]
    # Simple count: for each j, count nnz in struct(j) by walking column j's pattern
    # and merging children's struct[c] minus c. Same as fill_pattern but counting only.
    indptr, indices = Sp.indptr, Sp.indices
    # Build children lists
    children = defaultdict(list)
    for j in range(n):
        if parent[j] != -1:
            children[parent[j]].append(j)
    cc = np.ones(n, dtype=np.int32)  # diagonal
    # Walk struct[j] using DFS-style: cs_counts is faster but this is OK for n<200k
    struct = [set() for _ in range(n)]
    for j in range(n):
        struct[j].add(j)
        for p in range(indptr[j], indptr[j+1]):
            i = indices[p]
            if i > j:
                struct[j].add(i)
        for c in children[j]:
            struct[j].update(struct[c] - {c})
        cc[j] = len(struct[j])
    return cc


def chain_panels(parent, cap):
    """Current relaxed_panels: chain merge in postorder."""
    n = len(parent)
    # Assume parent is in postorder index space (parent[j] > j).
    # This is the chain-merge logic.
    panel_of = np.full(n, -1, dtype=np.int32)
    pid = 0
    j = 0
    while j < n:
        start = j
        sz = 1
        while j + 1 < n and sz < cap and parent[j] == j + 1:
            j += 1
            sz += 1
        for k in range(start, j + 1):
            panel_of[k] = pid
        pid += 1
        j += 1
    return panel_of, pid


def panel_etree(parent, panel_of, num_panels):
    """Build panel etree from column etree."""
    n = len(parent)
    panel_parent = np.full(num_panels, -1, dtype=np.int32)
    for j in range(n):
        if parent[j] == -1:
            continue
        pj = panel_of[j]
        pp = panel_of[parent[j]]
        if pj != pp:
            # Last column of pj's parent is pp
            panel_parent[pj] = pp
    return panel_parent


def panel_levels(panel_parent):
    """Compute panel level (depth from leaves) for each panel."""
    P = len(panel_parent)
    level = np.zeros(P, dtype=np.int32)
    for p in range(P):  # parent > p in postorder space; one forward pass
        pp = panel_parent[p]
        if pp != -1:
            if level[pp] < level[p] + 1:
                level[pp] = level[p] + 1
    return level


def amalgamate_panels(parent, panel_of_init, num_panels_init, max_cap):
    """Etree-based bottom-up amalgamation: merge a panel with its etree parent if the parent
    has few etree children AND the merged column count stays <= max_cap.

    Implements a simplified Ashcraft-Grimes relaxed amalgamation.
    """
    n = len(parent)
    panel_of = panel_of_init.copy()

    # Build initial panel children list
    panel_parent = panel_etree(parent, panel_of, num_panels_init)
    children = defaultdict(list)
    for p in range(num_panels_init):
        pp = panel_parent[p]
        if pp != -1:
            children[pp].append(p)

    # Panel sizes (column count)
    sizes = np.zeros(num_panels_init, dtype=np.int32)
    for j in range(n):
        sizes[panel_of[j]] += 1

    # Greedy: walk panels in REVERSE postorder (bottom-up).
    # For each panel p with parent pp:
    #   if size[p] + size[pp] <= max_cap, merge.
    # When we merge, p's cols become pp's cols. The amalgamated panel is named pp's id.
    # Re-evaluate after one full pass; iterate until no merges happen.
    rep = np.arange(num_panels_init)  # union-find rep
    def find(x):
        while rep[x] != x:
            rep[x] = rep[rep[x]]
            x = rep[x]
        return x

    changed = True
    iters = 0
    while changed and iters < 20:
        changed = False
        for p in range(num_panels_init - 1, -1, -1):
            pr = find(p)
            ppr = panel_parent[pr]
            if ppr == -1:
                continue
            ppr_root = find(ppr)
            if pr == ppr_root:
                continue
            if sizes[pr] + sizes[ppr_root] <= max_cap:
                # Merge pr into ppr_root
                sizes[ppr_root] += sizes[pr]
                sizes[pr] = 0
                rep[pr] = ppr_root
                changed = True
        iters += 1

    # Final panel_of after merges
    new_panel_of = np.full(n, -1, dtype=np.int32)
    new_pid = {}
    cnt = 0
    for j in range(n):
        old = panel_of[j]
        root = find(old)
        if root not in new_pid:
            new_pid[root] = cnt
            cnt += 1
        new_panel_of[j] = new_pid[root]
    return new_panel_of, cnt


def analyze(path, cap=8):
    print(f"\n=== {path} ===")
    A = read_mtx(path)
    n = A.shape[0]
    print(f"  n={n}")
    perm = metis_nd(A.tocsr())
    Sp = permute_pattern(A.tocsr(), perm)
    p = etree(Sp)

    # Chain panels (current code)
    panel_of_chain, P_chain = chain_panels(p, cap=cap)
    panel_parent_chain = panel_etree(p, panel_of_chain, P_chain)
    levels_chain = panel_levels(panel_parent_chain)
    depth_chain = int(levels_chain.max() + 1)
    print(f"\n[chain cap={cap}]")
    print(f"  P={P_chain}  depth={depth_chain}  cols/panel_avg={n/P_chain:.2f}")

    # Etree-amalgamated panels
    for amal_cap in [16, 32, 64, 128, 256]:
        panel_of_a, P_a = amalgamate_panels(p, panel_of_chain, P_chain, max_cap=amal_cap)
        panel_parent_a = panel_etree(p, panel_of_a, P_a)
        levels_a = panel_levels(panel_parent_a)
        depth_a = int(levels_a.max() + 1)
        # Level-cnt distribution (cnt per level)
        lvl_cnts = np.bincount(levels_a)
        # Show first few + last few
        print(f"\n[etree-amalgamate cap={amal_cap}]")
        print(f"  P={P_a}  depth={depth_a}  cols/panel_avg={n/P_a:.2f}")
        if depth_a <= 30:
            for L in range(depth_a):
                if lvl_cnts[L] > 0:
                    print(f"    L{L:2d}  cnt={lvl_cnts[L]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs='?', default="/datasets/power_system/nr_linear_systems/case8387pegase/J.mtx")
    ap.add_argument("--cap", type=int, default=8, help="chain panel cap")
    args = ap.parse_args()
    analyze(args.path, cap=args.cap)
