#!/usr/bin/env python3
"""
v2: Contiguity-preserving amalgamation strategies for the multifrontal panel etree.

Same setup as v1, but adds amalgamation strategies that REQUIRE the merged panel's
columns to remain contiguous in postorder (GPU constraint of our current code).

Two strategies:
  (W) Whole-subtree compression: walk panels bottom-up; if total subtree size <= cap,
      collapse entire subtree into one panel. Always contiguous (subtree cols are
      contiguous in postorder).
  (T) Top-down spine + leaf compression: combine consecutive parent-child panels with
      single child relationship (chain merge), then for each branch point, try to
      whole-subtree compress.

Reports depth + cols/panel for each strategy on case8387 and USA.
"""
import sys, time
import numpy as np
from scipy.io import mmread
from collections import defaultdict


def read_perm_etree(path):
    A = mmread(path).tocsr()
    n = A.shape[0]
    print(f"  n={n}, nnz={A.nnz}")
    import pymetis
    S = (A + A.T).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    adjlist = [list(S.indices[S.indptr[i]:S.indptr[i+1]]) for i in range(n)]
    perm, _ = pymetis.nested_dissection(adjacency=adjlist)
    perm = np.array(perm, dtype=np.int32)
    iperm = np.empty(n, dtype=np.int32)
    iperm[perm] = np.arange(n)
    Sp = S[perm, :][:, perm].tocsc()
    # etree
    indptr, indices = Sp.indptr, Sp.indices
    parent = np.full(n, -1, dtype=np.int32)
    ancestor = np.full(n, -1, dtype=np.int32)
    for k in range(n):
        for p in range(indptr[k], indptr[k+1]):
            i = int(indices[p])
            while i != -1 and i < k:
                inext = int(ancestor[i])
                ancestor[i] = k
                if inext == -1: parent[i] = k
                i = inext
    # METIS gives a topological order (parent[j] > j) but NOT necessarily a postorder.
    # Run explicit postorder, then relabel parent[] in postorder index space so subtree
    # column ranges become contiguous.
    children = [[] for _ in range(n)]
    for j in range(n):
        if parent[j] != -1: children[parent[j]].append(j)
    post = []
    # Iterative DFS (postorder)
    visited = [0] * n
    stack = [j for j in range(n) if parent[j] == -1]
    out_stack = []
    while stack:
        x = stack.pop()
        if visited[x]:
            post.append(x)
            continue
        visited[x] = 1
        stack.append(x)  # re-enqueue for post-visit
        for c in children[x]:
            stack.append(c)
    # Validate
    assert len(post) == n
    # Relabel: post[k] is the OLD column index at NEW postorder position k.
    # New parent[k] = post-index of OLD parent[post[k]].
    pos_of = np.empty(n, dtype=np.int32)
    for k in range(n): pos_of[post[k]] = k
    new_parent = np.full(n, -1, dtype=np.int32)
    for k in range(n):
        old = post[k]
        if parent[old] != -1:
            new_parent[k] = pos_of[parent[old]]
    # Verify new_parent[j] > j (still topological in new indices)
    for j in range(n):
        if new_parent[j] != -1:
            assert new_parent[j] > j, f"j={j} new_parent={new_parent[j]}"
    return new_parent


def chain_panels(parent, cap):
    """Existing relaxed_panels: chain merge in postorder."""
    n = len(parent)
    panel_of = np.full(n, -1, dtype=np.int32)
    pid = 0
    j = 0
    while j < n:
        start = j; sz = 1
        while j + 1 < n and sz < cap and parent[j] == j + 1:
            j += 1; sz += 1
        for k in range(start, j + 1):
            panel_of[k] = pid
        pid += 1; j += 1
    return panel_of, pid


def panel_etree(parent, panel_of, num_panels):
    P = num_panels
    pp = np.full(P, -1, dtype=np.int32)
    n = len(parent)
    for j in range(n):
        pj_par = int(parent[j])
        if pj_par == -1: continue
        my = int(panel_of[j]); par = int(panel_of[pj_par])
        if my != par: pp[my] = par
    return pp


def panel_levels(panel_parent):
    P = len(panel_parent)
    lvl = np.zeros(P, dtype=np.int32)
    for p in range(P):
        if panel_parent[p] != -1:
            if lvl[panel_parent[p]] < lvl[p] + 1:
                lvl[panel_parent[p]] = lvl[p] + 1
    return lvl


def etree_amalgamate(parent, panel_of, num_panels, cap):
    """v1's algorithm: greedy bottom-up, merge panel with parent if combined size fits cap."""
    P = num_panels
    pp = panel_etree(parent, panel_of, P)
    sizes = np.bincount(panel_of, minlength=P).astype(np.int32)
    rep = np.arange(P)
    def find(x):
        while rep[x] != x:
            rep[x] = rep[rep[x]]; x = rep[x]
        return x
    changed = True; it = 0
    while changed and it < 30:
        changed = False
        for p in range(P - 1, -1, -1):
            pr = find(p)
            par = pp[pr]
            if par == -1: continue
            par_r = find(par)
            if pr == par_r: continue
            if sizes[pr] + sizes[par_r] <= cap:
                sizes[par_r] += sizes[pr]; sizes[pr] = 0
                rep[pr] = par_r
                changed = True
        it += 1
    n = len(panel_of)
    new_panel = np.full(n, -1, dtype=np.int32)
    pid_map = {}; cnt = 0
    for j in range(n):
        old = int(panel_of[j]); root = find(old)
        if root not in pid_map: pid_map[root] = cnt; cnt += 1
        new_panel[j] = pid_map[root]
    return new_panel, cnt


def whole_subtree_compress(parent, panel_of, num_panels, cap):
    """Walk panel etree bottom up; for each subtree whose TOTAL cols <= cap, collapse
    the whole subtree into one panel. Always produces contiguous panels (subtree cols
    are contiguous in postorder).

    Implementation: union-find with rep[p] = merged-panel root.
    Iterate: for each leaf-up panel p, if (size of subtree rooted at p) <= cap, merge
    all panels in that subtree into one. After this, p is the "root" of that supernode.
    """
    P = num_panels
    pp = panel_etree(parent, panel_of, P)
    sizes = np.bincount(panel_of, minlength=P).astype(np.int32)
    children = defaultdict(list)
    for p in range(P):
        if pp[p] != -1:
            children[pp[p]].append(p)

    # rep[p] = panel that p has been merged into (-1 if root or self)
    rep = np.arange(P)
    def find(x):
        while rep[x] != x:
            rep[x] = rep[rep[x]]
            x = rep[x]
        return x

    # Size of the subtree rooted at each panel (DP, leaves first)
    # Order panels by postorder of panel etree = max postorder col index in each panel
    last_col = np.zeros(P, dtype=np.int32)
    for j in range(len(panel_of)):
        last_col[panel_of[j]] = max(last_col[panel_of[j]], j)
    order = np.argsort(last_col)  # increasing -> bottom up

    subtree_size = sizes.copy()
    for p in order:
        if pp[p] != -1:
            subtree_size[pp[p]] += subtree_size[p]

    # Bottom-up: if subtree_size <= cap, merge entire subtree into p (collapse).
    # When we collapse subtree rooted at p, all descendants merge into p.
    # "absorbed[c]" = True means c has been swallowed by an ancestor and is no longer
    # a candidate to be the new supernode root. We DON'T skip merging at p when its
    # children are absorbed — we just don't visit absorbed panels as the outer p.
    absorbed = np.zeros(P, dtype=bool)
    for p in order:
        if absorbed[p]: continue
        if subtree_size[p] <= cap and subtree_size[p] > sizes[p]:  # has descendants to absorb
            # BFS all descendants, merge into p.
            stack = list(children[p])
            while stack:
                x = stack.pop()
                rep[x] = p
                absorbed[x] = True
                for c in children[x]:
                    stack.append(c)
        # If subtree_size == sizes[p] (i.e., p is a leaf), no merge needed.

    # Build new panel_of with re-numbered IDs
    n = len(panel_of)
    new_panel = np.full(n, -1, dtype=np.int32)
    pid_map = {}
    cnt = 0
    for j in range(n):
        old = int(panel_of[j])
        root = find(old)
        if root not in pid_map:
            pid_map[root] = cnt; cnt += 1
        new_panel[j] = pid_map[root]
    return new_panel, cnt


def check_contiguous(panel_of, num_panels):
    """Verify each panel's columns are a contiguous postorder range."""
    n = len(panel_of)
    first = np.full(num_panels, n, dtype=np.int32)
    last = np.full(num_panels, -1, dtype=np.int32)
    for j in range(n):
        p = int(panel_of[j])
        first[p] = min(first[p], j)
        last[p] = max(last[p], j)
    # All cols in [first[p], last[p]+1) must have panel == p
    for j in range(n):
        p = int(panel_of[j])
        # The range first[p]..last[p] should have all == p
    bad = 0
    for p in range(num_panels):
        for j in range(first[p], last[p] + 1):
            if panel_of[j] != p:
                bad += 1
                break
    return bad == 0


def repostorder_for_supernodes(parent, panel_of, num_panels):
    """Given an etree (parent[], in some index space) and a panel assignment that may put
    non-adjacent columns in the same panel, produce a NEW column ordering (a permutation
    new_pos[old_idx] = new_idx) such that every panel's cols become a contiguous range
    in the new order.

    Algorithm: build supernode etree (panel_parent), postorder it. For each panel in
    supernode-postorder, emit all its cols (in their relative postorder within the panel)
    before moving on.
    """
    n = len(panel_of)
    P = num_panels
    pp = panel_etree(parent, panel_of, P)

    # Children of each supernode
    sn_children = defaultdict(list)
    for p in range(P):
        if pp[p] != -1: sn_children[pp[p]].append(p)
    sn_roots = [p for p in range(P) if pp[p] == -1]

    # Iterative DFS supernode postorder
    sn_visited = [0] * P
    sn_post = []
    stack = list(sn_roots)
    while stack:
        x = stack.pop()
        if sn_visited[x]:
            sn_post.append(x); continue
        sn_visited[x] = 1
        stack.append(x)
        for c in sn_children[x]: stack.append(c)

    # Cols of each panel
    panel_cols = defaultdict(list)
    for j in range(n):
        panel_cols[int(panel_of[j])].append(j)

    new_order = []
    for sn in sn_post:
        new_order.extend(panel_cols[sn])
    assert len(new_order) == n

    new_pos = np.empty(n, dtype=np.int32)
    for new_idx, old in enumerate(new_order):
        new_pos[old] = new_idx

    # Renumber parent[] in new index space
    new_parent = np.full(n, -1, dtype=np.int32)
    for j in range(n):
        if parent[j] != -1:
            new_parent[new_pos[j]] = new_pos[parent[j]]
    # Renumber panel_of in new index space
    new_panel_of = np.empty(n, dtype=np.int32)
    for j in range(n):
        new_panel_of[new_pos[j]] = panel_of[j]
    return new_parent, new_panel_of


def col_etree_depth(parent):
    n = len(parent)
    depth = np.zeros(n, dtype=np.int32)
    for j in range(n):
        if parent[j] != -1:
            if depth[parent[j]] < depth[j] + 1:
                depth[parent[j]] = depth[j] + 1
    return int(depth.max() + 1)


def analyze(path):
    print(f"\n=== {path} ===")
    parent = read_perm_etree(path)
    n = len(parent)
    print(f"  COLUMN etree depth (theoretical lower bound) = {col_etree_depth(parent)}")

    # Baseline: chain (current code, adaptive cap)
    cap_chain = 8 if n < 16000 else (12 if n < 80000 else 20)
    panel_of_c, P_c = chain_panels(parent, cap=cap_chain)
    pp_c = panel_etree(parent, panel_of_c, P_c)
    lvl_c = panel_levels(pp_c)
    depth_c = int(lvl_c.max() + 1)
    print(f"\n[chain cap={cap_chain}]  P={P_c}  depth={depth_c}  cols/panel={n/P_c:.2f}")

    # v1: greedy bottom-up etree amalgamation (may break contiguity)
    print("\n--- etree-amalgamate (greedy) + postorder reordering ---")
    for cap in [16, 32, 64, 128, 256]:
        panel_of_a, P_a = etree_amalgamate(parent, panel_of_c, P_c, cap)
        new_parent, new_panel = repostorder_for_supernodes(parent, panel_of_a, P_a)
        pp_a = panel_etree(new_parent, new_panel, P_a)
        lvl_a = panel_levels(pp_a)
        depth_a = int(lvl_a.max() + 1)
        contig = check_contiguous(new_panel, P_a)
        print(f"  cap={cap:3d}: P={P_a:5d} depth={depth_a:3d} cols/panel={n/P_a:5.2f}  contig_after_repostorder={contig}")

    # Whole-subtree compression (always contiguous)
    print("\n--- whole-subtree compress (always contiguous) ---")
    for cap in [16, 32, 64, 128, 256, 512, 1024]:
        panel_of_w, P_w = whole_subtree_compress(parent, panel_of_c, P_c, cap)
        pp_w = panel_etree(parent, panel_of_w, P_w)
        lvl_w = panel_levels(pp_w)
        depth_w = int(lvl_w.max() + 1)
        contig = check_contiguous(panel_of_w, P_w)
        print(f"  cap={cap:4d}: P={P_w:5d} depth={depth_w:3d} cols/panel={n/P_w:5.2f}  contig={contig}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs='+')
    args = ap.parse_args()
    for p in args.paths:
        analyze(p)
