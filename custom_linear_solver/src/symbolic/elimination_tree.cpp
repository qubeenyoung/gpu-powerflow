#include "symbolic/elimination_tree.hpp"

#include <algorithm>
#include <thread>

namespace custom_linear_solver::symbolic {

std::vector<int> etree(int n, const int* col_ptr, const int* row_idx)
{
    std::vector<int> parent(n, -1);
    std::vector<int> ancestor(n, -1);  // path-compressed ancestor of each node

    for (int k = 0; k < n; ++k) {
        for (int p = col_ptr[k]; p < col_ptr[k + 1]; ++p) {
            int i = row_idx[p];
            // Walk from i up to the root, attaching the path to k (Liu 1986).
            while (i != -1 && i < k) {
                const int inext = ancestor[i];
                ancestor[i] = k;
                if (inext == -1) {
                    parent[i] = k;
                }
                i = inext;
            }
        }
    }
    return parent;
}

void symmetric_pattern(int n, const int* col_ptr, const int* row_idx,
                       std::vector<int>& sym_col_ptr, std::vector<int>& sym_row_idx)
{
    // Flat CSR two-pass (no per-node vector<vector> allocations): count adjacency
    // degrees (with duplicates — symmetric input double-counts), bucket-fill, then
    // sort + unique each node's slice into the compacted output. Same result as
    // the old adjacency-list build, much faster on large matrices.
    std::vector<int> off(static_cast<std::size_t>(n) + 1, 0);
    auto valid = [&](int row, int col) { return row >= 0 && row < n && row != col; };
    for (int col = 0; col < n; ++col)
        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            const int row = row_idx[p];
            if (!valid(row, col)) continue;
            ++off[row + 1];
            ++off[col + 1];
        }
    for (int i = 0; i < n; ++i) off[i + 1] += off[i];
    std::vector<int> adj(off[n]);
    std::vector<int> next(off.begin(), off.end());
    for (int col = 0; col < n; ++col)
        for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            const int row = row_idx[p];
            if (!valid(row, col)) continue;
            adj[next[row]++] = col;
            adj[next[col]++] = row;
        }
    // Parallelize the per-node sort+dedup+compact. On large power-grid Jacobians the serial
    // pass is non-trivial (tens of ms on n ~ 10^6). Each node's slice is independent → output
    // is byte-identical to serial (sort+dedup per col, concatenated in col order via prefix-sum).
    auto par_for = [](int lo, int hi, auto&& fn) {
        unsigned hw = std::thread::hardware_concurrency();
        const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
        if (hi - lo < 32768 || nth <= 1) { fn(lo, hi); return; }  // small: thread overhead > gain
        std::vector<std::thread> th;
        const int chunk = (hi - lo + nth - 1) / nth;
        for (int t = 0; t < nth; ++t) {
            const int a = lo + t * chunk, b = std::min(hi, a + chunk);
            if (a < b) th.emplace_back([&fn, a, b] { fn(a, b); });
        }
        for (auto& x : th) x.join();
    };
    std::vector<int> ucnt(n, 0);
    par_for(0, n, [&](int lo, int hi) {  // sort each slice in place + count uniques
        for (int col = lo; col < hi; ++col) {
            const int b = off[col], e = off[col + 1];
            std::sort(adj.begin() + b, adj.begin() + e);
            int u = 0, last = -1;
            for (int k = b; k < e; ++k)
                if (adj[k] != last) { ++u; last = adj[k]; }
            ucnt[col] = u;
        }
    });
    sym_col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int col = 0; col < n; ++col) sym_col_ptr[col + 1] = sym_col_ptr[col] + ucnt[col];
    sym_row_idx.resize(sym_col_ptr[n]);
    par_for(0, n, [&](int lo, int hi) {  // compact uniques into the prefix-summed slots
        for (int col = lo; col < hi; ++col) {
            int w = sym_col_ptr[col], last = -1;
            for (int k = off[col]; k < off[col + 1]; ++k)
                if (adj[k] != last) { sym_row_idx[w++] = adj[k]; last = adj[k]; }
        }
    });
}

std::vector<int> postorder(const std::vector<int>& parent, int n)
{
    // Build reversed child lists, then iterative DFS (mirrors CXSparse cs_post).
    std::vector<int> head(n, -1), next(n, -1), stack(n), post(n);
    for (int j = n - 1; j >= 0; --j) {
        if (parent[j] == -1) {
            continue;
        }
        next[j] = head[parent[j]];
        head[parent[j]] = j;
    }
    int k = 0;
    for (int j = 0; j < n; ++j) {
        if (parent[j] != -1) {
            continue;  // start DFS only from roots
        }
        int top = 0;
        stack[0] = j;
        while (top >= 0) {
            const int p = stack[top];
            const int child = head[p];
            if (child == -1) {
                post[k++] = p;  // no unvisited children: emit
                --top;
            } else {
                head[p] = next[child];   // advance child list
                stack[++top] = child;    // descend
            }
        }
    }
    return post;
}

namespace {

// Least-common-ancestor / leaf detection helper (CXSparse cs_leaf).
int leaf(int i, int j, const std::vector<int>& first, std::vector<int>& maxfirst,
         std::vector<int>& prevleaf, std::vector<int>& ancestor, int& jleaf)
{
    jleaf = 0;
    if (i <= j || first[j] <= maxfirst[i]) {
        return -1;
    }
    maxfirst[i] = first[j];
    const int jprev = prevleaf[i];
    prevleaf[i] = j;
    jleaf = (jprev == -1) ? 1 : 2;  // 1 = first leaf of subtree, 2 = subsequent
    if (jleaf == 1) {
        return i;
    }
    int q = jprev;
    while (q != ancestor[q]) {
        q = ancestor[q];
    }
    for (int s = jprev; s != q;) {
        const int sparent = ancestor[s];
        ancestor[s] = q;  // path compression
        s = sparent;
    }
    return q;  // least common ancestor of j and jprev
}

}  // namespace

std::vector<int> column_counts(int n, const int* col_ptr, const int* row_idx,
                               const std::vector<int>& parent,
                               const std::vector<int>& post)
{
    std::vector<int> delta(n, 0);
    std::vector<int> ancestor(n), maxfirst(n, -1), prevleaf(n, -1), first(n, -1);

    for (int k = 0; k < n; ++k) {
        int j = post[k];
        delta[j] = (first[j] == -1) ? 1 : 0;  // delta = 1 if j is a leaf of the etree
        for (; j != -1 && first[j] == -1; j = parent[j]) {
            first[j] = k;
        }
    }
    for (int i = 0; i < n; ++i) {
        ancestor[i] = i;
    }
    for (int k = 0; k < n; ++k) {
        const int j = post[k];
        if (parent[j] != -1) {
            --delta[parent[j]];  // j is not a root
        }
        for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            const int i = row_idx[p];
            int jleaf = 0;
            const int q = leaf(i, j, first, maxfirst, prevleaf, ancestor, jleaf);
            if (jleaf >= 1) {
                ++delta[j];
            }
            if (jleaf == 2) {
                --delta[q];
            }
        }
        if (parent[j] != -1) {
            ancestor[j] = parent[j];
        }
    }
    // Accumulate subtree contributions: parent[j] > j holds for an etree, so a
    // simple node-order pass sums children into parents.
    for (int j = 0; j < n; ++j) {
        if (parent[j] != -1) {
            delta[parent[j]] += delta[j];
        }
    }
    return delta;
}

long predicted_fill(int n, const int* col_ptr, const int* row_idx)
{
    if (n <= 0) {
        return 0;
    }
    std::vector<int> sym_cp, sym_ri;
    symmetric_pattern(n, col_ptr, row_idx, sym_cp, sym_ri);
    const std::vector<int> parent = etree(n, sym_cp.data(), sym_ri.data());
    const std::vector<int> post = postorder(parent, n);
    const std::vector<int> colcount = column_counts(n, sym_cp.data(), sym_ri.data(), parent, post);
    long lnz = 0;
    for (int c : colcount) {
        lnz += c;
    }
    return 2 * lnz - n;  // L + U proxy (symmetric: U mirrors L; subtract shared diagonal)
}

void permute_pattern(int n, const int* col_ptr, const int* row_idx,
                     const std::vector<int>& perm,
                     std::vector<int>& out_col_ptr, std::vector<int>& out_row_idx)
{
    std::vector<int> iperm(n);
    for (int k = 0; k < n; ++k) {
        iperm[perm[k]] = k;  // new index of old node perm[k]
    }
    out_col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int c = 0; c < n; ++c) {
        out_col_ptr[iperm[c] + 1] += col_ptr[c + 1] - col_ptr[c];
    }
    for (int c = 0; c < n; ++c) {
        out_col_ptr[c + 1] += out_col_ptr[c];
    }
    out_row_idx.assign(static_cast<std::size_t>(col_ptr[n]), 0);
    std::vector<int> next(out_col_ptr.begin(), out_col_ptr.end());
    for (int c = 0; c < n; ++c) {
        const int nc = iperm[c];
        for (int p = col_ptr[c]; p < col_ptr[c + 1]; ++p) {
            out_row_idx[next[nc]++] = iperm[row_idx[p]];
        }
    }
}

long predicted_fill_perm(int n, const int* col_ptr, const int* row_idx,
                         const std::vector<int>& perm)
{
    if (n <= 0) {
        return 0;
    }
    std::vector<int> pcp, pri;
    permute_pattern(n, col_ptr, row_idx, perm, pcp, pri);
    return predicted_fill(n, pcp.data(), pri.data());
}

void fill_pattern(int n, const int* col_ptr, const int* row_idx,
                  const std::vector<int>& parent,
                  std::vector<int>& Lp, std::vector<int>& Li)
{
    Lp.assign(static_cast<std::size_t>(n) + 1, 0);
    Li.clear();
    if (n <= 0) {
        return;
    }

    // Symbolic Cholesky (column-merge form): struct(j) = {j} ∪ {i>j : S(i,j)≠0}
    // ∪ over etree-children c of (struct(c) \ {c}). parent[j] > j, so children
    // are computed before j.
    //
    // Write the merge DIRECTLY into a FLAT Li (no vector<vector> col/head with per-column
    // small-vector allocs + regrowth). column_counts (proven cs_counts) gives the EXACT
    // per-column size (|L(:,j)| == |struct(j)|), so Lp is prefix-summed up front and each
    // column's slice [Lp[j],Lp[j+1]) is filled in place. Consumers re-sort the
    // pattern, so per-column order is free -> output is set-identical to the old version.
    auto flap = [](const char*) {};
    const std::vector<int> post = postorder(parent, n);
    flap("postorder");
    const std::vector<int> cc = column_counts(n, col_ptr, row_idx, parent, post);
    flap("column_counts");
    for (int j = 0; j < n; ++j) Lp[j + 1] = Lp[j] + cc[j];
    Li.assign(static_cast<std::size_t>(Lp[n]), 0);

    // Flat-CSR etree children (replaces the head vector<vector>; child order is
    // irrelevant to the set union).
    std::vector<int> choff(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j)
        if (parent[j] != -1) ++choff[parent[j] + 1];
    for (int j = 0; j < n; ++j) choff[j + 1] += choff[j];
    std::vector<int> chl(choff[n]);
    {
        std::vector<int> cnext(choff.begin(), choff.end());
        for (int j = 0; j < n; ++j)
            if (parent[j] != -1) chl[cnext[parent[j]]++] = j;
    }

    std::vector<int> mark(n, -1);
    for (int j = 0; j < n; ++j) {
        int w = Lp[j];
        mark[j] = j;
        Li[w++] = j;  // diagonal
        for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            const int i = row_idx[p];
            if (i > j && mark[i] != j) {
                mark[i] = j;
                Li[w++] = i;
            }
        }
        for (int t = choff[j]; t < choff[j + 1]; ++t) {
            const int c = chl[t];
            // Li[Lp[c]] is the child's diagonal c, and struct(c) \ {c} is what parent j needs.
            for (int u = Lp[c] + 1; u < Lp[c + 1]; ++u) {
                const int i = Li[u];
                if (mark[i] != j) {
                    mark[i] = j;
                    Li[w++] = i;  // child structure minus the child itself
                }
            }
        }
        // w == Lp[j+1] by construction (cc[j] is exact).
    }
    flap("merge");
}

}  // namespace custom_linear_solver::symbolic
