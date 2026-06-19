#include "analyze/symbolic/elimination_tree.hpp"

#include <algorithm>
#include <thread>

namespace custom_linear_solver::symbolic {

// Elimination tree of a symmetric pattern: parent[j] = the node that inherits
// column j's fill (-1 for roots). Path-compressed. In: CSC pattern. Out:
// parent[n].
std::vector<int> Etree(int n, const int* col_ptr, const int* row_idx) {
  std::vector<int> parent(n, -1);
  std::vector<int> ancestor(n, -1);  // path-compressed ancestor of each node

  // Process columns in order; each entry below the diagonal links a subtree.
  for (int k = 0; k < n; ++k) {
    for (int p = col_ptr[k]; p < col_ptr[k + 1]; ++p) {
      int i = row_idx[p];

      // Walk from i up to the root, attaching the path to k.
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

// Symmetric (A+Aᵀ) adjacency pattern in CSC, per column sorted + deduped,
// diagonal removed. In: CSC pattern. Out: sym_col_ptr[n+1] / sym_row_idx.
void SymmetricPattern(int n, const int* col_ptr, const int* row_idx,
                      std::vector<int>& sym_col_ptr,
                      std::vector<int>& sym_row_idx) {
  // 1. Count adjacency degrees (with duplicates; symmetric input double-counts)
  //    into a flat CSR offset array, then prefix-sum it.
  std::vector<int> off(static_cast<std::size_t>(n) + 1, 0);
  auto valid = [&](int row, int col) {
    return row >= 0 && row < n && row != col;
  };
  for (int col = 0; col < n; ++col)
    for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
      const int row = row_idx[p];
      if (!valid(row, col)) continue;
      ++off[row + 1];
      ++off[col + 1];
    }
  for (int i = 0; i < n; ++i) off[i + 1] += off[i];

  // 2. Bucket-fill both directed edges of each off-diagonal entry.
  std::vector<int> adj(off[n]);
  std::vector<int> next(off.begin(), off.end());
  for (int col = 0; col < n; ++col)
    for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
      const int row = row_idx[p];
      if (!valid(row, col)) continue;
      adj[next[row]++] = col;
      adj[next[col]++] = row;
    }

  // Parallelize the per-node sort+dedup+compact. Each node's slice is
  // independent, so the output is identical regardless of threading (sort+dedup
  // per col, concatenated in col order via prefix-sum).
  auto ParFor = [](int lo, int hi, auto&& fn) {
    unsigned hw = std::thread::hardware_concurrency();
    const int nth = static_cast<int>(std::max(1u, std::min(hw ? hw : 1u, 12u)));
    if (hi - lo < 32768 || nth <= 1) {
      fn(lo, hi);
      return;
    }  // small: thread overhead > gain
    std::vector<std::thread> th;
    const int chunk = (hi - lo + nth - 1) / nth;
    for (int t = 0; t < nth; ++t) {
      const int a = lo + t * chunk, b = std::min(hi, a + chunk);
      if (a < b) th.emplace_back([&fn, a, b] { fn(a, b); });
    }
    for (auto& x : th) x.join();
  };

  // 3. Sort each slice in place and count uniques per node.
  std::vector<int> ucnt(n, 0);
  ParFor(0, n,
         [&](int lo, int hi) {
           for (int col = lo; col < hi; ++col) {
             const int b = off[col], e = off[col + 1];
             std::sort(adj.begin() + b, adj.begin() + e);
             int u = 0, last = -1;
             for (int k = b; k < e; ++k)
               if (adj[k] != last) {
                 ++u;
                 last = adj[k];
               }
             ucnt[col] = u;
           }
         });

  // 4. Prefix-sum the unique counts into the output column pointers.
  sym_col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (int col = 0; col < n; ++col)
    sym_col_ptr[col + 1] = sym_col_ptr[col] + ucnt[col];
  sym_row_idx.resize(sym_col_ptr[n]);

  // 5. Compact uniques into the prefix-summed slots.
  ParFor(0, n,
         [&](int lo, int hi) {
           for (int col = lo; col < hi; ++col) {
             int w = sym_col_ptr[col], last = -1;
             for (int k = off[col]; k < off[col + 1]; ++k)
               if (adj[k] != last) {
                 sym_row_idx[w++] = adj[k];
                 last = adj[k];
               }
           }
         });
}

// Postorder of the Etree (children emitted before parents). In: parent[n]. Out:
// post[n].
std::vector<int> Postorder(const std::vector<int>& parent, int n) {
  // Build reversed per-parent child lists.
  std::vector<int> head(n, -1), next(n, -1), stack(n), post(n);
  for (int j = n - 1; j >= 0; --j) {
    if (parent[j] == -1) {
      continue;
    }
    next[j] = head[parent[j]];
    head[parent[j]] = j;
  }

  // Iterative DFS from each root, emitting a node once its children are done.
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
        head[p] = next[child];  // advance child list
        stack[++top] = child;   // descend
      }
    }
  }
  return post;
}

namespace {

// Least-common-ancestor / Leaf detection helper for the column-count pass.
int Leaf(int i, int j, const std::vector<int>& first,
         std::vector<int>& maxfirst, std::vector<int>& prevleaf,
         std::vector<int>& ancestor, int& jleaf) {
  jleaf = 0;
  if (i <= j || first[j] <= maxfirst[i]) {
    return -1;
  }
  maxfirst[i] = first[j];
  const int jprev = prevleaf[i];
  prevleaf[i] = j;
  jleaf = (jprev == -1) ? 1 : 2;  // 1 = first Leaf of subtree, 2 = subsequent
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

// Exact nonzero count of each column of the Cholesky factor L. In: CSC pattern
// + Etree parent + Postorder. Out: colcount[n] (= |struct(j)|).
std::vector<int> ColumnCounts(int n, const int* col_ptr, const int* row_idx,
                              const std::vector<int>& parent,
                              const std::vector<int>& post) {
  std::vector<int> delta(n, 0);
  std::vector<int> ancestor(n), maxfirst(n, -1), prevleaf(n, -1), first(n, -1);

  // 1. Record the first Postorder time each node is reached and seed leaf deltas.
  for (int k = 0; k < n; ++k) {
    int j = post[k];
    delta[j] =
        (first[j] == -1) ? 1 : 0;  // delta = 1 if j is a Leaf of the Etree
    for (; j != -1 && first[j] == -1; j = parent[j]) {
      first[j] = k;
    }
  }

  // 2. Accumulate per-column deltas via leaf detection over each column's rows.
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
      const int q = Leaf(i, j, first, maxfirst, prevleaf, ancestor, jleaf);
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

  // 3. Roll subtree contributions up: parent[j] > j in an Etree, so a single
  //    node-order pass sums children into parents.
  for (int j = 0; j < n; ++j) {
    if (parent[j] != -1) {
      delta[parent[j]] += delta[j];
    }
  }
  return delta;
}

// Predicted L+U fill of a pattern (for comparing orderings): 2*nnz(L) - n.
long PredictedFill(int n, const int* col_ptr, const int* row_idx) {
  if (n <= 0) {
    return 0;
  }

  // Build the symmetric pattern, its Etree, postorder, and column counts.
  std::vector<int> sym_cp, sym_ri;
  SymmetricPattern(n, col_ptr, row_idx, sym_cp, sym_ri);
  const std::vector<int> parent = Etree(n, sym_cp.data(), sym_ri.data());
  const std::vector<int> post = Postorder(parent, n);
  const std::vector<int> colcount =
      ColumnCounts(n, sym_cp.data(), sym_ri.data(), parent, post);

  // Sum column counts into nnz(L) and return the L+U proxy.
  long lnz = 0;
  for (int c : colcount) {
    lnz += c;
  }
  return 2 * lnz -
         n;  // L + U proxy (symmetric: U mirrors L; subtract shared diagonal)
}

// Relabel a CSC pattern by `perm` (out node iperm[c] = old node c). In/Out: CSC
// patterns.
void PermutePattern(int n, const int* col_ptr, const int* row_idx,
                    const std::vector<int>& perm, std::vector<int>& out_col_ptr,
                    std::vector<int>& out_row_idx) {
  // Invert the permutation to map each old node to its new index.
  std::vector<int> iperm(n);
  for (int k = 0; k < n; ++k) {
    iperm[perm[k]] = k;  // new index of old node perm[k]
  }

  // Size and prefix-sum the new column pointers from per-column nnz.
  out_col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (int c = 0; c < n; ++c) {
    out_col_ptr[iperm[c] + 1] += col_ptr[c + 1] - col_ptr[c];
  }
  for (int c = 0; c < n; ++c) {
    out_col_ptr[c + 1] += out_col_ptr[c];
  }

  // Scatter each old column's relabeled rows into its new column slot.
  out_row_idx.assign(static_cast<std::size_t>(col_ptr[n]), 0);
  std::vector<int> next(out_col_ptr.begin(), out_col_ptr.end());
  for (int c = 0; c < n; ++c) {
    const int nc = iperm[c];
    for (int p = col_ptr[c]; p < col_ptr[c + 1]; ++p) {
      out_row_idx[next[nc]++] = iperm[row_idx[p]];
    }
  }
}

// PredictedFill after applying `perm` (compares a candidate ordering's fill).
long PredictedFillPerm(int n, const int* col_ptr, const int* row_idx,
                       const std::vector<int>& perm) {
  if (n <= 0) {
    return 0;
  }
  std::vector<int> pcp, pri;
  PermutePattern(n, col_ptr, row_idx, perm, pcp, pri);
  return PredictedFill(n, pcp.data(), pri.data());
}

// Symbolic Cholesky: the nonzero structure of each column of L, exact-sized.
// In: CSC pattern + Etree parent. Out: Lp[n+1] / Li (per-column row indices,
// unsorted).
void FillPattern(int n, const int* col_ptr, const int* row_idx,
                 const std::vector<int>& parent, std::vector<int>& Lp,
                 std::vector<int>& Li) {
  Lp.assign(static_cast<std::size_t>(n) + 1, 0);
  Li.clear();
  if (n <= 0) {
    return;
  }

  // Symbolic Cholesky (column-merge form): struct(j) = {j} ∪ {i>j : S(i,j)≠0}
  // ∪ over Etree-children c of (struct(c) \ {c}). parent[j] > j, so children
  // are computed before j. ColumnCounts gives the exact per-column size
  // (|L(:,j)| == |struct(j)|), so Lp is prefix-summed up front and each
  // column's slice [Lp[j],Lp[j+1]) is filled in place in a flat Li. Consumers
  // re-sort, so per-column order is irrelevant.

  // 1. Postorder + exact column counts, prefix-summed into Lp.
  auto flap = [](const char*) {};
  const std::vector<int> post = Postorder(parent, n);
  flap("Postorder");
  const std::vector<int> cc = ColumnCounts(n, col_ptr, row_idx, parent, post);
  flap("ColumnCounts");
  for (int j = 0; j < n; ++j) Lp[j + 1] = Lp[j] + cc[j];
  Li.assign(static_cast<std::size_t>(Lp[n]), 0);

  // 2. Flat-CSR Etree children (child order is irrelevant to the set union).
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

  // 3. Merge each column's own rows and its children's structures into its slice.
  std::vector<int> mark(n, -1);
  for (int j = 0; j < n; ++j) {
    int w = Lp[j];
    mark[j] = j;
    Li[w++] = j;  // diagonal

    // Below-diagonal entries of column j from the input pattern.
    for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
      const int i = row_idx[p];
      if (i > j && mark[i] != j) {
        mark[i] = j;
        Li[w++] = i;
      }
    }

    // Union in each Etree child's structure (minus the child itself).
    for (int t = choff[j]; t < choff[j + 1]; ++t) {
      const int c = chl[t];
      // Li[Lp[c]] is the child's diagonal c, and struct(c) \ {c} is what parent
      // j needs.
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
