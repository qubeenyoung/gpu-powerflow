#include "analyze/symbolic/supernode.hpp"

namespace custom_linear_solver::symbolic {

// Group columns into Supernodes: merge a Postorder node into the previous one
// when it is that node's single child and its column structure nests (exact
// Supernodes, no padding).
//   In: Etree parent + Postorder + column counts.  Out: snode_of[n], sizes,
//   num_supernodes.
SupernodePartition Supernodes(int n, const std::vector<int>& parent,
                              const std::vector<int>& post,
                              const std::vector<int>& colcount) {
  SupernodePartition sp;
  sp.snode_of.assign(n < 0 ? 0 : n, -1);
  if (n <= 0) {
    return sp;
  }

  // Count each node's children (single-child is the merge precondition).
  std::vector<int> nchild(n, 0);
  for (int j = 0; j < n; ++j) {
    if (parent[j] != -1) {
      ++nchild[parent[j]];
    }
  }

  // Walk Postorder; start a new supernode unless the previous node nests.
  int id = -1;
  for (int k = 0; k < n; ++k) {
    const int j = post[k];
    bool start = true;
    if (k > 0) {
      const int jprev = post[k - 1];
      // Merge only when the previous Postorder node is j's single child and
      // its column structure nests inside j's (one extra nonzero: its diag).
      const bool nests = (parent[jprev] == j) && (nchild[j] == 1) &&
                         (colcount[jprev] == colcount[j] + 1);
      start = !nests;
    }
    if (start) {
      ++id;
      sp.sizes.push_back(0);
    }
    sp.snode_of[j] = id;
    ++sp.sizes[id];
  }
  sp.num_supernodes = id + 1;
  return sp;
}

// Amalgamate Etree chains into panels of up to `cap` columns (relaxed
// Supernodes). A wider cap means fewer fronts / fewer Solve levels, paid for in
// padded fill (each panel's dense block is sized to its widest member). In:
// Etree parent + column counts + cap. Out: PanelPartition.
PanelPartition RelaxedPanels(int n, const std::vector<int>& parent,
                             const std::vector<int>& colcount, int cap) {
  PanelPartition pp;
  pp.panel_of.assign(n < 0 ? 0 : n, -1);
  if (n <= 0) {
    return pp;
  }
  if (cap < 1) {
    cap = 1;
  }

  // Merge only along an Etree chain (parent[j]==j+1 in Postorder index space)
  // so each child's contribution block still nests in one parent front. Greedily
  // extend a panel along the chain until it would exceed `cap` columns; the
  // dense block is padded to the panel's widest member, so
  // padded_fill = Σ ncols*width is the dense storage cost.
  for (int j = 0; j < n;) {
    const int start = j;
    int sz = 1, w = colcount[j];
    while (j + 1 < n && sz < cap && parent[j] == j + 1) {
      ++j;
      ++sz;
      if (colcount[j] > w) {
        w = colcount[j];
      }
    }
    const int id = pp.num_panels++;
    pp.first.push_back(start);
    pp.ncols.push_back(sz);
    pp.width.push_back(w);
    pp.padded_fill += static_cast<long>(sz) * w;
    for (int c = start; c <= j; ++c) {
      pp.panel_of[c] = id;
    }
    ++j;
  }
  return pp;
}

}  // namespace custom_linear_solver::symbolic
