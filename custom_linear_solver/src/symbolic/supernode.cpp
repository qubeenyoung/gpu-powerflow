#include "symbolic/supernode.hpp"

namespace custom_linear_solver::symbolic {

// Group columns into supernodes: merge a postorder node into the previous one when it is that
// node's single child and its column structure nests (exact supernodes, no padding).
//   In: etree parent + postorder + column counts.  Out: snode_of[n], sizes, num_supernodes.
SupernodePartition supernodes(int n, const std::vector<int>& parent,
                              const std::vector<int>& post,
                              const std::vector<int>& colcount)
{
    SupernodePartition sp;
    sp.snode_of.assign(n < 0 ? 0 : n, -1);
    if (n <= 0) {
        return sp;
    }

    std::vector<int> nchild(n, 0);
    for (int j = 0; j < n; ++j) {
        if (parent[j] != -1) {
            ++nchild[parent[j]];
        }
    }

    int id = -1;
    for (int k = 0; k < n; ++k) {
        const int j = post[k];
        bool start = true;
        if (k > 0) {
            const int jprev = post[k - 1];
            // Merge only when the previous postorder node is j's single child and
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

// Amalgamate etree chains into panels of up to `cap` columns (relaxed supernodes). A wider cap
// means fewer fronts / fewer solve levels, paid for in padded fill (each panel's dense block is
// sized to its widest member). In: etree parent + column counts + cap. Out: PanelPartition.
PanelPartition relaxed_panels(int n, const std::vector<int>& parent,
                              const std::vector<int>& colcount, int cap)
{
    PanelPartition pp;
    pp.panel_of.assign(n < 0 ? 0 : n, -1);
    if (n <= 0) {
        return pp;
    }
    if (cap < 1) {
        cap = 1;
    }
    // Tensor-core-oriented deep-K amalgamation belongs in the analyze path. A naive "merge any
    // consecutive columns" version is invalid -- a child's contribution block must nest in one
    // parent front -- so relaxed_panels itself only does the safe chain merge below.
    // Postorder index space: an etree chain is a run of consecutive columns with
    // parent[j]==j+1. Greedily extend a panel along such a chain until it would
    // exceed `cap` columns; the panel's dense block is padded to its widest member
    // (max colcount), so padded_fill = Σ ncols*width is the dense storage cost.
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
