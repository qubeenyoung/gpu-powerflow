#include "analyze/symbolic/supernode.hpp"

#include <algorithm>

namespace custom_linear_solver::symbolic {

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

PanelPartition deep_k_panels(int n, const std::vector<int>& parent,
                             const std::vector<int>& colcount, int cap_nc)
{
    if (cap_nc < 1) cap_nc = 1;
    if (cap_nc > 32) cap_nc = 32;  // TC pivot cap + single-warp solve substitution limit

    // Base chain partition (already valid, nc <= cap_nc). We coarsen it by absorbing whole child
    // subtrees into the parent, bottom-up, while contiguity + the nc cap hold.
    const PanelPartition base = relaxed_panels(n, parent, colcount, cap_nc);
    PanelPartition pp;
    pp.panel_of.assign(n < 0 ? 0 : n, -1);
    if (n <= 0) return pp;
    const int Pb = base.num_panels;

    std::vector<int> bfirst(Pb), blast(Pb);
    for (int p = 0; p < Pb; ++p) {
        bfirst[p] = base.first[p];
        blast[p] = base.first[p] + base.ncols[p] - 1;
    }
    // Base panel-tree parent: the panel owning the etree-exit of the chain's last column.
    std::vector<int> ppar(Pb, -1);
    for (int p = 0; p < Pb; ++p) {
        const int e = parent[blast[p]];           // parent[j] > j in postorder space (or -1 at root)
        ppar[p] = (e < 0) ? -1 : base.panel_of[e];
    }
    std::vector<std::vector<int>> children(Pb);
    for (int p = 0; p < Pb; ++p)
        if (ppar[p] != -1) children[ppar[p]].push_back(p);
    // Sort each child list ascending by last column; we absorb closest-to-parent first (descending).
    for (int p = 0; p < Pb; ++p)
        std::sort(children[p].begin(), children[p].end(),
                  [&](int a, int b) { return blast[a] < blast[b]; });

    // Grown state per base panel. A panel only ever grows DOWNWARD (cur_first shrinks); cur_last
    // stays blast. cur_nc == cur_last - cur_first + 1 (contiguous, no gaps).
    std::vector<int> cur_first = bfirst;
    std::vector<int> cur_nc = base.ncols;
    std::vector<int> cur_width = base.width;
    std::vector<char> absorbed(Pb, 0);

    // Process children before parents: increasing last column.
    std::vector<int> order(Pb);
    for (int p = 0; p < Pb; ++p) order[p] = p;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return blast[a] < blast[b]; });

    for (int oi = 0; oi < Pb; ++oi) {
        const int P = order[oi];
        // Absorb children closest-first; stop at the first non-adjacent / over-cap child so the
        // merged column range stays contiguous and pivot-count stays within cap.
        for (int ci = static_cast<int>(children[P].size()) - 1; ci >= 0; --ci) {
            const int C = children[P][ci];
            if (blast[C] != cur_first[P] - 1) break;            // contiguity: C must sit right below
            if (cur_nc[P] + cur_nc[C] > cap_nc) break;          // pivot-column cap
            cur_first[P] = cur_first[C];
            cur_nc[P] += cur_nc[C];
            cur_width[P] = std::max(cur_width[P], cur_width[C]);
            absorbed[C] = 1;
        }
    }

    // Emit survivors in column order. Their extended ranges re-partition [0, n) with no overlap.
    for (int p = 0; p < Pb; ++p) {
        if (absorbed[p]) continue;
        const int id = pp.num_panels++;
        pp.first.push_back(cur_first[p]);
        pp.ncols.push_back(cur_nc[p]);
        pp.width.push_back(cur_width[p]);
        pp.padded_fill += static_cast<long>(cur_nc[p]) * cur_width[p];
        for (int c = cur_first[p]; c <= blast[p]; ++c) pp.panel_of[c] = id;
    }
    return pp;
}

}  // namespace custom_linear_solver::symbolic
