#include "symbolic/amalgamate.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace custom_linear_solver::symbolic {

namespace {

// Build supernode parent (in chain-panel space) by walking the etree edges and grouping
// by which chain panel each col belongs to.
std::vector<int> build_chain_panel_parent(int n, const std::vector<int>& parent,
                                          const PanelPartition& chain_panels)
{
    const int P = chain_panels.num_panels;
    std::vector<int> chain_parent(P, -1);
    for (int j = 0; j < n; ++j) {
        if (parent[j] == -1) continue;
        const int my = chain_panels.panel_of[j];
        const int par = chain_panels.panel_of[parent[j]];
        if (my != par) chain_parent[my] = par;  // chain panels are nested; assignment overwrites
                                                 // are fine — all CB rows of `my` resolve to `par`
    }
    return chain_parent;
}

// Union-find with path compression.
struct UF {
    std::vector<int> rep;
    explicit UF(int n) : rep(n) { std::iota(rep.begin(), rep.end(), 0); }
    int find(int x)
    {
        while (rep[x] != x) {
            rep[x] = rep[rep[x]];
            x = rep[x];
        }
        return x;
    }
};

}  // namespace

AmalgamateResult amalgamate_and_repostorder(int n, const std::vector<int>& parent,
                                            const PanelPartition& chain_panels, int amal_cap)
{
    AmalgamateResult result;
    if (n <= 0 || chain_panels.num_panels <= 0) {
        result.panels = chain_panels;
        result.new_parent = parent;
        result.perm.resize(n);
        std::iota(result.perm.begin(), result.perm.end(), 0);
        result.iperm = result.perm;
        return result;
    }
    if (amal_cap < 1) amal_cap = 1;

    const int P = chain_panels.num_panels;

    // ---- Step 1: chain-panel etree ---------------------------------------------------------
    const std::vector<int> chain_parent = build_chain_panel_parent(n, parent, chain_panels);

    // ---- Step 1.5: compute panel level (depth from root, so root=0; leaves have largest)
    // and panel child count. We use these for cost-aware merging:
    //   - top-of-tree (low depth-from-root, i.e., near root) panels have few siblings and
    //     small fronts -> merging saves a kernel launch with little extra work. ALWAYS merge.
    //   - bottom-of-tree (deep, many siblings) panels have wide leaf set -> merging would
    //     grow per-front work as O(merged_fsz^3), dominating any launch savings.
    //
    // CLS_AMAL_MIN_DEPTH selects the depth-from-leaves cutoff: only merge panels whose
    // depth-from-leaves >= this threshold (i.e., closer to root). Default 0 = merge all.
    int min_depth_from_leaves = 0;
    if (const char* md = std::getenv("CLS_AMAL_MIN_DEPTH")) min_depth_from_leaves = std::atoi(md);

    std::vector<int> depth_from_leaves(P, 0);
    if (min_depth_from_leaves > 0) {
        // panels are roughly in postorder -> single forward pass propagates child max+1 to parent
        for (int p = 0; p < P; ++p) {
            const int par = chain_parent[p];
            if (par >= 0 && depth_from_leaves[par] < depth_from_leaves[p] + 1) {
                depth_from_leaves[par] = depth_from_leaves[p] + 1;
            }
        }
    }

    // ---- Step 2: greedy bottom-up etree amalgamation ---------------------------------------
    // For each panel in reverse postorder (panels are numbered in postorder by
    // relaxed_panels), try to merge into its parent if combined column count <= amal_cap.
    // Iterate until no more merges happen (or hit safety cap).
    std::vector<int> sizes(P, 0);
    for (int j = 0; j < n; ++j) ++sizes[chain_panels.panel_of[j]];

    UF uf(P);
    bool changed = true;
    int iters = 0;
    constexpr int MAX_ITERS = 30;
    while (changed && iters < MAX_ITERS) {
        changed = false;
        // Reverse panel-id iteration ≈ reverse postorder ≈ leaves first
        for (int p = P - 1; p >= 0; --p) {
            const int pr = uf.find(p);
            const int par = chain_parent[pr];
            if (par < 0) continue;
            const int par_r = uf.find(par);
            if (pr == par_r) continue;
            // Cost-aware gate: only merge if BOTH endpoints are at/above the depth-from-leaves
            // threshold. Leaves (depth_from_leaves[p]=0) are excluded when threshold > 0 ->
            // wide flat bottom of the tree stays as fundamental supernodes, while the spine
            // (large depth-from-leaves) gets aggressively merged.
            if (min_depth_from_leaves > 0 &&
                (depth_from_leaves[pr] < min_depth_from_leaves ||
                 depth_from_leaves[par_r] < min_depth_from_leaves)) {
                continue;
            }
            if (sizes[pr] + sizes[par_r] <= amal_cap) {
                sizes[par_r] += sizes[pr];
                sizes[pr] = 0;
                uf.rep[pr] = par_r;
                changed = true;
            }
        }
        ++iters;
    }

    // ---- Step 3: supernode etree (collapse rep groups) -------------------------------------
    // Each surviving rep is one supernode. Assign sequential local IDs in *original
    // chain-panel order* of their root (will be re-ordered by supernode-etree postorder
    // shortly).
    std::vector<int> sn_local_id(P, -1);
    int sn_count = 0;
    for (int p = 0; p < P; ++p) {
        if (uf.find(p) == p) {
            sn_local_id[p] = sn_count++;
        }
    }

    // Supernode parent (in local IDs) from chain_parent of each root.
    std::vector<int> sn_parent(sn_count, -1);
    for (int p = 0; p < P; ++p) {
        if (uf.find(p) != p) continue;
        const int par = chain_parent[p];
        if (par < 0) continue;
        const int par_root = uf.find(par);
        if (par_root == p) continue;
        sn_parent[sn_local_id[p]] = sn_local_id[par_root];
    }

    // ---- Step 4: supernode-etree postorder -------------------------------------------------
    // Build flat children offsets (avoid vector<vector> for large P).
    std::vector<int> ch_off(sn_count + 1, 0);
    for (int s = 0; s < sn_count; ++s) {
        if (sn_parent[s] != -1) ++ch_off[sn_parent[s] + 1];
    }
    for (int s = 0; s < sn_count; ++s) ch_off[s + 1] += ch_off[s];
    std::vector<int> ch_list(ch_off[sn_count]);
    {
        std::vector<int> cnext(ch_off.begin(), ch_off.end());
        for (int s = 0; s < sn_count; ++s) {
            if (sn_parent[s] != -1) ch_list[cnext[sn_parent[s]]++] = s;
        }
    }

    std::vector<int> sn_post;
    sn_post.reserve(sn_count);
    {
        // Iterative DFS postorder. Stack holds (node, child_index_to_visit_next).
        std::vector<int> stack;
        std::vector<int> next_child(sn_count, 0);
        for (int root = 0; root < sn_count; ++root) {
            if (sn_parent[root] != -1) continue;
            stack.push_back(root);
            while (!stack.empty()) {
                const int x = stack.back();
                const int kidx = next_child[x];
                if (kidx < ch_off[x + 1] - ch_off[x]) {
                    ++next_child[x];
                    stack.push_back(ch_list[ch_off[x] + kidx]);
                } else {
                    sn_post.push_back(x);
                    stack.pop_back();
                }
            }
        }
    }
    assert(static_cast<int>(sn_post.size()) == sn_count);

    // Mapping from sn_local_id -> position in postorder (= final supernode id)
    std::vector<int> sn_post_pos(sn_count);
    for (int k = 0; k < sn_count; ++k) sn_post_pos[sn_post[k]] = k;

    // ---- Step 5: group cols by supernode (in chain-panel post-order, which mirrors
    //              within-supernode topological order: parent col indices in the chain
    //              panel space are > child indices, so emitting cols in increasing old-idx
    //              keeps new_parent[j] > j within each supernode).
    std::vector<int> col_sn_local(n);
    for (int j = 0; j < n; ++j) {
        col_sn_local[j] = sn_local_id[uf.find(chain_panels.panel_of[j])];
    }

    // Count cols per supernode
    std::vector<int> sn_col_count(sn_count, 0);
    for (int j = 0; j < n; ++j) ++sn_col_count[col_sn_local[j]];

    // Compute first[final_sn_id] = starting new_idx (offset in the new ordering)
    std::vector<int> first(sn_count, 0);
    for (int k = 0; k < sn_count; ++k) {
        const int sn = sn_post[k];
        first[k] = (k == 0) ? 0 : first[k - 1] + sn_col_count[sn_post[k - 1]];
    }

    // Emit cols in order: for each final supernode (postorder), walk j=0..n-1 and emit
    // those belonging to it. To avoid an O(sn_count * n) scan, instead walk j once and
    // place each col into its supernode's slot via per-sn write cursor.
    std::vector<int> write_cursor(sn_count);
    for (int k = 0; k < sn_count; ++k) write_cursor[k] = first[k];

    result.perm.assign(n, -1);
    result.iperm.assign(n, -1);
    result.panels.panel_of.assign(n, -1);
    result.panels.first.assign(sn_count, 0);
    result.panels.ncols.assign(sn_count, 0);
    result.panels.width.assign(sn_count, 0);  // populated by caller from colcount
    result.panels.num_panels = sn_count;
    result.panels.padded_fill = 0;

    for (int old_j = 0; old_j < n; ++old_j) {
        const int sn_local = col_sn_local[old_j];
        const int final_id = sn_post_pos[sn_local];
        const int new_idx = write_cursor[final_id]++;
        result.perm[new_idx] = old_j;
        result.iperm[old_j] = new_idx;
        result.panels.panel_of[new_idx] = final_id;
    }
    for (int k = 0; k < sn_count; ++k) {
        result.panels.first[k] = first[k];
        result.panels.ncols[k] = sn_col_count[sn_post[k]];
    }

    // ---- Step 6: parent[] in new index space -----------------------------------------------
    result.new_parent.assign(n, -1);
    for (int old_j = 0; old_j < n; ++old_j) {
        if (parent[old_j] == -1) continue;
        const int new_j = result.iperm[old_j];
        const int new_par = result.iperm[parent[old_j]];
        result.new_parent[new_j] = new_par;
        // Topological property: within one supernode, old-idx order preserves
        // parent > child because chain_panels were built in postorder of the etree and
        // new-idx within a supernode follows old-idx order. Across supernodes, postorder
        // of the supernode etree guarantees descendant supernodes come before ancestors.
        // assert(new_par > new_j);  // (enforced by construction; left as a debug check)
    }

    return result;
}

}  // namespace custom_linear_solver::symbolic
