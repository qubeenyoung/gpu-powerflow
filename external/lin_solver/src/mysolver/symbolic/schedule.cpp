#include "mysolver/symbolic/schedule.hpp"

#include <algorithm>

namespace mysolver::symbolic {

SupernodeSchedule build_schedule(int n, const std::vector<int>& parent,
                                 const SupernodePartition& sp)
{
    SupernodeSchedule sched;
    const int s = sp.num_supernodes;
    sched.num_supernodes = s;
    if (s <= 0) {
        return sched;
    }
    sched.super_parent.assign(s, -1);
    sched.level.assign(s, 0);

    // Supernode etree: the only column of a supernode whose etree-parent leaves
    // the supernode is its top, and it defines the supernode's parent.
    for (int j = 0; j < n; ++j) {
        const int p = parent[j];
        if (p != -1 && sp.snode_of[p] != sp.snode_of[j]) {
            sched.super_parent[sp.snode_of[j]] = sp.snode_of[p];
        }
    }

    // super_parent[c] > c (postorder numbering), so a forward pass gives levels.
    for (int c = 0; c < s; ++c) {
        const int p = sched.super_parent[c];
        if (p != -1) {
            sched.level[p] = std::max(sched.level[p], sched.level[c] + 1);
        }
    }

    sched.num_levels = 0;
    for (int c = 0; c < s; ++c) {
        sched.num_levels = std::max(sched.num_levels, sched.level[c] + 1);
    }
    sched.levels.assign(sched.num_levels, {});
    for (int c = 0; c < s; ++c) {
        sched.levels[sched.level[c]].push_back(c);
    }
    return sched;
}

}  // namespace mysolver::symbolic
