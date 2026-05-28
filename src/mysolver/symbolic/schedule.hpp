#pragma once

#include <vector>

#include "mysolver/symbolic/supernode.hpp"

namespace mysolver::symbolic {

// Supernode dependency / level schedule (PLAN §M2 dependency + schedule).
//
// The supernode elimination tree captures the factorization dependencies: a
// supernode can be factored once all its descendants are done. Supernodes at the
// same level are mutually independent and form one GPU launch (the forward /
// factorize order processes levels 0..L; triangular backward solve walks them in
// reverse).
struct SupernodeSchedule {
    int num_supernodes = 0;
    int num_levels = 0;
    std::vector<int> super_parent;          // supernode etree parent (-1 = root)
    std::vector<int> level;                 // level[s]: 0 = leaf; equal level => independent
    std::vector<std::vector<int>> levels;   // levels[L] = supernodes at level L (forward order)
};

// Build the schedule from the elimination tree parent[] and a supernode
// partition. super_parent[s] > s holds (supernodes are numbered in postorder),
// so a single forward pass computes the levels.
SupernodeSchedule build_schedule(int n, const std::vector<int>& parent,
                                 const SupernodePartition& sp);

}  // namespace mysolver::symbolic
