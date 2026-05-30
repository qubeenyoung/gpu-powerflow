#pragma once

#include <vector>

#include "symbolic/supernode.hpp"

namespace custom_linear_solver::symbolic {

// Multifrontal symbolic analysis (dense-panel large-path, PLAN §M3). For each
// relaxed panel it builds the DENSE FRONT row pattern, the panel elimination tree,
// and the EXTEND-ADD map that places each panel's contribution block (CB) into its
// parent front. All value-independent (depends only on the fill pattern + panel
// partition) -> built once in analyze, reused every factorize.
//
// Front layout per panel P (ncols = panels.ncols[P] pivot columns):
//   front_rows[P] = sorted union of the member columns' filled L structures.
//   The first `ncols` entries are exactly the panel's pivot columns (contiguous
//   integers first..last); the remaining entries are the CB rows (rows > last).
// After eliminating the ncols pivots, the trailing (front-ncols) rows form the CB,
// a dense block that is extend-added into the parent front via asm_idx.
struct MultifrontalSymbolic {
    int num_panels = 0;
    std::vector<int> panel_parent;  // panel id -> parent panel id (-1 = root)
    std::vector<int> front_ptr;     // panel id -> [start,end) in front_rows (size P+1)
    std::vector<int> front_rows;    // flattened sorted front row indices (pivots first)
    std::vector<int> asm_ptr;       // panel id -> [start,end) in asm_idx (size P+1)
    std::vector<int> asm_idx;       // for each CB row of P, its position in the parent
                                    //   front_rows (so extend-add is an indexed scatter)
};

// Build the multifrontal symbolic structure from the filled pattern (Lp/Li, the
// lower factor structure incl. diagonal) and a relaxed panel partition. The
// columns must be in postorder index space (panels contiguous), matching
// relaxed_panels(). asm_idx entries are >= 0 iff the multifrontal invariant
// (cb_rows[child] subset of front_rows[parent]) holds.
MultifrontalSymbolic multifrontal_symbolic(int n, const std::vector<int>& Lp,
                                           const std::vector<int>& Li,
                                           const PanelPartition& panels);

}  // namespace custom_linear_solver::symbolic
