#pragma once

#include <vector>

#include "symbolic/supernode.hpp"

namespace custom_linear_solver::symbolic {

// Etree-aware amalgamation + postorder reordering result.
//
// Background: the existing `relaxed_panels` does *chain merge* (postorder columns whose
// etree edges form `parent[j] == j+1`). For power-grid Jacobians this captures only
// fundamental supernodes (avg 2 cols/panel), leaving the panel etree 30+ deep on
// case8387 (vs cuDSS depth=10) and 39 on USA. The bottleneck per factor is then a long
// chain of small per-level kernel launches.
//
// `amalgamate_and_repostorder` does etree-aware bottom-up greedy merge: it walks the
// chain panels bottom-up and merges each panel with its etree parent when the combined
// size stays <= amal_cap. This is sibling-merge friendly (multiple siblings can fold
// into one parent), which is what halves the depth.
//
// Merging children with a parent makes the union of their cols *non-contiguous* in the
// current postorder (other children sit between them). To restore the contiguity that
// the multifrontal kernels assume, we then postorder the supernode etree and emit each
// supernode's cols (in their original relative order) as one contiguous range. The
// emitted column permutation must be composed with the existing METIS perm by the
// caller (Solver::analyze).
//
// Python prototype results (case8387 starting from chain cap=8):
//   amal_cap= 32 -> depth=12 (cuDSS-class for power-grid)
//   amal_cap= 64 -> depth= 9 (cuDSS-class)
// On USA:  amal_cap=128 -> depth=12, amal_cap=256 -> depth=11.
//
// Safety: amal_cap is bounded by kernel-side limits. The single-batch invert-pivot
// kernel (factorize/multifrontal.cu MF_REG_NC=16 today) and batched invert
// (batched/factor_kernels.cuh MF_REG_NC=32) require nc <= those bounds. With
// amal_cap <= 32 the batched/TC paths stay safe; >32 needs lifting MF_REG_NC.
struct AmalgamateResult {
    // `perm[new_idx] = old_idx` -- additional permutation, composes with the existing
    // METIS perm to give the final column ordering. iperm is its inverse.
    std::vector<int> perm;
    std::vector<int> iperm;

    // Panels in the FINAL (post-repostorder) index space. Cols of each panel are a
    // contiguous range [first[p], first[p]+ncols[p]). Suitable to feed to
    // `multifrontal_symbolic` as the panel partition.
    PanelPartition panels;

    // Etree parent in the FINAL index space. new_parent[j] is the etree parent of col j
    // (in new index space) or -1 if root. Satisfies new_parent[j] > j (topological).
    std::vector<int> new_parent;
};

// Build amalgamated + repostorder result. `parent` and `chain_panels.panel_of` are in
// the same input index space (typically: METIS perm + an explicit postorder pass if
// the caller wants chain merge to capture all chains). `amal_cap` is the merge cap.
AmalgamateResult amalgamate_and_repostorder(int n, const std::vector<int>& parent,
                                            const PanelPartition& chain_panels, int amal_cap);

}  // namespace custom_linear_solver::symbolic
