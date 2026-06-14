#pragma once

#include <vector>

namespace custom_linear_solver::symbolic {

// Fundamental supernode partition (PLAN §M2 supernode).
// A supernode is a maximal set of columns, contiguous in postorder, whose
// below-diagonal nonzero structures nest, so they can be factored as one dense
// block (the unit of work for the M3 GPU factorization).
struct SupernodePartition {
    int num_supernodes = 0;
    std::vector<int> snode_of;  // original column index -> supernode id
    std::vector<int> sizes;     // supernode id -> number of columns
};

// Build the fundamental supernode partition (Liu-Ng-Peyton) from the elimination
// tree parent[], a postorder of it, and the column counts of L. Column j merges
// with the preceding postorder node iff that node is j's only child and the
// structures nest (colcount[child] == colcount[j] + 1).
SupernodePartition supernodes(int n, const std::vector<int>& parent,
                              const std::vector<int>& post,
                              const std::vector<int>& colcount);

// Relaxed-amalgamation panel partition (the dense-panel multifrontal foundation,
// PLAN §M3 large-path). Fundamental supernodes here average ~2 columns, too small
// for dense BLAS; RELAXED amalgamation merges etree chains into wider
// panels (mean ~3.4 cols at cap=8), each treated as one dense trapezoidal block
// padded to the panel's widest column structure. The padded-fill cost stays modest
// (~1.2x at cap=8 on power-grid Jacobians, amalg_stats), making dense GPU panels
// viable where the per-op atomicAdd scatter is contention-bound. `parent` and
// `colcount` must be in postorder index space (etree chains are consecutive), as
// produced after an AMD/METIS reorder; a chain merges while it stays a single-child
// path (parent[j]==j+1) and the panel stays <= cap columns.
struct PanelPartition {
    int num_panels = 0;
    std::vector<int> panel_of;  // column -> panel id
    std::vector<int> first;     // panel id -> first column (panels are contiguous)
    std::vector<int> ncols;     // panel id -> number of columns
    std::vector<int> width;     // panel id -> widest member colcount (dense block rows)
    long padded_fill = 0;       // Σ ncols*width  (dense storage; vs true_fill = Σ colcount)
};

PanelPartition relaxed_panels(int n, const std::vector<int>& parent,
                              const std::vector<int>& colcount, int cap);

// Deep-K amalgamation (exp_260612 compute-bound experiment, CLS_AMALG_K). relaxed_panels only
// merges single-child etree CHAINS, so power-grid leaf regions stay thin (nc ~ colcount ~ 2-3);
// the batched trailing GEMM is then thin-K (K=nc) and memory-bound. deep_k_panels additionally
// absorbs whole child SUBTREES into a parent panel to thicken nc toward `cap_nc` (capped at 32 by
// the kTensorCorePivotColumnCap + single-warp solve substitution limits), raising arithmetic
// intensity so tensor cores can fire.
//
// Validity (the nesting invariant relaxed_panels guards against): a panel must be a postorder-
// CONTIGUOUS column range so its member columns sort to the leading nc front rows, and each
// merged child's contribution block must nest in the single parent front. Both hold here because
// (a) we only ever absorb a child whose grown range is immediately adjacent below the parent's
// current first column (contiguity preserved), and (b) by the elimination-tree theorem a column's
// CB nests in its parent's structure, so absorbing it leaves the parent's CB unchanged -> a single
// parent front. multifrontal_symbolic still re-validates via asm_idx==-1.
//
// `cap_nc` bounds pivot columns per panel (clamped to [1,32]). Built on top of relaxed_panels(cap_nc)
// so chain merges are retained; only the extra sibling/subtree absorption is new.
PanelPartition deep_k_panels(int n, const std::vector<int>& parent,
                             const std::vector<int>& colcount, int cap_nc);

}  // namespace custom_linear_solver::symbolic
