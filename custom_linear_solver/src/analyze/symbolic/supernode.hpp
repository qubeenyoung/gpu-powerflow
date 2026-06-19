#pragma once

#include <vector>

namespace custom_linear_solver::symbolic {

// Fundamental supernode partition (PLAN §M2 supernode).
// A supernode is a maximal set of columns, contiguous in Postorder, whose
// below-diagonal nonzero structures nest, so they can be factored as one dense
// block (the unit of work for the M3 GPU factorization).
struct SupernodePartition {
  int num_supernodes = 0;
  std::vector<int> snode_of;  // original column index -> supernode id
  std::vector<int> sizes;     // supernode id -> number of columns
};

// Build the fundamental supernode partition (Liu-Ng-Peyton) from the
// elimination tree parent[], a Postorder of it, and the column counts of L.
// Column j merges with the preceding Postorder node iff that node is j's only
// child and the structures nest (colcount[child] == colcount[j] + 1).
SupernodePartition Supernodes(int n, const std::vector<int>& parent,
                              const std::vector<int>& post,
                              const std::vector<int>& colcount);

// Relaxed-amalgamation panel partition (the dense-panel multifrontal
// foundation). Fundamental Supernodes are often too small for dense BLAS;
// relaxed amalgamation merges Etree chains into wider panels, each treated as
// one dense trapezoidal block padded to the panel's widest column structure.
// Padding keeps the dense GPU panels viable where the per-op atomicAdd scatter
// is contention-bound, at a modest padded-fill cost.
// `parent` and `colcount` must be in Postorder index space (Etree chains are
// consecutive), as produced after an AMD/METIS reorder; a chain merges while it
// stays a single-child path (parent[j]==j+1) and the panel stays <= cap
// columns.
struct PanelPartition {
  int num_panels = 0;
  std::vector<int> panel_of;  // column -> panel id
  std::vector<int> first;  // panel id -> first column (panels are contiguous)
  std::vector<int> ncols;  // panel id -> number of columns
  std::vector<int>
      width;  // panel id -> widest member colcount (dense block rows)
  long padded_fill =
      0;  // Σ ncols*width  (dense storage; vs true_fill = Σ colcount)
};

PanelPartition RelaxedPanels(int n, const std::vector<int>& parent,
                             const std::vector<int>& colcount, int cap);

}  // namespace custom_linear_solver::symbolic
