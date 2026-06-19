#pragma once

#include <vector>

// METIS nested-dissection fill-reducing ordering.
namespace custom_linear_solver::reordering {

// ND ordering from a prebuilt symmetric graph (xadj size n+1, adjncy = directed
// unique off-diagonal edges, neighbors sorted+deduped per vertex). The inputs
// may be consumed (moved from) when METIS's idx_t is 32-bit. Fills `perm`
// (size n) in METIS convention: perm[new_pos] = old_vertex. Returns false only
// on invalid input (n < 0).
//
// parallel=true uses parallel nested dissection (vertex-separator split, halves
// recurse on separate threads) for a large wall-time win with fill comparable to
// serial METIS. METIS's RNG is a thread-unsafe global, so the parallel ordering
// is non-deterministic run-to-run; reproducible-benchmark callers must pass
// parallel=false (serial METIS_NodeND).
//
// par_nd_depth / par_nd_base_small / par_nd_base_large tune the parallel-ND
// recursion (only used when parallel=true): max recursion depth, and the
// subgraph-size base-case threshold below which a branch drops to serial METIS
// (small value for n < ~20k, large above).
bool MetisNdFromGraph(int n, std::vector<int>& xadj, std::vector<int>& adjncy,
                      std::vector<int>& perm, bool parallel, int seed = 42,
                      int par_nd_depth = 4, int par_nd_base_small = 4000,
                      int par_nd_base_large = 20000);

}  // namespace custom_linear_solver::reordering
