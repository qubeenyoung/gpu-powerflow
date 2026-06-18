#pragma once

#include <vector>

// METIS nested-dissection fill-reducing ordering.
namespace custom_linear_solver::reordering {

// ND ordering from a prebuilt symmetric graph (xadj size n+1, adjncy = directed
// unique off-diagonal edges, neighbors sorted+deduped per vertex) — e.g. one
// built on the GPU by matrix::BuildSymmetricGraphDevice. The inputs may be
// moved from (consumed) when METIS's idx_t is 32-bit. Fills `perm` (size n) in
// METIS convention: perm[new_pos] = old_vertex. Returns false only on invalid
// input (n < 0).
//
// parallel=true uses parallel nested dissection: a vertex separator splits the
// graph into two independent halves that recurse on separate threads (~-40%
// Analyze wall on large Jacobians; fill ~= serial METIS, so the downstream
// factor stays competitive). NOTE: METIS's RNG is a thread-unsafe global, so
// the parallel ordering is NON-deterministic run-to-run (within a single run /
// NR loop the ordering is computed once and fixed). Production enables it for
// the wall win; reproducible-benchmark callers pass parallel=false (serial
// METIS_NodeND).
bool MetisNdFromGraph(int n, std::vector<int>& xadj, std::vector<int>& adjncy,
                      std::vector<int>& perm, bool parallel, int seed = 42);

}  // namespace custom_linear_solver::reordering
