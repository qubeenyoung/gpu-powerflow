#pragma once

#include <vector>

// METIS nested-dissection fill-reducing ordering.
namespace custom_linear_solver::reordering {

// Fill `perm` (size n) with a METIS ND ordering of the symmetric pattern
// (A + Aᵀ, diagonal excluded) of the n×n matrix given in CSC/CSR-pattern form
// (col_ptr/row_idx; the pattern is symmetric in structure either way).
// Falls back to the natural order if METIS cannot order the graph.
// Returns false only on invalid input.
// parallel=true uses the cy155 parallel nested dissection (recurse separator halves across
// cores, ~-40% A on large matrices, fill ~= serial so F kept competitive). NOTE: METIS's
// RNG is a thread-unsafe global -> the parallel ordering is NON-deterministic run-to-run
// (within a run / NR loop the ordering is computed once and fixed). Production enables it
// for the A win; reproducible-benchmark callers leave it false (serial).
bool metis_nd(int n, const int* col_ptr, const int* row_idx, std::vector<int>& perm,
              bool parallel = false, std::vector<int>* sym_col_ptr = nullptr,
              std::vector<int>* sym_row_idx = nullptr);

// ND ordering from a prebuilt symmetric graph (xadj size n+1, adjncy = directed unique
// off-diagonal edges, neighbors sorted+deduped per vertex) — e.g. one built on the GPU by
// matrix::build_symmetric_graph_device. The inputs may be moved from (consumed) when the
// METIS idx_t is 32-bit. Same ordering as metis_nd given the same graph.
bool metis_nd_from_graph(int n, std::vector<int>& xadj, std::vector<int>& adjncy,
                         std::vector<int>& perm, bool parallel);

}  // namespace custom_linear_solver::reordering
