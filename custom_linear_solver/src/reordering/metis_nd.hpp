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
// for the A win; reproducible-benchmark callers leave it false (serial). The env PAR_ND
// also forces parallel (depth = its value) for A/B sweeps.
bool metis_nd(int n, const int* col_ptr, const int* row_idx, std::vector<int>& perm,
              bool parallel = false, std::vector<int>* sym_col_ptr = nullptr,
              std::vector<int>* sym_row_idx = nullptr);

}  // namespace custom_linear_solver::reordering
