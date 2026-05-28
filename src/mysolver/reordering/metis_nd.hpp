#pragma once

#include <vector>

// METIS nested-dissection fill-reducing ordering (PLAN §M1).
//
// M1 step: mysolver generates the permutation; the numeric factorization is
// still delegated to KLU. The ordering is fed to KLU through its user_order
// hook (klu_common.ordering = 3), so KLU keeps doing BTF + numeric while the
// fill-reducing order on each block comes from METIS ND.
namespace mysolver::reordering {

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
              bool parallel = false);

}  // namespace mysolver::reordering

// KLU user-ordering callback (klu_common.ordering = 3). C-compatible signature
// matching klu_common::user_order. Returns a positive lnz estimate on success,
// 0 on failure (KLU treats 0 as failure).
struct klu_common_struct;
int klu_metis_user_order(int n, int* Ap, int* Ai, int* Perm, struct klu_common_struct* common);
