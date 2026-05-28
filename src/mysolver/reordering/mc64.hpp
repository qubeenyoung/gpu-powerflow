#pragma once

#include <vector>

// MC64-style maximum-product bipartite matching (PLAN §M1 matching). Finds a row
// permutation that places large entries on the diagonal so the no-pivot numeric
// factorization stays stable on matrices with poor natural diagonals (circuit
// matrices: rajat27/onetone2 hit a zero pivot, rajat15 is inaccurate, without it).
//
// Solves the linear assignment problem maximizing sum_j log|A(match[j], j)| via
// successive shortest augmenting paths (Dijkstra with dual potentials) on the
// sparse bipartite graph. Equivalent to MC64 job=4 (the matching part).
namespace mysolver::reordering {

// Match columns of the n×n CSC matrix (Ap/Ai/Ax) to distinct rows, maximizing the
// product of matched |entries|. On success fills match_col (size n): match_col[j]
// is the row assigned to column j (a permutation). Returns false if the matrix
// has no perfect matching (structurally singular).
bool mc64_match(int n, const int* Ap, const int* Ai, const double* Ax,
                std::vector<int>& match_col);

}  // namespace mysolver::reordering
