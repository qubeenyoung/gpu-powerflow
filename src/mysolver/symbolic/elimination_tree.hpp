#pragma once

#include <vector>

// Symbolic stage building blocks (PLAN §M2).
namespace mysolver::symbolic {

// Symmetric elimination tree of a CSC pattern, treating the pattern as
// symmetric (identical semantics to CXSparse cs_etree(A, ata=0)).
// Returns parent[] of size n; parent[k] is k's parent, or -1 if k is a root.
std::vector<int> etree(int n, const int* col_ptr, const int* row_idx);

// Build the symmetric pattern S = A + Aᵀ (off-diagonal, deduped) of a CSC
// matrix, as CSC arrays. Diagonal is omitted (irrelevant for the etree and for
// fill counts). Feed the result to etree() / column counts.
void symmetric_pattern(int n, const int* col_ptr, const int* row_idx,
                       std::vector<int>& sym_col_ptr, std::vector<int>& sym_row_idx);

// Postorder of the elimination tree given its parent array (matches the node
// ordering produced by CXSparse cs_post).
std::vector<int> postorder(const std::vector<int>& parent, int n);

// Column counts of the Cholesky factor L of a symmetric CSC pattern, given the
// etree parent and a postorder (matches CXSparse cs_counts(A, ata=0)).
// colcount[k] = number of nonzeros in column k of L, including the diagonal.
std::vector<int> column_counts(int n, const int* col_ptr, const int* row_idx,
                               const std::vector<int>& parent,
                               const std::vector<int>& post);

// Predicted symmetric fill proxy for a CSC matrix: builds S = A+Aᵀ, its etree,
// postorder and column counts, and returns 2*nnz(L) - n (an L+U estimate used to
// compare candidate orderings without factorizing).
long predicted_fill(int n, const int* col_ptr, const int* row_idx);

// Pattern of P·A·Pᵀ where perm[new_position] = old_index. Output is CSC pattern
// arrays (no values). Used to score a candidate ordering via predicted_fill.
void permute_pattern(int n, const int* col_ptr, const int* row_idx,
                     const std::vector<int>& perm,
                     std::vector<int>& out_col_ptr, std::vector<int>& out_row_idx);

// Predicted L+U fill proxy for a CSC matrix reordered by perm (= predicted_fill
// of P·A·Pᵀ). Lets analyze() compare candidate orderings without factorizing.
long predicted_fill_perm(int n, const int* col_ptr, const int* row_idx,
                         const std::vector<int>& perm);

// Symbolic Cholesky fill pattern of L (lower triangle incl. diagonal) for an
// already-ordered symmetric CSC pattern whose elimination order is the natural
// order 0..n-1 (so parent[j] > j). Output is CSC arrays (Lp, Li), row indices
// sorted per column. Total nnz(Li) equals sum(column_counts). This is the static
// pattern the M3 numeric factorization fills in.
void fill_pattern(int n, const int* col_ptr, const int* row_idx,
                  const std::vector<int>& parent,
                  std::vector<int>& Lp, std::vector<int>& Li);

}  // namespace mysolver::symbolic
