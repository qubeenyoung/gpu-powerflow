#ifndef EXP_20260519_METIS_ND_H
#define EXP_20260519_METIS_ND_H

#include "matrix/csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Function: build_metis_graph
 *
 * Purpose:
 *   Build the undirected METIS graph directly from a structurally symmetric
 *   sparse matrix pattern.
 *
 * Inputs:
 *   matrix                      - Square CSR matrix. Numerical values are not
 *                                 used by this function.
 *   validate_symmetric_pattern  - If nonzero, verify that every off-diagonal
 *                                 edge i -> j also has edge j -> i.
 *
 * Outputs:
 *   graph                       - CSR graph with diagonal entries removed.
 *                                 The caller owns the allocated arrays.
 *
 * Returns:
 *   SDS_OK on success.
 *   SDS_ERR_BAD_INPUT if the matrix is invalid or symmetry validation fails.
 *   SDS_ERR_ALLOC if allocation fails.
 *
 * Notes:
 *   This function does not build A + A^T. It assumes the input pattern is
 *   already structurally symmetric and only removes self-loops plus row-local
 *   duplicates.
 */
int build_metis_graph(
    const CSRMatrix *matrix,
    CSRMatrix *graph,
    int validate_symmetric_pattern);

/*
 * Function: compute_metis_nd_ordering
 *
 * Purpose:
 *   Compute a nested-dissection ordering using METIS_NodeND.
 *
 * Inputs:
 *   graph     - Square undirected CSR adjacency graph without diagonal
 *               self-loops.
 *
 * Outputs:
 *   perm      - METIS permutation array. Caller owns the storage.
 *               perm[old_index] = new_index.
 *   inv_perm  - Inverse permutation array. Caller owns the storage.
 *               inv_perm[new_index] = old_index.
 *
 * Returns:
 *   SDS_OK on success.
 *   SDS_ERR_BAD_INPUT if graph or arrays are invalid.
 *   SDS_ERR_ALLOC if temporary allocation fails.
 *   SDS_ERR_METIS if METIS_NodeND fails.
 *
 * Notes:
 *   The graph must be undirected. Use
 *   build_metis_graph() for matrices whose sparsity pattern is structurally
 *   symmetric.
 */
int compute_metis_nd_ordering(const CSRMatrix *graph, int *perm, int *inv_perm);

/*
 * Function: apply_symmetric_permutation
 *
 * Purpose:
 *   Apply the symmetric permutation A_perm = P A P^T.
 *
 * Inputs:
 *   matrix   - Square CSR matrix.
 *   perm     - Permutation with perm[old_index] = new_index.
 *
 * Outputs:
 *   matrix_perm - Permuted CSR matrix. The caller owns the allocated arrays.
 *
 * Returns:
 *   SDS_OK on success.
 *   SDS_ERR_BAD_INPUT if inputs are invalid or perm is not a bijection.
 *   SDS_ERR_ALLOC if allocation fails.
 *
 * Notes:
 *   The common path is O(nnz): count rows, prefix sum, scatter. Rows are then
 *   sorted and duplicate coordinates, if any, are merged row-locally.
 */
int apply_symmetric_permutation(const CSRMatrix *matrix,
                                const int *perm,
                                CSRMatrix *matrix_perm);

#ifdef __cplusplus
}
#endif

#endif
