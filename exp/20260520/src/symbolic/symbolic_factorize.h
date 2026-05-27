#ifndef EXP_20260520_SYMBOLIC_FACTORIZE_H
#define EXP_20260520_SYMBOLIC_FACTORIZE_H

#include "matrix/csc_matrix.h"
#include "matrix/csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;
    int l_nnz;
    int u_nnz;
} SymbolicFactorizeStats;

/*
 * Build no-pivoting LU fill patterns after reordering.
 *
 * Input:
 *   a_reordered - Square reordered matrix in CSR format. Numerical values are
 *                 ignored; only the sparsity pattern is used.
 *
 * Outputs:
 *   l_pattern   - Lower-triangular L pattern in CSC format. The diagonal is
 *                 included.
 *   u_pattern   - Upper-triangular U pattern in CSC format. The diagonal is
 *                 included.
 *
 * The caller owns both output matrices and must release them with csc_destroy().
 * On failure, outputs are left untouched.
 */
int symbolic_factorize_reordered_csr(const CSRMatrix *a_reordered,
                                     CSCMatrix *l_pattern,
                                     CSCMatrix *u_pattern);

int symbolic_factorize_reordered_csr_with_stats(
    const CSRMatrix *a_reordered,
    CSCMatrix *l_pattern,
    CSCMatrix *u_pattern,
    SymbolicFactorizeStats *stats);

#ifdef __cplusplus
}
#endif

#endif
