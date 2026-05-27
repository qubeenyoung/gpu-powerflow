#include "matrix/csc_matrix.h"

#include <stdlib.h>
#include <string.h>

int csc_create(CSCMatrix *A, int nrows, int ncols, int nnz)
{
    if (!A || nrows <= 0 || ncols <= 0 || nnz < 0) {
        return SDS_ERR_BAD_INPUT;
    }

    memset(A, 0, sizeof(*A));
    A->nrows = nrows;
    A->ncols = ncols;
    A->nnz = nnz;

    A->colptr = (int *)calloc((size_t)ncols + 1u, sizeof(int));
    A->rowind = nnz > 0 ? (int *)malloc((size_t)nnz * sizeof(int)) : NULL;
    A->values = nnz > 0 ? (double *)malloc((size_t)nnz * sizeof(double)) : NULL;

    if (!A->colptr || (nnz > 0 && (!A->rowind || !A->values))) {
        csc_destroy(A);
        return SDS_ERR_ALLOC;
    }

    return SDS_OK;
}

void csc_destroy(CSCMatrix *A)
{
    if (!A) {
        return;
    }
    free(A->colptr);
    free(A->rowind);
    free(A->values);
    memset(A, 0, sizeof(*A));
}

int csr_to_csc(const CSRMatrix *A, CSCMatrix *B)
{
    int *next = NULL;
    int rc;

    if (!A || !B || A->nrows <= 0 || A->ncols <= 0 || A->nnz < 0 ||
        !A->rowptr || (A->nnz > 0 && (!A->colind || !A->values))) {
        return SDS_ERR_BAD_INPUT;
    }

    rc = csc_create(B, A->nrows, A->ncols, A->nnz);
    if (rc != SDS_OK) {
        return rc;
    }

    for (int p = 0; p < A->nnz; ++p) {
        const int col = A->colind[p];
        if (col < 0 || col >= A->ncols) {
            csc_destroy(B);
            return SDS_ERR_BAD_INPUT;
        }
        ++B->colptr[col + 1];
    }
    for (int col = 0; col < B->ncols; ++col) {
        B->colptr[col + 1] += B->colptr[col];
    }

    next = (int *)malloc(((size_t)B->ncols + 1u) * sizeof(int));
    if (!next) {
        csc_destroy(B);
        return SDS_ERR_ALLOC;
    }
    memcpy(next, B->colptr, ((size_t)B->ncols + 1u) * sizeof(int));

    for (int row = 0; row < A->nrows; ++row) {
        for (int p = A->rowptr[row]; p < A->rowptr[row + 1]; ++p) {
            const int col = A->colind[p];
            const int dst = next[col]++;
            B->rowind[dst] = row;
            B->values[dst] = A->values[p];
        }
    }

    free(next);
    return SDS_OK;
}
