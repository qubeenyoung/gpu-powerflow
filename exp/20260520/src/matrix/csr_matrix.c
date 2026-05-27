#include "matrix/csr_matrix.h"

#include <stdlib.h>
#include <string.h>

int csr_create(CSRMatrix *A, int nrows, int ncols, int nnz)
{
    if (!A || nrows <= 0 || ncols <= 0 || nnz < 0) {
        return SDS_ERR_BAD_INPUT;
    }

    memset(A, 0, sizeof(*A));
    A->nrows = nrows;
    A->ncols = ncols;
    A->nnz = nnz;

    A->rowptr = (int *)calloc((size_t)nrows + 1u, sizeof(int));
    A->colind = nnz > 0 ? (int *)malloc((size_t)nnz * sizeof(int)) : NULL;
    A->values = nnz > 0 ? (double *)malloc((size_t)nnz * sizeof(double)) : NULL;

    if (!A->rowptr || (nnz > 0 && (!A->colind || !A->values))) {
        csr_destroy(A);
        return SDS_ERR_ALLOC;
    }

    return SDS_OK;
}

void csr_destroy(CSRMatrix *A)
{
    if (!A) {
        return;
    }
    free(A->rowptr);
    free(A->colind);
    free(A->values);
    memset(A, 0, sizeof(*A));
}
