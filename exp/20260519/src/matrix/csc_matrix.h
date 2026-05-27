#ifndef EXP_20260519_CSC_MATRIX_H
#define EXP_20260519_CSC_MATRIX_H

#include "matrix/csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nrows;
    int ncols;
    int nnz;
    int *colptr;
    int *rowind;
    double *values;
} CSCMatrix;

int csc_create(CSCMatrix *A, int nrows, int ncols, int nnz);
void csc_destroy(CSCMatrix *A);
int csr_to_csc(const CSRMatrix *A, CSCMatrix *B);

#ifdef __cplusplus
}
#endif

#endif
