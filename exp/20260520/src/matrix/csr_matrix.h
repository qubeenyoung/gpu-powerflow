#ifndef EXP_20260519_CSR_MATRIX_H
#define EXP_20260519_CSR_MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
    SDS_OK = 0,
    SDS_ERR_BAD_INPUT = -1,
    SDS_ERR_ALLOC = -2,
    SDS_ERR_METIS = -3,
    SDS_ERR_SYMBOLIC = -4,
    SDS_ERR_NUMERIC = -5,
    SDS_ERR_ZERO_PIVOT = -6,
    SDS_ERR_CUDA = -7
};

typedef struct {
    int nrows;
    int ncols;
    int nnz;
    int *rowptr;
    int *colind;
    double *values;
} CSRMatrix;

int csr_create(CSRMatrix *A, int nrows, int ncols, int nnz);
void csr_destroy(CSRMatrix *A);

#ifdef __cplusplus
}
#endif

#endif
