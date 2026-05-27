#include "symbolic/symbolic_factorize.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int *data;
    int size;
    int capacity;
} IntList;

static int validate_square_csr_pattern(const CSRMatrix *A);
static int build_csc_pattern_from_csr(const CSRMatrix *A, CSCMatrix *A_csc);
static int int_list_push(IntList *list, int value);
static void int_list_destroy(IntList *list);
static void destroy_columns(IntList *columns, int n);
static int materialize_csc_pattern(const IntList *columns, int n, CSCMatrix *pattern);

int symbolic_factorize_reordered_csr(const CSRMatrix *a_reordered,
                                     CSCMatrix *l_pattern,
                                     CSCMatrix *u_pattern)
{
    return symbolic_factorize_reordered_csr_with_stats(
        a_reordered, l_pattern, u_pattern, NULL);
}

int symbolic_factorize_reordered_csr_with_stats(
    const CSRMatrix *a_reordered,
    CSCMatrix *l_pattern,
    CSCMatrix *u_pattern,
    SymbolicFactorizeStats *stats)
{
    CSCMatrix a_csc;
    CSCMatrix l_tmp;
    CSCMatrix u_tmp;
    IntList *l_columns = NULL;
    IntList *u_columns = NULL;
    int *mark = NULL;
    int rc;
    const int n = a_reordered ? a_reordered->nrows : 0;

    memset(&a_csc, 0, sizeof(a_csc));
    memset(&l_tmp, 0, sizeof(l_tmp));
    memset(&u_tmp, 0, sizeof(u_tmp));

    if (!l_pattern || !u_pattern || l_pattern == u_pattern) {
        return SDS_ERR_BAD_INPUT;
    }
    rc = validate_square_csr_pattern(a_reordered);
    if (rc != SDS_OK) {
        return rc;
    }

    rc = build_csc_pattern_from_csr(a_reordered, &a_csc);
    if (rc != SDS_OK) {
        return rc;
    }

    l_columns = (IntList *)calloc((size_t)n, sizeof(IntList));
    u_columns = (IntList *)calloc((size_t)n, sizeof(IntList));
    mark = (int *)malloc((size_t)n * sizeof(int));
    if (!l_columns || !u_columns || !mark) {
        rc = SDS_ERR_ALLOC;
        goto fail;
    }
    for (int i = 0; i < n; ++i) {
        mark[i] = -1;
    }

    /*
     * Left-looking symbolic LU without pivoting.
     *
     * For column k, mark starts with A(:,k). Every active row j < k means
     * U(j,k) is present, so L(:,j) contributes additional fill to this column.
     * Since L is lower triangular, a linear j=0..k-1 scan is already a
     * topological traversal: newly marked rows greater than j will still be
     * visited later in the same scan.
     */
    for (int k = 0; k < n; ++k) {
        for (int p = a_csc.colptr[k]; p < a_csc.colptr[k + 1]; ++p) {
            mark[a_csc.rowind[p]] = k;
        }
        mark[k] = k;

        for (int j = 0; j < k; ++j) {
            if (mark[j] != k) {
                continue;
            }
            rc = int_list_push(&u_columns[k], j);
            if (rc != SDS_OK) {
                goto fail;
            }
            for (int p = 0; p < l_columns[j].size; ++p) {
                const int row = l_columns[j].data[p];
                if (row > j) {
                    mark[row] = k;
                }
            }
        }

        rc = int_list_push(&u_columns[k], k);
        if (rc != SDS_OK) {
            goto fail;
        }
        for (int row = k; row < n; ++row) {
            if (mark[row] == k) {
                rc = int_list_push(&l_columns[k], row);
                if (rc != SDS_OK) {
                    goto fail;
                }
            }
        }
    }

    rc = materialize_csc_pattern(l_columns, n, &l_tmp);
    if (rc == SDS_OK) {
        rc = materialize_csc_pattern(u_columns, n, &u_tmp);
    }
    if (rc != SDS_OK) {
        goto fail;
    }

    if (stats) {
        stats->n = n;
        stats->l_nnz = l_tmp.nnz;
        stats->u_nnz = u_tmp.nnz;
    }

    *l_pattern = l_tmp;
    *u_pattern = u_tmp;

    csc_destroy(&a_csc);
    destroy_columns(l_columns, n);
    destroy_columns(u_columns, n);
    free(mark);
    return SDS_OK;

fail:
    csc_destroy(&a_csc);
    csc_destroy(&l_tmp);
    csc_destroy(&u_tmp);
    destroy_columns(l_columns, n);
    destroy_columns(u_columns, n);
    free(mark);
    return rc;
}

static int validate_square_csr_pattern(const CSRMatrix *A)
{
    if (!A || A->nrows <= 0 || A->ncols <= 0 || A->nrows != A->ncols ||
        A->nnz < 0 || !A->rowptr || (A->nnz > 0 && !A->colind)) {
        return SDS_ERR_BAD_INPUT;
    }
    if (A->rowptr[0] != 0 || A->rowptr[A->nrows] != A->nnz) {
        return SDS_ERR_BAD_INPUT;
    }
    for (int row = 0; row < A->nrows; ++row) {
        if (A->rowptr[row] > A->rowptr[row + 1]) {
            return SDS_ERR_BAD_INPUT;
        }
        for (int p = A->rowptr[row]; p < A->rowptr[row + 1]; ++p) {
            if (A->colind[p] < 0 || A->colind[p] >= A->ncols) {
                return SDS_ERR_BAD_INPUT;
            }
        }
    }
    return SDS_OK;
}

static int build_csc_pattern_from_csr(const CSRMatrix *A, CSCMatrix *A_csc)
{
    int *next = NULL;
    int rc;

    rc = csc_create(A_csc, A->nrows, A->ncols, A->nnz);
    if (rc != SDS_OK) {
        return rc;
    }

    for (int row = 0; row < A->nrows; ++row) {
        for (int p = A->rowptr[row]; p < A->rowptr[row + 1]; ++p) {
            ++A_csc->colptr[A->colind[p] + 1];
        }
    }
    for (int col = 0; col < A_csc->ncols; ++col) {
        A_csc->colptr[col + 1] += A_csc->colptr[col];
    }

    next = (int *)malloc(((size_t)A_csc->ncols + 1u) * sizeof(int));
    if (!next) {
        csc_destroy(A_csc);
        return SDS_ERR_ALLOC;
    }
    memcpy(next, A_csc->colptr, ((size_t)A_csc->ncols + 1u) * sizeof(int));

    for (int row = 0; row < A->nrows; ++row) {
        for (int p = A->rowptr[row]; p < A->rowptr[row + 1]; ++p) {
            const int dst = next[A->colind[p]]++;
            A_csc->rowind[dst] = row;
            A_csc->values[dst] = 1.0;
        }
    }

    free(next);
    return SDS_OK;
}

static int int_list_push(IntList *list, int value)
{
    int new_capacity;
    int *new_data;

    if (list->size == list->capacity) {
        new_capacity = list->capacity == 0 ? 8 : list->capacity * 2;
        if (new_capacity < list->capacity || new_capacity > INT_MAX / (int)sizeof(int)) {
            return SDS_ERR_ALLOC;
        }
        new_data = (int *)realloc(list->data, (size_t)new_capacity * sizeof(int));
        if (!new_data) {
            return SDS_ERR_ALLOC;
        }
        list->data = new_data;
        list->capacity = new_capacity;
    }
    list->data[list->size++] = value;
    return SDS_OK;
}

static void int_list_destroy(IntList *list)
{
    if (!list) {
        return;
    }
    free(list->data);
    memset(list, 0, sizeof(*list));
}

static void destroy_columns(IntList *columns, int n)
{
    if (!columns) {
        return;
    }
    for (int col = 0; col < n; ++col) {
        int_list_destroy(&columns[col]);
    }
    free(columns);
}

static int materialize_csc_pattern(const IntList *columns, int n, CSCMatrix *pattern)
{
    long long nnz = 0;
    int rc;
    int offset = 0;

    for (int col = 0; col < n; ++col) {
        nnz += columns[col].size;
        if (nnz > INT_MAX) {
            return SDS_ERR_ALLOC;
        }
    }

    rc = csc_create(pattern, n, n, (int)nnz);
    if (rc != SDS_OK) {
        return rc;
    }

    pattern->colptr[0] = 0;
    for (int col = 0; col < n; ++col) {
        for (int p = 0; p < columns[col].size; ++p) {
            pattern->rowind[offset] = columns[col].data[p];
            pattern->values[offset] = 1.0;
            ++offset;
        }
        pattern->colptr[col + 1] = offset;
    }

    return SDS_OK;
}
