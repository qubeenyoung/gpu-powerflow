#include "symbolic/symbolic_supernode.h"

#include <stdlib.h>
#include <string.h>

static int validate_csc_pattern(const CSCMatrix *A);
static void column_pattern_range(const CSCMatrix *A,
                                 int col,
                                 SymbolicSupernodePatternMode mode,
                                 int *begin,
                                 int *end);
static int same_column_pattern(const CSCMatrix *A,
                               int left_col,
                               int right_col,
                               SymbolicSupernodePatternMode mode);

int symbolic_supernodes_find_strict(const CSCMatrix *factor_pattern,
                                    SymbolicSupernodeSet *supernodes)
{
    return symbolic_supernodes_find_strict_with_mode(
        factor_pattern, SYMBOLIC_SUPERNODE_PATTERN_STRICT_TAIL, supernodes);
}

int symbolic_supernodes_find_strict_with_mode(
    const CSCMatrix *factor_pattern,
    SymbolicSupernodePatternMode mode,
    SymbolicSupernodeSet *supernodes)
{
    int *ptr = NULL;
    int *column_to_supernode = NULL;
    int n;
    int num_supernodes = 0;
    int max_width = 0;
    int rc;

    if (!supernodes ||
        (mode != SYMBOLIC_SUPERNODE_PATTERN_FULL_COLUMN &&
         mode != SYMBOLIC_SUPERNODE_PATTERN_STRICT_TAIL)) {
        return SDS_ERR_BAD_INPUT;
    }
    rc = validate_csc_pattern(factor_pattern);
    if (rc != SDS_OK) {
        return rc;
    }

    memset(supernodes, 0, sizeof(*supernodes));
    n = factor_pattern->ncols;

    ptr = (int *)malloc(((size_t)n + 1u) * sizeof(int));
    column_to_supernode = (int *)malloc((size_t)n * sizeof(int));
    if (!ptr || !column_to_supernode) {
        free(ptr);
        free(column_to_supernode);
        return SDS_ERR_ALLOC;
    }

    ptr[0] = 0;
    column_to_supernode[0] = 0;
    num_supernodes = 1;

    for (int col = 1; col < n; ++col) {
        if (!same_column_pattern(factor_pattern, col - 1, col, mode)) {
            ptr[num_supernodes] = col;
            ++num_supernodes;
        }
        column_to_supernode[col] = num_supernodes - 1;
    }
    ptr[num_supernodes] = n;

    for (int s = 0; s < num_supernodes; ++s) {
        const int width = ptr[s + 1] - ptr[s];
        if (width > max_width) {
            max_width = width;
        }
    }

    supernodes->n = n;
    supernodes->num_supernodes = num_supernodes;
    supernodes->max_width = max_width;
    supernodes->supernode_ptr = ptr;
    supernodes->column_to_supernode = column_to_supernode;
    return SDS_OK;
}

int symbolic_supernodes_validate(const SymbolicSupernodeSet *supernodes)
{
    int current_supernode = 0;

    if (!supernodes || supernodes->n <= 0 || supernodes->num_supernodes <= 0 ||
        !supernodes->supernode_ptr || !supernodes->column_to_supernode) {
        return SDS_ERR_BAD_INPUT;
    }
    if (supernodes->supernode_ptr[0] != 0 ||
        supernodes->supernode_ptr[supernodes->num_supernodes] != supernodes->n) {
        return SDS_ERR_BAD_INPUT;
    }
    for (int s = 0; s < supernodes->num_supernodes; ++s) {
        const int begin = supernodes->supernode_ptr[s];
        const int end = supernodes->supernode_ptr[s + 1];
        if (begin < 0 || begin >= end || end > supernodes->n) {
            return SDS_ERR_BAD_INPUT;
        }
        if (end - begin > supernodes->max_width) {
            return SDS_ERR_BAD_INPUT;
        }
    }
    for (int col = 0; col < supernodes->n; ++col) {
        while (current_supernode + 1 < supernodes->num_supernodes &&
               col >= supernodes->supernode_ptr[current_supernode + 1]) {
            ++current_supernode;
        }
        if (supernodes->column_to_supernode[col] != current_supernode) {
            return SDS_ERR_BAD_INPUT;
        }
    }
    return SDS_OK;
}

void symbolic_supernodes_destroy(SymbolicSupernodeSet *supernodes)
{
    if (!supernodes) {
        return;
    }
    free(supernodes->supernode_ptr);
    free(supernodes->column_to_supernode);
    memset(supernodes, 0, sizeof(*supernodes));
}

static int validate_csc_pattern(const CSCMatrix *A)
{
    if (!A || A->nrows <= 0 || A->ncols <= 0 || A->nrows != A->ncols ||
        A->nnz < 0 || !A->colptr || (A->nnz > 0 && !A->rowind)) {
        return SDS_ERR_BAD_INPUT;
    }
    if (A->colptr[0] != 0 || A->colptr[A->ncols] != A->nnz) {
        return SDS_ERR_BAD_INPUT;
    }
    for (int col = 0; col < A->ncols; ++col) {
        int previous_row = -1;
        if (A->colptr[col] > A->colptr[col + 1]) {
            return SDS_ERR_BAD_INPUT;
        }
        for (int p = A->colptr[col]; p < A->colptr[col + 1]; ++p) {
            const int row = A->rowind[p];
            if (row < 0 || row >= A->nrows || row <= previous_row) {
                return SDS_ERR_BAD_INPUT;
            }
            previous_row = row;
        }
    }
    return SDS_OK;
}

static void column_pattern_range(const CSCMatrix *A,
                                 int col,
                                 SymbolicSupernodePatternMode mode,
                                 int *begin,
                                 int *end)
{
    int first = A->colptr[col];
    const int last = A->colptr[col + 1];

    if (mode == SYMBOLIC_SUPERNODE_PATTERN_STRICT_TAIL) {
        while (first < last && A->rowind[first] <= col) {
            ++first;
        }
    }

    *begin = first;
    *end = last;
}

static int same_column_pattern(const CSCMatrix *A,
                               int left_col,
                               int right_col,
                               SymbolicSupernodePatternMode mode)
{
    int left_begin;
    int left_end;
    int right_begin;
    int right_end;
    int left_len;
    int right_len;

    column_pattern_range(A, left_col, mode, &left_begin, &left_end);
    column_pattern_range(A, right_col, mode, &right_begin, &right_end);

    left_len = left_end - left_begin;
    right_len = right_end - right_begin;
    if (left_len != right_len) {
        return 0;
    }
    if (left_len == 0) {
        return 1;
    }
    return memcmp(A->rowind + left_begin,
                  A->rowind + right_begin,
                  (size_t)left_len * sizeof(int)) == 0;
}
