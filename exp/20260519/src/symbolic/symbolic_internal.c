#include "symbolic/symbolic_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int symbolic_validate_csc_pattern(const CSCMatrix *A)
{
    if (!A || A->nrows <= 0 || A->ncols <= 0 || A->nrows != A->ncols ||
        A->nnz < 0 || !A->colptr || (A->nnz > 0 && !A->rowind)) {
        return SDS_ERR_BAD_INPUT;
    }
    if (A->colptr[0] != 0 || A->colptr[A->ncols] != A->nnz) {
        return SDS_ERR_BAD_INPUT;
    }
    for (int col = 0; col < A->ncols; ++col) {
        if (A->colptr[col] > A->colptr[col + 1]) {
            return SDS_ERR_BAD_INPUT;
        }
        for (int p = A->colptr[col]; p < A->colptr[col + 1]; ++p) {
            if (A->rowind[p] < 0 || A->rowind[p] >= A->nrows) {
                return SDS_ERR_BAD_INPUT;
            }
        }
    }
    return SDS_OK;
}

int symbolic_validate_permutation_array(const int *perm, int n)
{
    int *seen = NULL;

    if (!perm || n <= 0) {
        return SDS_ERR_BAD_INPUT;
    }
    seen = (int *)calloc((size_t)n, sizeof(int));
    if (!seen) {
        return SDS_ERR_ALLOC;
    }
    for (int i = 0; i < n; ++i) {
        if (perm[i] < 0 || perm[i] >= n || seen[perm[i]]) {
            free(seen);
            return SDS_ERR_BAD_INPUT;
        }
        seen[perm[i]] = 1;
    }
    free(seen);
    return SDS_OK;
}

int symbolic_int_list_append(SymbolicIntList *list, int value)
{
    int *next;

    if (!list) {
        return SDS_ERR_BAD_INPUT;
    }
    if (list->count == list->capacity) {
        int new_capacity = list->capacity ? 2 * list->capacity : 8;
        next = (int *)realloc(list->values, (size_t)new_capacity * sizeof(int));
        if (!next) {
            return SDS_ERR_ALLOC;
        }
        list->values = next;
        list->capacity = new_capacity;
    }
    list->values[list->count++] = value;
    return SDS_OK;
}

void symbolic_int_list_destroy(SymbolicIntList *list)
{
    if (!list) {
        return;
    }
    free(list->values);
    memset(list, 0, sizeof(*list));
}

int symbolic_int_list_add_marked(SymbolicIntList *list,
                                 int *marker,
                                 int marker_id,
                                 int value)
{
    if (!list || !marker || value < 0) {
        return SDS_ERR_BAD_INPUT;
    }
    if (marker[value] == marker_id) {
        return SDS_OK;
    }
    marker[value] = marker_id;
    return symbolic_int_list_append(list, value);
}

/* 셸 정렬(shell sort): 전력 시스템 행렬에서 전형적인 소규모 update set에서
 * qsort의 함수 포인터 호출 오버헤드 없이 충분한 성능을 낸다. */
void symbolic_sort_values_by_position(int *values, int count, const int *position)
{
    if (!values || count <= 1 || !position) {
        return;
    }

    for (int gap = count / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < count; ++i) {
            const int value = values[i];
            int j = i;
            while (j >= gap && position[values[j - gap]] > position[value]) {
                values[j] = values[j - gap];
                j -= gap;
            }
            values[j] = value;
        }
    }
}

int symbolic_build_position_from_fronts(const SeparatorTree *tree,
                                        const FrontSymbolic *fronts,
                                        int n,
                                        int *position)
{
    if (!tree || !fronts || !position || n <= 0) {
        return SDS_ERR_BAD_INPUT;
    }

    for (int i = 0; i < n; ++i) {
        position[i] = -1;
    }
    for (int sep = 0; sep < tree->num_separators; ++sep) {
        const FrontSymbolic *front = &fronts[sep];
        for (int i = 0; i < front->num_pivots; ++i) {
            const int var = front->pivot_vars[i];
            if (var < 0 || var >= n || position[var] != -1) {
                return SDS_ERR_SYMBOLIC;
            }
            position[var] = front->sep_begin + i;
        }
    }
    for (int var = 0; var < n; ++var) {
        if (position[var] == -1) {
            return SDS_ERR_SYMBOLIC;
        }
    }
    return SDS_OK;
}

int symbolic_find_front_var(const FrontSymbolic *front, int var)
{
    if (!front) {
        return -1;
    }
    for (int i = 0; i < front->num_front_vars; ++i) {
        if (front->front_vars[i] == var) {
            return i;
        }
    }
    return -1;
}

int symbolic_build_front_owner_map(const SymbolicFactorization *symbolic,
                                   int *owner)
{
    if (!symbolic || !owner || symbolic->n <= 0) {
        return SDS_ERR_BAD_INPUT;
    }
    for (int i = 0; i < symbolic->n; ++i) {
        owner[i] = -1;
    }
    for (int front_id = 0; front_id < symbolic->num_fronts; ++front_id) {
        const FrontSymbolic *front = &symbolic->fronts[front_id];
        for (int i = 0; i < front->num_pivots; ++i) {
            const int var = front->pivot_vars[i];
            if (var < 0 || var >= symbolic->n || owner[var] != -1) {
                return SDS_ERR_SYMBOLIC;
            }
            owner[var] = front_id;
        }
    }
    for (int var = 0; var < symbolic->n; ++var) {
        if (owner[var] == -1) {
            return SDS_ERR_SYMBOLIC;
        }
    }
    return SDS_OK;
}

size_t symbolic_square_size(int n)
{
    const size_t value = (size_t)n;
    return value * value;
}

void symbolic_front_destroy(FrontSymbolic *front)
{
    if (!front) {
        return;
    }
    free(front->pivot_vars);
    free(front->update_vars);
    free(front->front_vars);
    free(front->update_to_parent);
    memset(front, 0, sizeof(*front));
}

void symbolic_fronts_destroy(FrontSymbolic *fronts, int num_fronts)
{
    if (!fronts) {
        return;
    }
    for (int i = 0; i < num_fronts; ++i) {
        symbolic_front_destroy(&fronts[i]);
    }
}

void symbolic_print_int_array(const char *name, const int *values, int n)
{
    printf("%s:", name);
    for (int i = 0; i < n; ++i) {
        printf(" %d", values ? values[i] : -1);
    }
    printf("\n");
}
