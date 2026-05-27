#include "symbolic/front_symbolic.h"
#include "symbolic/symbolic_internal.h"

#include <stdlib.h>
#include <string.h>

/*
 * Algorithmic reference:
 *   third_party/lin_sol/strumpack/src/sparse/EliminationTree.hpp
 *   third_party/lin_sol/strumpack/src/sparse/EliminationTree.cpp
 *   third_party/lin_sol/strumpack/src/sparse/fronts/Front.hpp
 *   third_party/lin_sol/strumpack/src/sparse/fronts/Front.cpp
 *
 * Referenced concepts:
 *   - symbolic_factorization(): child updates are merged into parent fronts.
 *   - setup_tree(): one front per separator tree node.
 *   - Front::upd_to_parent(): child update variables map to parent front vars.
 *
 * This file reimplements those ideas in small C-style code. It does not copy
 * STRUMPACK source code.
 */

static int build_update_set_for_front(int sep,
                                      const CSCMatrix *A,
                                      FrontSymbolic *fronts,
                                      const int *position,
                                      int *marker,
                                      int marker_id);

int symbolic_build_front_tree(const SeparatorTree *tree,
                              const int *perm,
                              int n,
                              FrontSymbolic *fronts)
{
    if (!tree || !perm || !fronts ||
        separator_tree_validate(tree, n) != SDS_OK ||
        symbolic_validate_permutation_array(perm, n) != SDS_OK) {
        return SDS_ERR_BAD_INPUT;
    }

    for (int sep = 0; sep < tree->num_separators; ++sep) {
        FrontSymbolic *front = &fronts[sep];
        memset(front, 0, sizeof(*front));
        front->front_id = sep;
        front->separator_id = sep;
        front->parent = tree->parent[sep];
        front->left_child = tree->left_child[sep];
        front->right_child = tree->right_child[sep];
        front->sep_begin = tree->sizes[sep];
        front->sep_end = tree->sizes[sep + 1];
        front->num_pivots = front->sep_end - front->sep_begin;

        if (front->num_pivots > 0) {
            front->pivot_vars = (int *)malloc((size_t)front->num_pivots * sizeof(int));
            if (!front->pivot_vars) {
                return SDS_ERR_ALLOC;
            }
            for (int i = 0; i < front->num_pivots; ++i) {
                front->pivot_vars[i] = perm[front->sep_begin + i];
            }
        }
    }

    return SDS_OK;
}

int symbolic_build_update_sets(const CSCMatrix *a_perm_pattern,
                               const SeparatorTree *tree,
                               FrontSymbolic *fronts)
{
    int *position = NULL;
    int *marker = NULL;
    int marker_id = 1;
    int rc;
    const int n = a_perm_pattern ? a_perm_pattern->ncols : 0;

    if (!fronts || separator_tree_validate(tree, n) != SDS_OK ||
        symbolic_validate_csc_pattern(a_perm_pattern) != SDS_OK) {
        return SDS_ERR_BAD_INPUT;
    }

    position = (int *)malloc((size_t)n * sizeof(int));
    marker = (int *)calloc((size_t)n, sizeof(int));
    if (!position || !marker) {
        free(position);
        free(marker);
        return SDS_ERR_ALLOC;
    }

    rc = symbolic_build_position_from_fronts(tree, fronts, n, position);
    if (rc != SDS_OK) {
        free(position);
        free(marker);
        return rc;
    }

    for (int sep = 0; sep < tree->num_separators; ++sep) {
        rc = build_update_set_for_front(sep, a_perm_pattern, fronts,
                                        position, marker, marker_id++);
        if (rc != SDS_OK) {
            free(position);
            free(marker);
            return rc;
        }
    }

    free(position);
    free(marker);
    return SDS_OK;
}

int symbolic_build_update_to_parent_maps(const SeparatorTree *tree,
                                         FrontSymbolic *fronts)
{
    if (!tree || !fronts) {
        return SDS_ERR_BAD_INPUT;
    }

    for (int sep = 0; sep < tree->num_separators; ++sep) {
        FrontSymbolic *front = &fronts[sep];
        if (front->parent == -1) {
            continue;
        }

        front->num_update_to_parent = front->num_updates;
        if (front->num_updates == 0) {
            continue;
        }

        front->update_to_parent =
            (int *)malloc((size_t)front->num_updates * sizeof(int));
        if (!front->update_to_parent) {
            return SDS_ERR_ALLOC;
        }

        for (int i = 0; i < front->num_updates; ++i) {
            const int local =
                symbolic_find_front_var(&fronts[front->parent], front->update_vars[i]);
            if (local < 0) {
                return SDS_ERR_SYMBOLIC;
            }
            front->update_to_parent[i] = local;
        }
    }

    return SDS_OK;
}

/*
 * 한 프론트의 update set 구성.
 *
 * update 변수는 두 경로로 수집된다:
 *
 * 경로 1 — 행렬 채움(fill): 각 pivot 열의 비영값 행 중에서
 *   position[row] >= sep_end 인 변수. 이 separator 밖에 있는 변수이므로
 *   해당 프론트의 Schur 보완(C 블록)에 기여한다.
 *
 * 경로 2 — 자식 상속: 자식 프론트의 update 변수 중 이 프론트의
 *   separator 범위[sep_begin, sep_end)에 속하지 않는 변수.
 *   자식에서 소거되지 않고 더 위로 전파되어야 하는 fill이다.
 *
 * 두 경로에서 중복을 막기 위해 marker 배열을 사용한다.
 * 각 프론트에 고유한 marker_id를 부여하여 marker 배열 초기화 없이
 * 중복 여부를 O(1)로 검사한다.
 */
static int build_update_set_for_front(int sep,
                                      const CSCMatrix *A,
                                      FrontSymbolic *fronts,
                                      const int *position,
                                      int *marker,
                                      int marker_id)
{
    FrontSymbolic *front = &fronts[sep];
    SymbolicIntList updates;
    memset(&updates, 0, sizeof(updates));

    /* 경로 1: 행렬 비영값에서 fill 수집 */
    for (int i = 0; i < front->num_pivots; ++i) {
        const int col = front->pivot_vars[i];
        for (int p = A->colptr[col]; p < A->colptr[col + 1]; ++p) {
            const int row = A->rowind[p];
            const int pos = position[row];
            if (pos >= front->sep_end) {
                int rc = symbolic_int_list_add_marked(&updates, marker,
                                                       marker_id, row);
                if (rc != SDS_OK) {
                    symbolic_int_list_destroy(&updates);
                    return rc;
                }
            }
        }
    }

    /* 경로 2: 자식 프론트에서 상속 */
    if (front->left_child != -1) {
        const int children[2] = {front->left_child, front->right_child};
        for (int c = 0; c < 2; ++c) {
            const FrontSymbolic *child = &fronts[children[c]];
            for (int i = 0; i < child->num_updates; ++i) {
                const int var = child->update_vars[i];
                const int pos = position[var];
                if (!(front->sep_begin <= pos && pos < front->sep_end)) {
                    int rc = symbolic_int_list_add_marked(&updates, marker,
                                                           marker_id, var);
                    if (rc != SDS_OK) {
                        symbolic_int_list_destroy(&updates);
                        return rc;
                    }
                }
            }
        }
    }

    symbolic_sort_values_by_position(updates.values, updates.count, position);
    front->num_updates = updates.count;
    front->update_vars = updates.values;
    updates.values = NULL;
    updates.count = 0;
    updates.capacity = 0;

    front->num_front_vars = front->num_pivots + front->num_updates;
    if (front->num_front_vars > 0) {
        front->front_vars = (int *)malloc((size_t)front->num_front_vars * sizeof(int));
        if (!front->front_vars) {
            symbolic_int_list_destroy(&updates);
            return SDS_ERR_ALLOC;
        }
        for (int i = 0; i < front->num_pivots; ++i) {
            front->front_vars[i] = front->pivot_vars[i];
        }
        for (int i = 0; i < front->num_updates; ++i) {
            front->front_vars[front->num_pivots + i] = front->update_vars[i];
        }
    }

    symbolic_int_list_destroy(&updates);
    return SDS_OK;
}
