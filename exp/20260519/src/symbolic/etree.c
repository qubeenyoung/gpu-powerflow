#include "symbolic/etree.h"

#include <stdlib.h>

/*
 * Algorithmic reference:
 *   third_party/lin_sol/strumpack/src/src/sparse/SeparatorTree.hpp
 *   third_party/lin_sol/strumpack/src/src/sparse/SeparatorTree.cpp
 *
 * Specifically referenced concepts:
 *   - spsymetree(): Liu/Gilbert symmetric elimination tree construction.
 *   - etree_postorder(): DFS postordering of the elimination tree forest.
 *
 * This file reimplements the ideas in small C-style code for this experiment.
 * It does not copy STRUMPACK source code.
 */

static int validate_csc_pattern(const CSCMatrix *A);
static int find_disjoint_set(int i, int *parent);
static int build_child_lists(int n, const int *parent, int **child_ptr, int **child_ind);
static int emit_postorder_dfs(int node, const int *child_ptr, const int *child_ind,
                              int *postorder, int *count, int *seen);

int symbolic_build_elimination_tree(const CSCMatrix *a_perm_pattern, int *parent)
{
    int *root = NULL;
    int *set_parent = NULL;
    int rc;
    const int n = a_perm_pattern ? a_perm_pattern->ncols : 0;

    rc = validate_csc_pattern(a_perm_pattern);
    if (rc != SDS_OK || !parent) {
        return rc != SDS_OK ? rc : SDS_ERR_BAD_INPUT;
    }

    root = (int *)malloc((size_t)n * sizeof(int));
    set_parent = (int *)malloc((size_t)n * sizeof(int));
    if (!root || !set_parent) {
        free(root);
        free(set_parent);
        return SDS_ERR_ALLOC;
    }

    /*
     * Liu/Gilbert 대칭 소거 트리 구성.
     *
     * 각 열 col에 대해 대각 위쪽(row < col)의 비영값만 처리한다.
     * 행렬이 구조적으로 대칭이므로 하삼각 부분은 상삼각과 동일한
     * 소거 트리 엣지를 생성하기 때문에 중복 처리할 필요가 없다.
     *
     * 핵심 불변식: 열 col 처리 후 root[find(row_set)] == col 이면
     * row의 소거 트리 루트가 이미 col을 가리키고 있으므로 새 엣지 불필요.
     * 서로소 집합(disjoint-set)으로 이미 방문한 경로를 압축하여
     * 밀집 기호적 채움(symbolic fill)을 피한다.
     */
    for (int col = 0; col < n; ++col) {
        int col_set = col;
        set_parent[col] = col;
        root[col_set] = col;
        parent[col] = -1;

        for (int p = a_perm_pattern->colptr[col]; p < a_perm_pattern->colptr[col + 1]; ++p) {
            const int row = a_perm_pattern->rowind[p];
            int row_set;
            int row_root;

            if (row >= col) {
                continue;
            }
            row_set = find_disjoint_set(row, set_parent);
            row_root = root[row_set];
            if (row_root != col) {
                parent[row_root] = col;
                set_parent[col_set] = row_set;
                col_set = row_set;
                root[col_set] = col;
            }
        }
    }

    free(root);
    free(set_parent);
    return SDS_OK;
}

int symbolic_etree_postorder(int n, const int *parent, int *postorder)
{
    int *child_ptr = NULL;
    int *child_ind = NULL;
    int *seen = NULL;
    int count = 0;
    int rc;

    if (n <= 0 || !parent || !postorder) {
        return SDS_ERR_BAD_INPUT;
    }
    for (int i = 0; i < n; ++i) {
        if (parent[i] < -1 || parent[i] >= n || parent[i] == i) {
            return SDS_ERR_BAD_INPUT;
        }
    }

    rc = build_child_lists(n, parent, &child_ptr, &child_ind);
    if (rc != SDS_OK) {
        return rc;
    }
    seen = (int *)calloc((size_t)n, sizeof(int));
    if (!seen) {
        free(child_ptr);
        free(child_ind);
        return SDS_ERR_ALLOC;
    }

    for (int node = 0; node < n; ++node) {
        if (parent[node] == -1) {
            rc = emit_postorder_dfs(node, child_ptr, child_ind, postorder, &count, seen);
            if (rc != SDS_OK) {
                free(child_ptr);
                free(child_ind);
                free(seen);
                return rc;
            }
        }
    }
    if (count != n) {
        free(child_ptr);
        free(child_ind);
        free(seen);
        return SDS_ERR_BAD_INPUT;
    }

    free(child_ptr);
    free(child_ind);
    free(seen);
    return SDS_OK;
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

/* 경로 반분(path halving): 올라가면서 각 노드의 부모를 조부모로 교체한다.
 * 완전 경로 압축(full compression)과 동일한 상각 복잡도를 가지면서
 * 단일 패스만 사용하므로 스택이 필요 없다. */
static int find_disjoint_set(int i, int *parent)
{
    int p = parent[i];
    int gp = parent[p];
    while (gp != p) {
        parent[i] = gp;
        i = gp;
        p = parent[i];
        gp = parent[p];
    }
    return p;
}

static int build_child_lists(int n, const int *parent, int **child_ptr, int **child_ind)
{
    int *ptr = NULL;
    int *ind = NULL;
    int *next = NULL;

    ptr = (int *)calloc((size_t)n + 1u, sizeof(int));
    ind = (int *)malloc((size_t)n * sizeof(int));
    next = (int *)malloc(((size_t)n + 1u) * sizeof(int));
    if (!ptr || !ind || !next) {
        free(ptr);
        free(ind);
        free(next);
        return SDS_ERR_ALLOC;
    }

    for (int node = 0; node < n; ++node) {
        if (parent[node] != -1) {
            ++ptr[parent[node] + 1];
        }
    }
    for (int node = 0; node < n; ++node) {
        ptr[node + 1] += ptr[node];
    }
    for (int i = 0; i <= n; ++i) {
        next[i] = ptr[i];
    }

    for (int node = 0; node < n; ++node) {
        if (parent[node] != -1) {
            ind[next[parent[node]]++] = node;
        }
    }

    free(next);
    *child_ptr = ptr;
    *child_ind = ind;
    return SDS_OK;
}

static int emit_postorder_dfs(int node, const int *child_ptr, const int *child_ind,
                              int *postorder, int *count, int *seen)
{
    if (seen[node]) {
        return SDS_ERR_BAD_INPUT;
    }
    seen[node] = 1;
    for (int p = child_ptr[node]; p < child_ptr[node + 1]; ++p) {
        int rc = emit_postorder_dfs(child_ind[p], child_ptr, child_ind,
                                    postorder, count, seen);
        if (rc != SDS_OK) {
            return rc;
        }
    }
    postorder[(*count)++] = node;
    return SDS_OK;
}
