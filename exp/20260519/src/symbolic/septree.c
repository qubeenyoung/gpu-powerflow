#include "symbolic/septree.h"
#include "symbolic/etree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Algorithmic reference:
 *   third_party/lin_sol/strumpack/src/src/sparse/SeparatorTree.hpp
 *   third_party/lin_sol/strumpack/src/src/sparse/SeparatorTree.cpp
 *
 * Specifically referenced concepts:
 *   - SeparatorTree data layout: sizes, parent, left child, right child.
 *   - build_sep_tree_from_perm(): etree -> etree postorder -> separator tree.
 *   - separators_from_etree(): compress etree paths and form a binary
 *     separator tree.
 *   - SeparatorTree::check()/print(): validation and debug report style.
 *
 * This file reimplements the ideas in small C-style code for this experiment.
 * It does not copy STRUMPACK source code.
 */

typedef struct {
    int count;
    int capacity;
    int *values;
} IntList;

typedef struct {
    IntList pivots;
    int parent;
    int left_child;
    int right_child;
} TempSeparator;

typedef struct {
    int count;
    int capacity;
    TempSeparator *items;
} TempSeparatorArray;

static int validate_csc_pattern(const CSCMatrix *A);
static int validate_permutation_array(const int *perm, int n);
static int build_child_lists(int n, const int *parent, int **child_ptr, int **child_ind);
static int sort_child_lists_by_postorder(int n, const int *child_ptr, int *child_ind,
                                         const int *post_position);
static int compute_subtree_weights(int n, const int *child_ptr, const int *child_ind,
                                   int *subtree_weight);
static int compute_subtree_weight_dfs(int node, const int *child_ptr,
                                      const int *child_ind, int *subtree_weight);
static int build_compressed_separator_tree(int n, const int *parent,
                                           const int *post_position,
                                           SeparatorTree *tree,
                                           int *separator_order);
static int build_temp_separator_from_node(int node, const int *child_ptr,
                                          const int *child_ind,
                                          const int *subtree_weight,
                                          TempSeparatorArray *temps);
static int build_balanced_dummy_tree(int *roots, const int *weights, int count,
                                   TempSeparatorArray *temps);
static int partition_roots_by_weight(int *roots, const int *weights, int count,
                                     int **left_roots, int **left_weights,
                                     int *left_count,
                                     int **right_roots, int **right_weights,
                                     int *right_count);
static int append_temp_separator(TempSeparatorArray *temps, int *id);
static void destroy_temp_separators(TempSeparatorArray *temps);
static int append_int(IntList *list, int value);
static void destroy_int_list(IntList *list);
static int build_post_position(int n, const int *etree_postorder, int *post_position);
static int build_final_permutations(int n,
                                    const int *input_perm,
                                    const int *input_inv_perm,
                                    const int *separator_perm,
                                    int *separator_inv_perm,
                                    int *final_perm,
                                    int *final_inv_perm);

int separator_tree_create(SeparatorTree *tree, int num_separators)
{
    if (!tree || num_separators <= 0) {
        return SDS_ERR_BAD_INPUT;
    }

    memset(tree, 0, sizeof(*tree));
    tree->num_separators = num_separators;
    tree->root = -1;
    tree->sizes = (int *)calloc((size_t)num_separators + 1u, sizeof(int));
    tree->parent = (int *)malloc((size_t)num_separators * sizeof(int));
    tree->left_child = (int *)malloc((size_t)num_separators * sizeof(int));
    tree->right_child = (int *)malloc((size_t)num_separators * sizeof(int));
    if (!tree->sizes || !tree->parent || !tree->left_child || !tree->right_child) {
        separator_tree_destroy(tree);
        return SDS_ERR_ALLOC;
    }

    for (int i = 0; i < num_separators; ++i) {
        tree->parent[i] = -1;
        tree->left_child[i] = -1;
        tree->right_child[i] = -1;
    }

    return SDS_OK;
}

void separator_tree_destroy(SeparatorTree *tree)
{
    if (!tree) {
        return;
    }
    free(tree->sizes);
    free(tree->parent);
    free(tree->left_child);
    free(tree->right_child);
    memset(tree, 0, sizeof(*tree));
    tree->root = -1;
}

int separator_tree_validate(const SeparatorTree *tree, int n)
{
    int root_count = 0;
    int root = -1;
    int *seen = NULL;
    int stack_count = 0;
    int *stack = NULL;

    if (!tree || tree->num_separators <= 0 || !tree->sizes || !tree->parent ||
        !tree->left_child || !tree->right_child || n < 0) {
        return SDS_ERR_BAD_INPUT;
    }
    if (tree->sizes[0] != 0 || tree->sizes[tree->num_separators] != n) {
        return SDS_ERR_BAD_INPUT;
    }
    for (int sep = 0; sep < tree->num_separators; ++sep) {
        if (tree->sizes[sep] > tree->sizes[sep + 1]) {
            return SDS_ERR_BAD_INPUT;
        }
        if (tree->parent[sep] == -1) {
            ++root_count;
            root = sep;
        } else if (tree->parent[sep] < 0 || tree->parent[sep] >= tree->num_separators) {
            return SDS_ERR_BAD_INPUT;
        }
        if (tree->left_child[sep] < -1 || tree->left_child[sep] >= tree->num_separators ||
            tree->right_child[sep] < -1 || tree->right_child[sep] >= tree->num_separators) {
            return SDS_ERR_BAD_INPUT;
        }
        if ((tree->left_child[sep] == -1) != (tree->right_child[sep] == -1)) {
            return SDS_ERR_BAD_INPUT;
        }
        if (tree->left_child[sep] != -1 &&
            (tree->parent[tree->left_child[sep]] != sep ||
             tree->parent[tree->right_child[sep]] != sep)) {
            return SDS_ERR_BAD_INPUT;
        }
    }
    if (root_count != 1 || tree->root != root) {
        return SDS_ERR_BAD_INPUT;
    }

    seen = (int *)calloc((size_t)tree->num_separators, sizeof(int));
    stack = (int *)malloc((size_t)tree->num_separators * sizeof(int));
    if (!seen || !stack) {
        free(seen);
        free(stack);
        return SDS_ERR_ALLOC;
    }

    stack[stack_count++] = root;
    while (stack_count) {
        const int sep = stack[--stack_count];
        if (seen[sep]) {
            free(seen);
            free(stack);
            return SDS_ERR_BAD_INPUT;
        }
        seen[sep] = 1;
        if (tree->left_child[sep] != -1) {
            stack[stack_count++] = tree->left_child[sep];
            stack[stack_count++] = tree->right_child[sep];
        }
    }

    for (int sep = 0; sep < tree->num_separators; ++sep) {
        if (!seen[sep]) {
            free(seen);
            free(stack);
            return SDS_ERR_BAD_INPUT;
        }
    }

    free(seen);
    free(stack);
    return SDS_OK;
}

void separator_tree_print(const SeparatorTree *tree)
{
    if (!tree) {
        return;
    }
    printf("SeparatorTree: num_separators=%d root=%d\n",
           tree->num_separators, tree->root);
    printf("  id parent left right range\n");
    for (int sep = 0; sep < tree->num_separators; ++sep) {
        printf("  %d %d %d %d [%d,%d)\n",
               sep, tree->parent[sep], tree->left_child[sep],
               tree->right_child[sep], tree->sizes[sep], tree->sizes[sep + 1]);
    }
}

int separator_tree_build_from_etree_and_perm(int n,
                                             const int *etree_parent,
                                             const int *etree_postorder,
                                             const int *input_perm,
                                             const int *input_inv_perm,
                                             int *separator_perm,
                                             int *separator_inv_perm,
                                             int *final_perm,
                                             int *final_inv_perm,
                                             SeparatorTree *tree)
{
    int *post_position = NULL;
    int rc;

    if (!tree || n <= 0 || !etree_parent || !etree_postorder ||
        validate_permutation_array(input_perm, n) != SDS_OK ||
        validate_permutation_array(input_inv_perm, n) != SDS_OK ||
        !separator_perm || !separator_inv_perm || !final_perm || !final_inv_perm) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(tree, 0, sizeof(*tree));
    tree->root = -1;

    post_position = (int *)malloc((size_t)n * sizeof(int));
    if (!post_position) {
        return SDS_ERR_ALLOC;
    }

    rc = build_post_position(n, etree_postorder, post_position);
    if (rc == SDS_OK) {
        rc = build_compressed_separator_tree(n, etree_parent, post_position,
                                            tree, separator_perm);
    }
    if (rc == SDS_OK) {
        rc = build_final_permutations(n, input_perm, input_inv_perm,
                                      separator_perm, separator_inv_perm,
                                      final_perm, final_inv_perm);
    }
    if (rc != SDS_OK) {
        separator_tree_destroy(tree);
    }

    free(post_position);
    return rc;
}

int separator_tree_build_from_perm(const CSCMatrix *a_perm_pattern,
                                   const int *input_perm,
                                   const int *input_inv_perm,
                                   SeparatorTree *tree,
                                   int *etree_parent_out,
                                   int *etree_postorder_out,
                                   int *separator_perm,
                                   int *separator_inv_perm,
                                   int *final_perm,
                                   int *final_inv_perm)
{
    int *parent = NULL;
    int *postorder = NULL;
    int rc;
    const int n = a_perm_pattern ? a_perm_pattern->ncols : 0;

    if (!tree || validate_csc_pattern(a_perm_pattern) != SDS_OK ||
        validate_permutation_array(input_perm, n) != SDS_OK ||
        validate_permutation_array(input_inv_perm, n) != SDS_OK ||
        !separator_perm || !separator_inv_perm || !final_perm || !final_inv_perm) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(tree, 0, sizeof(*tree));
    tree->root = -1;

    parent = (int *)malloc((size_t)n * sizeof(int));
    postorder = (int *)malloc((size_t)n * sizeof(int));
    if (!parent || !postorder) {
        free(parent);
        free(postorder);
        return SDS_ERR_ALLOC;
    }

    rc = symbolic_build_elimination_tree(a_perm_pattern, parent);
    if (rc == SDS_OK) {
        rc = symbolic_etree_postorder(n, parent, postorder);
    }
    if (rc == SDS_OK) {
        rc = separator_tree_build_from_etree_and_perm(n, parent, postorder,
                                                      input_perm, input_inv_perm,
                                                      separator_perm,
                                                      separator_inv_perm,
                                                      final_perm,
                                                      final_inv_perm,
                                                      tree);
    }
    if (rc == SDS_OK && etree_parent_out) {
        memcpy(etree_parent_out, parent, (size_t)n * sizeof(int));
    }
    if (rc == SDS_OK && etree_postorder_out) {
        memcpy(etree_postorder_out, postorder, (size_t)n * sizeof(int));
    }

    free(parent);
    free(postorder);
    return rc;
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

static int validate_permutation_array(const int *perm, int n)
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

/*
 * 소거 트리 → 이진 분리자 트리 압축.
 *
 * 단계:
 *   1. 소거 트리의 각 포리스트 루트부터 재귀 탐색하여 TempSeparator 배열 구성.
 *   2. 포리스트 루트가 여럿이면 combine_separator_roots()로 단일 루트로 통합.
 *   3. TempSeparator 배열(동적 크기) → 고정 크기 SeparatorTree로 복사.
 *   4. pivot 변수 목록을 separator 순서대로 final_order에 기록.
 *      → final_order는 이후 etree 포스트오더를 대체하는 변수 순서가 된다.
 */
static int build_compressed_separator_tree(int n, const int *parent,
                                           const int *post_position,
                                           SeparatorTree *tree,
                                           int *separator_order)
{
    TempSeparatorArray temps;
    int *child_ptr = NULL;
    int *child_ind = NULL;
    int *subtree_weight = NULL;
    int *roots = NULL;
    int *root_weights = NULL;
    int num_roots = 0;
    int root_sep;
    int rc;
    int cursor = 0;

    memset(&temps, 0, sizeof(temps));
    rc = build_child_lists(n, parent, &child_ptr, &child_ind);
    if (rc != SDS_OK) {
        return rc;
    }
    rc = sort_child_lists_by_postorder(n, child_ptr, child_ind, post_position);
    if (rc != SDS_OK) {
        free(child_ptr);
        free(child_ind);
        return rc;
    }

    subtree_weight = (int *)malloc((size_t)n * sizeof(int));
    if (!subtree_weight) {
        free(child_ptr);
        free(child_ind);
        return SDS_ERR_ALLOC;
    }
    rc = compute_subtree_weights(n, child_ptr, child_ind, subtree_weight);
    if (rc != SDS_OK) {
        free(child_ptr);
        free(child_ind);
        free(subtree_weight);
        return rc;
    }

    roots = (int *)malloc((size_t)n * sizeof(int));
    root_weights = (int *)malloc((size_t)n * sizeof(int));
    if (!roots || !root_weights) {
        free(child_ptr);
        free(child_ind);
        free(subtree_weight);
        free(roots);
        free(root_weights);
        return SDS_ERR_ALLOC;
    }

    for (int node = 0; node < n; ++node) {
        if (parent[node] == -1) {
            int sep_root = build_temp_separator_from_node(node, child_ptr, child_ind,
                                                          subtree_weight, &temps);
            if (sep_root < 0) {
                destroy_temp_separators(&temps);
                free(child_ptr);
                free(child_ind);
                free(subtree_weight);
                free(roots);
                free(root_weights);
                return SDS_ERR_ALLOC;
            }
            roots[num_roots++] = sep_root;
            root_weights[num_roots - 1] = subtree_weight[node];
        }
    }

    root_sep = build_balanced_dummy_tree(roots, root_weights, num_roots, &temps);
    if (root_sep < 0) {
        destroy_temp_separators(&temps);
        free(child_ptr);
        free(child_ind);
        free(subtree_weight);
        free(roots);
        free(root_weights);
        return SDS_ERR_ALLOC;
    }

    rc = separator_tree_create(tree, temps.count);
    if (rc != SDS_OK) {
        destroy_temp_separators(&temps);
        free(child_ptr);
        free(child_ind);
        free(subtree_weight);
        free(roots);
        free(root_weights);
        return rc;
    }

    tree->root = root_sep;
    tree->sizes[0] = 0;
    for (int sep = 0; sep < temps.count; ++sep) {
        tree->parent[sep] = temps.items[sep].parent;
        tree->left_child[sep] = temps.items[sep].left_child;
        tree->right_child[sep] = temps.items[sep].right_child;
        tree->sizes[sep + 1] = tree->sizes[sep] + temps.items[sep].pivots.count;
        for (int i = 0; i < temps.items[sep].pivots.count; ++i) {
            separator_order[cursor++] = temps.items[sep].pivots.values[i];
        }
    }

    if (cursor != n || separator_tree_validate(tree, n) != SDS_OK) {
        separator_tree_destroy(tree);
        rc = SDS_ERR_SYMBOLIC;
    }

    destroy_temp_separators(&temps);
    free(child_ptr);
    free(child_ind);
    free(subtree_weight);
    free(roots);
    free(root_weights);
    return rc;
}

/*
 * 소거 트리 노드 하나를 분리자 트리 서브트리로 재귀 변환.
 *
 * - 자식 0개 (리프): pivot이 node 하나인 새 separator 생성.
 * - 자식 1개: 자식의 separator를 재사용하고 node를 pivot 끝에 추가.
 *   (단일 경로 압축 — 이 경우 새 separator 노드를 만들지 않는다.)
 * - 자식 2개 이상: 각 자식을 재귀 처리하여 서브트리 루트 배열을 얻은 뒤,
 *   첫 두 루트를 묶는 pivot-없는 더미 separator를 반복 생성하여
 *   count가 2가 될 때까지 좌-우 이진 트리로 축소한다.
 *   마지막으로 node 자신을 pivot으로 갖는 separator를 만들어 최종 루트로 반환.
 */
static int build_temp_separator_from_node(int node, const int *child_ptr,
                                          const int *child_ind,
                                          const int *subtree_weight,
                                          TempSeparatorArray *temps)
{
    const int first = child_ptr[node];
    const int last = child_ptr[node + 1];
    const int num_children = last - first;
    int id;

    if (num_children == 0) {
        if (append_temp_separator(temps, &id) != SDS_OK ||
            append_int(&temps->items[id].pivots, node) != SDS_OK) {
            return -1;
        }
        return id;
    }

    if (num_children == 1) {
        id = build_temp_separator_from_node(child_ind[first], child_ptr, child_ind,
                                            subtree_weight, temps);
        if (id < 0 || append_int(&temps->items[id].pivots, node) != SDS_OK) {
            return -1;
        }
        return id;
    }

    {
        int *roots = (int *)malloc((size_t)num_children * sizeof(int));
        int *weights = (int *)malloc((size_t)num_children * sizeof(int));
        int *left_roots = NULL;
        int *left_weights = NULL;
        int *right_roots = NULL;
        int *right_weights = NULL;
        int left_count = 0;
        int right_count = 0;
        int left_root;
        int right_root;
        if (!roots || !weights) {
            free(roots);
            free(weights);
            return -1;
        }
        for (int p = first; p < last; ++p) {
            roots[p - first] = build_temp_separator_from_node(child_ind[p], child_ptr,
                                                              child_ind,
                                                              subtree_weight, temps);
            if (roots[p - first] < 0) {
                free(roots);
                free(weights);
                return -1;
            }
            weights[p - first] = subtree_weight[child_ind[p]];
        }

        if (partition_roots_by_weight(roots, weights, num_children,
                                      &left_roots, &left_weights, &left_count,
                                      &right_roots, &right_weights, &right_count) != SDS_OK) {
            free(roots);
            free(weights);
            return -1;
        }
        left_root = build_balanced_dummy_tree(left_roots, left_weights, left_count, temps);
        right_root = build_balanced_dummy_tree(right_roots, right_weights, right_count, temps);
        free(left_roots);
        free(left_weights);
        free(right_roots);
        free(right_weights);
        if (left_root < 0 || right_root < 0) {
            free(roots);
            free(weights);
            return -1;
        }

        if (append_temp_separator(temps, &id) != SDS_OK ||
            append_int(&temps->items[id].pivots, node) != SDS_OK) {
            free(roots);
            free(weights);
            return -1;
        }
        temps->items[id].left_child = left_root;
        temps->items[id].right_child = right_root;
        temps->items[left_root].parent = id;
        temps->items[right_root].parent = id;
        free(roots);
        free(weights);
        return id;
    }
}

static int build_balanced_dummy_tree(int *roots, const int *weights, int count,
                                     TempSeparatorArray *temps)
{
    int *left_roots = NULL;
    int *left_weights = NULL;
    int *right_roots = NULL;
    int *right_weights = NULL;
    int left_count = 0;
    int right_count = 0;
    int left_root;
    int right_root;
    int id;

    if (!roots || !weights || !temps || count <= 0) {
        return -1;
    }
    if (count == 1) {
        return roots[0];
    }
    if (partition_roots_by_weight(roots, weights, count,
                                  &left_roots, &left_weights, &left_count,
                                  &right_roots, &right_weights, &right_count) != SDS_OK) {
        return -1;
    }
    left_root = build_balanced_dummy_tree(left_roots, left_weights, left_count, temps);
    right_root = build_balanced_dummy_tree(right_roots, right_weights, right_count, temps);
    free(left_roots);
    free(left_weights);
    free(right_roots);
    free(right_weights);
    if (left_root < 0 || right_root < 0) {
        return -1;
    }
    if (append_temp_separator(temps, &id) != SDS_OK) {
        return -1;
    }
    temps->items[id].left_child = left_root;
    temps->items[id].right_child = right_root;
    temps->items[left_root].parent = id;
    temps->items[right_root].parent = id;
    return id;
}

static int partition_roots_by_weight(int *roots, const int *weights, int count,
                                     int **left_roots, int **left_weights,
                                     int *left_count,
                                     int **right_roots, int **right_weights,
                                     int *right_count)
{
    int *order = NULL;
    int lcount = 0;
    int rcount = 0;
    int lweight = 0;
    int rweight = 0;

    if (!roots || !weights || count < 2 || !left_roots || !left_weights ||
        !left_count || !right_roots || !right_weights || !right_count) {
        return SDS_ERR_BAD_INPUT;
    }

    *left_roots = (int *)malloc((size_t)count * sizeof(int));
    *left_weights = (int *)malloc((size_t)count * sizeof(int));
    *right_roots = (int *)malloc((size_t)count * sizeof(int));
    *right_weights = (int *)malloc((size_t)count * sizeof(int));
    order = (int *)malloc((size_t)count * sizeof(int));
    if (!*left_roots || !*left_weights || !*right_roots || !*right_weights || !order) {
        free(*left_roots);
        free(*left_weights);
        free(*right_roots);
        free(*right_weights);
        free(order);
        *left_roots = NULL;
        *left_weights = NULL;
        *right_roots = NULL;
        *right_weights = NULL;
        return SDS_ERR_ALLOC;
    }

    for (int i = 0; i < count; ++i) {
        order[i] = i;
    }
    for (int gap = count / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < count; ++i) {
            const int item = order[i];
            int j = i;
            while (j >= gap && weights[order[j - gap]] < weights[item]) {
                order[j] = order[j - gap];
                j -= gap;
            }
            order[j] = item;
        }
    }

    for (int oi = 0; oi < count; ++oi) {
        const int idx = order[oi];
        if ((lweight <= rweight && lcount < count - 1) || rcount == count - 1) {
            (*left_roots)[lcount] = roots[idx];
            (*left_weights)[lcount] = weights[idx];
            lweight += weights[idx];
            ++lcount;
        } else {
            (*right_roots)[rcount] = roots[idx];
            (*right_weights)[rcount] = weights[idx];
            rweight += weights[idx];
            ++rcount;
        }
    }

    if (lcount == 0 || rcount == 0) {
        free(*left_roots);
        free(*left_weights);
        free(*right_roots);
        free(*right_weights);
        free(order);
        *left_roots = NULL;
        *left_weights = NULL;
        *right_roots = NULL;
        *right_weights = NULL;
        return SDS_ERR_SYMBOLIC;
    }

    *left_count = lcount;
    *right_count = rcount;
    free(order);
    return SDS_OK;
}

static int sort_child_lists_by_postorder(int n, const int *child_ptr, int *child_ind,
                                         const int *post_position)
{
    if (n <= 0 || !child_ptr || !child_ind || !post_position) {
        return SDS_ERR_BAD_INPUT;
    }

    for (int node = 0; node < n; ++node) {
        const int begin = child_ptr[node];
        const int end = child_ptr[node + 1];
        const int count = end - begin;
        for (int gap = count / 2; gap > 0; gap /= 2) {
            for (int i = begin + gap; i < end; ++i) {
                const int child = child_ind[i];
                int j = i;
                while (j >= begin + gap &&
                       post_position[child_ind[j - gap]] > post_position[child]) {
                    child_ind[j] = child_ind[j - gap];
                    j -= gap;
                }
                child_ind[j] = child;
            }
        }
    }
    return SDS_OK;
}

static int compute_subtree_weights(int n, const int *child_ptr, const int *child_ind,
                                   int *subtree_weight)
{
    if (n <= 0 || !child_ptr || !child_ind || !subtree_weight) {
        return SDS_ERR_BAD_INPUT;
    }

    for (int node = 0; node < n; ++node) {
        subtree_weight[node] = 0;
    }
    for (int node = 0; node < n; ++node) {
        if (subtree_weight[node] == 0 &&
            compute_subtree_weight_dfs(node, child_ptr, child_ind, subtree_weight) < 0) {
            return SDS_ERR_SYMBOLIC;
        }
    }
    return SDS_OK;
}

static int compute_subtree_weight_dfs(int node, const int *child_ptr,
                                      const int *child_ind, int *subtree_weight)
{
    int weight = 1;

    if (subtree_weight[node] > 0) {
        return subtree_weight[node];
    }
    for (int p = child_ptr[node]; p < child_ptr[node + 1]; ++p) {
        const int child_weight =
            compute_subtree_weight_dfs(child_ind[p], child_ptr, child_ind,
                                       subtree_weight);
        if (child_weight <= 0) {
            return -1;
        }
        weight += child_weight;
    }
    subtree_weight[node] = weight;
    return weight;
}

static int build_post_position(int n, const int *etree_postorder, int *post_position)
{
    int *seen = NULL;

    if (n <= 0 || !etree_postorder || !post_position) {
        return SDS_ERR_BAD_INPUT;
    }
    seen = (int *)calloc((size_t)n, sizeof(int));
    if (!seen) {
        return SDS_ERR_ALLOC;
    }
    for (int pos = 0; pos < n; ++pos) {
        const int node = etree_postorder[pos];
        if (node < 0 || node >= n || seen[node]) {
            free(seen);
            return SDS_ERR_BAD_INPUT;
        }
        seen[node] = 1;
        post_position[node] = pos;
    }
    free(seen);
    return SDS_OK;
}

static int build_final_permutations(int n,
                                    const int *input_perm,
                                    const int *input_inv_perm,
                                    const int *separator_perm,
                                    int *separator_inv_perm,
                                    int *final_perm,
                                    int *final_inv_perm)
{
    if (n <= 0 || !input_perm || !input_inv_perm || !separator_perm ||
        !separator_inv_perm || !final_perm || !final_inv_perm) {
        return SDS_ERR_BAD_INPUT;
    }
    if (validate_permutation_array(separator_perm, n) != SDS_OK) {
        return SDS_ERR_SYMBOLIC;
    }

    for (int pos = 0; pos < n; ++pos) {
        separator_inv_perm[separator_perm[pos]] = pos;
    }
    for (int original = 0; original < n; ++original) {
        const int metis_var = input_perm[original];
        final_perm[original] = separator_inv_perm[metis_var];
    }
    for (int final_pos = 0; final_pos < n; ++final_pos) {
        const int metis_var = separator_perm[final_pos];
        final_inv_perm[final_pos] = input_inv_perm[metis_var];
    }

    if (validate_permutation_array(separator_inv_perm, n) != SDS_OK ||
        validate_permutation_array(final_perm, n) != SDS_OK ||
        validate_permutation_array(final_inv_perm, n) != SDS_OK) {
        return SDS_ERR_SYMBOLIC;
    }
    for (int original = 0; original < n; ++original) {
        if (final_inv_perm[final_perm[original]] != original) {
            return SDS_ERR_SYMBOLIC;
        }
    }
    return SDS_OK;
}

static int append_temp_separator(TempSeparatorArray *temps, int *id)
{
    TempSeparator *next;
    if (temps->count == temps->capacity) {
        int new_capacity = temps->capacity ? 2 * temps->capacity : 16;
        next = (TempSeparator *)realloc(temps->items,
                                        (size_t)new_capacity * sizeof(TempSeparator));
        if (!next) {
            return SDS_ERR_ALLOC;
        }
        temps->items = next;
        temps->capacity = new_capacity;
    }
    *id = temps->count++;
    memset(&temps->items[*id], 0, sizeof(temps->items[*id]));
    temps->items[*id].parent = -1;
    temps->items[*id].left_child = -1;
    temps->items[*id].right_child = -1;
    return SDS_OK;
}

static void destroy_temp_separators(TempSeparatorArray *temps)
{
    if (!temps) {
        return;
    }
    for (int i = 0; i < temps->count; ++i) {
        destroy_int_list(&temps->items[i].pivots);
    }
    free(temps->items);
    memset(temps, 0, sizeof(*temps));
}

static int append_int(IntList *list, int value)
{
    int *next;
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

static void destroy_int_list(IntList *list)
{
    if (!list) {
        return;
    }
    free(list->values);
    memset(list, 0, sizeof(*list));
}
