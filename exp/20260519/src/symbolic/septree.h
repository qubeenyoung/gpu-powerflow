#ifndef EXP_20260519_SEPTREE_H
#define EXP_20260519_SEPTREE_H

#include "matrix/csc_matrix.h"

/*
 * 분리자 트리(separator tree) 구성 및 검증.
 *
 * 소거 트리에서 이진 분리자 트리로의 변환 과정:
 *   1. 단일 자식 경로 → 하나의 separator로 병합 (pivot 목록 집결)
 *   2. 자식이 3개 이상인 노드 → subtree weight를 사용해 좌/우 그룹으로
 *      균형 분할한 뒤 이진 더미(dummy) separator로 연결
 *   3. 소거 트리 포리스트의 루트가 여럿이면 → 같은 균형 분할 방식으로
 *      단일 루트로 통합
 *
 * 결과 트리는 항상 진이진 트리(full binary tree):
 *   내부 노드는 정확히 2개의 자식을 가지며, 리프는 자식이 없다.
 *
 * 알고리즘 참고:
 *   STRUMPACK sparse/SeparatorTree.hpp and SeparatorTree.cpp:
 *   SeparatorTree layout, build_sep_tree_from_perm(), etree_postorder(), and
 *   separators_from_etree().
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_separators;
    int *sizes;
    int *parent;
    int *left_child;
    int *right_child;
    int root;
} SeparatorTree;

int separator_tree_create(SeparatorTree *tree, int num_separators);
void separator_tree_destroy(SeparatorTree *tree);
int separator_tree_validate(const SeparatorTree *tree, int n);
void separator_tree_print(const SeparatorTree *tree);

/*
 * Function: separator_tree_build_from_perm
 *
 * Purpose:
 *   Build a STRUMPACK-style separator tree from a reordered sparse pattern.
 */
int separator_tree_build_from_etree_and_perm(int n,
                                             const int *etree_parent,
                                             const int *etree_postorder,
                                             const int *input_perm,
                                             const int *input_inv_perm,
                                             int *separator_perm,
                                             int *separator_inv_perm,
                                             int *final_perm,
                                             int *final_inv_perm,
                                             SeparatorTree *tree);

int separator_tree_build_from_perm(const CSCMatrix *a_perm_pattern,
                                   const int *input_perm,
                                   const int *input_inv_perm,
                                   SeparatorTree *tree,
                                   int *etree_parent_out,
                                   int *etree_postorder_out,
                                   int *separator_perm,
                                   int *separator_inv_perm,
                                   int *final_perm,
                                   int *final_inv_perm);

#ifdef __cplusplus
}
#endif

#endif
