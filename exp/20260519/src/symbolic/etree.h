#ifndef EXP_20260519_ETREE_H
#define EXP_20260519_ETREE_H

#include "matrix/csc_matrix.h"

/*
 * 소거 트리(elimination tree) 구성 및 포스트오더 계산.
 *
 * 소거 트리는 희소 직접 솔버의 기호 분석 첫 번째 단계다.
 * 노드 j의 부모는 열 j를 소거했을 때 최초로 채워지는(fill-in) 열 번호이며,
 * 이 구조가 인수분해 데이터 의존성을 결정한다.
 *
 * 알고리즘 참고:
 *   STRUMPACK sparse/SeparatorTree.hpp and SeparatorTree.cpp:
 *   spsymetree() and etree_postorder().
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Function: symbolic_build_elimination_tree
 *
 * Purpose:
 *   Build a Liu-style symmetric elimination tree from a structurally symmetric
 *   CSC sparsity pattern.
 */
int symbolic_build_elimination_tree(const CSCMatrix *a_perm_pattern, int *parent);

/*
 * Function: symbolic_etree_postorder
 *
 * Purpose:
 *   Compute a DFS postorder sequence of an elimination tree forest.
 */
int symbolic_etree_postorder(int n, const int *parent, int *postorder);

#ifdef __cplusplus
}
#endif

#endif
