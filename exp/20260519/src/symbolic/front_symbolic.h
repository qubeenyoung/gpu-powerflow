#ifndef EXP_20260519_FRONT_SYMBOLIC_H
#define EXP_20260519_FRONT_SYMBOLIC_H

#include "symbolic/symbolic_factorization.h"

/*
 * FrontSymbolic 구성: separator 트리 노드별로 프론트 메타데이터를 채운다.
 *
 * 호출 순서:
 *   1. symbolic_build_front_tree        — pivot_vars 초기화 (separator_perm에서 복사)
 *   2. symbolic_build_update_sets       — update_vars, front_vars 계산
 *   3. symbolic_build_update_to_parent_maps — update_to_parent 인덱스 맵 작성
 *
 * 세 함수를 모두 호출해야 FrontSymbolic이 완전히 초기화된다.
 */

int symbolic_build_front_tree(const SeparatorTree *tree,
                              const int *perm,
                              int n,
                              FrontSymbolic *fronts);

int symbolic_build_update_sets(const CSCMatrix *a_perm_pattern,
                               const SeparatorTree *tree,
                               FrontSymbolic *fronts);

int symbolic_build_update_to_parent_maps(const SeparatorTree *tree,
                                         FrontSymbolic *fronts);

#endif
