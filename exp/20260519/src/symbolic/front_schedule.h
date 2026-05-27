#ifndef EXP_20260519_FRONT_SCHEDULE_H
#define EXP_20260519_FRONT_SCHEDULE_H

#include "symbolic/symbolic_factorization.h"

/*
 * 프론트 트리의 순회 순서와 레벨 그룹을 계산한다.
 *
 * 레벨 정의: 리프 = 레벨 0, 부모 = max(자식 레벨) + 1.
 * 같은 레벨에 속하는 프론트들은 조상/자손 관계가 없으므로 병렬 처리 가능.
 *
 * factor_order / forward_order : 자식 → 부모 순서 (포스트오더). 수치 인수분해에 사용.
 * backward_order               : 부모 → 자식 순서. 역방향 치환(back-substitution)에 사용.
 * level_ptr / level_fronts     : CSR 형식으로 레벨별 프론트 목록 저장.
 */

int symbolic_build_front_schedule(const SeparatorTree *tree,
                                  FrontSchedule *schedule);

void symbolic_front_schedule_destroy(FrontSchedule *schedule);

#endif
