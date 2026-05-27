#ifndef EXP_20260519_ASSEMBLY_PLAN_H
#define EXP_20260519_ASSEMBLY_PLAN_H

#include "symbolic/symbolic_factorization.h"

/*
 * 수치 인수분해 시 데이터를 어디에 쓸지 미리 계산하는 두 가지 어셈블리 계획.
 *
 * EntryAssemblyPlan (엔트리 어셈블리):
 *   A_perm의 각 비영값 → 어느 프론트의 F 블록 어느 오프셋으로 가는가.
 *   수치 단계에서 이 계획을 따르면 CSC → 밀집 복사가 분기 없이 처리된다.
 *
 * ContributionAssemblyPlan (기여 어셈블리):
 *   STRUMPACK 방식처럼 자식 update_to_parent 맵만 사용한다.
 *   C 블록의 nupd*nupd 엔트리별 offset plan은 만들지 않는다.
 *   수치 단계에서 child C(i,j)를 순회하며 update_to_parent[i/j]로
 *   부모 front의 dense block offset을 즉석 계산한다.
 */

int symbolic_build_entry_assembly_plan(const CSCMatrix *a_perm_pattern,
                                       const SymbolicFactorization *symbolic,
                                       EntryAssemblyPlan *plan);

int symbolic_build_contribution_assembly_plan(const SymbolicFactorization *symbolic,
                                              ContributionAssemblyPlan *plan);

void symbolic_entry_assembly_plan_destroy(EntryAssemblyPlan *plan);
void symbolic_contribution_assembly_plan_destroy(ContributionAssemblyPlan *plan);

#endif
