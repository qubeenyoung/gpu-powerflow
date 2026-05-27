#ifndef EXP_20260519_SYMBOLIC_PRINT_H
#define EXP_20260519_SYMBOLIC_PRINT_H

#include "symbolic/symbolic_factorization.h"

/*
 * SymbolicFactorization의 모든 필드를 stdout에 출력한다.
 * 기호 분석 결과 디버깅용. 대형 문제에서는 출력량이 많으므로
 * EntryAssemblyPlan은 첫 16개 엔트리만 출력한다. Contribution assembly는
 * nupd*nupd 엔트리 맵을 만들지 않으므로 update_to_parent 기반 요약만 출력한다.
 */
void symbolic_factorization_print(const SymbolicFactorization *symbolic);

#endif
