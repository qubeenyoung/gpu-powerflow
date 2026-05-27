#ifndef EXP_20260519_SYMBOLIC_VALIDATE_H
#define EXP_20260519_SYMBOLIC_VALIDATE_H

#include "symbolic/symbolic_factorization.h"

/*
 * symbolic_factorization_analyze() 완료 후 전체 구조 무결성 검사.
 *
 * 검사 항목:
 *   - separator 트리 구조 유효성
 *   - 각 프론트의 pivot 소유권 (중복·누락 없음)
 *   - update 변수 정렬 순서
 *   - update_to_parent 인덱스 정합성
 *   - 스케줄 포스트오더 및 레벨 단조성
 *   - 저장 오프셋 연속성
 *   - 어셈블리 계획 오프셋 정합성
 */
int symbolic_factorization_validate(const SymbolicFactorization *symbolic);

#endif
