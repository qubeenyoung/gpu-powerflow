#ifndef EXP_20260519_STORAGE_PLAN_H
#define EXP_20260519_STORAGE_PLAN_H

#include "symbolic/symbolic_factorization.h"

/*
 * 멀티프론털 인수분해를 위한 밀집 저장 레이아웃 계산.
 *
 * 프론트별 블록 배치 (열 우선, double 단위 오프셋):
 *
 *   F       nfront × nfront   어셈블된 프론털 패널
 *   L11     npiv   × npiv     피벗 블록 하삼각 인수
 *   U11     npiv   × npiv     피벗 블록 상삼각 인수
 *   L21     nupd   × npiv     하부 오프-다이어그널 인수
 *   U12     npiv   × nupd     상부 오프-다이어그널 인수
 *   C       nupd   × nupd     Schur 보완(기여 블록)
 *
 * 여기서 nfront = npiv + nupd.
 * 모든 오프셋은 바이트가 아닌 double 엔트리 수로 표현된다.
 */

int symbolic_build_front_storage_plan(const FrontSymbolic *fronts,
                                      int num_fronts,
                                      FrontStoragePlan *storage);

void symbolic_front_storage_plan_destroy(FrontStoragePlan *storage);

#endif
