#ifndef EXP_20260519_SYMBOLIC_INTERNAL_H
#define EXP_20260519_SYMBOLIC_INTERNAL_H

#include "symbolic/symbolic_factorization.h"

#include <stddef.h>

/* symbolic 서브모듈 전용 내부 유틸리티. 외부에 노출하지 않는다. */

/* 동적 크기 정수 배열. realloc 기반 더블링 전략 사용. */
typedef struct {
    int count;
    int capacity;
    int *values;
} SymbolicIntList;

/* 입력 검증 */
int symbolic_validate_csc_pattern(const CSCMatrix *A);
int symbolic_validate_permutation_array(const int *perm, int n);

/* SymbolicIntList 조작 */
int symbolic_int_list_append(SymbolicIntList *list, int value);
void symbolic_int_list_destroy(SymbolicIntList *list);
/* marker[value] == marker_id 이면 중복으로 간주하여 추가하지 않는다. */
int symbolic_int_list_add_marked(SymbolicIntList *list,
                                 int *marker,
                                 int marker_id,
                                 int value);

/* values[]를 position[] 기준 오름차순으로 제자리 정렬한다. */
void symbolic_sort_values_by_position(int *values, int count, const int *position);

/* var → separator 내 연속 위치 맵 구성 (update set 정렬 기준으로 사용). */
int symbolic_build_position_from_fronts(const SeparatorTree *tree,
                                        const FrontSymbolic *fronts,
                                        int n,
                                        int *position);

/* front_vars에서 var의 로컬 인덱스를 반환. 없으면 -1. */
int symbolic_find_front_var(const FrontSymbolic *front, int var);
/* 각 변수를 pivot으로 갖는 프론트 id를 owner[]에 기록한다. */
int symbolic_build_front_owner_map(const SymbolicFactorization *symbolic,
                                   int *owner);

size_t symbolic_square_size(int n);

void symbolic_front_destroy(FrontSymbolic *front);
void symbolic_fronts_destroy(FrontSymbolic *fronts, int num_fronts);

void symbolic_print_int_array(const char *name, const int *values, int n);

#endif
