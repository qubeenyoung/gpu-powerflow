#include "reordering/metis_nd.h"

#include <metis.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* 이 파일은 희소 행렬(CSR 형식)에 대해 METIS의 중첩 분할(Nested Dissection,    */
/* ND) 정렬을 적용하기 위한 기능들을 제공한다.                                  */
/*                                                                            */
/* 전체 흐름은 다음과 같다:                                                     */
/*   1) build_metis_graph()        : 입력 행렬 → METIS용 그래프(CSR) 변환       */
/*   2) compute_metis_nd_ordering(): METIS_NodeND 호출로 정렬(순열) 계산        */
/*   3) apply_symmetric_permutation(): 계산된 순열을 행렬에 대칭적으로 적용      */
/* ------------------------------------------------------------------------- */

/* 정적(static) 헬퍼 함수 프로토타입 선언 */
static int validate_square_csr_matrix(const CSRMatrix *matrix);       /* 정사각 CSR 검증 */
static int validate_permutation(const int *perm, int n);              /* 순열 유효성 검증 */
static int validate_metis_graph(const CSRMatrix *graph);              /* METIS 그래프 검증 */
static int validate_structural_symmetry(const CSRMatrix *graph);      /* 구조적 대칭성 검증 */
static int sort_csr_rows_and_merge_duplicates(CSRMatrix *matrix, int sum_duplicates); /* 행별 정렬 + 중복 병합 */
static int find_column_in_sorted_csr_row(const CSRMatrix *matrix, int row, int col);  /* 정렬된 행에서 열 이진 탐색 */
static void sort_row_entries_by_column(int *cols, double *values, int count);         /* 단일 행 셸 정렬 */
static int make_metis_adjacency_arrays(const CSRMatrix *graph,
                                       idx_t **xadj_out,
                                       idx_t **adjncy_out);           /* METIS 인접 배열 생성 */

/* ------------------------------------------------------------------------- */
/* build_metis_graph                                                          */
/*                                                                            */
/* 입력 CSR 행렬로부터 METIS가 사용할 그래프를 CSR 형태로 만든다.              */
/* METIS 그래프는 대각 성분(자기 자신으로의 간선)을 포함하지 않으므로,         */
/* 비대각(off-diagonal) 항목만 간선으로 복사한다.                              */
/*                                                                            */
/* 매개변수:                                                                   */
/*   matrix                   : 입력 희소 행렬 (정사각, 구조적 대칭 가정)      */
/*   graph                    : 출력 그래프(CSR) — 호출 측에서 비어있는 구조체  */
/*   validate_symmetric_pattern: 0이 아니면 결과 그래프의 구조적 대칭성을 검사  */
/*                                                                            */
/* 반환값: SDS_OK 또는 오류 코드                                                */
/* ------------------------------------------------------------------------- */
int build_metis_graph(
    const CSRMatrix *matrix,
    CSRMatrix *graph,
    int validate_symmetric_pattern)
{
    int rc;
    int num_edges = 0;   /* 그래프에 들어갈 간선(비대각 항목)의 총 개수 */

    /* 출력 포인터가 NULL이면 잘못된 입력 */
    if (!graph) {
        return SDS_ERR_BAD_INPUT;
    }
    /* 출력 구조체를 0으로 초기화하여 부분 실패 시에도 안전한 상태 보장 */
    memset(graph, 0, sizeof(*graph));

    /* 입력 행렬이 정사각 CSR로서 유효한지 검증 */
    rc = validate_square_csr_matrix(matrix);
    if (rc != SDS_OK) {
        return rc;
    }

    /*
     * 입력 패턴은 구조적으로 대칭이라고 가정한다. 따라서 METIS 그래프는
     * 이미 비대각 항목에 모두 존재한다. 역방향 간선을 추가로 넣지 않는다.
     */
    /* 1차 패스: 대각이 아닌 항목 개수를 세어 간선 총 개수를 구한다. */
    for (int row = 0; row < matrix->nrows; ++row) {
        for (int p = matrix->rowptr[row]; p < matrix->rowptr[row + 1]; ++p) {
            if (matrix->colind[p] != row) {   /* 대각 성분 제외 */
                ++num_edges;
            }
        }
    }

    /* 그래프 CSR 구조체를 간선 개수만큼 메모리 할당 */
    rc = csr_create(graph, matrix->nrows, matrix->ncols, num_edges);
    if (rc != SDS_OK) {
        return rc;
    }

    /* 2차 패스: 각 행의 비대각 항목 개수를 rowptr[row+1]에 임시로 기록 */
    for (int row = 0; row < matrix->nrows; ++row) {
        for (int p = matrix->rowptr[row]; p < matrix->rowptr[row + 1]; ++p) {
            if (matrix->colind[p] != row) {
                ++graph->rowptr[row + 1];
            }
        }
    }
    /* 누적 합(prefix sum)을 취해 rowptr을 정상적인 CSR 오프셋으로 변환 */
    for (int row = 0; row < graph->nrows; ++row) {
        graph->rowptr[row + 1] += graph->rowptr[row];
    }

    /* 3차 패스: 실제 열 인덱스와 값을 그래프 배열에 복사 */
    for (int row = 0; row < matrix->nrows; ++row) {
        int write = graph->rowptr[row];   /* 현재 행의 쓰기 위치 */
        for (int p = matrix->rowptr[row]; p < matrix->rowptr[row + 1]; ++p) {
            const int col = matrix->colind[p];
            if (col != row) {             /* 대각 제외 */
                graph->colind[write] = col;
                graph->values[write] = 1.0;   /* 그래프 간선 가중치는 1.0으로 통일 */
                ++write;
            }
        }
    }

    /* 각 행을 열 기준으로 정렬하고 중복 열을 병합 (sum_duplicates=0: 가중치 합산 안 함) */
    rc = sort_csr_rows_and_merge_duplicates(graph, 0);
    if (rc != SDS_OK) {
        csr_destroy(graph);
        return rc;
    }

    /* 요청 시 결과 그래프가 구조적으로 대칭인지(무방향 그래프인지) 검증 */
    if (validate_symmetric_pattern) {
        rc = validate_structural_symmetry(graph);
        if (rc != SDS_OK) {
            csr_destroy(graph);
            return rc;
        }
    }

    return SDS_OK;
}

/* ------------------------------------------------------------------------- */
/* compute_metis_nd_ordering                                                  */
/*                                                                            */
/* METIS_NodeND를 호출하여 중첩 분할 기반의 채움(fill-in) 최소화 순열을        */
/* 계산한다.                                                                   */
/*                                                                            */
/* 매개변수:                                                                   */
/*   graph    : METIS용 그래프(CSR) — 대각 성분이 없어야 함                     */
/*   perm     : 출력 순열      (크기 = graph->nrows)                           */
/*   inv_perm : 출력 역순열    (크기 = graph->nrows)                           */
/*                                                                            */
/* 반환값: SDS_OK 또는 오류 코드                                                */
/* ------------------------------------------------------------------------- */
int compute_metis_nd_ordering(const CSRMatrix *graph, int *perm, int *inv_perm)
{
    idx_t n;                          /* METIS가 사용하는 정점 개수 */
    idx_t *xadj = NULL;               /* METIS 인접 구조: 행 시작 오프셋 */
    idx_t *adjncy = NULL;             /* METIS 인접 구조: 인접 정점 목록 */
    idx_t *metis_perm = NULL;         /* METIS가 반환하는 순열 (idx_t 타입) */
    idx_t *metis_inv_perm = NULL;     /* METIS가 반환하는 역순열 */
    int status;                       /* METIS 함수 반환 상태 */
    int rc;

    /* 출력 버퍼 포인터 검증 */
    if (!perm || !inv_perm) {
        return SDS_ERR_BAD_INPUT;
    }

    /* 그래프가 METIS 입력 규칙을 만족하는지 검증 (대각 없음, 정렬됨 등) */
    rc = validate_metis_graph(graph);
    if (rc != SDS_OK) {
        return rc;
    }

    /* CSR을 METIS가 요구하는 idx_t 기반 xadj/adjncy 배열로 변환 */
    rc = make_metis_adjacency_arrays(graph, &xadj, &adjncy);
    if (rc != SDS_OK) {
        return rc;
    }

    n = (idx_t)graph->nrows;
    /* METIS 결과를 담을 idx_t 버퍼 할당 */
    metis_perm = (idx_t *)malloc((size_t)graph->nrows * sizeof(idx_t));
    metis_inv_perm = (idx_t *)malloc((size_t)graph->nrows * sizeof(idx_t));
    if (!metis_perm || !metis_inv_perm) {
        /* 할당 실패 시 지금까지 잡은 모든 자원 해제 */
        free(xadj);
        free(adjncy);
        free(metis_perm);
        free(metis_inv_perm);
        return SDS_ERR_ALLOC;
    }

    /*
     * METIS_NodeND 호출.
     * 가중치 배열(vwgt)과 옵션(options)은 NULL로 전달하여 기본값 사용.
     */
    status = METIS_NodeND(&n, xadj, adjncy, NULL, NULL,
                          metis_perm, metis_inv_perm);
    if (status != METIS_OK) {
        /* METIS 실패 시 자원 해제 후 오류 반환 */
        free(xadj);
        free(adjncy);
        free(metis_perm);
        free(metis_inv_perm);
        return SDS_ERR_METIS;
    }

    /* idx_t 결과를 호출 측 int 배열로 복사 (타입 변환) */
    for (int i = 0; i < graph->nrows; ++i) {
        perm[i] = (int)metis_perm[i];
        inv_perm[i] = (int)metis_inv_perm[i];
    }

    /* 임시 버퍼 모두 해제 */
    free(xadj);
    free(adjncy);
    free(metis_perm);
    free(metis_inv_perm);
    return SDS_OK;
}

/* ------------------------------------------------------------------------- */
/* apply_symmetric_permutation                                                */
/*                                                                            */
/* 순열 perm을 행렬에 대칭적으로 적용한다. 즉 P * A * P^T 를 계산한다.         */
/* 행 row 는 새 위치 perm[row] 로, 열 col 은 perm[col] 로 옮겨진다.            */
/*                                                                            */
/* 매개변수:                                                                   */
/*   matrix      : 입력 행렬 (정사각, 값 배열 필수)                            */
/*   perm        : 적용할 순열 (크기 = matrix->nrows)                          */
/*   matrix_perm : 출력 행렬 — 호출 측에서 비어있는 구조체                     */
/*                                                                            */
/* 반환값: SDS_OK 또는 오류 코드                                                */
/* ------------------------------------------------------------------------- */
int apply_symmetric_permutation(const CSRMatrix *matrix,
                                const int *perm,
                                CSRMatrix *matrix_perm)
{
    int *next = NULL;   /* 각 새 행에 다음으로 쓸 위치를 추적하는 보조 배열 */
    int rc;

    if (!matrix_perm) {
        return SDS_ERR_BAD_INPUT;
    }
    /* 출력 구조체 0 초기화로 안전성 확보 */
    memset(matrix_perm, 0, sizeof(*matrix_perm));

    /* 입력 행렬 정사각 CSR 검증 */
    rc = validate_square_csr_matrix(matrix);
    if (rc != SDS_OK) {
        return rc;
    }
    /* 이 연산은 수치 값을 옮기므로 값 배열이 반드시 있어야 한다 */
    if (!matrix->values) {
        return SDS_ERR_BAD_INPUT;
    }
    /* perm이 0..n-1의 유효한 치환(permutation)인지 검증 */
    rc = validate_permutation(perm, matrix->nrows);
    if (rc != SDS_OK) {
        return rc;
    }

    /* 출력 행렬은 입력과 동일한 비영(非零) 원소 개수를 가진다 */
    rc = csr_create(matrix_perm, matrix->nrows, matrix->ncols, matrix->nnz);
    if (rc != SDS_OK) {
        return rc;
    }

    /* 1차 패스: 각 원래 행의 항목 수를 새 행 위치(perm[row])의 카운트로 누적 */
    for (int row = 0; row < matrix->nrows; ++row) {
        const int new_row = perm[row];
        matrix_perm->rowptr[new_row + 1] += matrix->rowptr[row + 1] - matrix->rowptr[row];
    }
    /* 누적 합으로 rowptr을 정상 CSR 오프셋으로 변환 */
    for (int row = 0; row < matrix_perm->nrows; ++row) {
        matrix_perm->rowptr[row + 1] += matrix_perm->rowptr[row];
    }

    /* 각 새 행의 현재 쓰기 위치를 추적할 보조 배열을 rowptr 복사로 초기화 */
    next = (int *)malloc(((size_t)matrix_perm->nrows + 1u) * sizeof(int));
    if (!next) {
        csr_destroy(matrix_perm);
        return SDS_ERR_ALLOC;
    }
    memcpy(next, matrix_perm->rowptr,
           ((size_t)matrix_perm->nrows + 1u) * sizeof(int));

    /* 2차 패스: 실제 항목을 새 위치로 복사. 행은 perm[row], 열은 perm[col]. */
    for (int row = 0; row < matrix->nrows; ++row) {
        const int new_row = perm[row];
        for (int p = matrix->rowptr[row]; p < matrix->rowptr[row + 1]; ++p) {
            const int col = matrix->colind[p];
            const int dst = next[new_row]++;        /* 쓰기 위치 확보 후 증가 */
            matrix_perm->colind[dst] = perm[col];   /* 열도 순열에 따라 재배치 */
            matrix_perm->values[dst] = matrix->values[p];
        }
    }
    free(next);

    /*
     * 재배치 후 각 행의 열 순서가 흐트러질 수 있으므로 행별로 정렬한다.
     * sum_duplicates=1: 동일 (행,열) 위치가 겹치면 값을 합산한다.
     */
    rc = sort_csr_rows_and_merge_duplicates(matrix_perm, 1);
    if (rc != SDS_OK) {
        csr_destroy(matrix_perm);
        return rc;
    }

    return SDS_OK;
}

/* ========================================================================= */
/* 정적(static) 헬퍼 함수 정의                                                 */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* validate_square_csr_matrix                                                 */
/*                                                                            */
/* 행렬이 유효한 정사각 CSR인지 검증한다:                                       */
/*   - 포인터/차원/nnz의 기본 무결성                                            */
/*   - rowptr 의 양 끝값(0, nnz) 일치                                          */
/*   - rowptr 의 단조 증가성                                                   */
/*   - 모든 열 인덱스가 [0, ncols) 범위 내에 있는지                            */
/* ------------------------------------------------------------------------- */
static int validate_square_csr_matrix(const CSRMatrix *matrix)
{
    /* 기본 필드 무결성: 비어있지 않고, 정사각이며, nnz가 음수가 아님 등 */
    if (!matrix || matrix->nrows <= 0 || matrix->nrows != matrix->ncols ||
        matrix->nnz < 0 || !matrix->rowptr ||
        (matrix->nnz > 0 && !matrix->colind)) {
        return SDS_ERR_BAD_INPUT;
    }
    /* CSR 규약: rowptr[0]==0, rowptr[nrows]==nnz */
    if (matrix->rowptr[0] != 0 || matrix->rowptr[matrix->nrows] != matrix->nnz) {
        return SDS_ERR_BAD_INPUT;
    }
    for (int row = 0; row < matrix->nrows; ++row) {
        /* rowptr은 단조 증가해야 함 (행 길이가 음수가 될 수 없음) */
        if (matrix->rowptr[row] > matrix->rowptr[row + 1]) {
            return SDS_ERR_BAD_INPUT;
        }
        /* 각 열 인덱스가 유효 범위 안에 있는지 확인 */
        for (int p = matrix->rowptr[row]; p < matrix->rowptr[row + 1]; ++p) {
            if (matrix->colind[p] < 0 || matrix->colind[p] >= matrix->ncols) {
                return SDS_ERR_BAD_INPUT;
            }
        }
    }
    return SDS_OK;
}

/* ------------------------------------------------------------------------- */
/* validate_permutation                                                       */
/*                                                                            */
/* perm 이 0..n-1 의 정확한 치환인지 검증한다.                                  */
/* 각 값이 범위 내에 있고 한 번씩만 등장하는지를 'seen' 배열로 확인한다.        */
/* ------------------------------------------------------------------------- */
static int validate_permutation(const int *perm, int n)
{
    int *seen;   /* 각 목적지 인덱스가 이미 사용되었는지 표시 */

    if (!perm || n <= 0) {
        return SDS_ERR_BAD_INPUT;
    }

    /* calloc으로 0 초기화된 방문 표시 배열 확보 */
    seen = (int *)calloc((size_t)n, sizeof(int));
    if (!seen) {
        return SDS_ERR_ALLOC;
    }

    for (int i = 0; i < n; ++i) {
        /* 범위를 벗어났거나 이미 등장한 값이면 치환이 아님 */
        if (perm[i] < 0 || perm[i] >= n || seen[perm[i]]) {
            free(seen);
            return SDS_ERR_BAD_INPUT;
        }
        seen[perm[i]] = 1;   /* 해당 목적지 사용 처리 */
    }

    free(seen);
    return SDS_OK;
}

/* ------------------------------------------------------------------------- */
/* validate_metis_graph                                                       */
/*                                                                            */
/* METIS 입력으로 적합한 그래프인지 검증한다:                                   */
/*   - 정사각 CSR로서 유효                                                      */
/*   - 자기 자신으로의 간선(대각) 없음                                          */
/*   - 각 행의 열 인덱스가 엄격히 증가(중복 없이 정렬됨)                        */
/* ------------------------------------------------------------------------- */
static int validate_metis_graph(const CSRMatrix *graph)
{
    int rc = validate_square_csr_matrix(graph);
    if (rc != SDS_OK) {
        return rc;
    }

    for (int row = 0; row < graph->nrows; ++row) {
        int previous_col = -1;   /* 직전 열 인덱스 (엄격 증가 검사용) */
        for (int p = graph->rowptr[row]; p < graph->rowptr[row + 1]; ++p) {
            const int col = graph->colind[p];
            if (col == row) {            /* 대각 간선은 METIS에서 허용 안 함 */
                return SDS_ERR_BAD_INPUT;
            }
            if (col <= previous_col) {   /* 정렬 위반 또는 중복 열 */
                return SDS_ERR_BAD_INPUT;
            }
            previous_col = col;
        }
    }

    return SDS_OK;
}

/* ------------------------------------------------------------------------- */
/* validate_structural_symmetry                                               */
/*                                                                            */
/* 그래프가 무방향(구조적으로 대칭)인지 검증한다.                               */
/* 즉, 간선 (row, col)이 존재하면 반대 간선 (col, row)도 존재해야 한다.        */
/* 행이 정렬되어 있으므로 이진 탐색으로 반대 간선을 확인한다.                   */
/* ------------------------------------------------------------------------- */
static int validate_structural_symmetry(const CSRMatrix *graph)
{
    /* 먼저 기본 METIS 그래프 규칙(정렬/대각없음 등)을 만족해야 함 */
    int rc = validate_metis_graph(graph);
    if (rc != SDS_OK) {
        return rc;
    }

    for (int row = 0; row < graph->nrows; ++row) {
        for (int p = graph->rowptr[row]; p < graph->rowptr[row + 1]; ++p) {
            const int col = graph->colind[p];
            /* (row, col)이 있으면 (col, row)도 있어야 대칭 */
            if (!find_column_in_sorted_csr_row(graph, col, row)) {
                return SDS_ERR_BAD_INPUT;
            }
        }
    }

    return SDS_OK;
}

/* ------------------------------------------------------------------------- */
/* sort_csr_rows_and_merge_duplicates                                         */
/*                                                                            */
/* 각 행을 독립적으로 정렬하고, 같은 행 안의 중복 열을 병합한다.               */
/* 이는 행 단위(row-local) 작업이므로, 모든 간선에 대한 기존의                  */
/* 전역 O(nnz log nnz) 정렬을 피한다.                                          */
/*                                                                            */
/* sum_duplicates:                                                            */
/*   0 → 중복 열을 하나로 합치되 값은 1.0으로 고정 (그래프 패턴 용도)          */
/*   1 → 중복 열의 값을 모두 더함 (수치 행렬 용도)                             */
/* ------------------------------------------------------------------------- */
static int sort_csr_rows_and_merge_duplicates(CSRMatrix *matrix, int sum_duplicates)
{
    int *old_rowptr;   /* 압축 전 원본 rowptr의 사본 */
    int write = 0;     /* 압축 후 배열에 쓸 전역 위치 */

    if (!matrix || !matrix->rowptr || (matrix->nnz > 0 && !matrix->colind)) {
        return SDS_ERR_BAD_INPUT;
    }

    /*
     * rowptr을 제자리에서 갱신하면서 원래 행 경계 정보가 필요하므로,
     * 원본 rowptr을 따로 복사해 둔다.
     */
    old_rowptr = (int *)malloc(((size_t)matrix->nrows + 1u) * sizeof(int));
    if (!old_rowptr) {
        return SDS_ERR_ALLOC;
    }
    memcpy(old_rowptr, matrix->rowptr,
           ((size_t)matrix->nrows + 1u) * sizeof(int));

    for (int row = 0; row < matrix->nrows; ++row) {
        const int start = old_rowptr[row];      /* 원본 행 시작 */
        const int end = old_rowptr[row + 1];    /* 원본 행 끝(배타적) */
        int read = start;                       /* 현재 읽기 위치 */

        /* 이 행의 (col, value) 쌍을 열 기준으로 정렬 */
        sort_row_entries_by_column(matrix->colind + start,
                                   matrix->values ? matrix->values + start : NULL,
                                   end - start);

        /* 압축된 결과의 이 행 시작 위치를 기록 */
        matrix->rowptr[row] = write;
        while (read < end) {
            const int col = matrix->colind[read];
            double value = matrix->values ? matrix->values[read] : 1.0;
            ++read;

            /* 정렬되어 있으므로 동일 열은 연속으로 나타난다 → 묶어서 병합 */
            while (read < end && matrix->colind[read] == col) {
                if (sum_duplicates && matrix->values) {
                    value += matrix->values[read];   /* 값 누적 */
                }
                ++read;
            }

            /* 병합된 단일 항목을 압축 위치에 기록 */
            matrix->colind[write] = col;
            if (matrix->values) {
                matrix->values[write] = sum_duplicates ? value : 1.0;
            }
            ++write;
        }
    }
    /* 마지막 rowptr과 nnz를 압축 결과에 맞게 갱신 */
    matrix->rowptr[matrix->nrows] = write;
    matrix->nnz = write;

    free(old_rowptr);
    return SDS_OK;
}

/* ------------------------------------------------------------------------- */
/* find_column_in_sorted_csr_row                                              */
/*                                                                            */
/* 정렬된 CSR 행 row 안에서 열 col을 이진 탐색한다.                            */
/* 찾으면 1, 없으면 0을 반환한다.                                              */
/* ------------------------------------------------------------------------- */
static int find_column_in_sorted_csr_row(const CSRMatrix *matrix, int row, int col)
{
    int lo = matrix->rowptr[row];          /* 탐색 구간 하한 */
    int hi = matrix->rowptr[row + 1] - 1;  /* 탐색 구간 상한 */

    while (lo <= hi) {
        /* 오버플로 방지를 위한 중간값 계산 */
        const int mid = lo + (hi - lo) / 2;
        const int mid_col = matrix->colind[mid];
        if (mid_col == col) {
            return 1;            /* 찾음 */
        }
        if (mid_col < col) {
            lo = mid + 1;        /* 오른쪽 절반 탐색 */
        } else {
            hi = mid - 1;        /* 왼쪽 절반 탐색 */
        }
    }

    return 0;                    /* 없음 */
}

/* ------------------------------------------------------------------------- */
/* sort_row_entries_by_column                                                 */
/*                                                                            */
/* 단일 CSR 행을 셸 정렬(shell sort)한다. 열 인덱스를 키로 정렬하면서          */
/* 대응되는 값(values)을 같은 순서로 함께 이동시켜 정렬 후에도 (열, 값) 쌍이   */
/* 어긋나지 않도록 한다.                                                       */
/* ------------------------------------------------------------------------- */
static void sort_row_entries_by_column(int *cols, double *values, int count)
{
    /* 셸 정렬: 간격(gap)을 절반씩 줄여가며 갭 단위 삽입 정렬 수행 */
    for (int gap = count / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < count; ++i) {
            const int col = cols[i];                       /* 삽입할 열 키 */
            const double value = values ? values[i] : 0.0; /* 대응 값 */
            int j = i;

            /* gap 간격 앞의 원소가 더 크면 뒤로 밀어낸다 */
            while (j >= gap && cols[j - gap] > col) {
                cols[j] = cols[j - gap];
                if (values) {
                    values[j] = values[j - gap];   /* 값도 함께 이동 */
                }
                j -= gap;
            }

            /* 적절한 위치에 키와 값을 안착 */
            cols[j] = col;
            if (values) {
                values[j] = value;
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* make_metis_adjacency_arrays                                                */
/*                                                                            */
/* CSR 그래프를 METIS가 요구하는 idx_t 기반 인접 배열로 변환한다.             */
/*   xadj   : 길이 nrows+1, 각 정점의 인접 목록 시작 오프셋                    */
/*   adjncy : 길이 nnz,     인접 정점 인덱스의 평탄화 배열                     */
/*                                                                            */
/* 성공 시 *xadj_out, *adjncy_out 에 새로 할당된 배열을 설정한다.             */
/* (nnz==0이면 adjncy는 NULL일 수 있다.) 호출 측이 free 책임을 진다.          */
/* ------------------------------------------------------------------------- */
static int make_metis_adjacency_arrays(const CSRMatrix *graph,
                                       idx_t **xadj_out,
                                       idx_t **adjncy_out)
{
    idx_t *xadj;
    idx_t *adjncy = NULL;

    if (!xadj_out || !adjncy_out) {
        return SDS_ERR_BAD_INPUT;
    }
    /* 실패 시 호출 측이 free하지 않도록 출력 포인터를 먼저 NULL로 설정 */
    *xadj_out = NULL;
    *adjncy_out = NULL;

    /* idx_t로의 안전한 변환을 위해 크기가 INT_MAX를 넘지 않는지 확인 */
    if (graph->nrows > INT_MAX || graph->nnz > INT_MAX) {
        return SDS_ERR_BAD_INPUT;
    }

    /* xadj는 항상 nrows+1 개 필요 */
    xadj = (idx_t *)malloc(((size_t)graph->nrows + 1u) * sizeof(idx_t));
    if (!xadj) {
        return SDS_ERR_ALLOC;
    }
    /* 간선이 있을 때만 adjncy 할당 (nnz==0이면 NULL 유지) */
    if (graph->nnz > 0) {
        adjncy = (idx_t *)malloc((size_t)graph->nnz * sizeof(idx_t));
        if (!adjncy) {
            free(xadj);
            return SDS_ERR_ALLOC;
        }
    }

    /* rowptr → xadj (int → idx_t 변환 복사) */
    for (int i = 0; i <= graph->nrows; ++i) {
        xadj[i] = (idx_t)graph->rowptr[i];
    }
    /* colind → adjncy (int → idx_t 변환 복사) */
    for (int p = 0; p < graph->nnz; ++p) {
        adjncy[p] = (idx_t)graph->colind[p];
    }

    *xadj_out = xadj;
    *adjncy_out = adjncy;
    return SDS_OK;
}
