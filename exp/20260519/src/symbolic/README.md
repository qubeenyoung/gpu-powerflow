# Symbolic Factorization

이 모듈은 멀티프론털(multifrontal) 희소 직접 솔버의 **기호 분석(symbolic analysis) 단계**를 구현합니다.
METIS로 재배열된 희소 행렬을 받아, 수치 인수분해(numeric factorization)에 필요한 모든 메타데이터를
실제 수치 연산 없이 미리 계산합니다.

알고리즘 설계는 STRUMPACK의 `SeparatorTree`, `EliminationTree`, `Front` 클래스에서 참조했으며,
C 스타일로 재구현한 실험용 코드입니다.

---

## 알고리즘 파이프라인

`symbolic_factorization_analyze()` 한 번 호출로 다음 7단계가 순서대로 실행됩니다.

```
A_perm (CSC 패턴)
    │
    ▼
[1] 소거 트리 (etree.c)
    ── Liu/Gilbert 알고리즘으로 열별 소거 트리 구성
    ── DFS 포스트오더 계산
    │
    ▼
[2] 분리자 트리 (septree.c)
    ── 소거 트리의 단일 자식 경로를 하나의 separator로 병합
    ── 2개 초과 자식은 이진 더미 노드로 축소 → 이진 트리 보장
    │
    ▼
[3] 프론트 트리 (front_symbolic.c)
    ── separator 노드별로 FrontSymbolic 생성
    ── pivot_vars: 해당 separator가 소유하는 변수
    ── update_vars: 해당 프론트에서 위로 전파되는 Schur 보완 변수
    ── update_to_parent: 자식 update_vars → 부모 front_vars 인덱스 맵
    │
    ▼
[4] 프론트 스케줄 (front_schedule.c)
    ── 하향식 레벨(level) 계산: 리프 = 레벨 0, 루트 = 최대 레벨
    ── factor_order / forward_order: 자식 → 부모 순서 (포스트오더)
    ── backward_order: 부모 → 자식 순서 (역방향 치환에 사용)
    ── level_fronts: 같은 레벨의 프론트는 병렬 처리 가능
    │
    ▼
[5] 저장 계획 (storage_plan.c)
    ── 프론트별 밀집 블록 크기와 전역 오프셋 계산
    ── 단일 연속 배열에 모든 프론트의 블록을 순서대로 배치
    │
    ▼
[6] 엔트리 어셈블리 계획 (assembly_plan.c)
    ── A_perm의 각 비영값 → (대상 프론트, F 내 로컬 오프셋) 매핑
    │
    ▼
[7] 기여 어셈블리 계획 (assembly_plan.c)
    ── 자식 C 블록의 각 엔트리 → 부모 F 블록의 오프셋 매핑
```

---

## 프론트의 밀집 블록 레이아웃

각 프론트는 `npiv`(pivot 변수 수)와 `nupd`(update 변수 수)로 크기가 결정됩니다.
`nfront = npiv + nupd`. 모든 블록은 열 우선(column-major) double 배열에 연속 배치됩니다.

```
오프셋        블록     크기              설명
─────────────────────────────────────────────────────────────────
F_offset   → F       nfront × nfront   어셈블된 프론털 패널
L11_offset → L11     npiv  × npiv      피벗 블록의 하삼각 인수
U11_offset → U11     npiv  × npiv      피벗 블록의 상삼각 인수
L21_offset → L21     nupd  × npiv      하부 오프-다이어그널 인수
U12_offset → U12     npiv  × nupd      상부 오프-다이어그널 인수
C_offset   → C       nupd  × nupd      Schur 보완 / 기여 블록
```

---

## 주요 자료구조

| 타입 | 정의 위치 | 설명 |
|------|-----------|------|
| `SeparatorTree` | `septree.h` | `sizes[]`, `parent[]`, `left_child[]`, `right_child[]`를 가진 이진 separator 트리 |
| `FrontSymbolic` | `symbolic_factorization.h` | 프론트별 pivot/update 변수 목록과 `update_to_parent` 인덱스 맵 |
| `FrontSchedule` | `symbolic_factorization.h` | 포스트오더 순회 배열과 CSR 형식의 레벨 그룹 |
| `FrontStoragePlan` | `symbolic_factorization.h` | 프론트별 밀집 블록 크기와 전역 오프셋 |
| `EntryAssemblyPlan` | `symbolic_factorization.h` | A 비영값 → 밀집 F 엔트리 매핑 |
| `ContributionAssemblyPlan` | `symbolic_factorization.h` | 자식 C 엔트리 → 부모 F 엔트리 매핑 |
| `SymbolicFactorization` | `symbolic_factorization.h` | 위의 모든 구조체를 소유하는 최상위 객체 |

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `etree.c/h` | Liu/Gilbert 소거 트리 + DFS 포스트오더 |
| `septree.c/h` | 분리자 트리 구성 및 검증 |
| `front_symbolic.c/h` | FrontSymbolic 생성, update set, update-to-parent 맵 |
| `front_schedule.c/h` | 순회 순서 및 레벨 그룹 계산 |
| `storage_plan.c/h` | 밀집 블록 크기 및 오프셋 계산 |
| `assembly_plan.c/h` | 엔트리 어셈블리 계획 및 기여 어셈블리 계획 |
| `symbolic_factorization.c/h` | 최상위 `symbolic_factorization_analyze()` 오케스트레이터 |
| `symbolic_internal.c/h` | 공유 유틸리티 (int 리스트, 정렬, 위치 맵, 검증 헬퍼) |
| `symbolic_print.c/h` | 디버그 출력 루틴 |
| `symbolic_validate.c/h` | 분석 후 무결성 검사 |

---

## 사용 예시

```c
SymbolicFactorization symbolic;

int rc = symbolic_factorization_analyze(a_perm_pattern, perm, inv_perm, &symbolic);
if (rc != SDS_OK) { /* 오류 처리 */ }

/* symbolic.fronts, symbolic.storage, symbolic.entry_assembly 등 사용 */

symbolic_factorization_destroy(&symbolic);
```

입력: METIS 재배열된 행렬 패턴 + 순열 배열
출력: 수치 인수분해 직전 상태의 완전히 초기화된 `SymbolicFactorization`
종료 후 반드시 `symbolic_factorization_destroy()` 호출.

---

## 알고리즘 참고 문헌

- J. W. H. Liu, "The Role of Elimination Trees in Sparse Factorization", *SIAM J. Matrix Anal. Appl.*, 1990.
- A. Pothen and C.-J. Fan, "Computing the Block Triangular Form of a Sparse Matrix", *ACM TOMS*, 1990.
- STRUMPACK: `SeparatorTree.hpp`, `EliminationTree.hpp`, `fronts/Front.hpp`.
