# 04 — 메모리 레이아웃 & 자료구조

> **층위**: 상세. device 메모리가 어떻게 잡히고, front 숫자가 어떻게 배치되며, 누가 무엇을 소유하는지.
> 두 자료구조가 중심이다: **`MultifrontalPlan`**(청사진, 값 0개) 와 **`State`**(이번 B개의 숫자).

---

## 1. 두 메모리 도메인

| | **plan 메타데이터** | **수치 front 저장** |
|---|---|---|
| 구조 | `MultifrontalPlan` (`internal/plan/`) | `State` (`internal/runtime/`) |
| 만드는 곳 | `Analyze()` (한 번) | `Setup(B)` (B 바뀔 때) |
| 내용 | 오프셋·인덱스·순서(symbolic) | 실제 `A`,`L`,`U` 값 + 작업 버퍼 |
| 배치 의존 | 없음(1부 공유) | B배(시스템마다 복제) |
| 수명 | analyze~소멸까지 | setup~다음 setup까지 |

핵심 직관: **plan은 "어디에 무엇을 놓고 어떤 순서로 launch할지"이고, State는 "이번 숫자가 사는 곳"이다.**

## 2. plan arena — 단일 정렬 아레나 (`analyze/plan/lower.cu`)

symbolic 산출물은 **하나의 연속 device 할당**(`MultifrontalPlan::arena`)에 256B 정렬(`kArenaAlignmentBytes`)로
sub-alloc된다(L2 cache-line straddle 회피). 한 번에 잡고 한 번에 free(소멸자)한다.

주요 device 배열(arena 안):

| 배열 | 크기 | 뜻 |
|---|---|---|
| `d_front_off[p]` | P+1 | panel `p`의 front 시작 오프셋 = `Σ fsz²` prefix. **front-major 레이아웃의 핵심** |
| `d_front_ptr[p]` | P+1 | panel `p`의 front 크기 `fsz`(행/열 수) |
| `d_ncols[p]` | P | panel `p`의 pivot 열 수 `nc` (→ `uc = fsz − nc`) |
| `d_plcols` | P | panel을 레벨 순서(Postorder)로 나열한 디스패치 순서 |
| `d_panel_parent[p]` | P | 부모 panel id(없으면 −1) |
| `d_a_pos[k]` | nnz | 입력 비영 `k`가 들어갈 front 내 위치(scatter 맵). [06 §2](06-factorization.md) |
| `d_asm_ptr / d_asm_local` | CSR | extend-add 조립 맵: 자식 CB의 각 원소가 부모 front의 어느 로컬 (행,열)로 가는지 |
| `d_front_rows` | Σfsz | 각 front의 전역 행 인덱스(정렬됨) — `a_pos` 빌드 시 binary search 대상 |

> `d_front_off`는 **Postorder**라 부모 front가 자식 바로 뒤에 놓여 extend-add 캐시 지역성이 좋다
> (`lower.cu` `LayoutFrontArena`). arena가 1G double(8GB)을 넘으면 analyze가 실패 반환(거대 FEM/circuit 가드).

`MultifrontalPlan`은 이 외에 **host mirror**(디스패치 결정용, device 왕복 없이 읽음)를 가진다: `h_front_ptr`,
`h_ncols`, `h_plcols`, `h_front_off`, `panel_level_ptr`(레벨 CSR), 그리고 tier/subtree 디스패치 메타(`h_plcols_tier`,
`h_level_tier_off`, subtree 배열들, `h_spine_panels`). 구조는 move-only(소멸자가 모든 device 할당 free).

## 3. 수치 front 저장 — front-major + batch-strided (`runtime/setup.cu`, `state.hpp`)

실제 숫자가 들어가는 front는 plan의 symbolic 템플릿이 아니라 **`State`**에 있다.

```
정밀도별 front 버퍼 (FP64: d_front_batch / FP32·TF32: d_front_batch_f):
    한 batch의 모든 front를 연속 배치 ( = front_total = Σ_p fsz² )

배치 b, panel p, front 원소 (i,j):
    F = d_front_batch[_f] + b*front_total + front_off[p]   ← 배치는 front_total 단위로 stride
    F[i*fsz + j]                                            ← fsz×fsz dense, ROW-MAJOR
```

한 front 내부 분할(pivot 블록 vs contribution 블록):

```
          ┌──── nc ────┬──────── uc ────────┐
          │            │                     │
   nc 행 ─┤  L_pp/U_pp  │   U12 (U-panel)     │   nc 행 = pivot 행: 소거되는 변수
          │  (pivot)    │                     │     좌상단 = L_pp(단위하삼각)·U_pp(상삼각)
          ├────────────┼─────────────────────┤
   uc 행 ─┤  L21        │   CB (uc×uc)         │   uc 행 = contribution 행
          │ (L-panel)   │   = Schur 잔차       │     CB는 extend-add로 부모 front에 더해짐
          └────────────┴─────────────────────┘
```

이 분할이 [06](06-factorization.md)의 4단계가 동작하는 좌표계다: panel LU는 `L_pp/U_pp`를, U-solve는 `U12`를,
trailing은 `CB`를 채운다.

## 4. State의 버퍼 (per-Setup)

`State`(`internal/runtime/state.hpp`)는 이번 B개 시스템의 모든 device 자원을 소유한다.

| 버퍼 | 크기 | 배치 | 용도 |
|---|---|---|---|
| `d_front_batch` (FP64) / `d_front_batch_f` (FP32/TF32) | `B × front_total` | ×B | front 저장(정밀도 하나만 할당) |
| `d_y_batch` / `d_y_batch_f` | `B × num_rows` | ×B | solve RHS/작업 벡터 |
| `d_sing` | 1 int | 공유 | 특이 pivot 검출(atomicAdd; 모든 배치 공유) |
| `stream` | — | — | 내부 graph 모드면 solver 소유, 외부면 caller 소유 |
| `subtree_streams[k]`, `fork_event`, `join_events[k]` | ≤8 | 멀티스트림 | 독립 서브트리 fork/join. [08 §3](08-runtime-and-batching.md) |
| `factor_graph_exec` | — | 공유 | Setup서 캡처한 factor graph(내부 모드) |
| `full_solve_graph_exec` (+ 캐시 키) | — | 공유 | `(rhs,sol,perm,iperm,type)` 키로 lazy 캡처한 solve graph |

스칼라: `batch_count`(B), `front_total`, `num_rows`, `precision`, `static_pivoting`/`pivot_threshold`/`pivot_shift`,
`num_subtree_streams`. **B가 바뀌면** `Setup(B)`가 옛 State를 파괴하고 위를 재할당·(내부 모드면) factor graph 재캡처
한다(`setup.cu` `AllocateState`).

## 5. 무엇이 ×B이고 무엇이 공유인가 (요약)

```
공유(1부): MultifrontalPlan 전체 — front_off/front_ptr/ncols/a_pos/asm/tier/subtree 메타
           State.d_sing, factor_graph_exec, full_solve_graph_exec, 스트림/이벤트 인프라
×B(복제): State.d_front_batch[_f]   (b*front_total + front_off[p])
          State.d_y_batch[_f]        (b*num_rows + i)
          등록 입력: values[b*nnz+·], rhs[b*n+·], solution[b*n+·]
```

즉 **symbolic은 한 번 만들어 B개가 공유하고, 숫자만 B배**다 — 이게 "한 symbolic, B numeric" 배치의 메모리적 의미다.
