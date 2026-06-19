# 08 — 런타임 & 배치 (analyze · setup · graph · B=1 vs B>1)

> **층위**: 상세. analyze 파이프라인이 무엇을 만들고, `Setup`이 무엇을 잡고, CUDA graph가 어떻게 캡처/replay되며,
> 멀티스트림과 **B=1 vs B>1** 두 체제가 어디서 갈리는지.

---

## 1. Analyze 파이프라인 (`analyze/pipeline.cpp` `BuildPlanSeed`)

패턴만 보고(값 무관) 한 번 돈다. 8단계:

```
1. BuildCscFromCsrDevice      CSR(A) → CSC (device)
2. BuildSymmetricGraphDevice   A + Aᵀ 대칭 인접 그래프 (METIS 입력)
3. (옵션) StructuralMatching    Hopcroft–Karp 완전 매칭 (matching=Structural 일 때)
4. MetisNdFromGraph            METIS nested dissection → perm/iperm (fill-reducing 재정렬)
5. PermuteCscDevice            치환 적용 + ordered_value_to_csr 맵 (값 scatter용)
6. Etree                       elimination tree (Liu)
7. FillPattern                 채움 패턴 L (Davis; METIS postorder라 fill-neutral)
8. AnalyzeMultifrontal         supernode 묶기 → front 레이아웃 → 레벨/tier/subtree → device 업로드
```

산출물은 `MultifrontalPlan` + perm/iperm + `d_ordered_value_to_csr`(매 Factorize가 쓰는 값 scatter 맵). 구조는
[04 §2](04-memory-layout.md), 재정렬 상세는 [02 §2](02-source-layout.md)의 `reorder/`·`symbolic/`.

> **METIS ND 결정성**: parallel ND는 분리자 후 양분을 별도 스레드로 재귀해 빠르지만 METIS RNG가 thread-unsafe
> global이라 run-to-run 비결정적(한 실행/NR 루프 안에선 한 번 계산해 고정). 재현 벤치는 `use_parallel_nested_dissection=false`
> (serial `METIS_NodeND`). 튜닝(`parallel_nd_depth`/`_base_small`/`_base_large`)은 [03 §2](03-api-config-build.md).

## 2. Setup & graph 캡처 (`internal/runtime/setup.cu`)

`Setup(B)`는 이번 B개 시스템의 런타임을 잡는다([04 §4](04-memory-layout.md)): front/벡터 버퍼, `d_sing`, (조건부)
서브트리 스트림·이벤트. **B가 바뀌면** 옛 State를 파괴하고 재할당한다.

graph 모드 두 가지(`CLS_INTERNAL_GRAPH`):

| | **내부 graph (ON, 기본 standalone)** | **외부 캡처 (OFF)** |
|---|---|---|
| factor graph | `Setup`에서 한 번 캡처(`CapturePhaseGraphs`) → `Factorize`마다 `cudaGraphLaunch`+sync | 캡처 없음. 커널을 caller 스트림에 직접 발행 |
| solve graph | `(rhs,sol,perm,iperm,type)` 키로 **lazy 캡처/캐시**, 같으면 replay | 직접 발행, host sync 없음 |
| 스트림 | solver 소유 | caller 소유(`SetStream`) |
| 멀티스트림 | 가능 | 비활성(외부 캡처 깨짐) |
| 용도 | 단독 실행 | cuPF 등 외부가 NR 한 iteration 전체를 graph로 기록 |

내부 모드의 이득: 매 NR 반복의 launch overhead 제거(host-free replay).

## 3. 멀티스트림 서브트리 (B>1, 내부 모드)

plan이 독립 서브트리를 노출하고 `1 < num_subtrees ≤ 8`(`kMaxSubtreeStreams`)이면, `Setup`이 서브트리당 스트림+join
이벤트를 만든다(`CreateSubtreeStreams`). factor/solve는:

```
fork:  main 스트림에 이벤트 기록 → 각 서브트리 스트림이 그 이벤트를 wait
sweep: 각 서브트리 스트림이 자기 패널 슬라이스의 leaf~spine_lo 레벨을 tier-split하며 처리
join:  각 서브트리의 join 이벤트를 main 스트림이 wait
spine: 남은 spine 레벨(cnt=1 직렬 체인)을 main 스트림이 처리
```

독립 서브트리를 Hyper-Q로 겹쳐 GPU를 더 채운다(`schedule.cuh` `IssueFactorBatched`, `solve/dispatch.cuh`).

## 4. B = 1 vs B > 1 — 두 일관 경로

분기점 하나, **`UseSingleSystem(st) == (B == 1)`**(`state.hpp`)이 factor/solve 전체를 가른다
(`schedule.cuh` `IssueFactorLevels`).

| 측면 | **B = 1 (단일 시스템)** | **B > 1 (배치)** |
|---|---|---|
| factor 진입 | `IssueFactorSingleSchedule` (`single.cuh`) | `IssueFactorBatched` (`schedule.cuh`) |
| 매핑 | 1 block/front, factor+extend-add 융합 | 레벨별 tier-split, `grid.y = batch` |
| 멀티스트림 | 없음(단일 시스템) | 독립 서브트리 fork/join(num_subtrees∈[2,8]) |
| big front | `FactorSingleBigPanel/Trail[Tf32]/Extend` 멀티블록 | `FactorBig`(FP32/TF32) / pivot·panel·trail 3-launch(FP64) |
| **solve 준비** | factor 후 **pivot 블록 partitioned-inverse**(selinv, `FactorSingleInvertPivot`) | 없음 |
| solve | 역행렬화된 pivot으로 **병렬 GEMV**(`solve/single.cuh`) | 레벨별 전진/후진 대입(`solve/dispatch.cuh`) |

### 왜 B=1이 다른가
배치 스케줄은 per-level launch/barrier latency를 **B개에 걸쳐 amortize**한다. B=1에선 그 latency가 그대로 노출되므로:
1. front당 1 block 융합 커널로 launch/barrier를 줄이고,
2. pivot 블록을 미리 역행렬화(selinv)해 **삼각 역대입을 병렬 GEMV로** 바꿔 직렬 의존을 없앤다([07 §4](07-solve.md)).

측정 배경: [`../_legacy/03-optimization-notes/06-b1-factorize-regime-2026-06-13.md`](../_legacy/03-optimization-notes/06-b1-factorize-regime-2026-06-13.md),
[`../_legacy/history/b1-single-system-optimization.md`](../_legacy/history/b1-single-system-optimization.md).

## 5. 왜 빠른가 (런타임 관점 요약)

- **호스트-프리 실행**: factor/solve를 graph로 캡처해 매 반복 launch overhead 제거.
- **점유 우선 매핑 + 멀티스트림**: 작은 front를 packing/multi-block/Hyper-Q로 GPU를 채움([05](05-front-tiers.md), [06](06-factorization.md)).
- **메모리 왕복 제거**: extend-add 융합, whole-front shared staging([06 §3](06-factorization.md)).
- **B=1 latency 절감**: 융합 커널 + selinv GEMV(§4).

정직한 성능 천장(벤치 수치·공정 통제)은 [`../README.md` §8](../README.md), 미래 레버는
[`../_legacy/03-optimization-notes/`](../_legacy/03-optimization-notes/).
