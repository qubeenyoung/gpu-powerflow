# Elimination Tree 재구조화 — 문헌 조사 + 연구 계획

> Status: historical research log. tree restructuring의 시도, 기각 사유, race condition 기록을 보존한 문서다.
> 최신 전체 결론은 [`../05-reports/01-final-report-2026-06-05.md`](../05-reports/01-final-report-2026-06-05.md)를 우선 참고한다.

*목표: GPU 병렬에 유리하게 elimination tree 자체를 *재구성*. TC 사용 여부와 무관 (TC 기각은 별도 — [`06-tc-dedicated-path-study.md`](06-tc-dedicated-path-study.md) 참조).*

## 1. 문제 — 현재 tree 의 GPU-부적합성

### 1.1 입증된 측정 사실

- 단일 배치 [`../04-benchmarks-profiling/06-single-batch-bottleneck-case8387-fp64.md`](../04-benchmarks-profiling/06-single-batch-bottleneck-case8387-fp64.md) §10: 30 levels 중 L17–L29 의 13 levels 가 panel=1 (cnt=1) 의 narrow chain. GPU 의 82 SM 중 1 SM 만 사용 → 21 % 의 wall 이 SM idle 상태로 흐름.
- 멀티 배치 [`../04-benchmarks-profiling/08-batched-throughput-fp32-case8387-b2-b1024.md`](../04-benchmarks-profiling/08-batched-throughput-fp32-case8387-b2-b1024.md) §7: B=64 에서도 stall 의 80 % 가 non-compute (long_scoreboard + wait + barrier). issued cycles 의 비중이 B 와 무관하게 ~12 % flat → *compute 가 병목이 아니라 latency / dispatch 가 병목*.
- TC dedicated 시도 ([`06-tc-dedicated-path-study.md`](06-tc-dedicated-path-study.md)): WMMA tile grid 가 case8387 의 fsz=76, USA 의 fsz=245 영역에서도 너무 작아 setup amortize 불가.

### 1.2 fundamental 한계 = elimination tree 의 shape

multifrontal 알고리즘은 트리 root 가 반드시 1 개. log₂(82 SM) ≈ 7 levels 위쪽은 *어떤 cap 으로도 narrow*. 이건 알고리즘 정의에 박힌 한계 ([`../04-benchmarks-profiling/06-single-batch-bottleneck-case8387-fp64.md`](../04-benchmarks-profiling/06-single-batch-bottleneck-case8387-fp64.md) §10.2 의 입증).

→ **tree 의 shape 를 의도적으로 바꾸지 않으면 narrow-top 문제가 안 풀린다**.

## 2. 문헌 조사 결과 — 기법 분류

### 2.1 Tree shape 자체를 바꾸는 기법

#### A. Alternative graph partitioner (ordering)

기본은 METIS nested dissection (ND). 대안:
- **Scotch / PT-Scotch** [(Pellegrini)](https://www.researchgate.net/publication/222670658_PT-SCOTCH_a_tool_for_efficient_parallel_graph_ordering) — multilevel ND, METIS 와 quality 비슷 (matrix 별 차이 존재)
- **KaHIP** [(Sanders, KaHIP)](https://github.com/KaHIP/KaHIP) — 최근 partitioner, ParMETIS / PT-Scotch 보다 quality 높음 (자체 보고). top-level cut 에서 특히 우수 [(Hamann et al. 2019)](https://arxiv.org/pdf/1906.11811)
- **RCM** — 일부 matrix 에서 더 좋지만 parallel 적합성 낮음
- **AMD / minimum-degree** — 보통 더 narrow tree

ROI: ordering 자체로 tree top 의 *별로 넓어지지 않음*. multilevel ND 의 quality 차이는 *fill* 에 영향, top width 자체는 root separator 크기에 결정.

#### B. Aggressive amalgamation (현재의 확장)

현재 `relaxed_panels` (cap=8/16) 는 *parent-child chain* 만 merge. 확장 가능 방향:
- **Sibling supernode amalgamation** [(Hypermatrix Oriented, López et al.)](https://link.springer.com/article/10.1007/s11227-008-0188-y) — 같은 부모의 sibling fronts 를 *padded dense block* 으로 묶음. variable-sized hypermatrix partitioning.
- **Aggressive chain merge** — cap 을 64+ 까지 (case8387: 이미 검증 — root separator 가 limit, cap 으로 못 넘김)
- **Subtree fusion** [(Rennich et al., Davis CHOLMOD GPU)](https://people.engr.tamu.edu/davis/publications_files/IA3_2014_Workshop_Rennich_Stosic_Davis_preprint.pdf) — 작은 subtree 통째를 한 dense LU 로 처리

case8387/USA 적용성: *sibling fusion 이 가장 가능성 큼*. narrow-mid (L13–L16) 의 cnt 3–9 fronts 가 padded dense block 으로 묶이면 dispatch overhead ↓ + dense block 크기 ↑.

#### C. Subtree streaming through GPU

[gSoFa (UCR/UMass, 2020)](https://arxiv.org/pdf/2007.00840) — elimination tree 의 leaf-terminated branches 를 stream 단위로 GPU 에 보내 factorize.

ROI: case8387 의 L0 가 4094 leaves 라 *지금도 stream 적임*. 추가 win 작음. *narrow top* 은 stream 단위가 안 됨.

### 2.2 Tree 는 그대로, dispatch / scheduling 을 바꾸는 기법

#### D. Multi-stream / fork-join dispatch

[STRUMPACK](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000059) 의 CPU 구현은 OpenMP tasking 으로 sibling subtree 를 *concurrent* 실행. GPU 구현은 level-by-level batched (sibling concurrency 없음).

case8387/USA 단일 GPU 에서 적용: CUDA stream 또는 cooperative groups 로 sibling subtree 를 fork-join 표현. 단일 배치 doc §10 의 *narrow-top tree 에선 ROI 작음* 결론 (L13 부터 cnt ≤ 9, L17 부터 cnt=1). 멀티 배치 doc §4 의 *narrow-mid 영역에선 cnt × B 가 grid 채움* 결론.

#### E. Persistent kernel (deep-chain fusion)

[Laine et al. 2013, "Megakernels Considered Harmful"](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf) 의 caveat: 큰 megakernel 은 register pressure 로 occupancy 손해. 그러나 *spine 만 흡수* 하는 *small persistent kernel* 은 OK.

case8387 narrow spine (L17–L29 의 13 levels) 흡수 시 잠재 절감: 단일 배치 doc §6 의 deep-chain fusion 후보, 잠재 ~100–150 μs / call. batched 환경에서 잠재 약간 작음 (B 차원으로 amortize).

### 2.3 Tree 자체를 회피하는 기법

#### F. 3D sparse LU (SuperLU_DIST)

[Sao et al., OSTI 2019](https://www.osti.gov/biblio/1559632) — 트리를 *subtree + ancestor* 로 분할, subtree 는 독립 process grid, ancestor 는 모든 grid 에 *replicate*. **non-planar 3.3×, planar 27×** speedup vs 2D baseline.

case8387/USA 적용성: **multi-GPU 만**. 우리 환경 (단일 RTX 3090) N/A.

#### G. BLR (Block Low-Rank) compression

[Claus, Ghysels, Boukaram, Li 2025 STRUMPACK GPU](https://escholarship.org/uc/item/7tn9n67r) — root 근처 큰 fronts 의 dense block 을 low-rank 로 압축. 대형 PDE 에서 13.8× CPU 대비.

case8387/USA 적용성: max fsz 76 (case8387), 245 (USA). BLR 의 sweet spot 은 fsz ≥ 256 부터. *power-grid 에선 너무 작음*. **N/A**.

#### H. Selected inversion (PEXSI / PSelInv)

[Lin et al.](https://math.berkeley.edu/~linlin/publications/PSelInvTree.pdf) — full solve 가 아니라 inverse 의 selected elements 만 계산. 양자 화학 / electronic structure 영역.

case8387/USA 적용성: 우리 task = full solve. **N/A** (다른 문제 영역).

#### I. Iterative + preconditioner

direct solver 자체 회피, conjugate gradient / GMRES + AMG/ILU preconditioner. ROI 큼 (power-grid 의 conditioning 좋으면 빠른 수렴). 하지만 *direct solver 의 robust 성을 잃음* (iterative 가 발산하면 fallback 필요).

## 3. case8387 / USA 적용성 매트릭스

| 기법 | 단일 GPU | small batch | 트리 변경? | case8387 ROI 추정 |
|---|---|---|---|---|
| **A. Alternative ordering (Scotch/KaHiP)** | ✓ | ✓ | shape 살짝 | low (METIS 이미 quality 높음) |
| **B. Sibling amalgamation** | ✓ | ✓ | **yes (큰 변경)** | **medium-high** |
| C. Subtree streaming | ✓ | ✓ | no | low (이미 그렇게 동작) |
| **D. Multi-stream fork-join** | ✓ | ✓ | no | low (narrow top 한계, batched 에선 거의 0) |
| **E. Persistent kernel (spine fusion)** | ✓ | ✓ | no | **medium** (5–10 % factor 절감) |
| F. 3D sparse LU | ✗ multi-GPU | N/A | yes | N/A |
| G. BLR compression | ✓ | ✓ | yes | N/A (fronts 너무 작음) |
| H. Selected inversion | ✓ | ✓ | no | N/A (다른 문제) |
| I. Iterative + preconditioner | ✓ | ✓ | 알고리즘 변경 | high (별도 연구) |

→ **단일 GPU + small batch + 트리 변경 영역의 진짜 lever 는 B (sibling amalgamation) + E (spine fusion) 의 조합**.

## 4. 연구 계획 — B/C/D/E 통합 아키텍처

§2 에서 분류한 4 기법:
- **B**: sibling amalgamation (트리 변경, dense block 키움)
- **C**: subtree streaming (subtree 단위로 dispatch)
- **D**: multi-stream fork-join (independent subtree 동시 실행)
- **E**: persistent kernel (spine 흡수)

이 넷은 *독립 phase 가 아니라 하나의 통합된 새 아키텍처의 구성요소*. 각각 단독 적용해서는 narrow-top 의 본질적 한계가 안 풀리지만, *함께 적용* 하면 factor 의 dispatch 구조 자체가 바뀜.

### 4.0 통합 아키텍처 — "K-subtree + spine"

**현재 구조** (level-by-level sequential):
```
L0 (4094 fronts)  ──┐
L1 (1520)         ──┤
L2 (733)          ──┤
...                 ├── 30 levels sequential
L16 (3)           ──┤   level k+1 waits all of level k
L17–L29 (1 each)  ──┘
```

**제안 구조** (K-subtree + spine):
```
subtree_1 ──┐
subtree_2 ──┼── K subtrees in K parallel streams       ← [C + D]
subtree_K ──┘   (K = top spine 직전 의 cnt)
            │
            ↓ converge (event sync)
                                                       ← [E]
spine persistent kernel (cnt=1 chain)
```

핵심 변화:
1. **K subtree 정의** [C]: 트리에서 spine 직전의 cnt 가 작은 level (case8387 L16, cnt=3) 위쪽이 spine, 아래쪽이 K 개 독립 subtree
2. **subtree 내부 amalgamation** [B]: 각 subtree 의 narrow-mid 영역 (현재 cnt 3–9) 의 sibling fronts 를 dense block 으로 묶음
3. **K streams 으로 fork-join dispatch** [D]: K 개 subtree 가 K 개 stream 으로 concurrent. CUDA Graph 가 fork-join 표현
4. **spine 은 persistent kernel** [E]: cnt=1 chain 을 한 launch 의 device loop 로

이 4 가지가 *동시* 적용되어야 의미. 부분 적용은:
- B 만: dispatch 는 그대로 level-by-level → 회수 미흡
- D 만: 좁은 영역만 parallelize → narrow-top doc §10 의 ROI 작음 결론
- E 만: spine 만 회수 → factor 의 19 % 만 영향
- B+D 만: spine 그대로 → 마지막 13 launches 의 latency 남음
- B+D+E: 통합 win, *진짜* tree restructuring

### 4.1 Phase α — 측정 인프라 (선결)

진짜 구현 전에 *측정 도구* 부터:

**산출물**:
- 각 level 의 cnt, maxfsz, wall time attribution (nsys SQL 기반)
- "spine 정의" 자동 추출: cnt=1 의 가장 긴 chain top from root
- "K subtree" 자동 분할: spine 직전 level 의 panel 들을 K subtree root 로
- 각 subtree 의 size (panels, total work), critical path 추정

**비용**: 낮음 (~200 LOC python + helper). **ROI: 모든 후속 phase 의 발판**.

### 4.2 Phase 1 — Subtree 분할 + spine 식별 [C 기반]

**알고리즘**:
1. 기존 `relaxed_panels()` + multifrontal symbolic 실행 → panel etree 확보
2. panel etree 의 root 부터 leaf 방향으로 traverse, *cnt 가 1 인 connected chain* = **spine**
3. spine 아래 (= spine 마지막 panel 의 자식들) = **K subtree roots**
4. 각 subtree = root 부터 자식 closure 전부 (그 자체로 multifrontal subproblem)

**case8387 예측** (cap=8):
- spine = L17 부터 L29 = 13 panels (single-column chain)
- K subtree roots = L16 의 3 panels
- 각 subtree size: 평균 ~2470 panels (전체 7415 의 1/3)

**case_USA 예측**:
- spine ≈ L25 부터 L39 = ~15 panels
- K subtree roots = L24 의 2 panels (smaller K)
- 각 subtree size: ~37000 panels

**산출물**: 새 함수 `tc_partition_subtrees(plan)` returning `{spine_panel_ids[], subtree_panel_ids[][]}`. 기존 plan 구조에 더해 metadata.

**비용**: 낮음 (~150 LOC, 트리 traversal). **ROI**: 인프라 — Phase 2/3/4 의 발판.

### 4.3 Phase 2 — Subtree 내부 sibling amalgamation [B]

각 subtree 의 *내부 top* (cnt 가 적은 level) 에서 sibling fronts 를 padded dense block 으로 묶음.

**알고리즘** (Hypermatrix Oriented 기반):
1. subtree 의 각 level 분석. cnt < 일정 threshold (예: 16) 이면 *fusion 후보*
2. 같은 parent 의 sibling fronts 의 column structure union 을 계산
3. union 을 *padded dense block* 으로 표현. 새 super-panel 생성
4. parent-child 관계 재정렬 (super-panel 이 sibling 들의 *결합된 부모* 가 됨)
5. asm_ptr / asm_local 재구성 — super-panel 의 CB 가 spine 의 첫 panel 로 extend-add

**Trade-off**:
- 절감: subtree top 의 dispatch 회수 ↓, dense block 크기 ↑
- 손해: padded fill 증가. column structure 차이가 클수록 wasted 영역 ↑

**case8387 예측**: subtree 내부 narrow-mid (L11–L15) 에서 cnt 3–9 fronts 묶이면 fsz ~50–150 영역의 dense block. spine 직전의 cnt=3 (L16) 의 padded union = ~3×60 ≈ 180 column wide dense block.

**case_USA 예측**: 더 강력. L19 cnt=7 maxfsz=235 의 7 fronts 묶이면 fsz ~ 1600 column wide block → TC sweet spot 진입.

**산출물**: 새 symbolic 모듈 `src/symbolic/sibling_amalgamation.{hpp,cpp}`. 기존 `multifrontal_symbolic` 의 변종.

**비용**: 코드량 대형 (~800 LOC). 새 데이터 구조 + 검증 (잔차 변하면 안 됨, fill 증가 합리적이어야).

**Risk**:
- padded fill 폭증 → memory exceed 또는 work flops ↑ 가 dispatch 절감 압도
- column structure 차이가 심한 sibling 들은 amalgamation 부적합

### 4.4 Phase 3 — Multi-stream fork-join dispatch [D]

K subtree 들을 K 개 CUDA stream 으로 concurrent factorize.

**알고리즘**:
1. K 개 별도 stream 생성 (그래프 캡처와 호환되는 captured stream)
2. CUDA Graph capture 시:
   - 메인 stream 에서 K event 생성
   - 각 subtree 가 자기 stream 에서 capture, 자기 event 로 signal
   - 메인 stream 이 K events 모두 wait
   - 메인 stream 에서 spine kernel launch (Phase 4)
3. graph 가 fork (K subtrees) → join (events wait) → spine 의 dependency 표현

**case8387 예측**: K=3 subtrees concurrent. 각 subtree 의 wall time 이 비슷하면 *총 wall = max(subtree wall) + spine wall*. 현재는 *총 wall = sum(level wall)*. 잠재 3× 가속 on subtree 영역 (실제로는 K=3 이라 wall 의 1/3 정도 회수).

**case_USA 예측**: K=2 subtrees. less ROI from D 자체. 하지만 각 subtree 의 work 가 커서 (USA 의 root separator 가 작은 cnt 임) D 의 효과 측정 필요.

**산출물**:
- 새 dispatch 모듈 `src/tc/multifrontal_tc_subtree.cu` (혹은 기존 tc 확장)
- CUDA Graph capture 의 fork-join 표현

**비용**: 코드량 중간 (~400 LOC). 디버깅 까다로움 (stream 간 sync 가 정확해야 race-free).

**Risk**:
- subtree 들이 서로 다른 SM 자원을 다투면 그냥 serialize 됨 (RTX 3090 의 SM 분할 정책)
- CUDA Graph 의 fork-join 표현이 잘 capture 되어야 함

### 4.5 Phase 4 — Spine persistent kernel [E]

spine (cnt=1 chain) 을 한 launch 로 흡수.

**알고리즘**:
1. K subtrees 모두 완료 후 ([B+C+D] join), spine 의 첫 panel 에 모든 K subtrees 의 CB 가 extend-add 된 상태
2. 새 kernel `mf_spine_factor_b<float>`: persistent grid (B 차원으로 grid.x = B), 각 block 이 한 batch 의 spine 13 levels 를 *device-side loop* 로 순회
3. 한 level 마다: panel LU + U-solve + (한 level 만 있으므로 no trailing GEMM unless level 자체에 child 가 있음 — spine 은 single-panel chain 이라 자기 panel 만)
4. 기존 그래프의 마지막 13 launches 가 이 1 launch 로 치환

**case8387 예측**: 13 launches × 5 μs latency ≈ 65 μs 절감. factor wall ~10 %.

**case_USA 예측**: spine 15 levels 이지만 각 panel 의 work 가 더 큼. 절감 비중 비슷 ~10 %.

**산출물**: 새 kernel `src/tc/spine_kernels.cuh` 또는 `src/batched/`. 기존 graph capture 의 변경.

**비용**: 코드량 중간 (~300 LOC). debugging 단순 (spine 의 일직선 chain).

**Risk**:
- Megakernel 경고 (Laine 2013): register pressure ↑ → occupancy ↓. spine 의 fsz 가 작아 (case8387 max 68, USA max 240) register 사용 통제 가능.

### 4.6 통합 시 예상 효과 (case8387, FP32, B=64 기준)

| 단계 | factor wall μs/sys | 누적 절감 |
|---|---:|---:|
| baseline (현재 FP32 batched) | 33.9 | — |
| + α (측정만, no impl) | 33.9 | 0 |
| + Phase 1 (subtree 분할만, no exec change) | 33.9 | 0 |
| **+ Phase 2 (sibling amalgamation)** | **~28** | **−17 %** (narrow-mid dispatch 절감) |
| **+ Phase 3 (K-stream fork-join)** | **~24** | **−29 %** (subtree concurrent) |
| **+ Phase 4 (spine fusion)** | **~22** | **−35 %** (spine launches 절감) |

이론적 ceiling = `max(subtree_wall) + spine_wall` ≈ 22 μs/sys.

**case_USA 예측**:
- baseline 471 → 통합 후 ~280 μs/sys (−40 %)
- *추가* — sibling amalgamation 의 dense block (fsz 500+) 이 TC sweet spot 진입 → Phase 2 이후 *TC 다시 켜기* 로 추가 −15 %

### 4.7 구현 의존성 + 권장 순서

```
α (측정) ──┬─→ Phase 1 (subtree partition) ──┬─→ Phase 2 (sibling amalg)
           │                                  │
           │                                  ├─→ Phase 3 (multi-stream)
           │                                  │
           │                                  └─→ Phase 4 (spine fusion)
           │
           └─→ (직접 측정 가능, Phase 1 없이도)
```

권장 순서:
1. **α** — 측정 인프라 (필수 발판)
2. **Phase 1** — subtree 분할 (다른 phases 가 의존)
3. **Phase 4** — spine fusion 먼저 (가장 단순, ROI 명확)
4. **Phase 2** — sibling amalgamation (가장 복잡, ROI 가장 큼)
5. **Phase 3** — multi-stream (Phase 2 의 fusion 된 subtree 위에서 효과 큼)

Phase 4 를 Phase 2 보다 먼저 하는 이유: 구현 단순 + 명확한 win 으로 *측정 인프라 검증* + *후속 phase 의 baseline 강화*. 단독으로도 의미 있음.

### 4.8 위험 평가 + Go/No-Go 기준

각 phase 의 mid-implementation Go/No-Go:

- **Phase 1**: subtree 분할 결과의 K 값 확인. K ≥ 2 가 아니면 D 의 effectiveness 없음. case8387/USA 둘 다 K ≥ 2 예상되어 OK.
- **Phase 4**: spine kernel 구현 후 측정. 단독 5 % 이상 절감 안 보이면 *megakernel 경고* (occupancy 손실) 의심, 재검토.
- **Phase 2**: amalgamation 후 fill 증가율 측정. *2× 이상 증가* 면 work flops 가 절감 압도 가능, 다시 평가.
- **Phase 3**: K-stream capture 가 *진짜 concurrent* 측정 (nsys timeline). 직렬화 되면 SM 자원 경쟁 의심.

### 4.9 별도 trajectory: ζ (iterative)

위 통합이 *direct solver 의 최대 한계까지 끌어올리는 것*. 그래도 부족하면 (예: 더 큰 problem 에서 wall 이 여전히 binding), iterative + preconditioner 로 전환.

이건 별도 prototype: `src/iterative/` 모듈, CG/GMRES + AMG/block-Jacobi. direct solver fallback 유지. 본 연구 범위 외.

## 6. case-별 ROI 예측 (B/C/D/E 통합 적용)

### 6.1 case8387 (n=14908)

- max fsz 76, sibling group 영역 small (L13 cnt=9 maxfsz=56, L14 cnt=5 maxfsz=46, ..., L16 cnt=3)
- K subtree roots = L16 의 3 panels → K=3 fork-join
- spine = L17–L29 (cnt=1 chain, 13 levels)
- sibling amalgamation [B] 잠재: subtree top 의 5–9 fronts 의 padded union → fsz ≈ 50–150
- TC sweet spot (fsz ≥ 256) 까지는 못 도달, but dense block work 자체로 충분
- **통합 ROI (§4.6): factor wall −35 % 예상** (B+C+D+E 모두)

### 6.2 case_SyntheticUSA (n=156k)

- max fsz 245, L19 cnt=7 maxfsz=235, L22 cnt=3 maxfsz=245, ...
- K subtree roots = spine 직전 (L24 추정 cnt=2)
- spine = L25 부터 ~L39 (~15 levels)
- sibling amalgamation [B] 잠재: subtree top 의 3–10 fronts → fsz ≈ 500–2000 영역
- **TC sweet spot 도달 가능** — sibling amalgamation 결과의 dense block 이 TC 다시 의미 있게 만듦
- **통합 ROI (§4.6): factor wall −40 % 예상**, 추가로 TC 다시 켜면 추가 −15 %

### 6.3 더 큰 dense problems

3D PDE / Helmholtz 같은 *큰 dense block 보유* matrix:
- 본 솔버의 narrow-top 문제 자체가 덜 심각
- BLR / Compression 이 더 의미 있음 (B/C/D/E 와 별도)
- 본 연구 범위 밖

## 8. 진행 로그 (autonomous execution)

이 섹션은 phase 별 진행 상황을 시간 순으로 기록.

### Phase α — 측정 인프라 (완료)

**산출물**: `tests/analyze_tree.py` — CLS_DUMP 출력을 파싱해 spine / K subtree / sibling amalgamation 후보 식별 + ROI 추정.

**case8387 (n=14908) 분석 결과**:
- **Spine**: L20–L29 (10 levels, cnt=1). spine f² = 2.6 % of total
- **K = 2** subtree roots at L19 (cnt=2)
- Subtree 가 f² 의 97.4 % 보유 (거의 모든 work)
- **Sibling amalgamation 후보** (L11–L19, cnt 2–14):
  - L11: 14 fronts 묶으면 686 col dense block
  - L13: 10 fronts → 580 col
  - L19: 2 fronts → 112 col
  - 후보 f² 총합 = 14.4 % of total

**case_SyntheticUSA (n=156k) 분석 결과**:
- **Spine**: L30–L39 (10 levels, cnt=1). f² = 1.1 % of total
- **K = 2** subtree roots at L29 (cnt=2)
- Subtree 가 f² 의 98.9 %
- **Sibling amalgamation 후보** (L16–L29, cnt 2–15):
  - **L16: 15 fronts → 2535 col dense block** (TC sweet spot 의 10×)
  - L17: 11 fronts → 2310 col
  - L19: 9 fronts → 1719 col
  - 후보 f² 총합 = 18.3 % of total

**핵심 발견**:
1. 두 case 모두 K=2 + spine 10 levels 의 *동일한 패턴*. power-grid Jacobian 의 ND 가 root 근처에서 거의 항상 2-way split 만듦
2. **Phase 3 (K=2 streams) 의 잠재 ideal = 49 % wall 절감** (below-spine work 가 거의 100 %, 2× ideal speedup)
3. **Phase 2 의 dense block 크기가 USA 에서 2535 col 까지** — TC 의 *진짜 sweet spot* 진입 가능. case8387 은 686 col 까지로 마지널.
4. Phase 4 spine 자체는 f² 의 2.6 % / 1.1 % 만 차지 → 단독 ROI 작지만 *dispatch latency 절감* 으로 fixed overhead 회수

→ Phase 진행 우선순위 재확인: **Phase 2 가 가장 큰 lever** (특히 USA), Phase 3 가 그 위에서 K=2 평행화 추가, Phase 4 는 finishing touch.

### Phase 1 — Spine + subtree 코드 노출 (완료)

**산출물**:
- `MultifrontalPlan` 에 `h_spine_panels`, `spine_start_level`, `num_subtrees`, `h_subtree_roots`, `h_subtree_of_panel` 필드 추가
- `analyze_multifrontal` 에서 panel etree 분석 → spine 식별 (cnt=1 chain from top) → K subtree partition (BFS from roots)
- `CLS_TREE_INFO=1` env 로 출력

**실측 결과**:

case8387 (METIS run-to-run 변동 있음):
```
P=7431 levels=32  spine=[L18..L31] (14 panels)  K=2
  subtree 0: root panel 2937, 2935 panels
  subtree 1: root panel 4286, 301 panels  ← imbalance 9:1!
```

case_SyntheticUSA:
```
P=74280 levels=39  spine=[L30..L38] (9 panels)  K=2
  subtree 0: root panel 39294, 30243 panels  ← 48 %
  subtree 1: root panel 74273, 33236 panels  ← 52 %  (well-balanced)
```

**중요 발견**: case8387 의 K=2 subtree 가 **9:1 unbalanced**. Phase 3 (K-stream) 의 ideal speedup 이 (대략) `(2935+301) / max(2935, 301) ≈ 1.1×` 로 *거의 의미 없음*. case_SyntheticUSA 는 ~2× ideal (균형).

→ Phase 3 의 ROI 가 case 별로 크게 다름. case8387 에서는 Phase 3 보다 Phase 2 (sibling amalgamation) + Phase 4 (spine fusion) 가 더 의미. USA 에서는 Phase 3 도 큰 win.

### Phase 4 — Spine persistent kernel (완료, opt-in)

**산출물**:
- `src/tc/spine_kernel.cuh` — `mf_spine_factor_b` kernel (block-per-batch, device-side spine loop)
- `multifrontal_tc.cu` dispatch 수정: spine levels skip + 끝에서 spine kernel launch
- `CLS_USE_SPINE=1` 로 opt-in

**측정 결과 (5 trials mean)**:

| case | B | FP32 baseline | TC | TC + spine | Δspine |
|---|---:|---:|---:|---:|---:|
| case8387 | 64 | 31 μs/sys | 41 | 49 | **+19 %** ❌ |
| case8387 | 256 | 29 | 33 | 33 | 0 |
| case8387 | 512 | 28 | 34 | 33 | −3 % |
| USA | 64 | 492 | 565 | 564 | −0.2 % |
| USA | 256 | 478 | 551 | 540 | −2 % |
| USA | 512 | 454 | 536 | 540 | +0.7 % |

**판정**: spine fusion 이 **net loss 또는 무영향**. Laine et al. 2013 의 "Megakernels Considered Harmful" 경고가 정확히 적용됨 — persistent block 안의 sequential loop 가 register pressure / barrier overhead 를 추가, 작은 spine fronts (case8387 max fsz=60) 에서 dispatch latency 절감을 못 회수.

**결정**: spine fusion 을 *default OFF* 로 두고 `CLS_USE_SPINE=1` 로만 opt-in. case 가 더 클 때 (spine fsz ≥ 200+) 재평가.

### Phase 2 — Sibling amalgamation (재검토, 보류)

조사 후 *기각*. 이유:

1. **데이터 구조 한계**: `PanelPartition` 이 *contiguous 컬럼 panel* 만 표현 가능. Sibling panels 의 컬럼은 postorder 에서 *non-contiguous* (sibling 의 subtree 가 그 사이에 들어감). non-contiguous panel 지원하려면 fundamental refactor.
2. **Padded fill 의 ROI 손해**: case8387 USA L19 cnt=9 maxfsz=191 → padded union ~1720 col 의 dense block. 실제 nonzero 는 ~10 % → *10× 의 work 증가*, 그 중 90 % 가 padding (wasted). WMMA 의 4× 가속이 회수 못 함.
3. **Literature 의 "amalgamation" 은 chain merge 뿐**: Hypermatrix Oriented Supernode Amalgamation (López et al.) 도 fundamental supernode 의 chain 확장만. *sibling merge* 는 standard multifrontal 의 일부가 아님.

Sibling subtree 의 *진짜* 활용은 **Phase 3 (multi-stream concurrent dispatch)**. fusion 하지 말고 *동시 실행*.

### Phase 3 — Multi-stream fork-join (구현 완료, race condition 발견)

**산출물**:
- `TCState` 에 K subtree streams + fork-join events 추가
- `analyze` 단계에서 plcols 를 subtree id 별로 정렬 (subtree 0 → subtree 1 → spine) + per-(subtree, level) range 계산
- `issue_factor_levels` 에 multi-stream 분기: fork_event → K streams concurrent → join_events → main spine
- `CLS_USE_MULTISTREAM=1` 로 opt-in (default OFF, race 문제 발견)

**Race condition 발견**:

`CLS_USE_MULTISTREAM=1` 로 활성화 시 wall 은 빨라지지만 결과가 garbage:

| B | factor μs/sys | batch_relres |
|---:|---:|---:|
| 4 | 112 | 0 (suspect) |
| 16 | 37 | **1.5 × 10³¹** ❌ |
| 64 | 22 | **3.0 × 10²⁸** ❌ |
| 256 | 18 | **6.2 × 10³⁵** ❌ |

(single-stream 동일 setup 의 relres ≈ 0.008 — FP32 정상)

**원인 추정**:
- CUDA Graph stream capture 의 multi-stream fork-join pattern 이 capture 됐지만 replay 시 dependency 가 제대로 enforce 안 됨
- 또는 scatter (graph 밖에서 실행) 와 subtree streams 의 동기 누락
- 디버깅에 시간 더 필요 (gdb / nsys timeline 확인 등). 시간 제약상 후속 작업으로 미룸.

**Wall 측정 (relres 무시 시)**:
- case8387 B=64: tc-default 44 μs → multi-stream 22 μs (−50 %). *만약* 정확하면 매우 큰 win.
- 정확도 회복 후 같은 wall 절감을 유지할 수 있는지가 핵심

→ Phase 3 디버깅이 가장 큰 미해결 작업. fork-join graph capture 의 알려진 corner case 또는 우리 코드의 race 둘 중 하나.

## 9. 최종 결과 (5 trials mean, B=64/256/512)

| case | B | FP32 baseline | TC default | TC + spine |
|---|---:|---:|---:|---:|
| case8387 | 64 | 30 | 44 (+47%) | 47 (+57%) |
| case8387 | 256 | 26 | 34 (+31%) | 34 (+31%) |
| case8387 | 512 | 27 | 34 (+26%) | 51 (+89%) ❌ |
| USA | 64 | 494 | 560 (+13%) | 562 (+14%) |
| USA | 256 | 476 | 546 (+15%) | 540 (+13%) |
| USA | 512 | 460 | 531 (+15%) | 543 (+18%) |

**결과 요약**:
- TC dedicated path 자체는 *case8387 / USA 의 power-grid Jacobian* 에서 net win 없음 (FP32 batched 가 일관적으로 빠름). 이는 `06-tc-dedicated-path-study.md` 의 결론과 일치 — power-grid 의 root separator ceiling 이 TC sweet spot 못 도달.
- **Phase 4 (spine fusion) 도 무영향 또는 손해** — Laine 2013 의 megakernel 경고 적용.
- **Phase 3 (multi-stream) 만 큰 wall 절감 잠재 (−50 %), 단 correctness 깨짐** → 디버깅이 핵심 후속 작업.

## 10. 산출물 종합 요약 (잠자기 전 보고)

### 코드 변경
- `src/plan/multifrontal_plan.{hpp,cu}` — `h_spine_panels`, `spine_start_level`, `num_subtrees`, `h_subtree_roots`, `h_subtree_of_panel`, `h_subtree_level_off/cnt`, `d_spine_panels` 추가
- `src/factorize/multifrontal.cu` — analyze 단계에서 spine + K subtree 식별, plcols 를 subtree id 별로 재정렬, `CLS_TREE_INFO=1` env 로 디버그 출력
- `src/tc/spine_kernel.cuh` — 새 파일, `mf_spine_factor_b` persistent kernel (Phase 4)
- `src/tc/multifrontal_tc.{hpp,cu}` — `TCState` 에 K streams + fork-join events. dispatch 에 multi-stream / spine fusion 분기 추가
- `tests/analyze_tree.py` — Tree analysis tool (Phase α)

### Env flags (전부 default OFF)
- `CLS_TREE_INFO=1` — 트리 분석 정보 출력
- `CLS_USE_SPINE=1` — Phase 4 spine fusion 활성화 (default OFF, +19 % regression 측정됨)
- `CLS_USE_MULTISTREAM=1` — Phase 3 multi-stream 활성화 (default OFF, **race condition — relres garbage**)

### 측정 결과 요약 (case8387 B=64, 5 trials mean)

| 모드 | factor μs/sys | 평가 |
|---|---:|---|
| FP32 baseline | 30 | (기준) |
| TC default | 36 | +20 % (TC sweet spot 부적합) |
| TC + Phase 4 spine | 38 | +27 % (megakernel 경고 적용) |
| TC + Phase 3 multi-stream | 20 | wall is **−33 %**, but **CORRECTNESS BROKEN** ⚠️ |

### 핵심 발견

1. **Phase α + Phase 1 (인프라)**: 모든 후속 phase 의 발판으로 성공적으로 자리잡음. `CLS_TREE_INFO=1` 로 K subtree partition 검증 가능.

2. **Phase 2 (sibling amalgamation) — 기각**: PanelPartition 의 contiguous column 가정 + padding fill 의 ROI 음수. literature 의 standard 기법 아님.

3. **Phase 4 (spine fusion) — 미세 효과**: 측정상 +19 % 손해 (case8387 B=64). Laine 2013 megakernel 경고가 적용됨. spine 의 work 가 dispatch overhead 절감보다 작음. **default OFF**.

4. **Phase 3 (multi-stream) — wall 큰 win + correctness 문제**:
   - case8387 B=64 에서 wall 36 → 20 μs/sys = **−44 % factor wall** (만약 정확하면 *최대의 lever*)
   - 모든 B 에서 relres = 1e+28 ~ 1e+35 (NaN propagation)
   - 가설: CUDA Graph 의 multi-stream capture 에서 scatter (graph 밖, main stream) 와 subtree streams 사이의 memory ordering 이 enforce 안 됨. fork_event 가 sufficient 한 sync 가 아닐 가능성
   - **잠재 lever 가 크니 디버깅 가치 높음**

## 11. 후속 작업 (잠에서 일어나면)

### Critical (Phase 3 디버깅)
1. nsys timeline 으로 multi-stream 의 실제 execution 확인 — concurrent 인지, race 패턴 파악
2. CUDA Graph 의 multi-stream capture pattern 검증 — fork_event / join_events 의 graph node 가 제대로 dependency 표현하는지
3. 가능 fix:
   - subtree streams 의 Memset 전 sync 추가
   - scatter 를 graph 안에 포함 시키기 (모든 prep work 를 graph 안에)
   - capture mode 를 `cudaStreamCaptureModeThreadLocal` 로 변경 시도

### 후속 lever (Phase 3 작동 시)
- Phase 4 spine fusion 을 multi-stream 의 join 후 활성화 (이건 빠르게 측정 가능)
- 더 큰 problems (onetone2 같이 root separator ≥ 500 인 경우) 에서 TC + Phase 3 결합 측정 — 이 영역에서 진짜 TC sweet spot 진입 가능

### 별도 trajectory
- `06-tc-dedicated-path-study.md` 의 결론: case8387/USA 같은 power-grid 는 TC 효과 본질적으로 작음
- 본 문서의 결론: power-grid 에서 *진짜* 의미 있는 lever 는 **Phase 3 (multi-stream)** — TC 와 무관, race 문제만 풀면 단일 GPU + small batch 환경에서 큰 win




---

## 7. References

- Davis, T.A. — "The Multifrontal Method for Sparse Matrix Solution" SIAM Review 1996. [link](https://epubs.siam.org/doi/10.1137/1034004)
- Liu, J.W.H. — "The Role of Elimination Trees" SIAM J Matrix Anal Appl 1990. [link](https://epubs.siam.org/doi/10.1137/0611010)
- Anzt, H. et al. — "High performance sparse multifrontal solvers on modern GPUs" Parallel Computing 2022. [link](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000059)
- Boukaram, W. et al. — "Batched Sparse Direct Solver Design and Evaluation in SuperLU_DIST" 2024. [link](https://escholarship.org/uc/item/20h717s9)
- Sao, P. et al. — "A communication-avoiding 3D algorithm for sparse LU factorization on heterogeneous systems" 2019. [link](https://www.osti.gov/biblio/1559632)
- López, J.M. et al. — "Hypermatrix Oriented Supernode Amalgamation" J Supercomputing 2008. [link](https://link.springer.com/article/10.1007/s11227-008-0188-y)
- Rennich, S.C. et al. — "Accelerating Sparse Cholesky Factorization on GPUs" 2014. [link](https://people.engr.tamu.edu/davis/publications_files/IA3_2014_Workshop_Rennich_Stosic_Davis_preprint.pdf)
- Claus, L., Ghysels, P., Boukaram, W., Li, X.S. — "A graphics processing unit accelerated sparse direct solver and preconditioner with block low rank compression" 2025. [link](https://journals.sagepub.com/doi/full/10.1177/10943420241288567)
- Laine, S. et al. — "Megakernels Considered Harmful: Wavefront Path Tracing on GPUs" 2013. [link](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- Hamann, M. et al. — "Faster and Better Nested Dissection Orders" 2019. [link](https://arxiv.org/pdf/1906.11811)
- gSoFa team — "Scalable Sparse Symbolic LU Factorization on GPUs" 2020. [link](https://arxiv.org/pdf/2007.00840)
- Mapping Sparse Triangular Solves to GPUs via Fine-grained Domain Decomposition (2026). [link](https://arxiv.org/abs/2508.04917)
