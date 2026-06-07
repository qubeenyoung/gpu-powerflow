# [DEPRECATED] Architecture R&D: Persistent subtree-walking kernel (M2) — design + 측정

> **2026-06-07 폐기.** 본 문서의 design 단계 (Step 1) 후속인 prototype 구현 (docs/20) 에서 **wall +240~820% 회귀** 발생. parallel-within-subtree redesign 없이는 회생 불가하나, 비용/위험 대비 ROI 불명확 → 무기한 보류. 관련 코드 (host `compute_micro_subtrees`, kernel `factor_small_subtree`, plan 의 micro-subtree 필드, dispatch hook) 전체 삭제. 본 문서는 historical design log 로만 보존.

**작성일**: 2026-06-07
**대응 계획**: [docs/18 §3.3](18-small-mid-memory-bound-2026-06-07.md) M2 lever
**환경**: RTX 3090 (sm_86), CUDA 12.8
**상태**: ~~design 단계 + analyze prototype 시작. kernel 미구현.~~ **DEPRECATED 2026-06-07** — docs/20 prototype 실패 후 코드 삭제.

## 0. TL;DR

- **R&D 동기**: docs/18 의 small/mid memory-bound (DRAM 580 GB/s, peak 의 69%) 의 simple micro-opt 가 ceiling 도달. architecture-level lever 만 남음. literature (Rennich-Davis 2014) 의 persistent subtree-walking 이 후보 1순위.
- **R&D Step 1 — etree shape 측정**: case8387/USA/ACTIVSg25k 의 panel etree 를 dump 후 offline 분석. 의외의 발견 — **30 KB budget 으로 panel 의 94-96% 가 mean-21-42-크기 subtree 로 묶임, stage-in 의 90% 잠재 제거**.
- **잠재 wall 게인 추정**:
  - case8387 B=64 (small dominant): **-30% 잠재**
  - ACTIVSg25k B=64 (mid+small mix): **-15% 잠재**
  - USA B=64 (big dominant): **-12% 잠재**
- 본 게인은 docs/15-17 의 micro-opt 모든 시리즈 합 (-5~-10% on USA) 보다 큼. **architecture R&D 가 실효성 있는 첫 결과**.
- 단, sync→wall 변환률 ~1/10 (docs/14, 17) 패턴 적용 시 보수적 추정 **실제 게인 절반 또는 그 이하**. 예: case8387 30% 잠재 → 실제 10-15%.
- 본 라운드는 **(1) etree 측정, (2) design doc, (3) analyze-side subtree 분해 prototype** 까지. kernel 구현 별 round.

## 1. 동기 재정리

### 1.1 docs/15-18 까지의 ceiling 영역

- docs/15 V9h: GEMM PTX 시리즈 — USA B=1 -10.4%, B=64 -2.7%
- docs/17 LB(512,2): big_tf32 occupancy — USA B=64 -2.7% (V9h 와 합쳐 -5.7%)
- docs/18 EXP-D: small kernel warps/block 8→16 — **회귀 +10%** (가설 reject)

→ simple micro-opt (kernel-level) 의 wall 게인 합산 ~5-10% 가 본 codebase 의 ceiling. architecture-level 으로 가야 더 큰 lever.

### 1.2 docs/18 의 lever 카탈로그 재확인

| Lever | 예상 wall | 적용 가능성 |
|-------|-----------|------------|
| M1 (front packing) | 5-10% | mid 의 small front 묶기. 복잡도 중-상 |
| **M2 (persistent subtree-walking)** | **10-30%** | 본 문서의 R&D 대상 |
| M5 (정밀도 강하 + correction) | 5-10% | accuracy R&D 필요. orthogonal |

M2 가 잠재 가장 큼. 단 prior 기대값은 "Rennich-Davis 2014 의 sparse Cholesky 의 결과" 였고 우리 codebase 의 etree 가 그와 같은 구조인지 미확인.

### 1.3 R&D 진행 방향

1. etree 측정 → M2 의 applicability 검증
2. design 문서 작성 → kernel architecture 설계
3. analyze prototype → host-side subtree decomposition
4. (별도 라운드) kernel 구현 + 측정

## 2. R&D Step 1 — etree shape 측정

### 2.1 CLS_DUMP_FRONTS 확장

`src/plan/build.cpp` 의 `maybe_dump_fronts` 에 `parent` (panel_parent 의 device→host 복사) 와 `subtree` (h_subtree_of_panel) 컬럼 추가. 기존 `q,p,fsz,nc,uc,level` 에 두 컬럼 추가.

```cpp
std::fprintf(f, "q,p,fsz,nc,uc,level,parent,subtree\n");
// ...
const int parent = (p < (int)h_parent.size()) ? h_parent[p] : -2;
const int sub = (p < (int)plan.h_subtree_of_panel.size())
                    ? plan.h_subtree_of_panel[p] : -2;
std::fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%d\n", q, p, fsz, nc, fsz - nc, L, parent, sub);
```

### 2.2 etree 기본 shape

| case | panels | depth | leaves (no children) | non-leaves | avg children of non-leaf |
|------|--------|-------|---------------------|-----------|--------------------------|
| case8387 | 7409 | 30 | 4100 (55%) | 3309 (45%) | 2.24 |
| USA | 74271 | 40 | 41986 (57%) | 32285 (43%) | 2.30 |
| ACTIVSg25k | 22739 | 39 | 12753 (56%) | 9986 (44%) | 2.28 |

children-count 분포 (non-leaf):

| count | case8387 | USA | ACTIVSg25k |
|-------|----------|-----|------------|
| **1 (chain)** | **1237 (37%)** | **9952 (31%)** | **3395 (34%)** |
| 2 | 1180 | 12394 | 3520 |
| 3 | 432 | 6416 | 1832 |
| 4 | 218 | 1669 | 581 |
| 5+ | 242 | 1854 | 658 |

→ **30-37% 의 non-leaf 가 chain (1 child)** — chain-walking pattern 의 기본 단위.

### 2.3 micro-subtree 후보 측정 (3가지 알고리즘)

#### Algorithm A: chain-only (parent has exactly 1 child)

| case | clusters | covered % | mean size | max size |
|------|----------|-----------|-----------|----------|
| case8387 | 1099 | 31% | **2.07** | 5 |
| USA | 9269 | 26% | 2.07 | 5 |
| ACTIVSg25k | 3050 | 28% | 2.07 | 6 |

→ chain 만 으로는 mean size 2. chain 이 짧다.

#### Algorithm B: subtree-budget-fit (모든 descendants 포함, fsz ≤ 32, smem budget)

```
Budget 10 KB:  91-93% covered, mean 14-16, **85% stage-in saving**
Budget 30 KB:  94-96% covered, mean 21-42, **90% stage-in saving**
Budget 60 KB:  94-97% covered, mean 21-53, **90-95% stage-in saving**
```

→ **이 결과가 critical**. 30 KB budget 으로 panel 의 95% 가 mean 20+ 크기 subtree 로 묶임. 즉 **stage-in 의 90% 가 잠재 제거 가능**.

#### Algorithm B 의 size distribution (Budget 30 KB)

| case | subtrees | size 2-5 | size 6-15 | size 16-30 | size 30+ |
|------|----------|---------|----------|------------|----------|
| case8387 | 170 | 47 | 24 | 32 | 67 |
| USA | 3390 | 1013 | 685 | 921 | 771 |
| ACTIVSg25k | 999 | 415 | 313 | 175 | 96 |

→ subtree 의 절반 이상이 size ≥ 10. 큰 subtree 의 dominant 효과 큼.

### 2.4 algorithm A vs B 차이 해석

Algorithm A 의 strict 조건 ("parent has exactly 1 child") 가 chain-only — sibling 이 있으면 chain 끊김.

Algorithm B 는 ALL siblings together 가능. 즉 한 block 이 N 개 sibling 의 모두를 가족 단위 로 처리 → siblings 의 CB 들이 모두 shared 에 있으면 parent 의 extend-add 가 in-shared.

→ **B 가 architectural 으로 정합** (parent 의 extend-add 받으려면 ALL children 처리 끝나야).

### 2.5 잠재 wall 게인 추정

#### Stage-in/Writeback DRAM traffic 절약

mid tier DRAM 통계 (docs/18 §1.1): read 313 + write 267 = 580 GB/s, peak 의 69%. 이게 stage-in/writeback bulk traffic 의 burst.

M2 적용 시:
- subtree 의 root 만 stage-in (1 / 21 만 traffic). DRAM read **95% 감소** 가능.
- subtree 의 root 만 writeback (1 / 21 만 traffic). DRAM write **95% 감소** 가능.
- 결과: DRAM 사용 580 → 30 GB/s = peak 의 3.5%. compute 자원 이용 가능.

#### Wall 게인 예측

| case | dominant tier | M2 의 영향 범위 | 잠재 wall |
|------|---------------|----------------|----------|
| case8387 (B=64) | small (~70% of wall) | small 의 stage-in 95% 감소 → small wall -50% | **-30%** |
| ACTIVSg25k (B=64) | mid+small (50% mid + 25% small) | small + mid 의 일부 -30% | **-15%** |
| USA (B=64) | big (~40%) + mid (~25%) + small (~25%) | small/mid 절약 만 | **-12%** |

#### 보수적 추정 (sync→wall 변환률 적용)

docs/14/17 의 패턴: 큰 metric 변화 (sync 64% 감소, barrier 26% 감소) 가 wall 의 1/10 정도만 변환.

M2 의 stage-in 90% 절감 → 보수적 변환률 적용 시 **잠재 의 1/3 ~ 1/2 실현**:
- case8387: -30% → **실측 -10 ~ -15%**
- ACTIVSg25k: -15% → **실측 -5 ~ -8%**
- USA: -12% → **실측 -4 ~ -6%**

여전히 docs/17 의 LB(512,2) (-3.8% USA B=64) 보다 큰 lever.

### 2.6 측정의 한계

- offline analysis 만 — 실제 kernel 동작은 다를 수 있음
- subtree 의 max smem 추정이 conservative (sum of all panels) — 실제 working set 은 작을 것
- DRAM traffic 의 모든 90% 가 wall 로 변환 안 됨 (compute 가 새 binding 으로 등장 가능)
- **prototype 미구현, 본 추정은 실측 아님**

## 3. M2 Architecture 설계

### 3.1 핵심 아이디어

기존: 1 block = 1 (front, batch). per-block stage-in/writeback → global. parent block 가 다음 (front, batch) 의 stage-in 시 child 의 CB 를 global 에서 읽음.

새: 1 block = 1 (subtree, batch). subtree 의 모든 front 가 block 의 lifecycle 안에서 처리. child front 의 CB → shared (no global). parent 가 in-shared 로 extend-add 받음.

### 3.2 block 의 lifecycle

```
factor_small_subtree<<<num_subtrees, B, blk, smem>>>:
    Block(b, batch):
        # b = subtree index, batch = batch index
        subtree_members[] = decoded from device array
        for p in members (topological order leaf-first):
            if p is leaf:
                stage_in_subtree(p) — global F → shared (small)
                # OR: leaf data 도 host-side 에서 preload 가능 (analyze 단계 trick)
            else:
                # p's children 의 CB 들 이 이미 shared 에 (children 처리 끝 직후)
                # extend-add 를 in-shared 수행
                for child in children(p):
                    extend_add_in_shared(child.CB → p.front, child.uc, p.fsz, child.asm_local_in_p)
            factorize(p) — panel LU + U-solve + trailing GEMM
            store_CB_in_shared(p) — CB 영역만 shared 에 남김 (parent 사용 위해)
            if p is subtree_root:
                writeback_CB_to_global(p)  # parent 가 다른 subtree 일 경우
            else:
                # p 의 L/U writeback 도 가능 (solve 시 read 위해)
                writeback_LU_to_global(p)
```

### 3.3 smem layout

block 안에서 동시 보유:
- 현재 처리 중인 panel 의 working space (fsz² × 4 bytes)
- 이미 처리 끝났지만 parent 가 아직 안 가져간 children 의 CB (uc² × 4 bytes each)

max concurrent smem = depth × max(working) + sum(children CBs 가 시한적으로 보유). 균형 등 등급 etree 에서는 working space ≈ smem dominant.

예: case8387 mean subtree 21 panel, max fsz=32 → working 4 KB. children CB 평균 작음. 보유 시한 짧음. **total smem ~ 5-10 KB / block** (분석 §2.3 의 mean 2-3 KB 와 일치).

### 3.4 block dimension 과 work mapping

옵션 A: **1 thread group = 1 panel processing**. 모든 256 thread 가 한 panel 의 작업에 동참 (= 기존 factor_mid 패턴). panel 간 순서는 host 가 알아서. → barrier overhead 그대로.

옵션 B: **multiple panels per thread group, in parallel where possible**. parallel-process leaves 동시 — 다른 thread group 이 다른 leaf. parent 시점에 sync. → 더 복잡. barrier 만들기 까다로움.

→ 본 round 에서 옵션 A 가 simpler. 옵션 B 는 follow-up.

### 3.5 device-side data structure

새로 필요:
- `int d_subtree_offsets[num_subtrees + 1]`: 각 subtree 의 member 시작 offset
- `int d_subtree_members[total_members]`: subtree-별 panel id 들 (topo order)
- `int d_subtree_root[num_subtrees]`: 각 subtree 의 root panel id
- 기존 d_panel_parent / d_ncols / d_front_off 등 그대로 활용

### 3.6 dispatch 변경

```cpp
// in src/factorize/dispatch.cuh 또는 별도 함수
if (level_max_fsz <= SMALL_THRESH && CLS_M2_PERSISTENT) {
    // 1. small tier subtree 들 한꺼번에 launch
    factor_small_subtree<<<num_microst, B, smem>>>(...);
    // 2. covered 안 된 small panel 들 은 기존 factor_small
    factor_small<<<...>>>(...);   // 미커버 fallback
}
```

### 3.7 analyze 변경

`src/plan/analyze.cu` 의 subtree 분해 logic 옆에 새 함수 `compute_micro_subtrees`:
- input: P, fsz[], children[], smem budget
- output: subtrees[] (root, members[] in topo order)
- algorithm: DFS bottom-up, greedy budget-fit (§2.3 Algorithm B)

본 라운드의 prototype 은 이 단계까지 수행.

### 3.8 CUDA Graph capture 호환성

새 kernel 은 일반 `__global__` → graph capture 자연스러움. 단 dispatch 가 conditional branch (`if covered ... else ...`) 인 점이 capture-replay 의 deterministic 요구와 호환되는지 검증 필요. analyze 가 결정한 분기는 capture 시점에 고정 → OK.

## 4. R&D Step 3 — analyze prototype

본 라운드는 host-side micro-subtree 분해 함수만 작성. device-side 분배 + kernel 은 follow-up.

### 4.1 새 host 함수 (signature)

```cpp
// src/plan/analyze.cu 또는 src/plan/build.cpp 의 새 함수.
// 입력: 기존 MultifrontalPlan 의 etree 정보 (h_front_ptr, h_ncols, panel_parent_host)
// 출력: 두 vector (host-side, host-only — device 복사 follow-up):
//   h_microst_offsets[K+1], h_microst_members[total]
// where K = number of micro-subtrees.
void compute_micro_subtrees(const MultifrontalPlan& plan,
                             const std::vector<int>& parent,  // host copy
                             int small_fsz, int smem_budget_kb,
                             std::vector<int>& h_microst_offsets,
                             std::vector<int>& h_microst_members);
```

### 4.2 implementation 노트

algorithm (§2.3 B):
1. children[] build from parent[]
2. bottom-up DFS: 각 panel 의 sub_smem (subtree total) 와 can_be_root (fsz ≤ small + sub_smem ≤ budget) 계산
3. top-down: 가장 큰 can_be_root panel 부터 cluster — 자기 subtree 의 모든 descendants 묶기
4. visited set 으로 중복 방지
5. members 는 leaf-first topological order

### 4.3 본 prototype 의 한계

- host-only — device 배열 작성 + 복사 미구현
- kernel 미작성 — dispatch 가 새 path 안 부르도록 default OFF
- 측정 불가 — wall 변화 없음 (kernel 이 호출 안 됨)

→ R&D Step 2 의 결과 (architecture 검증) 만 본 라운드의 deliverable. Step 3-5 는 별도 round.

### 4.4 prototype 의 검증 방법

CLS_DUMP_MICROSUBTREES env var 로 host 함수 가 만든 subtree 목록을 CSV 로 dump. 오프라인 비교 (Python script 의 결과와 일치 확인) → 알고리즘 correctness 검증.

## 5. risk / 알려진 한계

### 5.1 wall 게인 의 conversion 률

§2.5 의 추정은 잠재 wall (-12 ~ -30%) — 실제 변환률 1/2 ~ 1/3 적용 시 5-10%. docs/15-17 시리즈의 ceiling 보다 큰 lever 이긴 함.

### 5.2 등록되지 않은 cost

- subtree 분해 의 analyze cost 추가 (한 번만 발생, amortize 가능)
- 새 kernel 의 register pressure 가 small kernel 보다 크면 occupancy 손실 가능
- dispatch 의 conditional branch 가 CUDA Graph capture 의 효율을 약간 떨어뜨릴 가능성

### 5.3 implementation 복잡도

design 의 §3 만 으로도 코드 ~400 줄 예상:
- analyze: ~150 줄
- kernel: ~200 줄
- dispatch: ~50 줄

본 codebase 의 multi-tier dispatch 와 호환 유지하면서 새 path 추가는 carefully. 의존: extend-add 의 in-shared variant 새 buildlng block 필요.

### 5.4 prior 실패의 경고

- docs/9-10 의 spine 메가커널 시도 +19% 회귀 — 큰 persistent kernel 의 register pressure / barrier overhead 위험
- docs/9 의 multi-stream subtree race (relres NaN) — fine-grained sync 의 위험성

→ M2 의 kernel 은 race-free + barrier 최소화 design 필요. 단계적 prototype 권장.

## 6. 본 라운드의 deliverable

1. **CLS_DUMP_FRONTS 확장** (parent, subtree 컬럼 추가) — ship.
2. **etree shape 측정 + micro-subtree feasibility** — 본 문서의 §2 가 결과.
3. **design 문서** (§3) — implementation 의 roadmap.
4. **prototype design** — analyze 의 host 함수 signature + algorithm (§4). 실제 코드 별 round.

본 라운드 **kernel 미구현, wall 측정 불가**. 다음 round 의 work scope:
1. analyze 의 host-side compute_micro_subtrees 구현 + 검증
2. device array 작성 + 복사
3. factor_small_subtree kernel 의 minimum-viable prototype (1 subtree per launch first, scale up)
4. small dispatch 의 conditional 경로 추가
5. wall 측정 + 비교

## 7. 결론 + 권고

### 7.1 architecture R&D Step 1 성과

- etree 측정으로 **M2 의 power-grid 적용성 검증 완료**: 30 KB budget 으로 panel 의 95% 가 mean 20+ subtree 로 묶임.
- **잠재 wall**: case8387 -10~-15%, ACTIVSg25k -5~-8%, USA -4~-6% — docs/15-17 ceiling 보다 큰 lever.
- **prior 실패 의 경고** (spine 메가커널 +19%) 에 비춰 단계적 prototype 권장.

### 7.2 권고

- **M2 R&D 계속 진행 권장** — Step 2 (analyze 구현) → Step 3 (kernel prototype) → Step 4 (측정) 의 multi-round 진행.
- **OR**: 사용자 의 V0 best 입장 존중 시 **본 R&D 종료**, M2 의 잠재 wall 만 marking + architecture R&D 의 ceiling 검증 결과로 보존.
- middle-ground: Step 2 의 analyze 구현 까지만 다음 round 에서 진행. kernel implementation 은 별도 R&D project.

### 7.3 본 codebase 의 lever ceiling 재정의

| Lever 시리즈 | 이론적 ceiling | 실측 | 본 codebase 의 변환률 |
|-------------|---------------|------|---------------------|
| docs/15 (GEMM PTX) | 15% | 5-10% | ~1/2 |
| docs/17 (occupancy) | 5-15% | 2-4% | ~1/4 |
| docs/18 (memory micro-opt) | reject | reject | – |
| **docs/19 (M2)** | **10-30%** | (미측정) | ~1/3 (보수 추정) |
| **realistic M2 게인** | **3-10%** | – | – |

→ M2 가 본 codebase 의 last realistic lever. 그 이후는 algorithm/problem class 변경 영역 (e.g., 다른 ordering, BLR compression).

## 8. 참고

- **Rennich, Stosic, Davis. "Accelerating sparse Cholesky factorization on GPUs." PMAA'14 / Parallel Computing 2016**: subtree-walking persistent kernel pattern of origin.
- **Sao, Kannan, Vuduc, Li. SuperLU_DIST 의 communication-avoiding triangular solve**: 비슷한 subtree-batching 패턴
- docs/9-10: spine 메가커널의 +19% 회귀 (prior 실패)
- docs/13-14: panel LU sync, P5 warp-spec 미실행
- docs/15-17: GEMM/occupancy micro-opt 시리즈 (ceiling 도달)
- docs/18: small/mid memory-bound 진단 + M1/M2/M5 lever 카탈로그
