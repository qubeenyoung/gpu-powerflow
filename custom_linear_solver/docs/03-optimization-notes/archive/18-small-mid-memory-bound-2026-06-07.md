# Small/Mid tier 의 memory-bound 해결책 분석

**작성일**: 2026-06-07
**대응 계획**: [docs/16 §4.1, §4.2](16-large-batch-bottleneck-analysis-2026-06-06.md)
**환경**: RTX 3090 (sm_86), CUDA 12.8

## 0. TL;DR

- **세부 진단**: small 은 **DRAM 32% util** + L1 hit 60% — **DRAM 여유 있지만 per-thread memory latency 가 dominate**. mid 는 **DRAM 64% util** + L1 hit 38% — **DRAM throughput bound**. 두 tier 의 memory-bound 가 다른 종류.
- **atomic 은 dominant 아님**: ncu 의 `l1tex__t_sectors_pipe_lsu_mem_global_op_atom` = 0 sectors/ns for mid → extend_add 의 atomicAdd 가 memory-bound 의 주범 아님. **stage-in + writeback bulk traffic** 이 dominant (313 GB/s read + 267 GB/s write = 580 GB/s, sm_86 peak 840 GB/s 의 69%).
- **EXP-D (small warps/block 8→16) 실측: 회귀 +4~10% wall**. docs/16 의 S1 가설 ("more warps in flight to hide latency") REJECTED — warp 들이 **독립적**이라 block 크기 키워도 같은 SM 안의 동시 warp 수 안 늘어남 (block per SM 만 절반).
- **남는 lever**: small 은 **block per SM 늘리기 (smem 절약)**, mid 는 **stage-in/writeback 부피 감소** (persistent kernel pattern, Rennich-Davis 2014). 둘 다 본 라운드 미실행 (높은 복잡도, 다른 lever 와 직교성).
- **권고**: 본 codebase 의 small/mid memory lever 는 simple micro-opt 로 짜낼 게 거의 없음. 진짜 lever 는 architecture-level (front-packing, persistent subtree-walking).

## 1. memory-bound 의 세부 출처 진단

docs/16 의 ncu 가 boundary stall 만 보여줌 (long_scoreboard %). 본 문서는 **트래픽 출처 까지 breakdown**.

### 1.1 ncu 메트릭 (factor_small case8387 B=64 + factor_mid_tf32 ACTIVSg25k B=64)

| 메트릭 | Small | Mid |
|--------|-------|-----|
| dram__throughput.avg.pct | **32.05%** | **63.60%** |
| dram__bytes_read.sum / s | 측정 안 함 (수치 작음) | 313 GB/s |
| dram__bytes_write.sum / s | – | 267 GB/s |
| l1tex__t_sector_hit_rate | 61.67% | **38.22%** |
| lts__t_sector_hit_rate | 59.31% | 37.32% |
| l1tex__atom_sectors / s | (n/a) | **0** sector/ns |
| launch__shared_mem_per_block_allocated | 10.37 KB | 17.79 KB |

### 1.2 해석

**Small (case8387 B=64)**:
- DRAM 32% — peak 의 1/3 만 활용. **DRAM 여유 충분**.
- L1 hit 62% — moderate.
- smem 10 KB / block — **L1 cache 에 ~118 KB 남음**, 충분.
- 진단: **per-warp memory request 의 latency 가 throughput 으로 흡수 안 됨** — 각 warp 가 1 (front, batch) 처리하면서 미해결 memory request 가 stall 로 노출. 동시 warp-in-flight 가 적어 latency hiding 실패.

**Mid (ACTIVSg25k B=64)**:
- DRAM **580 GB/s** (read+write) = peak 840 GB/s 의 69% → **DRAM-bound**.
- L1 hit **38%** — capacity miss 가 아님 (smem 18 KB / block, L1 ~110 KB 사용 가능). **access pattern 의 capacity-non-cacheable 영역** (stage-in 64 KB/front 가 L1 보다 큼).
- atomicAdd = 0 sectors/ns — **extend_add atomic 이 dominant 아님**. 이전 가정 (docs/16 §3.2 의 "extend_add 의 atomicAdd contention") 부분 reject.

### 1.3 mid 의 DRAM traffic 출처 추정

mid_tf32 의 per-(front, batch) memory traffic:
- **stage-in**: F → shared, ~fsz² × 4 bytes (read). avg ACTIVSg25k mid fsz~60 → 60² × 4 = 14 KB.
- **writeback**: L+U → global, ~(fsz² - uc²) × 4 bytes (write). ~10 KB.
- **extend_add**: read CB from shared (no DRAM), atomicAdd to parent F (~uc² × 4 bytes write, atomic).

per (front, batch) 평균 ~24 KB DRAM traffic. ACTIVSg25k 377 mid fronts × 64 batch × 24 KB = ~580 MB per repeat. kernel time ~0.115 ms × 64 = 7.4 ms → 78 GB/s.

ncu 가 580 GB/s 라고 한 것은 **kernel-active duration 동안** 의 instantaneous throughput. 78 GB/s 는 wall-averaged. 대략 14% 시간 동안 580 GB/s 로 DRAM 활용 → 나머지 86% 시간은 compute 또는 sync wait. **DRAM 활용은 burst 형태**.

→ stage-in/writeback 이 dominant burst source 임을 추정.

### 1.4 atomic 가 dominant 가 아닌 이유

ACTIVSg25k 의 extend_add 는 parent 의 CB 영역에 atomicAdd. **batched 모드** 에서 batch b 의 child 는 batch b 의 parent 로만 가므로 cross-batch contention 없음. 형제 child 가 같은 parent 의 같은 위치에 동시 atomicAdd 가능하나 power-grid 에서 sibling 수는 평균 2-5 → contention 작음.

따라서 atomic = 0 sectors/ns 는 합리적. atomic latency 가 있으나 throughput 측면에선 무시 가능.

## 2. EXP-D — small kernel 의 warps/block 8 → 16 (docs/16 S1)

### 2.1 가설

docs/16 S1: 8 warps/block → 16 warps/block 으로 늘리면 "more warps in flight to hide latency". 예상 -5 ~ -15% small wall.

### 2.2 구현

`src/factorize/dispatch.cuh`:
```cpp
#if defined(CLS_SMALL_WARPS_16)
    constexpr int SMALL_WARPS = 16;
#else
    constexpr int SMALL_WARPS = 8;
#endif
```

block size = 256 → 512. shb per block = 8 × fsz²·4 → 16 × fsz²·4 (2배).

### 2.3 결과 — **회귀**

ncu factor_small case8387 B=64:

| build | block_size | warps_active | inst/cycle | DRAM | L1 hit | long_scoreboard |
|-------|-----------|--------------|-----------|------|--------|-----------------|
| V0 (8 warps) | 256 | **42.98%** | 0.56 | 23% | 60% | 242% |
| SW16 (16 warps) | 512 | 28.11% **↓** | **0.43 ↓** | 16% **↓** | 64% (+4%) | 230% (-5%) |

scoreboard / L1 hit 둘 다 약간 개선됐으나 **warps_active 가 43% → 28% 로 추락**. inst/cycle 도 떨어짐.

wall (median-of-6, --repeat 50):

| case | B | V0 | SW16 | delta |
|------|---|-----|------|-------|
| case8387 | 1 | 0.443 | 0.441 | -0.4% (noise) |
| case8387 | 64 | 0.027 | 0.030 | **+10.1% 회귀** |
| case8387 | 256 | 0.024 | 0.025 | +7.7% |
| ACTIVSg25k | 1 | 0.802 | 0.779 | -2.9% |
| ACTIVSg25k | 64 | 0.115 | 0.120 | +4.6% |
| ACTIVSg25k | 256 | 0.111 | 0.118 | +6.2% |
| USA | 1 | 2.258 | 2.263 | +0.2% |
| USA | 64 | 0.488 | 0.512 | +4.9% |
| USA | 256 | 0.483 | 0.499 | +3.4% |

case8387 (small dominant) 의 B=64+ 에서 **+10% 회귀** — best wall regression target 인데 가설이 틀림.

### 2.4 가설 실패 원인 분석

small kernel 의 핵심 특성: **각 warp = 1 independent (front, batch)**. warp 간 cooperation 없음.

`warps_in_flight_per_SM = blocks_per_SM × warps_per_block`.

- V0 (8 warps × 256 thread): 1536 thread/SM ÷ 256 = 6 blocks/SM. warps_in_flight = 6 × 8 = **48**.
- SW16 (16 warps × 512 thread): 1536 ÷ 512 = 3 blocks/SM. warps_in_flight = 3 × 16 = **48**. **동일**.

warps_in_flight 이 그대로니 latency hiding 도 같음. **그러나 SW16 는 block 당 smem 2× → L1 carveout 줄어듬 → cache miss 약간 ↑ → 더 느림**. 실측한 warps_active 추락은 measure 변동성 일부.

**S1 reject**: warps_per_block 늘리는 게 small kernel 에서 latency hiding 에 도움 안 됨. **warp 간 cooperation 가 없어서 block-level grouping 이 그저 smem 만 더 먹음**.

### 2.5 진짜 lever 추정

small 의 latency hiding 을 늘리려면 **warps_in_flight_per_SM 자체** 를 늘려야 함:
- thread cap 은 이미 1536 = max.
- 즉 **`thread_per_block × blocks_per_SM` 의 product 한계**.
- 늘리려면: warps_per_thread (= 1) 외 다른 lever 없음.

(또는 SM 의 architectural limit — sm_86 의 max warp scheduler 가 처리 가능한 inflight slot 자체가 cap. 1536 thread = 48 warps 가 max instruction-issue slot.)

→ **small 에서 memory-bound 의 simple 한 lever 없음**. 진짜 lever: (a) memory 양 줄이기 (front-packing, persistent kernel 로 cache reuse), (b) memory access pattern 개선 (coalescing already optimal), (c) 다른 phase 의 compute 와 memory overlap (cp.async pipelining 이미 적용 — 추가 lever 어려움).

## 3. mid 의 DRAM-bound 해결 lever 카탈로그

mid 는 DRAM 64% 활용 + L1 hit 38%. 본질적 lever:

### 3.1 lever 우선순위

| Lever | 설명 | 예상 wall | 난이도 | 본 라운드 |
|-------|------|-----------|--------|----------|
| **M1: front packing (multi-front per block)** | 4-8 small mid fronts 를 한 block 에 묶어 stage-in 의 metadata overhead amortize. 같은 block 안에서 cache reuse 가능 | 10-20% | 상 | 미실행 |
| **M2: persistent kernel + subtree walking** | 한 block 이 etree 의 subtree 를 walk, child→parent CB 가 shared 에 머묾 (no global round-trip) | 15-30% | 매우 상 | 미실행 |
| **M3: explicit `__launch_bounds__(256, 6)` for mid** | 4-6 blocks/SM 강제 → DRAM 의 burst 분산 | 0-5% | 낮음 | 미실행 (mid 의 smem 18 KB × 5 = 90 KB, 이미 거의 max) |
| **M4: writeback async via cp.async or post-sync** | factorize_front 종료 직후 다음 phase 의 stage-in 와 overlap | 3-8% | 중 | 미실행 |
| **M5: 정밀도 강하** (mid 의 추가 quantization 등) | smem/DRAM 의 traffic 절반화 | 5-10% | 중 (accuracy 검증 필요) | 미실행 |

### 3.2 M1 (front packing) 세부

문제: mid 의 stage-in 이 (front, batch) 마다 64 KB+ 의 DRAM read 를 burst — small (fsz<48) 인 mid front 들이 같은 SM block 에서 묶이지 못 함.

해결: 같은 etree level 의 작은 mid front N개를 한 block 의 shared 에 packing. block 의 256 thread 중 N개 sub-group 이 각자 front 처리.

복잡도:
- analyze 단계에서 같은 level 의 작은 front 들을 group ID 부여
- dispatch 가 grouped front list 를 kernel 에 전달
- kernel 의 thread 가 sub-group ID 로 자기 front 찾기
- stage-in 의 packing layout 조정 (front 간 padding 등)

위험: front 간 work 가 다양 (fsz 6 vs 64 같이 묶이면 idle thread 많음). work-balanced grouping 필요.

문헌: Rennich-Davis 2014, MAGMA batched LU (Haidar et al. 2018).

### 3.3 M2 (persistent kernel) 세부

문제: mid_tf32 의 stage-in 이 child 의 extend_add 결과를 global 통해 받음. **child→parent 의 round-trip 이 global DRAM** 을 소비.

해결: 한 block 이 etree 의 subtree 를 (DFS or BFS) walk. child 의 CB 를 직접 shared 에 두고 parent 가 in-shared 로 받음. global F 의 stage-in 이 root 의 first 만 발생.

복잡도:
- dispatch 가 per-subtree 단위로 launch (level-by-level dispatch 와 다름)
- subtree 의 dependency 가 single block 안에서 처리 (no inter-block sync 필요)
- subtree 의 시작 front 는 외부 input 만, 내부 child 는 shared 에서 CB 받음

위험: subtree 의 깊이 / 폭이 한 block 의 smem 으로 처리 가능해야. analyze 가 subtree decomposition 추가 필요. CUDA Graph capture 호환성.

문헌: Rennich-Davis 2014 (PMAA / Parallel Computing 2016).

### 3.4 M5 (정밀도 강하) 세부

문제: mid 의 stage-in/writeback 이 fp32. front 데이터 양 자체 가 fsz² × 4 bytes.

해결: stage-in 시 fp32 → bf16 quantization (or tf32 의 압축 form). 절반 trafffic.

복잡도:
- accuracy 영향 큼 (이미 TF32 라서). quantization vs full FP32 difference 측정 필요.
- writeback 도 같이 quantize 해야.

문헌: Ootomo-Yokota 2022 의 correction scheme 으로 정확도 복구 가능 (docs/15 §17 surveyed).

### 3.5 lever 합산 가능성

M1 (front packing) + M3 (`__launch_bounds__`) 직교 — 합칠 가치 있음.
M2 (persistent subtree) 는 dispatch 구조 자체를 바꿈 — 다른 lever 와 호환성 검증 필요.

## 4. small tier 의 lever 카탈로그 (warps/block 외)

EXP-D 실패 후 남는 small lever:

### 4.1 lever 우선순위

| Lever | 설명 | 예상 wall | 난이도 |
|-------|------|-----------|--------|
| **S2: read-only metadata 를 `__ldg`** | plcols, front_off, asm_local 등 access 를 `__ldg` 로 강제 read-only L1 path | 0-3% (이미 `__restrict__` 로 비슷한 효과) | 낮음 |
| **S3: 작은 front 의 packing into mid kernel** | 4-8 SMALL_THRESH 이하 front 를 한 mid block 에 묶기 | 5-10% | 중-상 |
| **S4: cp.async pipelined writeback** | writeback 도 async 화, 다음 warp 의 stage-in 과 overlap | 0-5% (writeback 작아서 영향 작음) | 중 |
| **S5: 정밀도 강하 (small 만 half)** | half precision 의 fused LU (accuracy 영향 검증) | 10-20% small wall | 중 |

### 4.2 S3 — small 을 mid 의 packing 으로 흡수 (가장 흥미)

문제: small 은 1 warp = 1 (front, batch), block 마다 8 warps. warp 간 cooperation 없음 → memory-latency 의 한계.

해결: small 을 별도 kernel 로 두지 않고 **mid kernel 의 front packing path** 로 흡수. 같은 mid kernel 의 block 안에서 4-8 small front 처리.

장점: mid kernel 의 256 thread 가 collaborate 가능 → cache locality 향상 + latency hiding 더 잘 됨.
단점: small/mid tier 의 boundary 가 흐려짐 → dispatch 단순화? 아니면 복잡화? unclear.

→ M1 (front packing for mid) 과 같이 묶어 한 번에 설계.

## 5. 본 codebase 의 memory lever 한계 평가

### 5.1 docs/14 의 메타-교훈 재적용

docs/14: sync 64% 감소 → wall 1-4% only. ratio ≈ 1/16.

본 문서 의 memory analog 가능 추정:
- DRAM throughput 64% 활용 — 절반 줄여도 (DRAM 32%) wall 의 lever 가 sync 케이스 처럼 작을 가능성
- 이유: 본 codebase 가 memory-bound 라기보다 **memory-bound + sync-bound + compute-bound 가 wave 단위로 overlap**. 한 component 줄여도 다른 component 가 새 bottleneck.

따라서 M1 / M2 의 예상 wall 게인 (10-30%) 는 BG3 의 5-15% 예측이 실측 2-3% 였던 것 처럼 **하향 조정** 가능성 — **실제 5-10% 예상**.

### 5.2 simple micro-opt 의 ceiling

- EXP-D (warps/block 8→16): **회귀** — gen-1 micro-opt 의 마지막 lever 도 reject
- S2 (`__ldg`): 이미 `__restrict__` 로 거의 같은 효과 — wall 게인 거의 0
- M3 (`__launch_bounds__` for mid): mid smem 18 KB × 5 blocks = 90 KB, 이미 거의 max → 게인 작을 것
- M4 (writeback async): writeback 작아서 영향 작음

→ **본 codebase 의 small/mid memory-bound 의 simple micro-opt 는 거의 다 시도됨 또는 무가치**. 진짜 lever 는 architecture-level (M1, M2, S3).

### 5.3 architecture-level lever 의 ROI 평가

M1 (front packing) + M2 (persistent subtree):
- 예상 wall 게인: 5-10% (sync ratio 적용 후 보수적)
- 복잡도: 매우 큼 (analyze 단계 부터 dispatch / kernel 까지 모두 변경, CUDA Graph 호환성)
- 위험: 기존 multi-tier dispatch 구조를 깨지 않으면서 통합 어려움

ROI:
- gain / complexity ≈ 5-10% / "weeks of work" → 작음
- 단, V9h / LB(512,2) 같은 micro-opt 의 ceiling 에 도달한 시점에서는 **유일한 남은 lever**

## 6. 권고

### 6.1 시도하지 말 것 (이번 라운드 reject)

- **SMALL_WARPS = 16** (EXP-D): docs/16 S1 의 가설 reject. block 키워도 warps_in_flight 그대로, smem 만 추가 → 회귀.
- **`__ldg` for read-only metadata** (S2): 이미 `const __restrict__` 로 거의 같은 효과, 추가 게인 거의 0.
- **`__launch_bounds__` for mid** (M3): smem 이 거의 max 라 register cap 으로 binding 못 바꿈.

### 6.2 시도할 만한 후속 (단계적)

1. **mid 의 stage-in/writeback breakdown 정확 측정** — 어느 phase 가 정확히 어떤 DRAM byte 소비하는지 ncu source-level 또는 manual instrumentation 으로 분리. wall 의 어느 비중인지 정확히 측정 후 M1/M2 의 ROI 재평가.
2. **M1 (front packing) 의 prototype** — 같은 level 의 작은 mid fronts (fsz ≤ 48) 만 묶어 한 block 에서 처리. small 과 mid 의 boundary 영역에서 시작. 위험 적고 측정 가능.
3. **M2 (persistent subtree)** 의 architecture 검토 — analyze 의 subtree 분해 부분 부터 prototype. 큰 변경 이라 별도 R&D.
4. **M5 (정밀도 강하)** 의 accuracy 영향 측정 — Ootomo-Yokota correction 적용 후 wall vs accuracy trade-off.

### 6.3 본 라운드의 최종 결론

본 codebase 의 small/mid memory-bound 는 **simple micro-opt 로 해결 불가**. EXP-D 의 negative result 가 이를 확정. 진짜 lever 는 architecture-level (front packing, persistent subtree) — 모두 weeks of work 단위.

**docs/15-17 의 V0-V9h-LB512 시리즈 는 GEMM/occupancy micro-opt 의 ceiling 에 도달**. 다음 단계는 architecture R&D 또는 다른 phase (panel LU sync, BG1) 의 micro-opt 시도.

**사용자 직관 (V0 가 best) 의 strong evidence**:
- docs/15: GEMM micro-opt ceiling = trailing GEMM 의 wall 비중 (~30%) × 이론 가속 (2x) = 15% → 실측 5-10%
- docs/17: occupancy micro-opt = sync↓의 wall conversion 1/10 × barrier 26% = 2.6% → 실측 -2.7%
- docs/18 (본 문서): memory micro-opt = 시도 가능 lever 거의 모두 reject 또는 ceiling 도달

V9h + LB(512,2) 결합이 현재 codebase 의 **realistic ceiling** (~5-10% wall on USA B=64). 이상은 architecture 재설계 필요.

## 7. 참고

- docs/9-10: T4 plan 의 original, mid_warp 실패 (small 의 warp-per-front 패턴이 mid 에 fail)
- docs/13-14: panel LU sync 분석 + P1-P5, P5 (warp-spec) 미실행
- docs/16: B-sweep + ncu stall 분석, S1 의 origin
- docs/17: BG3 (launch_bounds) 실행, sync↓→wall↓ 변환률 ~1/10 측정
- Rennich-Davis PMAA'14: subtree-walking persistent kernel for sparse Cholesky
- MAGMA Haidar et al. 2018: batched LU, front packing patterns
- Ootomo-Yokota arXiv 2203.03341: TC accuracy correction for precision drops
