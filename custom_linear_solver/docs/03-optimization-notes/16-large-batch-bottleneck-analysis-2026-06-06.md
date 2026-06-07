# 대-batch 영역의 병목 분석과 tier 별 해소 방향

**작성일**: 2026-06-06
**대상**: factor 의 small / mid / big tier 가 B 증가에 따라 saturate 하는 지점과 saturation 이후의 병목 원인
**환경**: RTX 3090 (sm_86), CUDA 12.8, panel_cap=8, V0 default 빌드
**측정**: tf32, --repeat 32, 3-trial median; ncu B=64 1-launch sample
**동기**: docs/15 시리즈에서 trailing GEMM micro-optimization 의 wall 게인이 (B=64 throughput 시나리오에서) 2-5% noise floor 수준에 머무름. **trailing GEMM 자체가 wall 의 11-41% 만 차지** (docs/11), 따라서 GEMM 만 짜내선 ceiling 이 분명. B 가 커질 때 진짜 wall lever 가 어디인지 측정 → tier 별 분리 진단 → 제안.

## 0. TL;DR

- **3 tier 가 서로 다른 패턴으로 saturate**:
  - small: B≈256 (가장 늦음) — memory-latency bound (long scoreboard 203%, barrier 0)
  - mid : B≈64 — sync + memory bound (barrier 94%, scoreboard 257%, warps_active 31%)
  - big : B≈16 (가장 빠름) — **극단적 sync bound** (barrier **1907%**, DRAM 23%, inst/cycle 0.26)
- **GEMM 가속의 ceiling**: docs/15 의 V9h 가 USA B=64 -2.7% 한 이유는 trailing GEMM 의 wall 비중 자체가 작기 때문. 진짜 lever 는 phase 1 (panel LU) 의 sync wait — big tier 의 barrier 1907% 이 직접 증거.
- **권고 우선순위**: (1) big tier 의 warp-specialized panel LU (P5 from docs/13, 미구현 deferred), (2) mid tier 의 multi-front per block (front packing), (3) small tier 의 warps-per-block 확장. GEMM PTX 미세조정은 ROI 한계 도달.

## 1. 측정 — B sweep wall

`batch_factor_per_sys_ms` (V0 default, tf32, --repeat 32, 3-trial median):

| case | B=1 | B=4 | B=16 | B=64 | B=256 |
|------|-----|-----|------|------|-------|
| case8387 | 0.447 | 0.134 | 0.044 | 0.027 | **0.023** |
| ACTIVSg25k | 0.791 | 0.244 | 0.137 | 0.115 | **0.116** |
| USA | 2.270 | 0.842 | 0.535 | 0.496 | **0.493** |

per-sys ms 비율 (B=k → B=k+1 단계, 4× batch 증가 당 per-sys 감소율):

| case | 1→4 | 4→16 | 16→64 | 64→256 |
|------|-----|------|-------|--------|
| case8387 | 3.34× | 3.03× | 1.62× | 1.17× |
| ACTIVSg25k | 3.24× | 1.78× | 1.19× | **0.99×** |
| USA | 2.69× | 1.57× | 1.08× | **1.01×** |

**Saturation 정의**: per-sys 감소율 ≈ 1.0 (= 추가 batch 가 일대일 cost 만 추가, throughput 게인 0).

- **case8387**: B=64 에 1.62×, B=256 에 1.17× — 아직 saturate 안 함. **small tier 중심 case 가 가장 늦게 saturate**.
- **ACTIVSg25k**: B=64 에 1.19×, B=256 에 0.99× — **B=64 에 saturate**.
- **USA**: B=16 에 1.08×, B=64 에 1.08× — **B=16 에 이미 saturate**. **big tier 중심 case 가 가장 빨리 saturate**.

총 wall (`factor_per_sys × B`, ms) 도 확인:

| case | B=1 | B=4 | B=16 | B=64 | B=256 |
|------|-----|-----|------|------|-------|
| case8387 | 0.45 | 0.54 | 0.71 | 1.74 | 5.98 |
| ACTIVSg25k | 0.79 | 0.98 | 2.19 | 7.34 | 29.66 |
| USA | 2.27 | 3.37 | 8.56 | 31.77 | 126.14 |

USA B=64→256 (4× batch): 31.77 → 126.14 = **3.97× cost**. 거의 perfect-linear → throughput 측면 추가 게인 0.

## 2. 진단 — ncu stall reason (B=64)

| Tier (대상 kernel, case) | warps_active | barrier stall % | long_scoreboard % | DRAM throughput | inst/cycle | 진단 |
|--------------------|--------------|----------|--------------|-----|----------|------|
| Small (`factor_small`, case8387) | 72.22% | **0.00%** | 202.86% | – | 0.78 | memory-latency |
| Mid (`factor_mid_tf32`, ACTIVSg25k) | 31.49% | **93.89%** | 257.47% | – | 0.42 | sync + memory |
| Big (`factor_big_tf32`, USA) | 66.21% | **1906.78%** | 259.32% | 23.35% | 0.26 | **극단 sync** |

해석:
- **`barrier_stall_per_issue_active`** 가 100% 면 매 active issue 당 1 warp 가 barrier 대기, 1907% 면 19 warp 동시 대기. warp 32 개 중 절반 이상이 항상 sync wait.
- **`long_scoreboard`** 200%+ 는 global memory dependency 의 multiple-warp 동시 대기.
- **`inst/cycle`** Ampere peak = 4. 0.26 (big) → peak 의 6.5% 만 활용. 컴퓨트 자원 거의 노는 상태.
- **DRAM 23%** (big) → 메모리 대역폭 한계 아님. 컴퓨트 자원이 sync 로 묶여 발행 안 됨.

이론적 upper bound (barrier stall 완전 제거 시):
- 1907% / 100% ≈ **19× kernel time 단축 잠재** for big_tf32. 실현 가능성은 phase 1 panel LU 의 본질적 sequential 성격에 의해 제한. 보수적 추정 2-5× kernel 단축 → 20-40% wall 단축.

## 3. tier 별 saturation 메커니즘

### 3.1 Small tier — memory-latency bound

**구조**: 1 warp = 1 (front, batch). 8 warps/block = 256 thread. block grid = `((level_size × B + 7) / 8, 1)`. warp 들은 독립적 — block 내 sync 없음 (barrier 0%).

**Saturation point**: 거의 안 함. case8387 의 B=256 에서도 아직 ~17% 추가 게인. 이유: warp 단위로 매우 작은 work (small front, fsz≤32) 이라 latency 가 throughput 의 dominant cost.

**Bottleneck (B=64+ saturation 근처)**: long scoreboard 203% — global memory access (asm_local, plcols, front_ptr 등 read-only metadata) 의 latency. 8 warps × 32 lanes × stride pattern 으로 인해 memory transaction coalescing 불완전.

**케이스 별**:
- case8387 mid + big 적으니 small 이 wall dominant. small saturate 안 됐으니 B 더 키울 여지 있음 (계산 자원 남음).
- USA / ACTIVSg25k 의 small 은 wall 비중 작음 (큰 tier 가 먼저 saturate).

### 3.2 Mid tier — sync + memory bound

**구조**: 1 block = 1 (front, batch). 256 thread. grid = (level_size, B). 256 thread × 6 blocks/SM = 1536 thread/SM (cap).

**Saturation point**: B=64 (ACTIVSg25k). ACTIVSg25k mid fronts ~377 × 64 = 24K blocks. SM 84 × 6 = 504 동시 block. wave 수 = 24000 / 504 = 48 wave. 더 키워도 (B=256) wave 만 늘어 throughput 일정.

**Bottleneck**: barrier stall **94%** — Phase 1 (panel LU) 의 sequential 패널 처리. 매 패널 (nc=8-20) 마다 `__syncthreads`. nc thread 만 useful, 256-nc thread (= 236-248개) idle 하지만 sync 대기. 추가로 scoreboard 257% — global memory 의 stage-in/writeback 도 latency.

**warps_active 31%** = 8 warp 중 평균 2.5 warp 만 issue. 나머지 5.5 warp 는 sync 대기 + memory 대기.

### 3.3 Big tier — extreme sync bound

**구조**: 1 block = 1 (front, batch). 1024 thread. grid = (level_size, B). thread cap = 1536/SM → 1 block/SM only.

**Saturation point**: B=16 (USA). USA big fronts ~50 × 16 = 800 blocks. SM 84 × 1 = 84 동시. wave = 10. B=64 → 200 × wave 40 vs B=16 wave 10: 거의 linear → no per-sys 게인.

**Bottleneck**: barrier stall **1907%** — 1024 thread × 30+ syncthreads per front (panel LU 의 nc panels + writeback + extend-add) × 매 sync 마다 most threads idle. dram_throughput 23% 이고 inst/cycle 0.26 — 메모리도 컴퓨트도 활용도 매우 낮음, 거의 sync 대기 만.

**이론**: 1024 thread × nc useful per panel × 5-20 panels × per-panel sync. 평균 thread utilization = nc/1024 ≈ 1-2%. 나머지 98-99% thread 가 sync 대기. **이게 big tier 가 가장 빨리 saturate 하면서 wall delta 가 가장 크게 안 남는 이유**.

## 4. 해소 방향 제안 (tier 별)

각 제안에 (난이도, 예상 wall 게인) 표기. 게인은 측정된 stall % 기반 conservative estimate.

### 4.1 Small tier 제안

| 제안 | 설명 | 예상 wall 게인 | 난이도 |
|------|------|---------------|--------|
| **S1: warps/block 8 → 16** | 1 block = 16 warps × 32 thread = 512 thread (cap 1024 안). warp 독립 — 더 많은 latency hiding. | 5-15% small wall | 낮음 (block dim 한 줄 변경 + smem alloc 조정) |
| **S2: read-only metadata 를 `__ldg` / texture cache** | asm_local, plcols, front_ptr 같은 read-only int 배열을 `__ldg` 또는 ROC 강제로 hit rate 개선. | 3-8% small wall | 낮음 (인덱싱 함수 한정 수정) |
| **S3: inner-loop 의 fma scheduling 재배치** | `lu_small_warp` 의 rank-1 update 의 dependency chain 끊기 → ILP 증가. | 2-5% | 중 (kernel asm 수준 튜닝) |
| **S4: 정밀도 강하 (small 만)** | small 은 currently fp32 scalar. half/bf16 으로 강하 가능 (accuracy 영향 측정 필요). | 10-20% small wall | 중 (정밀도 변환 + accuracy 검증) |

**case8387 같이 small 비중 큰 case 에 S1+S2 결합** 권장. 우선순위: **S1 > S2 > S4 > S3**.

### 4.2 Mid tier 제안

barrier 94% 가 dominant → sync 감소가 정공법.

| 제안 | 설명 | 예상 wall 게인 | 난이도 |
|------|------|---------------|--------|
| **M1: multi-front per block (FPB, front packing)** | 1 block = N (front, batch). small mid front (fsz ≤ 48) N개를 한 block 에서 처리. sync wait 가 N개 front 사이 amortize. | **15-25% mid wall** | 중-상 (kernel 구조 변경 + dispatch 분기) |
| **M2: warp-specialized panel LU** | 1 warp 가 panel LU 전담 (nc thread, 28 thread idle). 나머지 7 warp 는 previous panel 의 trailing 을 pipeline. | 10-20% mid wall | 상 (warp role assignment + 양방향 dependency) |
| **M3: mblk 256 → 128** | per-block thread 절반 → sync wait 절반. 단 per-block 의 panel LU / U-solve parallelism 도 절반. trailing 은 거의 같음 (warp 별 작업). | 0-10% | 낮음 (한 줄 변경, 측정) |
| **M4: panel LU 의 TC 화 (Lopez-Mary 2023)** | panel 의 rank-nc update 도 trailing GEMM 처럼 TC. K dim 작아 m16n8k4 활용 여지. | 5-10% mid wall | 중-상 (panel LU 본질이 column elimination 이라 mma 적용 까다로움) |

**ACTIVSg25k 같이 mid 비중 큰 case 에 M1 우선**. 우선순위: **M1 > M2 > M4 > M3**.

**M1 의 detail**: ACTIVSg25k mid 377 front 의 fsz 분포를 봐야 함. 작은 mid (fsz 32-48) 가 다수 면 packing 효과 큼. dump_fronts 로 측정 후 결정.

### 4.3 Big tier 제안

barrier **1907%** 는 다른 어떤 metric 보다 압도적. sync wait 가 본질적 lever.

| 제안 | 설명 | 예상 wall 게인 | 난이도 |
|------|------|---------------|--------|
| **BG1: warp-specialized panel LU (P5 from docs/13)** | 1 warp (32 thread) 가 panel LU 만, 나머지 31 warp 는 previous panel 의 trailing/U-solve. pipeline. **barrier wait 의 본질적 해소**. | **20-40% big wall** | **상** (warp role, dependency, smem coordination) |
| **BG2: split front across multiple blocks (cooperative)** | 1 (front, batch) → 2-4 block 으로 split. 각 block 이 front 의 일부 column 담당. block 당 thread 줄어 → 1 block/SM thread cap 풀림 → 더 많은 (front, batch) 동시. | 15-30% | 상 (cooperative groups + 동기화) |
| **BG3: bigT 1024 → 256 with `__launch_bounds__`** | V4 (bigT=512) 실패는 register cap 이 새 binding. `__launch_bounds__` 로 register/thread 명시적 제한해 occupancy 게인 강제. | 5-15% | 중 (launch_bounds + spill 측정) |
| **BG4: phase fusion P2 (panel LU + U-solve)** | docs/14 에서 측정: sync 67% 절감 vs wall -1~-4%. **sync ≠ wall ceiling 증명** — 본 tier 의 다른 sync 가 새 병목으로 떠오름. | 1-4% (이미 알려진 수치) | 중 (이미 deprecated/mid_opt 으로 시도됨) |
| **BG5: streaming panel LU across B** | batch dim 의 ILP 활용. batch b 의 panel LU 동안 batch b-1 의 trailing 진행. | 10-20% | **매우 상** (B-축 의 dependency 그래프 재구성, cooperative kernel) |

USA 같이 big 비중 큰 case 에 **BG1 이 단연 highest-impact**. 본 연구 시리즈에서 검토한 GEMM micro-optimization (V0-V9h) 전체 게인 보다 BG1 한 변형의 잠재 게인이 클 것.

우선순위: **BG1 >> BG2 > BG3 > BG5 > BG4**.

## 5. case 별 권고

| case | dominant tier | 1순위 lever | 추가 |
|------|---------------|-------------|------|
| case8387 (small dominant, B=256 까지 scale) | small | **S1** (warps/block 16) | S2, panel_cap 12 (docs/15 V8) |
| ACTIVSg25k (mixed) | mid | **M1** (FPB) | M2, V9h 의 mid k4 hybrid 유지 |
| USA (big dominant, B=16 부터 saturate) | big | **BG1** (warp-spec panel LU) | BG2, V9h 의 big PTX 유지 |

## 6. 본 연구 시리즈 (docs/15) 의 위치 재평가

docs/15 의 V0-V9h 는 모두 **trailing GEMM 의 within-kernel micro-optimization**. trailing GEMM 자체가 wall 의 11-41% (docs/11), V9h 가 trailing 을 5-15% 가속 → wall 0.5-6% 게인. 측정 noise 와 자릿수 비슷.

**ncu 의 inst/cycle = 0.26 (big), 0.42 (mid)** — Ampere peak 4 의 6-10% 만 활용. **활용 안 되는 90% 의 컴퓨트 자원** 은 GEMM 의 K-padding 절약 으로 회복되지 않음. sync wait 으로 묶여있음.

본 문서의 BG1 (warp-spec panel LU) 이 sync wait 의 본질을 풀면 inst/cycle 0.26 → 1.0+ 가 현실적 → **kernel time 4×, wall 1.5-2× 단축** 잠재. GEMM PTX 시리즈의 10× 이상 wall lever.

따라서 docs/15 의 결론 **"V9h 의 wall 게인이 noise 와 자릿수 비슷 → V0 유지"** 의 더 근본적 이유:

> trailing GEMM 자체가 wall 의 ~30% 만 차지하고 그 GEMM 의 이론적 max 가속도 2× 정도. wall 영향 30% × 50% = **15% 가 GEMM 가속의 ceiling**. 실제 게인 5-10% 는 이 ceiling 의 절반 — micro-opt 의 한계.

진짜 lever 는 **나머지 70% 의 non-GEMM wall, 특히 phase 1 panel LU 의 sync wait**.

## 7. 후속 연구 우선순위 (전체 lever 비교)

| Lever | 잠재 wall 게인 (USA B=64) | 구현 난이도 | ROI |
|-------|---------------------------|-------------|-----|
| **BG1 — warp-spec panel LU (big)** | **20-40%** | 상 | **highest** |
| M1 — front packing (mid) | 10-15% | 중-상 | 높음 |
| BG2 — split front cooperative (big) | 10-20% | 상 | 중-높음 |
| S1 — warps/block 16 (small) | 1-3% (USA 는 small 작음) | 낮음 | case-dependent (case8387 에 높음) |
| BG3 — bigT 256 + launch_bounds | 3-8% | 중 | 중 |
| **V9h — trailing GEMM PTX (현재)** | 2-3% (이미 채택) | 중 | low (ceiling 도달) |

## 8. 실현 가능성과 위험

- **BG1 의 위험**: warp role assignment 가 fragile. cuda cooperative_groups 의 named_barrier (`__syncwarp` + custom `__syncthreads_count` 조합) 필요. CUDA Graph capture 와의 호환성 확인 필요.
- **M1 (FPB) 의 위험**: front 별 fsz/nc 가 다양 → packing 알고리즘 complex. analyze 단계에서 front 묶기 (similar fsz 끼리) pre-process 필요.
- **모두 공통**: 새 kernel 추가 시 `cudaFuncSetAttribute(.., MaxDynamicSharedMemorySize, 99 KB)` 등록 필수 (docs/15 §10.5).

## 9. 참고

- docs/11: trailing GEMM 의 wall 비중 (11-41%) 측정
- docs/13: panel LU + U-solve sync 분석, P5 (warp-spec) deferred
- docs/14: phase fusion (P2) 실측: sync 67% 절감 vs wall -1~-4% — **"sync 절감 ≠ wall 단축" 의 nuanced 해석**: 본 문서의 1907% barrier stall 은 그것보다 훨씬 큰 sync wait 누적이라 풀면 wall 영향 클 것
- docs/15: trailing GEMM PTX 시리즈 (V0-V9h) — ceiling 도달, 본 문서로 lever 이동
- docs/01-orientation/06: factorize 의 file layout — kernel 들의 위치 reference
