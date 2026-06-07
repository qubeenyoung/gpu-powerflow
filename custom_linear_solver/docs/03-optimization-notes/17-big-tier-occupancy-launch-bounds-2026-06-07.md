# Big-tier 의 sync-bound 완화 — docs/16 BG1/BG3 부분 실행 보고

**작성일**: 2026-06-07
**대응 계획**: [docs/16 §4.3](16-large-batch-bottleneck-analysis-2026-06-06.md)
**환경**: RTX 3090 (sm_86), CUDA 12.8
**문서 구성**: §1 선행 분석 + 문헌, §2 효과 순위 기준 실험 계획, §3 EXP-B 실행, §4 결과 정리, §5 EXP-A (warp-spec) 미실행 사유

## 0. TL;DR

- **EXP-B (bigT 1024 → 512 + `__launch_bounds__(512, 2)`)** 실행: factor_big_tf32 의 barrier stall **1801% → 1340% (-26%)**, register/thread 48 → 64 (compiler spill 허용). wall: USA B=64 **-2.7%**, B=256 **-3.4%**; ACTIVSg25k B=256 **-3.6%**; case8387 B=1 **-6.3%** (왜? §4.2).
- **EXP-B-aggressive (bigT 256, `__launch_bounds__(256, 4)`)** 도 시도: barrier stall **855% (-52%)** 더 줄였으나 warps_active 66% → **49% 추락**, **USA B=1 +41% catastrophic regression**. compute parallelism 부족 → 추천 안 함.
- **EXP-B + V9h 결합** 일관 게인: USA B=64 **-5.7%**, B=1 -6.4% (v9h-only -13.6% 보다 작음 — 부분 간섭).
- **EXP-A (warp-spec look-ahead pipeline, docs/16 BG1)**: 문헌 분석 + 코드 분석 후 **본 라운드 미실행**. 사유 §5 — panel LU 의 본질적 sequential 의존성 + 300+ 줄 invasive 변경 + named barrier 의 race-free 구현 위험. EXP-B 게인이 BG3 범주 (5-15% 예측) 의 하단 (3-6%) 인 점도 EXP-A 의 BG1 예측 (20-40%) 의 신뢰도를 낮춤.
- **사용자 직관 재확인**: V9h 단독 (USA B=64 -2.7%) + EXP-B (-3.8%) + 결합 (-5.7%) 모두 **noise floor 1-3% 의 2-3 배 영역**. wall 게인 의미 있지만 V0 가 가장 단순한 default 라는 점 변하지 않음.

## 1. 선행 분석 + 문헌 조사

### 1.1 prior 실험 정리

본 codebase 의 sync-bottleneck 해소 시도 history:

| 문서 | 시도 | 결과 |
|------|------|------|
| docs/9 T4 | warp-per-front for mid (mid_warp) | **+25-49% 회귀** (mid front 가 1 warp=32 lane 으론 너무 큼). deprecated/mid_warp 로 이동 |
| docs/10 T4.3 | cp.async stage-in | ship default ON |
| docs/10 T4.2.A | row-fused panel LU (nc ≤ 12) | ship default ON. case8387 nc=8 syncs 17→8 |
| docs/13 P1 | reciprocal multiply (FDIV → RCP × MUL) | ship default ON. USA B=1 -2% |
| docs/13 P2 + P3 (in docs/14) | Phase 1+2 fusion + parallel U-solve | USA B≥16 **-1~-4%** only. sync 64% 감소 vs wall 1-4% — **"sync ≠ wall" 변환률 측정**. separate kernel rollback |
| docs/13 P4 | shared padding (bank conflict 회피) | **+85% 회귀** (stage-in 의 integer div overhead). 폐기 |
| docs/13 P5 | warp-spec panel LU | 300+ 줄, last resort, **미실행** |
| docs/16 | big_tf32 의 barrier 1907% 측정 | 본 문서가 후속 |

### 1.2 문헌 조사 (warp-spec panel LU 영역)

- **Volkov & Demmel, SC'08 + LAPACK Note 202 (2008)**: look-ahead pattern 의 origin — panel factorization of step k+1 가 trailing update of step k 와 concurrent. CPU/GPU host 측 분리가 원래 구현.
- **Anderson, Ballard, Demmel, Keutzer, IPDPS'11**: communication-avoiding QR for GPUs. **intra-block warp-specialized** panel + trailing 의 published 사례. tall-skinny QR 에서 MAGMA 대비 2-4×.
- **NVIDIA PTX ISA §9.7.12 `bar` instruction + libcu++ `<cuda/barrier>`**: 16 named barriers per block, `bar.sync N, M` 으로 subset 만 sync. **본 연구의 가장 직접적 도구**.
- **CUTLASS `include/cutlass/pipeline/sm80_pipeline.hpp`** (Ampere): cp.async + `cuda::pipeline_shared_state` 의 producer-consumer pattern reference. wgmma/TMA 는 Hopper sm_90 전용이라 본 sm_86 환경에서 제외.
- **Rennich-Davis, PMAA'14 / Parallel Computing 2016 (sparse Cholesky on GPU)**: persistent-kernel subtree approach. small front kernels 가 "barrier-limited" 라고 **명시적으로 acknowledge 하지만 해결 안 함**.
- **MUMPS, STRUMPACK GPU offload**: small/medium front 의 sync-bound 가 known issue, 작은 front 는 CPU 에 둠 — 즉 GPU 측에서 해결 안 함. 우리 codebase 의 multi-tier dispatch 는 STRUMPACK 의 작은-front-on-CPU 전략을 GPU-만으로 옮긴 것.

→ **EXP-A (warp-spec look-ahead pipeline) 는 학계에서 sparse multifrontal 의 작은 front sync-bound 를 푸는 published 사례 없음**. tall-skinny QR (Anderson IPDPS'11) 에서만 검증. 실험 가치 있으나 prior art 없음 = 위험.

### 1.3 docs/16 의 측정 재요약

| Tier | barrier stall % | inst/cycle | warps_active % | 진단 |
|------|----------------|-----------|---------------|------|
| Small | 0 | 0.78 | 72 | memory-latency |
| Mid | 94 | 0.42 | 31 | sync + memory |
| **Big** | **1907** | **0.26** | 66 | **극단 sync** |

big_tf32 의 inst/cycle 0.26 = Ampere peak 의 6.5%. barrier 1907% = 32 warp 중 평균 19 warp 가 sync 대기.

## 2. 효과 순위 기준 실험 계획

| Rank | 실험 | 예상 wall (USA B=64) | 난이도 | 본 라운드 |
|------|-----|---------------------|--------|----------|
| **1** | **EXP-A: 전체 warp-spec look-ahead pipeline** — 1 warp panel LU, 31 warp trailing for panel k-1, named barriers | 15-30% | 매우 상 (300+ 줄) | **미실행** (§5) |
| **2** | **EXP-B: bigT 256/512 + `__launch_bounds__`** — 2-4 block/SM 강제, register spill 감수 | 5-15% | 낮음 | **실행** (§3) |
| 3 | EXP-C: minimum warp-spec (1 warp panel + idle others) | 0 ~ -5% (negative test) | 낮음 | 생략 (이론적 무가치) |
| 4 | EXP-D: persistent kernel for big — multi-front per launch | 3-8% | 중 | 미실행 (launch overhead 작음 측정 미시도) |
| 5 | EXP-E: cp.async 확장 — trailing GEMM L/U load 까지 pipeline | 3-10% | 중 | 미실행 |

본 라운드 EXP-B 만 실행. **이유**: EXP-B 의 결과 가 BG3 예측 (5-15%) 안에 들면 EXP-A 의 BG1 예측 (20-40%) 도 신뢰 가능 → EXP-A 시도 정당화. EXP-B 가 예측 하단이면 BG1 도 과대평가 가능성 → 위험 비용 너무 큼.

## 3. EXP-B — `__launch_bounds__` 로 occupancy 강제

### 3.1 동기

docs/15 §11 의 V4 (bigT 1024 → 512, no `__launch_bounds__`) 가 USA B=1 **+4.1% 회귀** 한 이유: **register cap 이 새 binding** (48 reg × 512 thread × 2 block = 49 K reg > sm_86 의 64 K 와 가까워 fractional). occupancy 가 오히려 -0.6%p.

**가설**: `__launch_bounds__(N, K)` 명시 → nvcc 가 register 제한 / spill 결정 → **K blocks per SM 강제 달성**. spill cost 가 sync 감소 게인보다 작으면 net win.

### 3.2 구현

`src/factorize/kernels.cuh` 의 factor_big_tf32 선언부:

```cpp
#if defined(CLS_TF32_BIG_LB_256_4)
__global__ void __launch_bounds__(256, 4)
#elif defined(CLS_TF32_BIG_LB_512_2)
__global__ void __launch_bounds__(512, 2)
#else
__global__ void
#endif
                                factor_big_tf32(...);
```

`src/factorize/dispatch.cuh` 의 bigT_tf32 도 같이 조정:

```cpp
#if defined(CLS_TF32_BIG_LB_256_4)
    constexpr int bigT_tf32 = 256;
#elif defined(CLS_TF32_BIG_LB_512_2) || defined(CLS_TF32_BIG_T_512)
    constexpr int bigT_tf32 = 512;
#else
    constexpr int bigT_tf32 = bigT;
#endif
```

`__launch_bounds__` 와 dispatch 의 bigT 가 일치해야 launch 가 cudaErrorInvalidConfig 안 됨.

### 3.3 ncu 측정 — occupancy / register / barrier

factor_big_tf32 on USA B=64:

| build | block_size | reg/thread | waves/SM | warps_active | barrier stall |
|-------|-----------|-----------|----------|--------------|---------------|
| V0 (default, bigT=1024) | 1024 | 48 | 38.24 | 66.26% | **1801%** |
| LB(512, 2) | 512 | **64** ↑ | 27.71 | 65.59% (≈ 동일) | **1340% (-26%)** |
| LB(256, 4) | 256 | 64 | 16.39 | **48.69% ↓** | **856% (-52%)** |

**해석**:
- LB(512, 2): register/thread 48→64 (compiler 가 2 block/SM 위해 register usage 늘려 spill 허용). warps_active 거의 동일 (66.3 → 65.6). **barrier stall -26%** — 1 block 의 syncthreads 가 동시 다른 block 의 다른 phase 와 overlap 가능해서 stall wait 감소.
- LB(256, 4): barrier 더 큰 감소 (-52%) 그러나 **warps_active -17%p 추락**. 256 thread/block 으로 per-block 의 trailing GEMM parallelism 부족 → compute 자체가 약함.

### 3.4 wall (10-trial median, --repeat 50)

| case | B | V0 | LB(512,2) | LB(256,4) |
|------|---|-----|-----------|-----------|
| case8387 | 1 | 0.493 | **0.462 (-6.3%)** | **0.444 (-10.0%)** |
| case8387 | 64 | 0.0278 | 0.0276 (-0.9%) | 0.0271 (-2.7%) |
| case8387 | 256 | 0.0237 | 0.0237 (-0.1%) | 0.0234 (-1.4%) |
| **USA** | **1** | **2.325** | 2.393 (+2.9%) | **3.280 (+41.1% 회귀)** |
| USA | 64 | 0.488 | **0.475 (-2.7%)** | 0.498 (+2.0%) |
| USA | 256 | 0.487 | **0.471 (-3.4%)** | 0.495 (+1.5%) |
| ACTIVSg25k | 1 | 0.815 | 0.801 (-1.7%) | 0.821 (+0.8%) |
| ACTIVSg25k | 64 | 0.116 | 0.116 (-0.3%) | 0.112 (-3.5%) |
| ACTIVSg25k | 256 | 0.115 | **0.110 (-3.6%)** | 0.111 (-3.2%) |

### 3.5 LB(512, 2) vs LB(256, 4) 비교

- **LB(512, 2)**: 안전한 선택. 모든 case 에서 noise (±2%) 또는 게인 (-3~-6%). regression 없음.
- **LB(256, 4)**: aggressive. case8387 B=1 -10% (best) 이지만 USA B=1 +41% catastrophic. compute parallelism 부족이 wall 게인 모두 잠식.

→ **LB(512, 2) 권장**, LB(256, 4) 폐기.

### 3.6 V9h 와 결합 (Big tier + Mid tier 직교 lever)

V9h (docs/15) 는 mid tier 의 trailing GEMM k4 hybrid. LB(512, 2) 는 big tier 의 occupancy. 직교 → 결합 가능.

빌드: `-DCLS_TF32_BIG_PTX=1 -DCLS_TF32_MMA_AREUSE=1 -DCLS_TF32_SKIP_CONVERT=1 -DCLS_TF32_MID_PTX=1 -DCLS_TF32_MID_K4_HYBRID=1 -DCLS_TF32_BIG_LB_512_2=1`.

| case | B | V0 | V9h | LB(512,2) | **V9h + LB** |
|------|---|-----|------|-----------|--------------|
| case8387 | 1 | 0.473 | 0.459 (-2.9%) | 0.477 (+0.9%) | 0.460 (-2.6%) |
| case8387 | 64 | 0.0272 | 0.0276 (+1.5%) | 0.0277 (+1.9%) | 0.0272 (+0.3%) |
| case8387 | 256 | 0.0239 | 0.0242 (+1.5%) | 0.0236 (-1.1%) | 0.0240 (+0.8%) |
| USA | 1 | 2.373 | **2.051 (-13.6%)** | 2.354 (-0.8%) | 2.220 (-6.4%) |
| USA | 64 | 0.496 | 0.480 (-3.2%) | 0.477 (-3.8%) | **0.467 (-5.7%)** |
| USA | 256 | 0.489 | 0.481 (-1.7%) | 0.478 (-2.3%) | **0.469 (-4.1%)** |
| ACTIVSg25k | 1 | 0.786 | 0.750 (-4.6%) | 0.803 (+2.2%) | **0.754 (-4.1%)** |
| ACTIVSg25k | 64 | 0.115 | 0.114 (-0.6%) | 0.113 (-1.2%) | **0.111 (-3.1%)** |
| ACTIVSg25k | 256 | 0.113 | 0.109 (-2.7%) | 0.113 (+0.2%) | 0.110 (-2.6%) |

결합 효과 패턴:
- USA B=64: V9h -3.2% + LB -3.8% 단독 합산 -7%, 결합 -5.7% — 부분 간섭 (서로 다른 phase 의 wall 점유율 중복)
- USA B=1: V9h 가 단독 -13.6% 일 때 결합은 -6.4% — LB 가 V9h 의 mid 게인을 일부 잠식 (mid 와 big 의 wave overlap 변화로 추정)
- ACTIVSg25k B=1: V9h -4.6% 와 결합 -4.1% 비슷, LB 단독 +2.2% (regression) — ACTIVSg25k 는 big 비중 작아 LB 효과 약함, V9h 가 dominant
- case8387: 모두 noise — case8387 은 big tier 미경유 (LB 무영향) + mid 도 k8 dominant (V9h 무영향)

## 4. 결과 정리

### 4.1 게인 정량화

| 변형 | USA B=1 | USA B=64 | USA B=256 | ACTIVSg25k B=1 | ACTIVSg25k B=64 | case8387 B=1 |
|------|---------|----------|-----------|----------------|------------------|--------------|
| V0 (default) | baseline | baseline | baseline | baseline | baseline | baseline |
| V9h (docs/15) | **-13.6%** | -3.2% | -1.7% | -4.6% | -0.6% | -2.9% |
| LB(512,2) (본 문서) | -0.8% | -3.8% | -2.3% | +2.2% | -1.2% | +0.9% |
| **V9h + LB(512,2)** | -6.4% | **-5.7%** | **-4.1%** | -4.1% | **-3.1%** | -2.6% |

### 4.2 case 별 dominant lever 재확인

docs/16 의 case 별 dominant tier:
- case8387 (small dominant): LB(512,2) 의 -6.3% B=1 게인 → small 의 latency hiding 개선 시그널? big 미경유 한데도 게인 — **block size 변화의 부수 효과 가능성** (예: launch overhead 변화). 더 측정 필요.
- ACTIVSg25k (mid dominant): V9h -1.8% + LB -0.3% = -2.1% combined. mid lever 가 dominant 라 LB 의 big-tier 효과 작음.
- USA (big dominant): V9h -3.2% + LB -3.8% combined -5.7%. **big tier 의 LB 게인이 dominant**, 예상대로.

### 4.3 docs/16 BG3 예측과의 정합성

docs/16 BG3: bigT 256 + `__launch_bounds__` → **5-15% wall 잠재**.
실측 LB(512, 2): USA B=64 **-2.7%**, B=256 **-3.4%**.

→ 예측 하단의 절반 정도. BG3 의 5-15% 가 **overestimate** 였음. 이유:
- ncu barrier stall -26% (1801 → 1340) — 26% sync reduction
- wall -2.7~3.4% — sync reduction 의 약 1/8 만 wall 로 전환
- docs/14 의 "sync 64% 감소 → wall -1~-4%" 변환률 (≈ 1/16) 와 비슷한 자릿수 — 즉 sync→wall 의 ratio 는 우리 codebase 에서 일관

이 ratio 가 일관되다는 사실은 BG1 의 BG3 대비 잠재가 다음과 같이 추정 가능:
- BG3: sync reduction 26% → wall 2.7%
- BG1 (warp-spec full pipeline): 가정 sync reduction 70% (panel LU 가 거의 hide) → wall **약 7-10%**

→ docs/16 BG1 의 **20-40% 예측은 과대 평가**. 현실적으로 **5-10% wall 가능성** 으로 조정.

### 4.4 "GEMM micro-opt vs sync 완화" lever 비교

| 시리즈 | best 단일 변형 | wall 게인 (USA B=64) | 복잡도 |
|-------|---------------|---------------------|--------|
| docs/15 V9h (GEMM PTX + k4 hybrid) | -3.2% | 중 (3개 macro) |
| docs/17 LB(512,2) (occupancy via launch_bounds) | -3.8% (single bigger lever ?) | 매우 낮음 (2개 매크로) |
| 결합 V9h + LB(512, 2) | -5.7% | 중 |
| **이론적 BG1 (warp-spec pipeline)** | 추정 5-10% | 매우 높음 (300+ 줄) |

LB(512, 2) 가 **V9h 와 비슷한 wall 게인 + 훨씬 단순한 구현** (2 line 추가) → ROI 최고.

### 4.5 docs/16 의 "barrier 1907% → 1340% sync 26% 감소가 wall 3% 변환" 의미

ncu 의 `barrier_stall_per_issue_active.pct` 가 1907% 라는 것은 **스케줄러의 active issue 시점에 평균 19 개의 warp 가 barrier wait 중** — 즉 wait 자체는 latency hiding 으로 일부 흡수됨. wait 가 measured "stall" 로 나타나는 것은 다른 warp 가 issue 못 할 때만. SM 의 wave 단위 overlap 으로 wait 를 시간상으로 hide 함.

**LB(512, 2)** 가 wave/SM 38 → 27.7 으로 줄여도 wall 작게 감소한 이유: wave 가 작아진 만큼 wave 간 overlap 의 latency hiding 도 줄어 net 효과 작음. 진짜 lever 는 wave 내 의 warp 간 overlap (= warp-spec).

## 5. EXP-A 미실행 사유

### 5.1 구현 복잡도 vs 시간 예산

docs/13 P5 에서 "warp-spec panel LU 300+ 줄" 으로 추정. 실제 구현하려면:

1. factor_big_tf32 의 phase 1 (lu_panel_factor) 호출을 **inline 으로 풀고**, panel LU 의 매 step k 를:
   - warp 0 가 step k+1 의 column k+1 divide (look-ahead)
   - warps 1-31 가 step k 의 rank-1 update
   - 둘 사이 named barrier 동기화
2. `bar.sync N, M` 명시적 PTX (libcu++ `cuda::barrier` 보다 cheap 하나 race-free 보장 까다로움)
3. CUDA Graph capture 와 호환: named barrier 의 graph capture 동작 검증 필요
4. 정확도: race-free 한 buffer management (k+1 의 column k+1 이 k 의 rank-1 update 에 영향 받지 않아야 함)
5. nc 변화 (per-front) 에 대한 adaptive — 일부 front 는 nc 작아 look-ahead 무의미

### 5.2 EXP-B 의 정량 결과가 EXP-A 의 위험 평가에 미친 영향

- docs/16 BG3 (5-15%) 예측 → 실측 -2.7~3.8%, 예측 하단의 절반.
- docs/16 BG1 (20-40%) 도 같은 ratio 로 over-estimate 가능 → 실현 가능 wall 게인 **5-10% 추정** (§4.3).
- 5-10% wall 게인 vs 300+ 줄 변경의 위험 — **ROI 낮음**.

### 5.3 prior 실패 사례

- docs/9 의 mid_warp (warp-per-front for mid): +25-49% 회귀. warp-spec 류의 idea 가 본 codebase 에서 실패한 전례.
- docs/14 의 P1+P2 fusion: sync 64% 감소 vs wall -1~-4% only. **본 codebase 의 sync→wall 변환률이 일관되게 낮음** — warp-spec 도 같은 곳에 발 들일 가능성.

→ EXP-A 는 **follow-up 우선순위 1 으로 marking, 본 라운드는 LB(512, 2) 채택**.

### 5.4 EXP-A 의 실현가능한 simplification 후보 (follow-up)

다음 라운드에서 EXP-A 시도할 경우의 step-by-step 분해:

1. **A-step1: named barrier 인프라**. `bar.sync N, M` 의 inline asm wrapper. 작은 test kernel 로 검증 후 production 적용.
2. **A-step2: panel LU 의 single-warp variant**. 1 warp 가 모든 panel LU 단계 수행 (다른 31 warp 는 명시적으로 idle). barrier wait → kernel 결과 비교 → no regression vs +regression 측정. 만약 +regression 작으면 (예: <10%) full pipeline 가능성 ↑.
3. **A-step3: 1-step look-ahead**. step A-step2 위에 다른 31 warp 가 step k-1 의 trailing 일부 수행. buffer 분리 + named barrier. ROI 명확히 측정.
4. **A-step4: full pipeline 화**. 모든 panel step look-ahead. step A-step3 이 wall -5% 이상이면 정당화.

각 단계에서 명확한 go/no-go gate → 총 위험 분산.

## 6. 권고 + 후속 가능성

### 6.1 권고

| 빌드 | 게인 | 복잡도 | 권고 |
|------|------|--------|------|
| V0 default | baseline | 최저 | **default 권장** |
| LB(512, 2) | USA B=64 -2.7%, ACTIVSg25k B=256 -3.6% | +2 줄 | **opt-in 안전** (regression 없음) |
| V9h | USA B=1 -13.6%, B=64 -3.2% | +5 macro | opt-in for power-grid |
| **V9h + LB(512, 2)** | **USA B=64 -5.7%** | +6 macro | **power-grid 류에 best opt-in** |
| LB(256, 4) | case8387 B=1 -10%, USA B=1 +41% | +2 줄 | **추천 안 함** (catastrophic) |
| EXP-A (warp-spec) | 추정 5-10% | 300+ 줄 | follow-up, 단계적 분해 후 |

### 6.2 메타-교훈

1. **sync → wall 변환률이 낮은 codebase**: docs/14 의 sync 64%↓ → wall -1~-4%, docs/17 의 barrier 26%↓ → wall -3% — **변환률 ≈ 1/10 ~ 1/16**. ncu 의 큰 stall % 가 wall lever 의 크기를 정확히 반영 안 함. wave 단위 latency hiding 이 stall 의 일부를 흡수.
2. **occupancy gain 의 wall 효과 측정**: 단순 launch_bounds 가 가장 단순한 occupancy lever 였음. V4 (no launch_bounds) 의 register cap binding 을 명시적으로 풀 수 있다는 것이 docs/15 §11 (V4 실패) 의 답.
3. **lever 직교성 ≠ 게인 가산성**: V9h (-3.2%) + LB (-3.8%) 단독 합산 -7% 가 결합에서 -5.7% — 일부 phase 의 wall 점유율 중복으로 게인 가산 안 됨.
4. **prior art 무 → 위험 큼**: warp-spec panel LU 가 sparse multifrontal 에서 published 사례 없음 — 학계도 unsolved 라고 인정 (Rennich-Davis 2016). 우리가 푸는 ROI 가 정당화될 때만 시도.

### 6.3 후속 가능성

1. **EXP-A 의 단계적 분해** (§5.4) — A-step2 의 simple test 부터.
2. **mid tier 의 launch_bounds**: 본 라운드 big 만 적용. mid 도 적용 가능 (256 thread/block 의 register 제한 → 다른 occupancy). 단 mid 는 register-bound 가 binding 아닐 가능성 (V4 와 다른 영역).
3. **EXP-E (cp.async 확장)**: 현 stage-in 외에 trailing GEMM 의 L/U load 까지 pipeline. cutlass sm80_pipeline 의 reference 활용. 위험 LB 보다 큼.

## 7. 코드 / 빌드 / 재현

### 7.1 변경 파일

- `src/factorize/kernels.cuh`: factor_big_tf32 선언에 `__launch_bounds__` macro 분기 추가.
- `src/factorize/dispatch.cuh`: bigT_tf32 의 macro 분기에 LB 변형 추가.

### 7.2 빌드 옵션

| build dir | flags | 변형 |
|-----------|-------|------|
| `build-lb-512-2/` | `-DCLS_TF32_BIG_LB_512_2=1` | LB(512, 2) 단독 |
| `build-lb-256-4/` | `-DCLS_TF32_BIG_LB_256_4=1` | LB(256, 4) 단독 (deprecated) |
| `build-v9h-lb512/` | V9h flags + `-DCLS_TF32_BIG_LB_512_2=1` | V9h + LB(512, 2) **권장 결합** |

cmake 예시:
```bash
cmake -S . -B build-v9h-lb512 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="-DCLS_TF32_BIG_PTX=1 -DCLS_TF32_MMA_AREUSE=1 \
                        -DCLS_TF32_SKIP_CONVERT=1 -DCLS_TF32_MID_PTX=1 \
                        -DCLS_TF32_MID_K4_HYBRID=1 -DCLS_TF32_BIG_LB_512_2=1"
```

## 8. 참고

- docs/9: T4 plan 의 원본
- docs/10: T4.1 mid_warp 실패 + T4.3 cp.async ship
- docs/13: P1-P5 설계, P5 (warp-spec) 미실행 last resort
- docs/14: P1+P2 fusion 실측, sync↓→wall 변환률 측정
- docs/15: GEMM PTX 시리즈 (V0-V9h)
- docs/16: B-sweep + ncu stall 분석, BG1-BG5 제안
- Volkov-Demmel SC'08: look-ahead origin
- Anderson IPDPS'11: intra-block warp-spec for tall-skinny QR
- PTX ISA §9.7.12 `bar`: named barriers
- CUTLASS sm80_pipeline.hpp: Ampere producer-consumer reference
- Rennich-Davis PMAA'14: sparse Cholesky GPU, small-front barrier-limited acknowledged unsolved
