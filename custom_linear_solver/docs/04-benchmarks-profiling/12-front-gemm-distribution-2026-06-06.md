# Front 크기와 trailing GEMM 분포 분석 (case8387 / USA)

**작성일**: 2026-06-06
**대상**: refactored `custom_linear_solver` (`41897cd`), panel_cap=8
**측정 데이터**: `src/solver.cpp`의 `CLS_DUMP_FRONTS` env-var hook으로 추출 (`/home/claude/prof/fronts_8387.csv`, `fronts_USA.csv`)
**분석 스크립트**: `/home/claude/prof/front_gemm_hist.py`

## 0. TL;DR

- **front 개수 분포는 극단적으로 small-heavy** (case8387 99.3%, USA 98.7%)
- 그러나 **FLOP은 mid/big에 집중** (case8387 mid 50.6%, USA mid+big 91.4%)
- **per-front trailing GEMM 크기 중간값**: case8387 small 32원소 / mid 8192원소 / USA small 32 / mid 17298 / big 353780
- **WMMA K-padding** (TC 활성 판단의 핵심):
  - FP16 K=16: mid median **50%** (case8387), **37.5%** (USA) — 큰 padding 손실
  - TF32 K=8: mid median **0%** (case8387!), **16.7%** (USA) — TF32 우월성의 정량 근거
- **nc=2와 nc=20**이 두 개의 모드 — small tier nc≤2 (75-88%), mid/big tier nc=8 또는 20 (case8387/USA 각각)

## 1. tier 정의와 dispatch 매핑

```cpp
// src/multifrontal.cu:issue_factor_level_range
SMALL_THRESH = 32     → factor_small (warp-per-front, 8 warps/block)
MID_THRESH   = 128    → factor_mid<T> (block-per-front, staged trailing)
                        / factor_mid_tc, factor_mid_tf32, factor_mid_warp (T4.1, opt-in)
> 128                 → factor_big<T> (1024-thread block, global memory)
```

Trailing GEMM 차원: 한 front당 `C(uc × uc) -= L(uc × nc) · U(nc × uc)`, 즉 (M, N, K) = (uc, uc, nc).
FLOP = 2·uc²·nc.

## 2. case8387pegase (n=14908, fronts=7427)

### 2.1 tier 합계

| tier | count | 비율 | trailing FLOP | FLOP 비율 |
|------|------:|----:|--------------:|----------:|
| small | 7372 | **99.26%** | 1.23 MF | 49.36% |
| mid | 55 | **0.74%** | 1.26 MF | **50.64%** |
| big | 0 | 0% | – | – |

- 99.3%의 front가 작지만 FLOP은 거의 반반. **55개의 mid front가 작업의 절반을 차지** — Amdahl 관점에서 mid 최적화의 가치 큼

### 2.2 small tier (7372 fronts)

| metric | min | max | mean | median |
|--------|----:|----:|----:|------:|
| fsz | 2 | 32 | 6.28 | **6** |
| nc | 1 | 8 | 1.97 | **2** |
| uc | 0 | 28 | 4.31 | 4 |
| uc²·nc (GEMM elts) | 0 | 4608 | 167 | 32 |

**fsz 분포** (70.3%가 fsz∈[4,8)):

```
[ 0, 4)    852  11.6%  *****
[ 4, 8)   5181  70.3%  ****************************
[ 8,12)    806  10.9%  ****
[12,16)    276   3.7%  *
[16,20)    109   1.5%  *
[20,24)     69   0.9%
[24,28)     42   0.6%
[28,32)     37   0.5%
```

**nc 분포** (75.5%가 nc=2):

```
nc=1: 1429 (19.4%)
nc=2: 5563 (75.5%)
nc=8:  102 ( 1.4%)
```

**Top 5 (fsz, nc) 쌍** (소수의 패턴이 압도적):

| fsz | nc | uc | GEMM | flops/front | 개수 | 비율 |
|----:|---:|---:|------|------------:|----:|----:|
| 6 | 2 | 4 | 4×4×2 | 64 | 2248 | **30.5%** |
| 4 | 2 | 2 | 2×2×2 | 16 | 1119 | 15.2% |
| 5 | 2 | 3 | 3×3×2 | 36 | 851 | 11.5% |
| 3 | 1 | 2 | 2×2×1 | 8 | 527 | 7.2% |
| 8 | 2 | 6 | 6×6×2 | 144 | 361 | 4.9% |

→ 상위 5개 패턴이 **69.3%** 차지. 가장 흔한 trailing GEMM은 **(4×4×2)** = 32 FMAs.

### 2.3 mid tier (55 fronts)

| metric | min | max | mean | median |
|--------|----:|----:|----:|------:|
| fsz | 33 | 79 | 46.24 | **43** |
| nc | 2 | 8 | 6.91 | **8** |
| uc | 25 | 71 | 39.33 | 37 |
| uc²·nc | 1922 | 40328 | 22934 | 8192 |

**fsz 분포** (절반이 fsz∈[32,48)):

```
[32,40)  20  36.4%  ***************
[40,48)  13  23.6%  *********
[48,56)  10  18.2%  *******
[56,64)   7  12.7%  *****
[64,72)   4   7.3%  ***
[72,80)   1   1.8%  *
```

**nc 분포** — **78.2%가 nc=8** (panel_cap=8의 상한에 도달):

```
nc=2:  7 (12.7%)
nc=4:  2
nc=5:  1
nc=6:  1
nc=8: 43 (78.2%)  ← 매우 dominant
```

→ mid tier는 cap=8에 거의 sat. 평균 nc≈7. trailing GEMM K 차원 = 8 이 표준.

### 2.4 WMMA padding overhead (case8387 mid)

| metric | FP16 (K=16) | TF32 (K=8) | M/N (16) |
|--------|------------:|-----------:|--------:|
| mean padding | 56.8% | **13.6%** | 14.4% |
| median padding | 50.0% | **0.0%** | 12.5% |

→ **TF32는 case8387 mid의 78%에 padding 0%** (nc=8이 K=8 tile에 완벽 fit). FP16은 K=16 tile에 nc=8 → 50% padding 손실. docs/11 T1 TF32 권장의 정량 근거.

## 3. case_SyntheticUSA (n=156255, fronts=74196)

### 3.1 tier 합계

| tier | count | 비율 | trailing FLOP | FLOP 비율 |
|------|------:|----:|--------------:|----------:|
| small | 73247 | **98.72%** | 10.30 MF | 8.65% |
| mid | 886 | **1.19%** | 61.75 MF | **51.81%** |
| big | 63 | **0.08%** | 47.12 MF | **39.54%** |

→ 99.7% front (small+mid)이 small이지만 FLOP은 mid가 절반, big이 약 40%. **886+63 = 949개 front가 91% FLOP**. 최적화 ROI는 mid/big에 집중.

### 3.2 small tier (73247 fronts) — case8387과 거의 동일 분포

case8387의 small과 평균/중간값 모두 거의 일치:

| metric | small case8387 | small USA |
|--------|---------------:|----------:|
| fsz median | 6 | 6 |
| nc median | 2 | 2 |
| uc median | 4 | 4 |
| nc=2 비율 | 75.5% | 88.1% |

USA 쪽이 nc=2 비율 더 높음 (88% vs 76%). Top 5 (fsz, nc) 패턴도 유사:

| fsz | nc | uc | 개수 | 비율 |
|----:|---:|---:|----:|----:|
| 6 | 2 | 4 | 24250 | 33.1% |
| 4 | 2 | 2 | 21515 | 29.4% |
| 8 | 2 | 6 | 9045 | 12.4% |
| 3 | 1 | 2 | 6885 | 9.4% |
| 10 | 2 | 8 | 3318 | 4.5% |

→ 상위 5개가 **88.8%**.

### 3.3 mid tier (886 fronts)

| metric | min | max | mean | median |
|--------|----:|----:|----:|------:|
| fsz | 33 | 128 | 57.44 | 51 |
| nc | 1 | 20 | 13.37 | **14** |
| uc | 15 | 111 | 44.07 | 38 |
| uc²·nc | 1024 | 233280 | 69702 | 17298 |

**nc 분포** — **35.4%가 nc=20** (USA의 panel_cap이 더 크게 동작?):

```
nc∈[2, 4)   60 ( 6.8%)
nc∈[4, 6)   56 ( 6.3%)
nc∈[6, 8)   63 ( 7.1%)
nc∈[8,10)   81 ( 9.1%)
nc∈[10,12)  77 ( 8.7%)
nc∈[12,14)  70 ( 7.9%)
nc∈[14,16)  69 ( 7.8%)
nc∈[16,18)  51 ( 5.8%)
nc∈[18,20)  39 ( 4.4%)
nc=20      314 (35.4%)
```

→ USA의 mid는 case8387보다 **nc가 더 크고 다양**. nc=20에 sharp peak (separator hat).

**fsz 분포**:

```
[32, 40) 203 (22.9%)
[40, 48) 178 (20.1%)
[48, 56) 116 (13.1%)
[56, 64) 105 (11.9%)
[64, 72)  91 (10.3%)
[72, 80)  56 ( 6.3%)
[80, 88)  42 ( 4.7%)
[88, 96)  27 ( 3.0%)
[96,128) 68 ( 7.7%)
```

상대적으로 long tail.

### 3.4 big tier (63 fronts)

| metric | min | max | mean | median |
|--------|----:|----:|----:|------:|
| fsz | 129 | 235 | 158.75 | 155 |
| nc | 4 | 20 | 18.29 | **20** |
| uc | 109 | 215 | 140.46 | 137 |
| uc²·nc | 76176 | 924500 | 423015 | 353780 |

**nc 분포** — **85.7%가 nc=20**:

```
nc=4:  2 ( 3.2%)
nc=6:  2
nc=8:  3
nc=10: 1
nc=14: 1
nc=20: 54 (85.7%)
```

big tier도 nc=20이 절대 다수. Top 5 (fsz, nc) 모두 nc=20.

### 3.5 WMMA padding overhead (USA mid/big)

| tier | metric | FP16 (K=16) | TF32 (K=8) | M/N (16) |
|------|--------|------------:|-----------:|--------:|
| mid | mean | 41.8% | **24.1%** | 14.9% |
| mid | median | 37.5% | **16.7%** | 14.1% |
| big | mean | 39.3% | **18.5%** | 4.9% |
| big | median | 37.5% | **16.7%** | 4.9% |

nc=20 dominant → K-padding: TF32는 24-tile (KP=24, 20/24 = 16.7% padding), FP16은 32-tile (20/32 = 37.5% padding). TF32가 FP16 대비 **2.2배 좋은 padding 효율**.

## 4. 종합 비교: case8387 vs USA

### 4.1 FLOP 집중도

| | case8387 mid | USA mid | USA big |
|---|--------:|------:|------:|
| 비율 (count) | 0.74% | 1.19% | 0.08% |
| 비율 (FLOPs) | 50.6% | 51.8% | 39.5% |
| count당 FLOPs | 22.9 kFLOP | 69.7 kFLOP | 748 kFLOP |

→ count당 FLOP 비교: case8387 mid (22.9k) vs USA mid (69.7k) → **USA mid의 GEMM이 3배 큼**.

### 4.2 GEMM 차원의 표준 형태

| 케이스 | 가장 흔한 (M,N,K) | 빈도 |
|--------|------------------|-----|
| case8387 small | (4, 4, 2) | 30.5% |
| case8387 mid | nc=8 dominant → 다양한 uc 값 ((25..71) × (25..71) × 8) | 78.2% nc=8 |
| USA small | (4, 4, 2) | 33.1% |
| USA mid | nc=14 또는 20 dominant. 평균 (38, 38, 13) | 35.4% nc=20 |
| USA big | nc=20 dominant. 평균 (137, 137, 20) | 85.7% nc=20 |

### 4.3 WMMA tile 활용도 (16×16×K 단위)

front당 WMMA tile 개수 = ceil(M/16) × ceil(N/16) × ceil(K/Ktile):

| 케이스 | 대표 GEMM | FP16 (K=16) tiles | TF32 (K=8) tiles | tile 활용도 |
|--------|----------|------------------:|-----------------:|------------|
| case8387 mid median | (37, 37, 8) | 3·3·1 = 9 tiles | 3·3·1 = 9 tiles | 동일 |
| USA mid median | (38, 38, 14) | 3·3·1 = 9 tiles | 3·3·2 = 18 tiles | TF32가 K 2개 (padding 적음) |
| USA big median | (137, 137, 20) | 9·9·2 = 162 tiles | 9·9·3 = 243 tiles | TF32가 K 3개 |

→ TF32는 K-tile이 작아 (8 vs 16) tile 개수 자체는 더 많지만, padding이 작아 **실효 FLOP 활용률이 ~25-30% 높음**.

## 5. sync 비용 추정 (per-tier)

panel-LU의 sync 개수 ≈ 3·nc + 5 (docs/09 §1.1):

| tier | nc 중간값 | per-front 추정 sync | per-front 전형 FLOP | sync/FLOP 비율 |
|------|--------:|--------------------:|--------------------:|--------------:|
| case8387 small | 2 | 11 | 64 (4×4×2) | **0.17 sync/FLOP** |
| case8387 mid | 8 | 29 | 8192 (median) | 0.0035 sync/FLOP |
| USA small | 2 | 11 | 64 | 0.17 |
| USA mid | 14 | 47 | 17298 | 0.0027 |
| USA big | 20 | 65 | 353780 | 0.00018 |

→ **small tier는 sync-dominated** (FLOP 대비 sync 비율 0.17). 그러나 factor_small이 이미 `__syncwarp` 사용하므로 sync 비용 자체가 작음.

→ **mid/big tier는 FLOP-dominated** (sync는 0.001-0.0001 수준). 즉 mid/big에서 sync를 더 줄여도 wall 효과 작다는 docs/10 §9 메타-결론 정량 확인.

## 6. 함의 (implication for next round)

### 6.1 TC 전략 — TF32가 정량적으로 옳음

- **case8387 mid**: nc=8 dominant → TF32 K-padding 0% vs FP16 50%. TF32는 case8387에서 trailing FLOP의 1.5-2배 효과 가능 (padding 절감 직접 환산).
- **USA mid/big**: nc=20 dominant → TF32 16.7% vs FP16 37.5%. TF32가 1.3배 효과.
- 이미 docs/index에 `factor_mid_tf32`, `factor_big_tf32` 가 추가됨 (T1 ship). 실측 TF32 wall 결과는 별도 후속 측정 필요.

### 6.2 small tier 추가 최적화의 ROI

small tier는 USA에서 **count 98.7% / FLOP 8.7%** — 즉 wall 시간의 비중은 (kernel별 SOL을 무시한다면) FLOP 비율 8.7%보다 클 수 없음. ncu 측정에서 USA B=64 small kernel 비중이 21% (docs/11 §2)인 이유는 **launch/dispatch overhead와 occupancy 부족** — FLOP 외 비용. 그러나 절대 우선순위는 mid (FLOP 50%+) > big (FLOP 40%) > small (FLOP 8%).

### 6.3 sync 절감의 이론 ceiling

mid tier 한 front 처리에 sync ~29개. case8387 mid B=64 wall 약 1.58 ms / 55 fronts / 64 batch = 0.45 μs/front. ncu barrier stall 41% 라면 barrier에 약 0.18 μs/front 소요. sync 1개당 ~6 ns → 29 sync ≈ 0.17 μs. 일치.

이걸 다 제거해도 0.18 μs/front 만 회수 → 전체 factor wall 단축 약 11% 한계 (USA 같은 더 큰 nc=20에서는 더 큼). docs/10 §9의 측정 결과 (실제 −5~−6%) 가 이 ceiling의 절반 수준 — sync 자체는 줄여도 SM 활용도가 그만큼 못 따라잡음을 정량 확인.

### 6.4 panel_cap 의 의미

panel_cap=8 → case8387 mid nc 중간값이 8에 sat (78.2%가 nc=8). USA mid는 nc=20까지 spread (35% nc=20). 같은 cap=8인데 USA에서 nc가 더 큰 이유는 USA의 elimination tree separator가 더 크기 때문 (METIS ND가 더 큰 separator을 만들어냄).

cap을 16, 24 등으로 올리면:
- case8387 mid의 nc=8 sat이 풀려 nc=16 가능 → fsz 증가 → fewer mid fronts (다른 fronts 흡수)
- TF32 K-tile (=8) 기준 nc=16 → 0% padding 유지
- 단 nc 증가 시 per-front FLOP은 nc^3로 증가 가능 — 측정 필요 (docs/11 T3)

## 7. 재현

```bash
# 1) front 데이터 dump (한 번)
CLS_DUMP_FRONTS=/tmp/fronts_8387.csv \
  build/custom_linear_solver_run /datasets/.../case8387pegase \
  --precision fp32 --batch 1 --batch-only --repeat 1

# 2) 히스토그램 분석
python3 docs_supplementary/front_gemm_hist.py /tmp/fronts_8387.csv case8387pegase
```

원본 데이터: `/home/claude/prof/fronts_8387.csv`, `/home/claude/prof/fronts_USA.csv`.
스크립트: `/home/claude/prof/front_gemm_hist.py`.
전체 출력: `/home/claude/prof/front_gemm_hist.out`.

## 8. 관련 문서

- 측정 기반: [`docs/11 — FP32 factorize GEMM vs non-GEMM`](11-fp32-factorize-gemm-vs-nongemm-2026-06-06.md) §3 (이론 FLOP 분해), §6.2 (tile geometry)
- TC 결정 root: [`docs/03-optimization-notes/06 — TC dedicated path study`](../03-optimization-notes/archive/06-tc-dedicated-path-study.md) (FP16 negative result)
- 비-GEMM 최적화 회고: [`docs/03-optimization-notes/10`](../03-optimization-notes/10-t4.1-t4.3-results-2026-06-06.md) §9 (meta-결론)
- amalgamation 회귀 history: [`docs/03-optimization-notes/07`](../03-optimization-notes/archive/07-symbolic-gemm-research.md) §10 (cap≥16 +72% 회귀)

---

## 9. tier별 factorize 구조 시각화

### 9.1 small tier — warp-per-front, fused right-looking LU

```
┌─ block: 256 thread = 8 warps ─────────────────────────────────────┐
│                                                                    │
│   warp 0      warp 1      warp 2      ...      warp 7              │
│   (front 0)   (front 1)   (front 2)            (front 7)           │
│      │           │           │                    │                │
│      ▼           ▼           ▼                    ▼                │
│   Fs[0..]    Fs[1..]     Fs[2..]              Fs[7..]              │
│   (per-warp shared, fsz²·4 bytes each = ~144B for fsz=6)           │
│                                                                    │
│   Each warp (32 lanes) processes ITS OWN front independently:      │
│                                                                    │
│   stage_in:   lane-coalesced read F → Fs                           │
│   __syncwarp                                                       │
│                                                                    │
│   ╔═══ FUSED right-looking LU ════════════════════════╗            │
│   ║                                                    ║            │
│   ║  for k in 0..nc-1:                                ║            │
│   ║      piv = Fs[k][k]   ← lane broadcast            ║            │
│   ║      for i in (k+1+lane; <fsz; +32):              ║            │
│   ║          Fs[i][k] /= piv         (divide column)  ║            │
│   ║      __syncwarp                                   ║            │
│   ║      for e in (lane; <m²; +32):  m = fsz-k-1      ║            │
│   ║          Fs[ii][jj] -= Fs[ii][k] · Fs[k][jj]      ║            │
│   ║          (FULL trailing — LU + U + GEMM merged)   ║            │
│   ║      __syncwarp                                   ║            │
│   ║                                                    ║            │
│   ╚════════════════════════════════════════════════════╝            │
│                                                                    │
│   writeback factored L/U → F (global)                              │
│   extend-add CB (Fs[(nc+a)*fsz+(nc+b)]) → parent (atomicAdd)       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
  per-front sync count: nc      (case8387/USA median nc=2 → 2 syncs)
  per-front compute:   ~nc·m²   (median 2·5² ≈ 50 FMAs/lane)
  per-warp shared:     fsz²·4   (median 144 B → cache-resident)
```

추상화: **"하나의 front을 32 lane이 통째로 처리, sync는 warp barrier로만"**. 한 block에 8개 front 동시 in-flight. 대부분 front가 너무 작아 lane의 1/2 가 idle하지만, **warp barrier가 block barrier 보다 ~8배 싸기 때문에** 점유율 손실보다 sync 비용 절감이 큼. nc=2 인 75-88%의 front에서 outer loop 2회로 끝나므로 LU + U-solve + trailing GEMM의 phase 분리가 의미 없어 fused 형태가 유리.

### 9.2 mid tier — block-per-front, split 3-phase + shared-staged trailing

```
┌─ block: 256 thread = 8 warps ─ ONE (front, batch) ────────────────┐
│                                                                    │
│   gridDim = (level_size, B)  → 한 block 당 한 (front, batch)        │
│                                                                    │
│   ┌── shared memory layout (≤ 96 KB) ─────────────────┐           │
│   │   Fs[fsz_cap²]      ← 전체 front 복사본          │           │
│   │   sh_L[uc·nc]       ← 좌측 L panel staging       │           │
│   │   sh_U[nc·uc]       ← 우측 U panel staging       │           │
│   └────────────────────────────────────────────────────┘           │
│                                                                    │
│   stage_in:  256 thread → cp.async F → Fs (T4.3)                  │
│   __syncthreads                                                    │
│                                                                    │
│   ┌─ Phase 1: lu_panel_factor ────────────────────────┐           │
│   │  for k in 0..nc-1:                                 │           │
│   │      divide column k    (256 lanes parallel)       │           │
│   │      __syncthreads                                 │           │
│   │      rank-1 update of (fsz-k-1)·(nc-k-1) panel block│          │
│   │      __syncthreads                                 │           │
│   │  → 2·nc syncs (T4.2.A로 nc로 줄임 if nc≤12)       │           │
│   └────────────────────────────────────────────────────┘           │
│   ┌─ Phase 2: u_panel_solve ──────────────────────────┐           │
│   │  for k in 1..nc-1:                                 │           │
│   │      row k trailing solve   (uc lanes parallel)    │           │
│   │      __syncthreads                                 │           │
│   │  → nc-1 syncs                                      │           │
│   └────────────────────────────────────────────────────┘           │
│   ┌─ Phase 3: trailing GEMM (staged) ─────────────────┐           │
│   │  stage L → sh_L  &  stage U → sh_U                │           │
│   │  __syncthreads                                     │           │
│   │  for e in (t; <uc²; +nt):    256 lanes parallel    │           │
│   │      C[i][j] -= Σ_k sh_L[i][k]·sh_U[k][j]          │           │
│   │  → 2 syncs (stage, GEMM-end)                       │           │
│   └────────────────────────────────────────────────────┘           │
│                                                                    │
│   writeback L/U → F (global)                                       │
│   extend-add CB → parent (atomicAdd)                               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
  per-front sync count: ~3·nc + 5  (case8387 nc=8 → 29, USA nc=20 → 65)
  per-front compute:    ~uc²·nc (median 38²·14 ≈ 20k FMAs in USA)
  shared usage:         (fsz_cap + 2·uc·nc)·4 bytes (~30 KB typical)
```

추상화: **"front 전체를 shared에 stage in 한 후 3 phase로 분리"**. front 크기가 커서 한 warp(32 lane)로 부족 → 256 thread가 한 front를 협력 처리. 대신 phase 경계마다 cross-warp `__syncthreads` 필요. 핵심 설계 결정: 
1. **front 전체 stage-in**: trailing GEMM 의 K 차원이 nc만큼 작아 staging 비용을 trailing이 amortize
2. **panel/trailing 분리**: panel LU는 직렬 (nc 직렬), trailing GEMM은 병렬 (uc²) — 분리해야 두 phase가 각각 최적화 가능 (T1 TF32 WMMA가 trailing에만 적용됨)

### 9.3 big tier — block-per-front, global-memory direct

```
┌─ block: 1024 thread = 32 warps ─ ONE (front, batch) ──────────────┐
│                                                                    │
│   gridDim = (level_size, B)                                        │
│                                                                    │
│   front size 너무 큼: fsz² · 4 > 96 KB shared budget               │
│   (USA big median fsz=155 → 96 KB, max fsz=235 → 220 KB)           │
│                                                                    │
│   → shared staging 포기, 모든 phase가 global memory F를 직접 RW    │
│                                                                    │
│   F (global, in front_arena[b·front_total + front_off[p]])          │
│      ↕                                                              │
│      ↕  Phase 1: lu_panel_factor on global F                       │
│      ↕              (256 lanes·write-through → DRAM bound)         │
│      ↕  __syncthreads (×2·nc)                                      │
│      ↕                                                              │
│      ↕  Phase 2: u_panel_solve on global F                         │
│      ↕  __syncthreads (×nc-1)                                      │
│      ↕                                                              │
│      ↕  Phase 3: trailing on global F                              │
│      ↕     TF32: 1024 thread WMMA + shared scratch for fragments   │
│      ↕     scalar: 1024 thread loop over uc² · nc                  │
│      ↕                                                              │
│   extend-add CB → parent (atomicAdd)                               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
  per-front sync count: ~3·nc + extras  (USA big nc=20 → 65+ syncs)
  per-front compute:    ~uc²·nc (median 137²·20 ≈ 376k FMAs)
  global traffic:       O(nc · fsz²) reads/writes (vs mid의 O(fsz²) stage-in only)
```

추상화: **"front이 shared에 안 들어가 global을 그대로 사용"**. 1024 thread block로 가능한 한 parallelism을 끌어올리지만 global memory traffic 비싸짐. TF32 WMMA는 fragment에 한해 shared 사용 (작은 trailing 스크래치). USA에서만 활성 — case8387은 max_fsz=79로 mid 영역에 머무름.

---

## 10. tier별 wall 비중과 ncu 병목 (fp32 측정, 종합)

### 10.1 factorize wall 차지 비율 (nsys median, refactored nograph build)

| 케이스 | B | small | mid | big | factor 합 |
|--------|--:|------:|----:|----:|----------:|
| case8387 | 1 | 15% (5μs) | **85%** (30μs) | 0 | 35 μs/call |
| case8387 | 64 | **40%** (632μs) | **60%** (944μs) | 0 | 1.58 ms/call |
| USA | 1 | 4.6% (138μs) | 19% (573μs) | **76%** (2305μs) | 3.0 ms/call |
| USA | 64 | 21% (6.3ms) | **47%** (14.1ms) | 32% (9.8ms) | 30.3 ms/call |

→ 최적화 ROI 의 절대 우선순위:
- **case8387**: B=1 → mid (85%), B=64 → mid+small (둘 다 큼)
- **USA**: B=1 → big (76%), B=64 → mid (47%) > big (32%) > small (21%)

### 10.1.1 dispatch 입자 — level 수 vs front 수

`issue_factor_level_range` 는 **레벨 단위로** kernel을 정한다 (각 레벨 안의 모든 front가 같은 kernel로 dispatch). 따라서 "small tier"의 의미는 두 가지로 구분:

| | case8387 | USA |
|---|--------:|----:|
| 총 레벨 수 | 29 | 40 |
| **순수 small 레벨** (max_fsz ≤ 32) | **7** | **5** |
| 그 안의 fronts | 5604 | 68526 |
| 전체 small front 중 비율 | 76.0% | 93.6% |
| 나머지 small front (mid 레벨에 섞임) | 1768 (24%) | 4721 (6.4%) |
| mid 레벨 / big 레벨 | 22 / 0 | 14 / 21 |

→ small front 의 76% (case8387) / 94% (USA) 는 max_fsz≤32 인 **순수 small 레벨**에 모여 있어 `factor_small` 로 처리. 나머지는 mid/big level에 끼어 `factor_mid`/`factor_big` 으로 처리되며 §10.1 의 "small wall" 비중에는 포함되지 않음.

dispatch 단위가 per-level이라는 점은 §11.1 (multi-front packing) 의 sort 가 per-(subtree, level) 슬라이스 안에서만 동작하는 이유, 그리고 [`docs/13`](13-multistream-tier-impact-2026-06-06.md) 의 multi-stream이 subtree 단위로 fork하는 이유의 root.

### 10.1.2 레벨 내 tier mixing — small front가 factor_mid/big에 끼는 양

front-기준 tier (개별 fsz) 와 level-기준 tier (level의 max_fsz) 는 **반드시 일치하지 않는다**. 한 레벨에 fsz가 4 부터 79 까지 섞여 있을 수 있고, 이때 모든 fronts는 max_fsz 가 mid range 라 `factor_mid` 로 dispatched. **즉 작은 front가 큰 block에 묻혀 비효율적으로 처리됨**.

**전체 mixing 통계**:

| | case8387 | USA |
|---|--------:|----:|
| 총 레벨 수 | 29 | 40 |
| **mixed 레벨 수** (≥ 2개 tier 동시 존재) | **16 (55%)** | **21 (53%)** |
| three-way mixed (small+mid+big 동시) | 0 | 4 (L=12, 14, 16, 20) |

**front 입장에서 본 mis-routing**:

| | case8387 | USA |
|---|--------:|----:|
| `factor_small` 로 dispatched | 5604 (100% small) | 68526 (100% small) |
| `factor_mid` 로 dispatched | 1823 (그 중 **1768 = 97% 가 실제 small front**) | 5458 (그 중 4721 = 87% small) |
| `factor_big` 로 dispatched | 0 | 212 (그 중 7 small + 142 mid = **70% 가 mis-routed**) |

**대표적인 mixed 레벨 예시 (case8387)**:

```
L=2:  716 fronts → factor_mid (max=33)
        구성: small=715 (fsz 2..20), mid=1 (fsz=33)
        → 715개 작은 front가 mid 커널 (256 thread block, 전체 stage-in 등) 로 처리됨
L=7:  61 fronts → factor_mid (max=71)
        구성: small=57 (fsz 8..32), mid=4
L=17: 4 fronts → factor_mid (max=54)
        구성: small=1, mid=3
```

**USA 의 worst case (small이 factor_big 으로)**:

```
L=14: 27 fronts → factor_big (max=162)
        구성: small=3 (fsz∈{26,28,30}), mid=20, big=4
        → 1024-thread block × fsz=26 front (36 entry stage-in) = 99% lane idle
L=16: 13 fronts → factor_big (max=166)
        구성: small=1, mid=7, big=5
L=20: 7 fronts → factor_big (max=235)
        구성: small=1 (fsz=18), mid=1, big=5
        → fsz=18 front을 1024 thread로 처리
```

### 10.1.3 mixing의 비용

mis-routed front 의 단위 비용 (개략):
- **small→mid 미스라우팅**: 256 thread block, 전체 fsz² shared stage, 3-phase split LU (panel/U/trailing). fsz=6 front 처리 시 lane 244 idle. per-front 비용 ~3-5x 의 factor_small.
- **small/mid→big 미스라우팅**: 1024 thread block, global memory 직접 RW. fsz=26 front 처리 시 lane 988 idle. per-front 비용 ~10x.

case8387 mid 레벨에서 처리되는 1768 small fronts 의 비용:
- 만약 별도 dispatch 가능하다면 factor_small 로 0.5 μs/front 정도
- 현재 factor_mid 에서는 5-10 μs/front (추정, ncu 측정 안 함)
- 차이 ~10 μs/front × 1768 fronts ≈ **17 ms의 잠재 절감**, 단 B=1 기준; B=64 에선 launch overhead 분산되어 효과 다름

### 10.1.4 잠재 최적화 — per-level tier split

`issue_factor_level_range` 가 한 레벨을 **fsz 기준으로 2-3 sub-range로 쪼개** 각각 별도 kernel dispatch 하면 mis-routing 제거 가능:
- 한 레벨 안 small 부분 → factor_small
- 한 레벨 안 mid 부분 → factor_mid
- 한 레벨 안 big 부분 → factor_big

전제 조건:
1. 같은 레벨 안 fronts 는 **서로 의존성 없음** (parent-child 관계 X) — 이미 보장됨 (level 정의)
2. 각 sub-range 가 contiguous in plcols 여야 함 → 사전 정렬 필요 (`docs/11` deprecated packing 의 sort 와 같은 메커니즘 재활용 가능)
3. mis-routed front 수가 충분히 많아 launch overhead 보다 이득이 커야 함

**ROI 추정**:
- case8387: 1768 small fronts in mid levels. 그러나 mid 자체 wall 이 60% factor (B=64) → 그 안 small 비중은 작음. 추정 −1~−3% factor wall.
- USA: 7+142 = 149 small/mid fronts in big levels. big 자체가 32-76% factor wall 차지 → 잠재 더 큼. 추정 −2~−5%.

작은 잠재. 단 sort 가 packing (docs/11) 처럼 multi-stream 과 충돌하지 않게 per-(subtree, level) 슬라이스 내에서만 정렬해야 함. 그리고 sub-range별 launch가 늘어 multi-stream 의 grid coalescing 이 약해질 수 있음 — packing 실험에서 본 launch fragmentation 회귀 (B=1 +120%) 가 재현될 위험.

→ **deferred** — `docs/13` 의 multi-stream 분석 + `docs/11` 의 packing 실패 합쳐 보면, "per-level tier split" 은 launch overhead 측면에서 작은 risk 의 큰 잠재 win 이 아닐 가능성 큼. 다른 lever (TF32 trailing, scatter_values 등) 우선.

### 10.2 ncu kernel-level bottleneck (fp32, factor_X<float>, B=1/64)

값은 ncu `.pct`(상대) 기준, 1순위/2순위만 표시:

| tier | 케이스 | B | dur/launch | SOL_SM | Occ% | 1순위 stall | 2순위 |
|------|--------|--:|----------:|------:|----:|------------|-------|
| **small** | 8387 | 1 | 13 μs | 7% | 16% | **long_scoreboard** (mem latency) | wait |
| small | 8387 | 64 | 104 μs | 31% | 38% | **wait** (FMA latency) | long_scoreboard |
| small | USA | 1 | 30 μs | 31% | 40% | **long_scoreboard** | wait |
| small | USA | 64 | 1447 μs | 49% | 51% | **wait** ≈ long_scoreboard | not_selected |
| **mid** | 8387 | 1 | 13 μs | 6.5% | 29% | **long_scoreboard** | barrier ≈ wait |
| mid | 8387 | 64 | 159 μs | 47% | 70% | **barrier** | wait |
| mid | USA | 1 | 47 μs | 20% | 36% | **wait** | barrier |
| mid | USA | 64 | 1040 μs | 43% | 45% | **wait** | barrier |
| **big** | USA | 1 | 110 μs | 5.5% | 66% | **barrier** | not_sel ≈ wait ≈ scoreboard ≈ math |
| big | USA | 64 | 876 μs | 42% | 66% | **barrier** | scoreboard ≈ wait |

### 10.3 병목 패턴 요약

- **B=1 모든 tier**: `long_scoreboard` (global memory latency) 지배. occupancy 16-40%로 latency hide 불가. **cp.async가 직접 타깃**.
- **B=64 small**: `wait` (fixed-latency FMA) — fused LU의 짧은 dependency chain. barrier=0 (warp-only).
- **B=64 mid**: `barrier` 지배 (8387) 또는 `wait` 지배 (USA — nc=20에서 GEMM 계산 자체가 길어 wait가 sync보다 큼).
- **모든 big**: `barrier` 압도적 — 1024 thread block에서 sync 한 번 비용이 매우 큼.

---

## 11. tier별 최적화 여지 분석

### 11.1 small tier (case8387 B=64 ~40%, USA B=64 ~21% wall)

**병목**: B=1 mem latency, B=64 FMA dependency chain (warp 내 직렬 → wait stall).

**여지**:
1. **Multi-front-per-warp packing** — 73-88%의 front가 nc=2/fsz≤8. 32-lane warp 안에 2개 front를 packing해 lane 활용률 ↑. wait stall (dep chain)도 단축. *추정 −10~15% on case8387 B=64.* 위험: 코드 복잡, lane 분리 로직.
2. **nc=2 specialization** — 75-88% front 의 nc=2 패턴을 outer-loop unroll한 fast path. *추정 −5~10%.* 위험: 코드 중복.
3. **Extend-add atomic 완화** — 형제 front들이 같은 parent에 atomic-add. warp-내 부분 합 → global atomic 1회로 contention 감소. *추정 −3~5%.*
4. **Selective cp.async** — fsz≤8은 overhead로 회귀, fsz>16에서만 활성. *추정 ~0% (fsz>16 front 5% 미만).*

→ **case8387 B=64에서 small 최적화는 잠재 −10~15%**. 전체 factor wall로는 −4~6% (small=40% × ~15%).

### 11.2 mid tier (case8387 B=1 85%, B=64 60%, USA B=64 47% wall)

**병목**: 8387 B=64 barrier, USA wait (큰 nc의 FMA chain).

**여지**:
1. **TF32 trailing GEMM** (T1, 이미 ship): nc=8 case8387에서 K-padding 50% → 0% 제거. *추정 −20~30% trailing wall = −2~5% factor wall.*
2. **Grouped GEMM over B** (T2): B=64 mid에서 batch 방향 GEMM fusion. *추정 −10~20% on USA B=64.* 미실시.
3. **Panel cap sweep** (T3): cap 8→16~24로 nc 키워서 trailing K util ↑. case8387 mid의 78% nc=8에서 nc=16 으로 가면 trailing efficiency 증가. 단 Σ.5 amalgamation 회귀 history 주의.
4. **T4.2 sub-block partial sync** (docs/10 §9 deferred): barrier 41%를 절반 정도 줄일 잠재 — 단 docs/10에서 row-fused 시도 결과 sync 절감이 wall로 변환 안 됨 (occupancy ceiling).
5. **mid_warp (T4.1 + variance gate)**: USA B=64 −4% (이미 opt-in 가능).

→ **mid 최적화의 메인 lever 는 GEMM (T1 TF32 + T2 grouped)**. 비-GEMM 추가 줄이기는 docs/10 결론대로 ROI 낮음.

### 11.3 big tier (USA B=1 76%, B=64 32% wall)

**병목**: barrier 압도적 (1024 thread × 65+ syncs).

**여지**:
1. **Shared staging for big** — fsz>96 KB이라 전체는 못 들어가지만, **panel column block + L/U strip만 shared로 stage**. global RW를 panel 영역만 shared로 격하 → wait stall ↓. 위험: shared budget 정교한 관리.
2. **Block size 축소** (1024 → 512 / 256): warps_per_block ↓ → barrier 비용 ↓. 단 compute per-front도 ↓하므로 wall 단축 미지. 측정 필요.
3. **TF32 big trailing** (이미 factor_big_tf32 ship): K-padding 50% → 16.7%. *추정 −20~25% big trailing wall, USA B=1 (big 76%) 전체로는 −15~20%.*
4. **Per-front persistent + cp.async cross-front overlap** (T4.4 deferred): big 25개 (USA) 만 처리 → persistent 1 SM당 ~3 front 처리 시 stage-in을 trailing과 overlap. 위험: spine-megakernel 회귀 history.

→ **big 최적화는 USA B=1의 −15%+ 잠재**. TF32 big 측정이 1순위, 그 이후 panel shared staging.

### 11.4 ROI 종합

전체 factor wall 단축 잠재 (가장 큰 sub-bottleneck × 최대 단축률):

| 케이스 | B | dominant tier | 잠재 단축 (factor wall) | 핵심 lever |
|--------|--:|:-------------|:----------------------:|:----------|
| case8387 | 1 | mid (85%) | **−5 ~ −10%** | T1 TF32 trailing |
| case8387 | 64 | mid (60%) + small (40%) | **−10 ~ −20%** | T1 TF32 + small packing |
| USA | 1 | big (76%) | **−15 ~ −20%** | TF32 big + shared staging |
| USA | 64 | mid (47%) + big (32%) | **−10 ~ −15%** | T1 TF32 + T2 grouped GEMM |

전체 진단: **T1 (TF32 trailing GEMM)이 모든 케이스에 걸쳐 핵심**. 이미 코드 ship됨, 실측 평가만 남음.

