# FP32 factorize 의 GEMM 비중 분석 — TC 가속 ceiling

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: trailing GEMM 은 이론 FLOP 으로는 factor 의 86~88% 지만 실측 wall 로는 21~43% 뿐 — trailing 의 GPU 실행 효율이 평균의 2.3× 라서, TC (~2× trailing 가속) 의 wall ceiling 은 1.15~1.27× (최대 ~15%) 에 불과하다.

목적: **"trailing GEMM 이 factor 시간의 얼마나 차지하는가"** 를 정량화하여, TC (WMMA) 가속 적용 시 wall time 상한을 보임. 이 문서가 **GEMM/TC wall fraction 과 TC speedup ceiling 의 canonical 근거** 다.

대상: case8387 (n=14908), USA (n=156255). batch size 1 ~ 256.
방법: 이론 FLOPS 분석 + nsys per-kernel 실측 + skip-trailing variant 직접 측정.

**크기 분류 기준** (`src/batched/multifrontal_batched.cu`):
```cpp
SMALL_THRESH = 32                              // small_warp kernel 경계
MID_THRESH   = (TC32) ? 128 : 159              // mid vs big 경계 (path 별 다름)
```
LEVEL 의 max_fsz 기준 routing:
- **small**: `max_fsz ≤ 32` → small_warp
- **mid**: `32 < max_fsz ≤ 159` (FP32) → mid_tc32 / mid_tiled (shared-resident block kernel)
- **big**: `max_fsz > 159` → extend_level (global-mem 1024-thread)

case8387 max_fsz = 80 → BIG 0개. USA max_fsz = 254 → BIG 25개.

자세한 임계값 근거는 [`05-tier-thresholds.md`](05-tier-thresholds.md).

---

## 1. 이론 — 한 panel 의 work 분해

frontal matrix 한 panel (fsz × fsz, pivot cols nc, CB rows uc = fsz − nc) 의 factor work 3 단계:

| 단계 | FLOPs | 설명 |
|---|---|---|
| **LU panel factor** | nc² × fsz / 2 | nc 회 col 제거 + intra-panel rank-1 |
| **U panel solve** | nc² × uc / 2 | U panel back-substitution |
| **Trailing GEMM** | 2 × uc² × nc | C(uc×uc) -= L(uc×nc) × U(nc×uc) |

전형 case (fsz=80, nc=20, uc=60): LU = 16,000 / Usolve = 12,000 / **Trailing = 144,000 flops (84%)**.

---

## 2. 실제 panel 분포 — 두 case 의 GEMM FLOPS 비중

per-panel (fsz, nc, uc) 를 `CLS_PANEL_DUMP=1` 로 dump, Python (`tests/gemm_fraction.py`) 집계.

### 2.1 case8387 (P = 7433)
| 카테고리 | #panels | LU% | U-solve% | **Trailing%** | 카테고리 비중 |
|---|---:|---:|---:|---:|---:|
| small (fsz ≤ 32) | 7373 | 11.8% | 7.8% | **80.4%** | 49.5% |
| mid (32 < fsz ≤ 159) | 60 | 4.7% | 3.9% | **91.3%** | 50.5% |
| big (fsz > 159) | 0 | — | — | — | 0.0% |
| **TOTAL** | **7433** | 8.2% | 5.8% | **86.0%** | **100%** |

총 FLOPs = 3.1 × 10⁶ flops/system. Trailing GEMM = **86.0%**.

### 2.2 USA (P = 74285)
| 카테고리 | #panels | LU% | U-solve% | **Trailing%** | 카테고리 비중 |
|---|---:|---:|---:|---:|---:|
| small (fsz ≤ 32) | 73375 | 10.1% | 6.9% | **83.0%** | 9.6% |
| mid (32 < fsz ≤ 159) | 885 | 7.7% | 5.8% | **86.5%** | 69.5% |
| big (fsz > 159) | 25 | 3.1% | 2.8% | **94.2%** | 21.0% |
| **TOTAL** | **74285** | 7.0% | 5.3% | **87.8%** | **100%** |

총 FLOPs = 1.34 × 10⁸ flops/system. Trailing GEMM = **87.8%**.

**관찰**: 양쪽 case 모두 trailing GEMM 이 전체 factor FLOPs 의 **86 ~ 88%**. 나머지 LU+Usolve 는 13 ~ 14%.

---

## 3. 실측 — FP32 factor kernel time breakdown (nsys)

`MF_FP32=1` 로 FP32 batched mode, nsys 로 kernel-level 시간 측정.

### 3.1 case8387 — FP32 factor kernel time (μs/sys)
| B | small_warp | mid_tc32 (FP32 scalar) | mid_tiled (staged FP32) | TOTAL |
|---:|---:|---:|---:|---:|
| 1 | 703.5 (44.4%) | 618.2 (39.1%) | 261.1 (16.5%) | **1582.8** |
| 4 | 215.1 (50.6%) | 134.3 (31.6%) | 76.0 (17.9%) | 425.3 |
| 16 | 50.2 (55.1%) | 23.3 (25.6%) | 17.5 (19.3%) | 91.0 |
| 64 | 26.6 (57.5%) | 9.0 (19.5%) | 10.6 (23.0%) | 46.2 |
| 256 | 16.7 (65.8%) | 5.4 (21.2%) | 3.3 (13.1%) | 25.5 |

(B 클수록 small_warp 비중 증가 → batch dim 으로 GPU 채워 mid kernel 효율 상승)

### 3.2 USA — FP32 factor kernel time (μs/sys)
| B | BIG (extend_level) | mid_tiled | mid_tc32 | small_warp | TOTAL |
|---:|---:|---:|---:|---:|---:|
| 1 | 2627.8 (44.3%) | 2268.3 (38.3%) | 650.4 (11.0%) | 379.7 (6.4%) | **5926.3** |
| 4 | 672.2 (38.7%) | 651.9 (37.5%) | 222.2 (12.8%) | 192.1 (11.0%) | 1738.4 |
| 16 | 206.3 (28.0%) | 296.0 (40.2%) | 111.5 (15.1%) | 122.9 (16.7%) | 736.7 |
| 64 | 138.5 (28.2%) | 204.8 (41.7%) | 38.4 (7.8%) | 109.1 (22.2%) | 490.8 |
| 256 | 106.0 (25.6%) | 159.6 (38.6%) | 43.5 (10.5%) | 104.4 (25.2%) | 413.5 |

(USA 는 BIG kernel 이 substantial — 작은 B 44%, 큰 B 도 25%)

---

## 3.5 직접 측정 — skip-trailing variant 로 본 trailing wall time ★

§3.1/3.2 의 nsys per-kernel 시간은 trailing 내부 비중을 *간접 추정* (이론 FLOPS × kernel time). **Σ.16-PROFILE**: 4 개 dominant FP32 kernel 의 trailing 제거 clone 만들어 **직접 측정** (`CLS_PROFILE_NO_TRAILING=1`).

### 3.5.1 NT (no-trailing) clone

| Original | NT clone | 변경 |
|---|---|---|
| `mf_factor_mid_tiled_b` | `mf_factor_mid_tiled_NT_b` | `trailing_update_staged` 제거 |
| `mf_factor_mid_tc32_b<false>` | `mf_factor_mid_tc32_NT_b` | `trailing_update_scalar` 제거 |
| `mf_factor_extend_level_b<float>` | `mf_factor_extend_level_NT_b` | `trailing_update_scalar` 제거 |
| `mf_factor_small_warp_b<float>` | `mf_factor_small_warp_NT_b` | `lu_small_front` → `lu_small_front_no_trailing` |

`src/tc/factor_no_trailing.cuh` (155 LOC). 결과는 WRONG factor — wall 측정 전용.

### 3.5.2 측정 결과 — **이론 FLOPS 분석 (§2) 와 큰 차이**

| B | case8387 full | (NT) | trailing | **trailing %** |
|---:|---:|---:|---:|---:|
| 1 | 400 | 247 | 153 | **38.3 %** |
| 4 | 133 | 76 | 57 | **43.1 %** |
| 16 | 46 | 30 | 16 | 35.0 % |
| 64 | 27 | 20 | 7 | 26.3 % |
| 256 | 23 | 18 | 5 | 21.1 % |

| B | USA full | (NT) | trailing | **trailing %** |
|---:|---:|---:|---:|---:|
| 1 | 2672 | 1652 | 1020 | **38.2 %** |
| 4 | 989 | 603 | 386 | **39.0 %** |
| 16 | 540 | 390 | 151 | 27.9 % |
| 64 | 477 | 349 | 128 | 26.8 % |
| 256 | 471 | 346 | 126 | 26.6 % |

**이론 (FLOPS): 86 ~ 88 %.  실측 (wall): 21 ~ 43 %.**

### 3.5.3 차이의 원인

이론은 "time ∝ FLOPs" 가정. 실제로는:
1. **Trailing 은 고병렬 (uc² 독립 outputs) → SM 효율 높음 → FLOPs 대비 적은 wall**
2. **LU panel factor 는 serial chain** (k 가 k−1 의존) → low occupancy → FLOPs 대비 많은 wall
3. **Stage + writeback + extend-add 의 memory traffic** 이 FLOPs 와 무관하게 wall 차지
4. **Kernel launch + sync overhead** 가 비례 분할 안 됨

per-step (case8387 B=1): Trailing FLOPs 86% 인데 wall 38% → trailing 의 wall/FLOP 효율이 평균의 **2.3 배**. 즉 non-trailing 1 FLOP 가 trailing 1 FLOP 의 **2.3 배 시간** 소요.

### 3.5.4 B 증가에 따라 trailing% 감소 — *왜*

B=1 → B=256 전체: **Trailing** 153 → 5 μs (**30× 감소**), **NT** 247 → 18 μs (**14× 감소**) → trailing 비중 38% → 21%.

원인: per-phase saturation 시점이 다름.

| Phase | Bottleneck | Saturation B | 큰 B 거동 |
|---|---|---|---|
| Stage / Writeback (F↔Fs) | memory bandwidth | 작은 B (~10) | plateau |
| Extend-add (atomicAdd) | atomic contention | 중간 B | contention 증가 |
| LU panel factor | serial dependency | 중간 B (~30-50) | 천천히 감소 |
| U panel solve | serial-ish | 중간 B | 천천히 감소 |
| **Trailing (C -= L*U)** | compute (high parallel) | 큰 B (> 256) | **계속 scaling** |

NT 의 stage/writeback BW-bound 부분이 큰 B 에서 dominant (case8387 B=256: stage 메모리 traffic 1.95 GB → ~30% wall). trailing 은 compute 로 계속 줄어듦. Per-block parallelism 도: trailing 은 thread 256 개 모두 활성 (100% 사용), LU 는 nc 개만 (nc=20 이면 ~8%). 큰 B 에서 multiple block 동시 실행으로 LU idle thread 가 보충되지만 trailing 은 원래 효율적이라 compute saturation 까지 추가 이득.

### 3.5.6 시사

1. **TC 가속의 ROI 는 작은 B (≤16) 에서 큼** — trailing 비중 35-43% 라서.
2. **큰 B (≥64) 에서는 stage/writeback memory BW 가 dominant lever** — TC 와 직교. BW 개선 (front_total compression, partial staging) 이 더 의미.
3. 이전 측정 "B=1 에서 TC 가 -17% win" 도 같은 메커니즘.

---

## 4. 결합 — kernel time × per-category trailing% → trailing wall time

각 kernel 의 panel category 기준 trailing FLOPS 비중을 곱해 trailing 실제 wall time 추정.

### 4.1 case8387 (small × 0.80, mid × 0.91)
| B | small | mid_tc32 | mid_tiled | **Trailing 합** | Factor total | **Trailing% of factor** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 562 | 562 | 237 | **1361** | 1583 | **86.0%** |
| 4 | 172 | 122 | 69 | **363** | 425 | **85.4%** |
| 16 | 40 | 21 | 16 | **77** | 91 | **84.6%** |
| 64 | 21 | 8 | 10 | **39** | 46 | **84.4%** |
| 256 | 13 | 5 | 3 | **21** | 26 | **82.4%** |

**Trailing 이 FP32 factor wall 의 82 ~ 86%** (FLOPS-weighted 추정).

### 4.2 USA (BIG × 0.94, mid × 0.87, small × 0.83)
| B | BIG | mid_tiled | mid_tc32 | small | **Trailing 합** | Factor total | **Trailing%** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2470 | 1973 | 565 | 315 | **5323** | 5926 | **89.8%** |
| 4 | 632 | 567 | 193 | 159 | **1551** | 1738 | **89.2%** |
| 16 | 194 | 257 | 97 | 102 | **650** | 737 | **88.2%** |
| 64 | 130 | 178 | 33 | 91 | **432** | 491 | **88.0%** |
| 256 | 100 | 139 | 38 | 87 | **364** | 413 | **88.0%** |

**Trailing 이 USA FP32 factor wall 의 88 ~ 90%** (FLOPS-weighted). 단 §3.5 의 *직접 측정* 은 21~43% — FLOPS-weighted 추정이 trailing 실행 효율을 과대 평가함을 §3.5.2 가 보임.

---

## 5. TC 가속의 wall 상한 (Amdahl) — **실측 f 기반 (§3.5)**

이전 (잘못된 추정): f = 0.86 (FLOPS 비중) → X=2 면 1.75x ceiling.
실측 f (wall 비중) = 0.21 ~ 0.43 (B 에 따라). 재계산:

| | f (실측) | X=2 ceiling | X=∞ ceiling | 현재 TC | 활용률 |
|---|---:|---:|---:|---:|---:|
| case8387 B=1 | 0.38 | **1.24×** | 1.62× | 1.10× | 42% |
| case8387 B=4 | 0.43 | 1.27× | 1.76× | 1.05× | 18% |
| case8387 B=64 | 0.26 | 1.15× | 1.36× | 1.03× | 21% |
| case8387 B=256 | 0.21 | 1.12× | 1.27× | 1.00× | 0% |
| USA B=1 | 0.38 | 1.24× | 1.62× | 1.21× | **88%** |
| USA B=4 | 0.39 | 1.24× | 1.64× | **1.21×** | **88%** |
| USA B=64 | 0.27 | 1.16× | 1.37× | 0.96× | (FP32 이김) |

**핵심 변화**:
- 이전 "X=2 면 1.75x" 는 *misleading* (FLOPS 가정).
- 실제 ceiling 은 **1.15-1.27× 수준** (훨씬 작음). ~2× trailing 가속 → ~15% wall max.
- **USA B=1/4 는 이미 ceiling 88% 활용** — 더 짜내기 어려움.
- case8387 큰 B 는 ceiling 자체가 1.12-1.15× → TC 의의 작음.

**WMMA 의 FP16 peak** = FP32 peak × 4. 실제 efficiency 30%~50% 면 effective X≈1.5-2.5 → factor wall **1.1-1.3× speedup 이 현실 상한**. 이전 추정이 틀린 이유: 이론 FLOPS 는 trailing 의 *알고리즘적* 비중을 봤지만 **GPU 의 trailing 실행 효율 (high parallelism) 이 평균보다 2.3× 높아** wall 비중이 FLOPS 비중보다 훨씬 낮음.

---

## 6. 실측 — 현재 TC vs FP32 의 wall 격차

`../05-reports/02-comprehensive-sweep.md` (factor + solve total):

| Case | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| case8387 | -9.1% | -9.7% | +5.0% | -8.5% | -4.9% |
| USA | -10.7% | **-17.5%** | -2.4% | +4.1% | +6.0% |

TC win 의 wall speedup: case8387 B=1 1.10×, USA B=4 **1.21×**. 현재 TC 가 wall 의 1.1 ~ 1.2× 가속, 이론 상한 1.4 ~ 2× 와 비교하면 **약 50% 의 잠재력만 활용**.

---

## 7. 왜 이론 ceiling 의 절반만 활용?

**현재 dispatch 의 한계**: mid_tiled_b (FP32 staged scalar) 가 대부분 mid front 처리하지만 WMMA 안 씀. WMMA 가 실제 fire 되는 곳: `mid_tc_lo<24>` (small mid fallback) + `extend_tc32_b` (BIG fronts). case8387 WMMA-fired panel ~20%, USA ~60%.

mid_tiled 도 WMMA 로 통합하면: case8387 1.10× → 잠재 1.4 ~ 1.6×, USA 1.21× → 잠재 1.7 ~ 1.8×. 이 격차는 *staged trailing 이 scalar 라서 WMMA 보다 빨라서 그렇게 dispatch* 한 결정 때문 — power-grid mid front (nc 10-30, uc 30-60) 에서 padding overhead 가 WMMA throughput 우위를 상쇄. 더 큰 front (fsz > 200) 일수록 WMMA 효율적.

---

## 8. 결론

### 8.1 정량 요약
| 항목 | case8387 | USA |
|---|---:|---:|
| **Trailing GEMM FLOPS 비중** | 86 % | 88 % |
| **Trailing time 비중 (직접 측정, §3.5)** | 21 ~ 43 % | 27 ~ 39 % |
| 실측 f 기반 TC ceiling (X=2) | 1.12 ~ 1.27× | 1.16 ~ 1.24× |
| 현재 측정 TC speedup | 1.10× | 1.21× |
| ceiling 활용률 | ~17 % | ~27 % |

### 8.2 설득 논점
1. **FP32 factor FLOP 의 85-90% 가 trailing GEMM** — 단 *wall* 로는 21~43% (실행 효율 2.3×).
2. **WMMA FP16 peak 가 FP32 peak 의 4×** 지만 실측 ceiling 은 1.15~1.27× (wall fraction 이 낮아서).
3. **현재 활용률 17-27%** — 미활용 영역 있으나 절대 ceiling 자체가 작음.
4. **USA-style 큰 grid 일수록 ROI 큼** (BIG fronts WMMA fire 비중 높음).

### 8.3 case8387 의 TC win 원천 — BIG 없는데 왜 가속되나

WMMA 가 fire 되는 위치 2 곳:
1. `mf_factor_extend_tc32_b` — BIG path (case8387 에 없음, USA 만)
2. **`mf_factor_mid_tc_lo_b<24>` — MID path fallback** (max_fsz < 48 OR shared overflow): case8387 의 fsz<48 mid level WMMA fire

TC dispatch:
```
max_fsz ≤ 32         → small_warp      (NO TC)
32 < max_fsz < 48    → mid_tc_lo<24>   ← WMMA fires (per-panel fsz≥24)
48 ≤ max_fsz ≤ 128   → mid_tiled_b     (NO TC, staged scalar)
max_fsz > 128        → extend_tc32_b   ← WMMA fires (BIG)
```

case8387 nsys (B=1 TC, 10 iter):
| Kernel | calls | μs/inst | total | WMMA? |
|---|---:|---:|---:|---|
| `mid_tc_lo<24>` | 50 | 13.6 | 0.68 ms | **YES** (10% of factor) |
| `mid_tiled_b` | 150 | 13.5 | 2.03 ms | NO (scalar) |
| `small_warp` | 60 | 9.3 | 0.56 ms | NO (scalar) |

case8387 TC win 구성 (~11%): mid_tc_lo<24> WMMA trailing ~3% + kernel granularity (multistream packing) ~5% + per-kernel overhead 차이 (mid_tc_lo 13.6 vs FP32 mid_tc32_b<false> 17.3 μs) ~3%. **대비 USA -22%**: BIG path extend_tc32_b WMMA 가 dominant (43% factor time × WMMA −39%/inst). case8387 은 그 lever 없음.

### 8.4 Σ.15 — FP16 register-blocked trailing GEMM 시도

`trailing_update_regblock_h16` (`src/tc/trailing_tiled.cuh`): Shared 에 L, U 를 FP16 stage, thread 별 4×4 FP32 accumulator tile, inner K loop = nc 회 (no padding, WMMA fragment 회피). `CLS_USE_REGBLOCK_H16=1`.

측정 (case8387 B=1):
| Mode | TOTAL μs | relres |
|---|---:|---:|
| default (mid_tiled_b FP32 scalar) | 685 | 2.0×10⁻³ |
| Σ.2 regblock_FP32 | **634 (-7.5%)** | 4.0×10⁻³ (OK) |
| Σ.15 regblock_h16 | 624 (-9.0%) | **2.9×10⁻² (15× 악화)** |

핵심: register-blocking 자체가 B=1 lever (-7.5%), FP16 추가 효과 marginal (-1.5%) 인데 precision loss 치명적 (relres 15× 악화, FP16 10-bit mantissa + 값 1e-3~1e3 범위 → 곱셈 underflow). B=64 에서 안 보였던 이유: batch 차원이 GPU 채워 per-thread efficiency 가 wall 결정 안 함. **결론: FP16 채택 안 함 (accuracy), Σ.2 regblock_FP32 는 B=1 영역 -7.5% win 으로 재고려 가치 (큰 B 에서는 무효 → B 기준 분기 필요).**

### 8.5 후속 lever
| Lever | 잠재 추가 win | 비용 |
|---|---|---|
| mid_tiled 를 WMMA staged 로 변경 | +5-15% factor | medium |
| nc 작은 panel 의 WMMA padding 회피 (16×16×8 fragment) | +5% | low |
| cuBLAS LtMatmul TF32 모드 | mid-front +5-10% | medium |

---

## 9. 작은 front 에 TC 사용 가능한가 — 측정 기반 negative

**cuBLAS sgemmGroupedBatched 의 MIN_FSZ 낮추기** (`CLS_USE_CUBLAS=1 CLS_CUBLAS_MIN_FSZ=0`): B=64 +8%, B=256 +11% **regression** — cuBLAS per-call overhead 가 amortize 안 됨.

**WMMA fragment 의 fundamental 한계**: fragment = 16×16×16 고정. nc=4, uc=4 panel: useful FMA = 64, WMMA 실행 = 4096 FMA cycle → **padding waste 98%**. TC 4× throughput 도 effective = 4 × 0.02 = 0.08× → scalar 보다 12× 느림.

**multi-panel packing** (4 panel 묶어 block-diagonal WMMA): useful 25%, waste 75% → per-panel (98%) 보다 좋지만 여전히 scalar 못 이김.

**직접 microbench** (`tests/wmma_pack_microbench.cu`, 8192 panels, RTX 3090 sm_86):
| uc, nc | Scalar GFLOPS | WMMA per-panel | packed K=2 | packed K=4 | 우승 |
|---|---:|---:|---:|---:|---|
| **4, 4** | **65** | 38 | 62 | 63 | scalar |
| **8, 4** | **258** | 124 | 216 | — | scalar |
| **8, 8** | **396** | 244 | 295 | — | scalar |
| 12, 8 | 329 | **386** | — | — | WMMA |
| 16, 8 | 446 | **495** | — | — | WMMA |
| **16, 16** | 630 | **985** ★ | — | — | WMMA −56% (full tile) |

실측이 이론 예측 ("packed 가 scalar 의 100× 느림") 보다 덜 비관적 (packing 으로 scalar 와 격차 0~25%). 이유: scalar 도 작은 panel 에서 occupancy 낮아 peak 의 일부만 활용. 그러나 **uc ≤ 8 영역에서는 여전히 scalar 가 packed WMMA 이김 25-100%**, uc ≥ 12 는 current dispatch 가 이미 WMMA fire.

**결론**: 작은 front 의 TC packing 은 measurement-validated negative. 진짜 lever = algorithmic 변경 (amalgamation → 큰 panel) 또는 scalar 유지. 현재 dispatch (small_warp scalar + mid_tiled scalar + WMMA only on small-mid fallback + extend_tc32 on BIG) 가 우리 분포에서 측정상 최적.

---

## 부록 A — Methodology

### A.1 이론 FLOPS
```python
LU_flops = nc * nc * fsz / 2
Usolve_flops = nc * nc * uc / 2
Trailing_flops = 2 * uc * uc * nc
```
`CLS_PANEL_DUMP=1` 로 per-panel (fsz, nc, uc) 수집 (`src/factorize/multifrontal.cu`).

### A.2 실측 kernel time
```
MF_FP32=1 nsys profile --trace=cuda --cuda-graph-trace=node ...
nsys stats --report cuda_gpu_kern_sum
```
Sum of kernel μs/sys ≠ wall (multistream concurrency factor).

### A.3 환경
- GPU: RTX 3090 (sm_86), 모든 lever 적용 (selinv OFF, multistream, Σ.1 staged), Median of 3 outer runs × `--repeat N` inner

### A.4 한계
- panel category ↔ kernel 매핑은 max_fsz_of_level 기준 — 카테고리 평균과 약간 다를 수 있음
- kernel 안 stage/writeback overhead 약 5-10% (메모리 bound) — trailing time 약간 overestimate 가능. 단 결론 ("trailing 이 dominant FLOP, 그러나 wall 은 21~43%") 에 영향 없음

---

## 관련 문서
- `../storyline.md` — 전체 서사 맥락
- [`05-tier-thresholds.md`](05-tier-thresholds.md) — SMALL_THRESH/MID_THRESH 근거
- [`../03-optimization-notes/03-tensor-core-investigation.md`](../03-optimization-notes/03-tensor-core-investigation.md) — TC32 negative result + 디자인
- [`../04-benchmarks-profiling/03-gemm-fraction-front-distribution.md`](../04-benchmarks-profiling/03-gemm-fraction-front-distribution.md) — front 분포 + tier별 wall
- [`01-why-custom-fast.md`](01-why-custom-fast.md) — GPU compute 가 leverage 아님 (TC ceiling 작은 이유의 상위 맥락)
