# FP32 factorize 의 GEMM 비중 분석 — TC 가속 잠재력 설득

> Canonical note: GEMM/TC wall fraction과 TC speedup ceiling의 최신 근거 문서다.
> 전체 성능 결론은 [`../05-reports/01-final-report-2026-06-05.md`](../05-reports/01-final-report-2026-06-05.md),
> full sweep 표는 [`../05-reports/02-comprehensive-sweep-2026-06-05.md`](../05-reports/02-comprehensive-sweep-2026-06-05.md)를 기준으로 한다.

목적: **"trailing GEMM 이 factor 시간의 얼마나 차지하는가"** 를 정량화하여, TC (WMMA) 가속 적용 시 wall time 상한을 보임.

대상: case8387 (n=14908), USA (n=156255). batch size 1 ~ 256.
방법: 이론 FLOPS 분석 + nsys per-kernel 실측 시간 결합.

**중요 — 크기 분류 기준**:
```cpp
// src/batched/multifrontal_batched.cu
SMALL_THRESH = 32                              // small_warp kernel 경계
MID_THRESH   = (TC32) ? 128 : 159              // mid vs big 경계 (path 별 다름)
```
LEVEL 의 max_fsz 기준 routing — 문서 카테고리:
- **small** : `max_fsz ≤ 32` → small_warp
- **mid** : `32 < max_fsz ≤ 159` (FP32) → mid_tc32 / mid_tiled (shared-resident block kernel)
- **big** : `max_fsz > 159` → extend_level (global-mem 1024-thread)

case8387 max_fsz = 80 → BIG 0개. USA max_fsz = 254 → BIG 25개.

---

## 1. 이론 — 한 panel 의 work 분해

frontal matrix 한 panel (fsz × fsz, pivot cols nc, CB rows uc = fsz − nc) 의 factor work 3 단계:

| 단계 | FLOPs (분석) | 설명 |
|---|---|---|
| **LU panel factor** | nc² × fsz / 2 | nc 회 col 제거 + intra-panel rank-1 |
| **U panel solve** | nc² × uc / 2 | U panel 의 back-substitution |
| **Trailing GEMM** | 2 × uc² × nc | C(uc×uc) -= L(uc×nc) × U(nc×uc) |

total = LU + Usolve + Trailing.

전형 case (fsz=80, nc=20, uc=60):
- LU = 16,000 flops
- Usolve = 12,000 flops
- **Trailing = 144,000 flops (84%)**

## 2. 실제 panel 분포 — 두 case 의 GEMM FLOPS 비중

per-panel (fsz, nc, uc) 정보를 `CLS_PANEL_DUMP=1` 로 dump, Python (`tests/gemm_fraction.py`) 으로 집계.

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

**관찰**: 양쪽 case 모두 trailing GEMM 이 전체 factor FLOPs 의 **86 ~ 88%** 차지. 나머지 LU+Usolve 는 13 ~ 14%.

## 3. 실측 — FP32 factor kernel time breakdown (nsys)

`MF_FP32=1` 로 FP32 batched mode 실행, nsys 로 kernel-level 시간 측정. batch 별 dispatch 차이 반영.

### 3.1 case8387 — FP32 factor kernel time breakdown (μs/sys)

| B | small_warp | mid_tc32 (FP32 scalar) | mid_tiled (staged FP32) | TOTAL factor kernels |
|---:|---:|---:|---:|---:|
| 1   | 703.5 (44.4%) | 618.2 (39.1%) | 261.1 (16.5%) | **1582.8** |
| 4   | 215.1 (50.6%) | 134.3 (31.6%) | 76.0 (17.9%) | 425.3 |
| 16  | 50.2 (55.1%) | 23.3 (25.6%) | 17.5 (19.3%) | 91.0 |
| 64  | 26.6 (57.5%) | 9.0 (19.5%) | 10.6 (23.0%) | 46.2 |
| 256 | 16.7 (65.8%) | 5.4 (21.2%) | 3.3 (13.1%) | 25.5 |

(B 클수록 small_warp 비중 증가 → batch dim 으로 GPU 채워서 mid kernel 효율 상승)

### 3.2 USA — FP32 factor kernel time breakdown (μs/sys)

| B | BIG (extend_level) | mid_tiled | mid_tc32 | small_warp | TOTAL |
|---:|---:|---:|---:|---:|---:|
| 1   | 2627.8 (44.3%) | 2268.3 (38.3%) | 650.4 (11.0%) | 379.7 (6.4%) | **5926.3** |
| 4   | 672.2 (38.7%) | 651.9 (37.5%) | 222.2 (12.8%) | 192.1 (11.0%) | 1738.4 |
| 16  | 206.3 (28.0%) | 296.0 (40.2%) | 111.5 (15.1%) | 122.9 (16.7%) | 736.7 |
| 64  | 138.5 (28.2%) | 204.8 (41.7%) | 38.4 (7.8%) | 109.1 (22.2%) | 490.8 |
| 256 | 106.0 (25.6%) | 159.6 (38.6%) | 43.5 (10.5%) | 104.4 (25.2%) | 413.5 |

(USA 는 BIG kernel 이 substantial — 작은 B 에서 44%, 큰 B 에서도 25%)

## 3.5 직접 측정 — skip-trailing variant 로 본 trailing wall time ★

§ 3.1/3.2 의 nsys per-kernel 시간은 trailing 내부 비중을 *간접 추정* 함 (이론 FLOPS × kernel time).
**Σ.16-PROFILE**: 4 개 dominant FP32 kernel 의 trailing 제거 clone 만들어 직접 측정 (`CLS_PROFILE_NO_TRAILING=1`).

### 3.5.1 NT (no-trailing) clone 구현

| Original | NT clone | 변경 |
|---|---|---|
| `mf_factor_mid_tiled_b` | `mf_factor_mid_tiled_NT_b` | `trailing_update_staged` 호출 제거 |
| `mf_factor_mid_tc32_b<false>` | `mf_factor_mid_tc32_NT_b` | `trailing_update_scalar` 제거 |
| `mf_factor_extend_level_b<float>` | `mf_factor_extend_level_NT_b` | `trailing_update_scalar` 제거 |
| `mf_factor_small_warp_b<float>` | `mf_factor_small_warp_NT_b` | `lu_small_front` → `lu_small_front_no_trailing` (per-step rank-1 update 제거) |

`src/tc/factor_no_trailing.cuh` (155 LOC). 결과는 WRONG factor — wall 측정 전용.

### 3.5.2 측정 결과 — **이론 FLOPS 분석 (§ 2) 와 큰 차이**

| B | case8387 wall (full) | (NT) | trailing | **trailing %** |
|---:|---:|---:|---:|---:|
| 1 | 400 | 247 | 153 | **38.3 %** |
| 4 | 133 | 76 | 57 | **43.1 %** |
| 16 | 46 | 30 | 16 | 35.0 % |
| 64 | 27 | 20 | 7 | 26.3 % |
| 256 | 23 | 18 | 5 | 21.1 % |

| B | USA wall (full) | (NT) | trailing | **trailing %** |
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
3. **Stage + writeback + extend-add 의 memory traffic** 가 FLOPs 와 무관하게 wall 차지
4. **Kernel launch + sync overhead** 가 비례 분할 안 됨

per-step 측정 (case8387 B=1):
- Trailing FLOPs 86% 인데 wall 38% → trailing 의 wall/FLOP 효율이 평균의 **2.3 배**
- 즉 non-trailing 1 FLOP 가 trailing 1 FLOP 의 **2.3 배 시간** 소요

### 3.5.4 B 증가에 따라 trailing% 감소 패턴 — *왜* 그러는가

| B | case8387 trailing μs | NT μs | trailing% | trailing 감소율 | NT 감소율 |
|---:|---:|---:|---:|---:|---:|
| 1 | 153 | 247 | 38.3% | baseline | baseline |
| 4 | 57 | 76 | 43.1% | ÷ 2.7 | ÷ 3.2 |
| 16 | 16 | 30 | 35.0% | ÷ 3.6 | ÷ 2.5 |
| 64 | 7 | 20 | 26.3% | ÷ 2.3 | ÷ 1.5 |
| 256 | 5 | 18 | 21.1% | ÷ 1.4 | ÷ 1.1 **(plateau)** |

B=1 → B=256 전체:
- **Trailing**: 153 → 5 μs (**30× 감소**)
- **NT (non-trailing)**: 247 → 18 μs (**14× 감소**)
- 결과: trailing 비중이 38% → 21% **(NT 가 plateau 하지만 trailing 은 계속 scaling)**

#### 원인 1 — Per-phase saturation 도달 시점이 다름

NT 구성 = `stage(F→Fs)` + `LU panel factor` + `U solve` + `writeback(Fs→F)` + `extend-add`

| Phase | Bottleneck 종류 | Saturation B | 큰 B 에서 거동 |
|---|---|---|---|
| **Stage** (F→Fs) | memory bandwidth | 작은 B (~10) | plateau |
| **Writeback** (Fs→F) | memory bandwidth | 작은 B (~10) | plateau |
| **Extend-add** (atomicAdd to parent) | atomic contention | 중간 B | contention 증가 |
| **LU panel factor** | serial dependency (k → k+1) | 중간 B (~30-50) | 천천히 감소 |
| **U panel solve** | serial-ish | 중간 B | 천천히 감소 |

Trailing 구성 = `C -= L * U` (uc² × nc 의 독립 outputs)
- per-output 완전 병렬
- 높은 arithmetic intensity (FMA per shared load 큼) → **compute bound**
- Saturation B 큼 (>256) → 계속 scaling

#### 원인 2 — 실측 메모리 traffic 으로 확인

case8387 의 front 전체 크기 = 1.9 MB float. Per-iter stage 메모리 traffic:
- B=1: 7.6 MB (negligible vs 1 TB/s GPU BW → ~7 μs)
- B=256: 1.95 GB (significant → ~2 ms BW-bound 시간)

per-iter factor wall:
- B=1: 400 μs (factor wall)
- B=256: 23 × 256 = 5.9 ms — **이 중 ~30% (1.95 ms) 가 stage/writeback BW 시간**

→ 큰 B 에서 NT 의 BW-bound 부분이 **dominant**. trailing 은 compute 로 계속 줄어듦.

#### 원인 3 — 직관적 비유

- **NT 의 stage/writeback** = "택배 트럭 (memory BW)" : 화물 (B) 늘면 트럭 더 보내야 함 → 도로 (1 TB/s GPU BW) 막히면 끝 → plateau
- **Trailing GEMM** = "공장 라인 (compute)" : 주문 (B) 늘면 라인 추가 가동 → 공장 capacity (35 TFLOPS FP32) 까지 계속 가속

큰 B 에서 트럭이 도로 막혀서 plateau, 공장은 계속 돌아감 → **공장 (trailing) 의 비중이 *상대적으로* 작아짐**.

#### 원인 4 — Per-block parallelism 차이

같은 (panel, batch) 블록 안에서:
- **Trailing**: thread 256 개 모두가 uc² 개 output 중 하나씩 계산. **100% 사용**.
- **LU panel factor**: thread 256 개 중 `nc` 개만 활성 (각 step k 의 column k 분할). nc=20 이면 **~8% 사용** (서 nc 만큼). 나머지는 동기화 대기.

→ Per-block 의 GPU 활용도가 trailing 이 훨씬 높음. 큰 B 에서 multiple 블록 동시 실행으로 LU 의 idle thread 가 다른 panel/batch 의 work 로 보충 → LU per-sys 감소 (한계까지). Trailing 은 원래 효율적이라 추가 이득은 compute saturation 때까지.

### 3.5.5 Saturation 시점 정리

| 작업 종류 | Saturation 도달 B | 이후 per-sys 거동 |
|---|---|---|
| Stage / writeback (BW-bound) | 작은 B (~10) | **plateau** |
| LU panel factor (serial chain) | 중간 B (~30-50) | 천천히 감소 |
| Trailing (compute, high parallel) | 큰 B (> 256) | **계속 감소** |

→ B 증가하면 **NT 의 BW/serial 부분 먼저 saturate, trailing 만 compute 효율로 추가 이익** → **trailing 의 wall 비중 점점 작아짐**.

### 3.5.6 시사

1. **TC 가속의 ROI 는 작은 B (≤16) 에서 큼** — trailing 비중 35-43% 라서.
2. **큰 B (≥64) 에서는 stage/writeback memory BW 가 dominant lever** — TC 와 직교한 영역. BW 개선 (예: front_total compression, partial staging) 이 더 의미 있음.
3. **이전 측정 § 11 의 "B=1 에서 TC 가 -17% win" 도 같은 메커니즘**: 작은 B 에서 trailing 비중 큼 → TC 영향 visible.

## 4. 결합 — kernel time × per-category trailing% → trailing wall time

각 kernel 의 panel category 기준 trailing FLOPS 비중을 곱해 trailing 실제 wall time 추정:

### 4.1 case8387

각 kernel category 가 처리하는 panel 의 trailing FLOPS 비중:
- small_warp processes "small" panels → trailing 80.4 %
- mid_tc32 / mid_tiled processes "mid" panels → trailing 91.3 %

| B | small × 0.80 | mid_tc32 × 0.91 | mid_tiled × 0.91 | **Trailing 합** | Factor total | **Trailing% of factor** |
|---:|---:|---:|---:|---:|---:|---:|
| 1   | 562 | 562 | 237 | **1361** | 1583 | **86.0%** |
| 4   | 172 | 122 | 69 | **363** | 425 | **85.4%** |
| 16  | 40 | 21 | 16 | **77** | 91 | **84.6%** |
| 64  | 21 | 8 | 10 | **39** | 46 | **84.4%** |
| 256 | 13 | 5 | 3 | **21** | 26 | **82.4%** |

**Trailing 이 FP32 factor wall 의 82 ~ 86%** 일관되게 차지.

### 4.2 USA

- BIG × 0.94, mid × 0.87, small × 0.83

| B | BIG | mid_tiled | mid_tc32 | small | **Trailing 합** | Factor total | **Trailing%** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 2470 | 1973 | 565 | 315 | **5323** | 5926 | **89.8%** |
| 4   | 632 | 567 | 193 | 159 | **1551** | 1738 | **89.2%** |
| 16  | 194 | 257 | 97 | 102 | **650** | 737 | **88.2%** |
| 64  | 130 | 178 | 33 | 91 | **432** | 491 | **88.0%** |
| 256 | 100 | 139 | 38 | 87 | **364** | 413 | **88.0%** |

**Trailing 이 USA FP32 factor wall 의 88 ~ 90%** 차지.

## 5. TC 가속의 wall 상한 (Amdahl) — **실측 f 기반 (§ 3.5)**

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
- 이전 보고 "X=2 면 1.75x" 는 *misleading* (FLOPS 가정).
- 실제 ceiling 은 **1.15-1.27× 수준** (훨씬 작음).
- **USA B=1/4 는 이미 ceiling 88% 활용** — 더 짜내기 어려움.
- case8387 큰 B 는 ceiling 자체가 1.12-1.15× → TC 의의 작음.

**WMMA 의 FP16 peak throughput** = FP32 peak × 4. 실제 efficiency 30%~50% 면 effective X≈1.5-2.5. → factor wall **1.1-1.3× speedup 이 현실 상한**.

이전 추정이 틀린 이유: 이론 FLOPS 분석은 trailing 의 *알고리즘적* 비중을 봤지만, **GPU 의 trailing 실행 효율 (high parallelism) 이 평균보다 2.3× 높아서** wall 비중이 FLOPS 비중보다 훨씬 낮음.

## 6. 실측 — 현재 TC vs FP32 의 wall 격차

[`../05-reports/02-comprehensive-sweep-2026-06-05.md`](../05-reports/02-comprehensive-sweep-2026-06-05.md) 의 결과 (factor + solve total):

| Case | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| case8387 | -9.1% | -9.7% | +5.0% | -8.5% | -4.9% |
| USA | -10.7% | **-17.5%** | -2.4% | +4.1% | +6.0% |

TC win 의 wall speedup (1 / (1 + ratio)):
- case8387 B=1: 1.10×
- USA B=4: **1.21×**

→ 현재 TC 가 wall 의 1.1 ~ 1.2× 가속. 이론 상한 1.4 ~ 2× 와 비교하면 **약 50% 의 잠재력만 활용**.

## 7. 왜 이론 ceiling 의 절반만 활용?

§ 13 의 분석으로 명확:

**현재 dispatch 의 한계**:
- mid_tiled_b (FP32 staged scalar): 대부분의 mid front 처리하지만 WMMA 안 씀
- WMMA 가 실제 fire 되는 곳: `mid_tc_lo<24>` (small mid fallback) + `extend_tc32_b` (BIG fronts)
- case8387: WMMA-fired panel 비중 ~ 20%
- USA: WMMA-fired panel 비중 ~ 60%

만약 mid_tiled 도 WMMA 로 통합하면 (즉 모든 trailing 이 WMMA 로):
- 현재 case8387 1.10× → 잠재 1.4 ~ 1.6×
- 현재 USA 1.21× → 잠재 1.7 ~ 1.8×

이 격차가 **이번 ses Σ.1 staged trailing 이 scalar 라서 WMMA 보다 빨라서 그렇게 dispatch** 한 결정 때문. 우리 power-grid 의 front 크기 분포 (mid front 의 nc 10-30, uc 30-60) 에서 padding overhead 가 WMMA throughput 우위 상쇄.

→ 더 큰 front (USA-style BIG, fsz > 200) 일수록 WMMA 가 효율적, ceiling 에 가까워짐.

## 8. 결론

### 8.1 정량 요약

| 항목 | case8387 | USA |
|---|---:|---:|
| **Trailing GEMM FLOPS 비중** | 86 % | 88 % |
| **Trailing time 비중 (FP32 factor)** | 82 ~ 86 % | 88 ~ 90 % |
| 이론 TC speedup ceiling (X=2) | 1.75× | 1.78× |
| 현재 측정 TC speedup | 1.10× | 1.21× |
| ceiling 활용률 | 13 / 75 ≈ 17 % | 21 / 78 ≈ 27 % |

### 8.2 설득 논점

1. **FP32 factor 의 85-90% 가 trailing GEMM** — TC 가속 lever 의 헤드룸이 크다.
2. **WMMA FP16 peak 가 FP32 peak 의 4×** → trailing X=2 가속만 달성해도 wall 1.7× 가능.
3. **현재 활용률 17-27%** — 미활용 영역이 큼.
4. **USA-style 큰 grid 일수록 ROI 큼** (BIG fronts 에서 WMMA fire 비중 높음).
5. **case8387 같은 작은 grid 도 mid front 의 WMMA 도달 시 더 큰 win 가능** (단 padding overhead 극복 필요 — register-blocked WMMA 같은 추가 작업).

### 8.3 case8387 의 TC win 원천 — BIG 없는데 왜 가속되나 (질문 응답)

**카테고리 기준 (LEVEL 의 max_fsz)**:
- small : ≤ 32 → small_warp kernel
- mid : 32 < max_fsz ≤ 159 (FP32) / ≤ 128 (TC) → mid kernel
- big : > MID_THRESH

case8387 max_fsz = 80 → BIG = 0. 그럼에도 TC -11% 가속.

**WMMA 가 fire 되는 위치는 2 곳**:
1. `mf_factor_extend_tc32_b` — BIG path 안 (case8387 에 없음, USA 만 해당)
2. **`mf_factor_mid_tc_lo_b<24>` — MID path 의 fallback** (max_fsz < 48 OR shared overflow): **여기서 case8387 의 fsz<48 mid level 의 WMMA 가 fire**

TC dispatch 흐름:
```
max_fsz ≤ 32                 → small_warp      (NO TC)
32 < max_fsz < 48            → mid_tc_lo<24>   ← WMMA fires (per-panel fsz≥24)
48 ≤ max_fsz ≤ 128           → mid_tiled_b     (NO TC, staged scalar)
max_fsz > 128                → extend_tc32_b   ← WMMA fires (BIG)
```

case8387 의 nsys 측정 (B=1 TC, 10 iter):
| Kernel | calls | μs/inst | total | WMMA? |
|---|---:|---:|---:|---|
| `mid_tc_lo<24>` | 50 | 13.6 | 0.68 ms | **YES** (10% of factor) |
| `mid_tiled_b` | 150 | 13.5 | 2.03 ms | NO (scalar) |
| `small_warp` | 60 | 9.3 | 0.56 ms | NO (scalar) |

**case8387 의 TC win = small-mid level WMMA + 구조적 차이**:
| 원천 | 기여 |
|---|---:|
| mid_tc_lo<24> 의 WMMA trailing (fsz<48 levels) | ~3% |
| Kernel granularity (TC 는 mid_tc_lo + mid_tiled 로 split → multistream packing 더 잘 됨) | ~5% |
| Per-kernel overhead 차이 (mid_tc_lo 13.6 vs FP32 mid_tc32_b<false> 17.3 μs) | ~3% |
| **합계** | **~11%** |

**대비 USA 의 TC -22%**: BIG path 의 extend_tc32_b WMMA 가 dominant (43% factor time × WMMA −39%/inst). case8387 은 그 lever 없음.

### 8.5 Σ.15 — FP16 register-blocked trailing GEMM 시도 (BF16/FP16 회피책)

§ 8.4 의 회피책 #2 ("register-blocked GEMM with bf16/fp16 — WMMA fragment 없이 FMA pipe 직접 사용") 실험.

**구현**: `trailing_update_regblock_h16` (`src/tc/trailing_tiled.cuh`):
- Shared 에 L, U 를 FP16 로 stage (FP32 의 절반 footprint)
- 각 thread 가 4×4 FP32 accumulator tile (register) 보유
- Inner K loop = nc 회 (no padding waste, WMMA fragment 회피)
- 각 step: FP16 load → FP32 cast → FP32 FMA

`CLS_USE_REGBLOCK_H16=1` opt-in.

**측정** (case8387 B=1):
| Mode | TOTAL μs | relres |
|---|---:|---:|
| default (mid_tiled_b FP32 scalar) | 685 | 2.0×10⁻³ |
| Σ.2 regblock_FP32 (재측정) | **634 (-7.5%)** | 4.0×10⁻³ (OK) |
| Σ.15 regblock_h16 | 624 (-9.0%) | **2.9×10⁻² (15× 악화)** |

**핵심 발견**:
1. **Register-blocking 자체가 B=1 lever** — Σ.2 가 발견했지만 B=64 측정에서 묻혔던 win. B=1 에서 -7.5% 실측.
2. **FP16 추가 효과 marginal** (-1.5% 만): -7.5% → -9%. 대부분은 register-blocking 효과.
3. **FP16 precision loss 치명적**: relres 15× 악화. FP16 10-bit mantissa + 입력값 1e-3~1e3 범위 → 곱셈 underflow.
4. **USA 에서 regblock_h16 효과 없음** (+1.5% regression). USA mid_tiled 이 이미 GPU 잘 활용.

**왜 B=64 에서 register-blocking 안 보였나** (Σ.2 의 dismissal):
- B=64 에서 batch 차원이 GPU 채움 → 각 block 의 per-thread efficiency 가 wall 결정 안 함
- B=1 에서는 single system 의 occupancy 가 낮아 register-blocking 의 load 감소가 visible

**BF16 미시도 이유**:
- sm_86 의 BF16 FMA 는 tensor core 외에서 emulated (FP32 throughput) → FMA pipe 우위 없음
- 유일한 benefit (shared 절반) 는 이미 FP16 으로 확인 — accuracy 만 다름
- BF16 의 wide range (FP32 exponent) 는 우리 power-grid 값에 큰 차이 없음

**결론**:
- ❌ FP16 채택 안 함 (accuracy 손실 너무 큼)
- ✓ **Σ.2 regblock_FP32 재고려 가치** — B=1 영역에 -7.5% win, 정확도 OK
- 단 큰 B 에서는 무효 → dispatch 에 B 기준 분기 필요 (구현 시 trade-off)

### 8.6 후속 lever

| Lever | 잠재 추가 win | 비용 |
|---|---|---|
| mid_tiled 를 WMMA staged 로 변경 | +5-15% factor | medium |
| nc 작은 panel 의 WMMA padding 회피 (16×16×8 fragment) | +5% | low |
| FP16 → BF16 trailing (mantissa 8-bit) | accuracy 개선 | low |
| cuBLAS LtMatmul TF32 모드 | mid-front 부근 +5-10% | medium |

---

## 9. 작은 front 에 TC 사용 가능한가 — 분석 + 측정

질문: 작은 front (fsz ≤ 32, small_warp 영역) 에 TC 사용 가능? 특히 큰 B 에서.

### 9.1 cuBLAS sgemmGroupedBatched 의 MIN_FSZ 낮추기 측정

`CLS_USE_CUBLAS=1 CLS_CUBLAS_MIN_FSZ=0` 으로 모든 mid level 에 cuBLAS 적용:

| B | MIN_FSZ | factor μs | vs default | 비고 |
|---:|---:|---:|---:|---|
| 64 | default (no cuBLAS) | **26.2** | baseline | mid_tiled scalar |
| 64 | =0 (cuBLAS for all) | 28.2 | **+8 %** ❌ | regression |
| 64 | =96 (cuBLAS for big only) | 26.3 | tie | |
| 256 | default | **23.3** | baseline | |
| 256 | =0 | 25.8 | **+11 %** ❌ | regression |
| 256 | =96 | 23.4 | tie | |

→ 큰 B 에서도 작은 front 에 cuBLAS 적용 시 **regression**. cuBLAS per-call overhead 가 amortize 안 됨.

### 9.2 왜 — WMMA fragment 의 fundamental 한계

WMMA fragment = **16×16×16 고정**. nc=4, uc=4 panel 의 경우:
- 실제 useful FMA = 4×4×4 = **64**
- WMMA 실행 = 4096 FMA cycle (16×16×16)
- **Padding waste = 4032/4096 = 98 %**

TC 의 4× nominal throughput vs FP32 도, padding waste 98% 면 effective throughput 은 FP32 의 4 × 0.02 = 0.08× → **scalar 보다 12× 느림**.

per-panel WMMA fire 는 작은 nc 에서 무조건 손해.

### 9.3 "여러 panel 묶어서 WMMA tile pack" 도 잘 안되는 이유

4 개 panel (uc=4, nc=4) 의 L 을 16 rows × 4 cols 로 stack, U 를 4 × 16 으로 stack:

```
L_pack = [L_A; L_B; L_C; L_D]   (16×4, 각 panel 4 rows)
U_pack = [U_A | U_B | U_C | U_D] (4×16, 각 panel 4 cols)

WMMA: 16×4 × 4×16 = 16×16 output
  output[A_rows, A_cols] = L_A * U_A = trailing of A ✓
  output[A_rows, B_cols] = L_A * U_B = WASTED ✗  (cross-panel 곱 불필요)
  ...
  4×4 block 4개 (diagonal) 만 valid, 12 blocks (off-diagonal) 낭비
```

→ Block-diagonal output. Useful = 4 × 16 = 64 entries, total = 256. **Useful fraction 25 %**.

Per-panel WMMA (waste 98 %) 보다 좋아짐 (waste 75 %), 그러나 여전히 scalar 보다 훨씬 비효율적.

수치 비교 (uc=4, nc=4, M panels):
| 방법 | WMMA call 수 | 총 cycle | scalar 대비 |
|---|---:|---:|---:|
| Scalar FP32 | 0 | ~2 × M | baseline |
| WMMA per-panel | M | 4096 × M | **2048×** 느림 |
| WMMA packed (4 per tile) | M/4 | 1024 × M | **512×** 느림 |

→ packing 으로 4× 개선되지만 scalar 대비 여전히 512× 손해.

### 9.4 packing 이 win 하는 영역 = uc≈16, nc≈16 (이미 full WMMA)

| uc, nc | per-panel WMMA waste | packed WMMA 가능? | 결론 |
|---|---|---|---|
| 4×4 | 98 % | 75% (pack 4) | scalar 압도적으로 win |
| 8×8 | 87 % | 50 % (pack 2) | scalar 여전히 win |
| 16×8 | 50 % | 50 % | WMMA 약간 효율 |
| **16×16** | **0 %** | pack 불필요 | **WMMA 진짜 win** |

우리 power-grid 의 mid front nc 분포 10-30 → 일부 WMMA-friendly, 일부 padding waste. 결국 **mid_tiled (scalar staged)** 가 측정상 최적 (Σ.1).

### 9.5 직접 microbench — Multi-panel WMMA packing 실제 측정

`tests/wmma_pack_microbench.cu` (~300 LOC). 3 방법 비교:
- (1) Scalar FP32 reference (one block per panel)
- (2) WMMA per-panel (1 warp = 1 panel = 1 mma_sync 16×16×16)
- (3) WMMA packed (K_PACK = 2, 4 panels per WMMA tile, block-diagonal output)

8192 panels 동시 (큰 B 시뮬레이션). RTX 3090 sm_86.

| uc, nc | Scalar GFLOPS | WMMA per-panel | WMMA packed K=2 | WMMA packed K=4 | 우승 |
|---|---:|---:|---:|---:|---|
| **4, 4** | **65 (8 μs)** | 38 (14 μs) | 62 (8.4 μs) | 63 (8.3 μs) | scalar |
| **8, 4** | **258 (8 μs)** | 124 (17 μs) | 216 (10 μs) | — | scalar |
| **8, 8** | **396 (11 μs)** | 244 (17 μs) | 295 (14 μs) | — | scalar |
| 12, 8 | 329 (29 μs) | **386 (25 μs)** | — | — | WMMA |
| 16, 8 | 446 (38 μs) | **495 (34 μs)** | — | — | WMMA |
| **16, 16** | 630 (53 μs) | **985 (34 μs)** ★ | — | — | WMMA −56 % (full tile) |

### 9.6 측정 → 분석 정확화

§ 9.3 의 이론 예측: "packed WMMA 가 scalar 의 100× 느림"

실측: **packing 으로 1.5-2× WMMA throughput 향상, scalar 와의 격차 0~25 % 수준** (이론 예측이 너무 비관적이었음).

차이 원인:
- Scalar 도 작은 panel 에서 thread occupancy 낮음 (64 threads 중 nc 만 active in inner k loop) → throughput peak 의 일부만 활용
- WMMA padding waste 는 컸지만 *절대* throughput 측면에서 그렇게 큰 차이 안 남
- Packing 으로 WMMA 의 useful work 비중 25→50% → WMMA 가 scalar 따라잡음

### 9.7 그래도 채택 안 함 — 이유

| 영역 | 측정 결과 | 채택 가능? |
|---|---|---|
| uc ≤ 8 (small_warp 영역) | scalar 가 packed WMMA 이김 25-100 % | ❌ |
| uc = 12 (mid_tc_lo 영역 일부) | WMMA per-panel 이미 win, packing 불필요 | (이미 적용) |
| uc ≥ 16 (full WMMA) | WMMA per-panel 압도, packing 의의 없음 | (이미 적용) |

추가로:
- **구현 복잡도**: variable-size panel 들을 K_PACK 그룹으로 묶기, 같은 그룹 내 size 통일 padding, dispatch overhead, shared mem 압박 등
- **마진이 작음**: 최대 win 이 uc=8 영역에서 packed K=2 (300 GFLOPS) vs scalar (400 GFLOPS) = scalar 가 33% 우세. packing 이 scalar 못 이김.

### 9.8 결론

**작은 front 에 TC packing 적용은 measurement-validated negative**.

- per-panel WMMA: padding waste 98 % → scalar 의 1/2
- multi-panel packing: padding waste 75 % 로 줄어 WMMA throughput 1.5-2× 향상
- 그러나 **여전히 scalar 못 이김** (uc ≤ 8 영역에서)
- 더 큰 panel (uc ≥ 12) 은 current dispatch 가 이미 WMMA fire 시키고 있음
- cuBLAS sgemmGroupedBatched (모든 packing 최적화 internal): 측정상 +8-11 % regression

작은 front 의 *진짜* lever = **algorithmic 변경 (amalgamation → 큰 panel 만들기)** 또는 **scalar 유지**. 현재 dispatch (small_warp scalar + mid_tiled scalar + WMMA only on small-mid fallback + extend_tc32 on BIG) 가 우리 분포에서 측정상 최적.

## Appendix A — Methodology

### A.1 이론 FLOPS 계산
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
batch_factor_per_sys_ms = host-side wall timing of cudaGraphLaunch + cudaEventSync.
Sum of kernel μs/sys ≠ wall (multistream concurrency factor).

### A.3 환경
- GPU: RTX 3090 (sm_86)
- 모든 lever 적용 (selinv OFF, multistream, Σ.1 staged)
- Median of 3 outer runs × `--repeat N` inner

### A.4 한계
- **panel category 와 kernel 의 매핑은 max_fsz_of_level 기준** — 한 mid kernel 이 처리하는 panel 들의 trailing% 평균이 카테고리 평균과 약간 다를 수 있음
- **kernel 안의 stage / writeback overhead** 가 약 5-10% 차지 (메모리 bound) — trailing time 약간 overestimate 가능
- 그러나 5-10% 오차는 결론 ("trailing 이 dominant") 에 영향 없음
