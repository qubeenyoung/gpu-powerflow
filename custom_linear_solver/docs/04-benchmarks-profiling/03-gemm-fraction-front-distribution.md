# Trailing GEMM 분율과 front 크기 분포 (case8387 / USA)

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: trailing GEMM은 이론 FLOP의 71–86%지만 실측 wall은 11–43%뿐 (비-GEMM이 per-FLOP 3–7배 느림); front 분포는 극단적 small-heavy지만 FLOP은 mid/big에 집중; TF32 K=8이 FP16 K=16 대비 padding 0–17% vs 37–50%.

**대상**: refactored `custom_linear_solver` (`41897cd`), panel_cap=8, RTX 3090, CUDA 12.8, fp32.
**측정 데이터**: `src/solver.cpp`의 `CLS_DUMP_FRONTS` hook → `/home/claude/prof/fronts_{8387,USA}.csv`. 스크립트 `analyze_flops.py`, `front_gemm_hist.py`.
**빌드 변형**: nsys/ncu는 `CLS_INTERNAL_GRAPH=OFF` (개별 커널 측정), `CLS_SKIP_TRAILING=1` (trailing stub해 차분).

## 0. TL;DR

- **이론**: trailing GEMM은 factor FLOP의 **71–86%** (big tier일수록 높음).
- **실측 wall (skip-trailing 차분)**: trailing은 wall의 **11–43%**뿐. 즉 *trailing이 효율적이라 비-GEMM이 wall을 빼앗음* — 비-GEMM phase가 per-FLOP 3–7배 느림.
- **front 개수 분포는 극단적 small-heavy** (8387 99.3%, USA 98.7%), 그러나 **FLOP은 mid/big에 집중** (8387 mid 50.6%, USA mid+big 91.4%).
- **nc 두 모드**: small tier nc=2 (75–88%), mid는 nc=8 (8387) 또는 nc=14~20 (USA), big은 nc=20 (USA).
- **WMMA K-padding**: FP16 K=16 → mid median 50%(8387)/37.5%(USA); TF32 K=8 → **0%(8387)** / 16.7%(USA). TF32 우월성의 정량 근거.
- **권장**: T1) TF32 WMMA → T2) panel_cap sweep → T3) B-방향 grouped GEMM → T4) sync-fused panel LU.

## 1. tier 정의와 dispatch

```cpp
// src/multifrontal.cu:issue_factor_level_range
SMALL_THRESH = 32   → factor_small (warp-per-front, 8 warps/block)
MID_THRESH   = 128  → factor_mid<T> (block-per-front, staged trailing)
                      / factor_mid_tc / factor_mid_tf32 (opt-in)
> 128               → factor_big<T> (1024-thread block, global memory)
```

Trailing GEMM: front당 `C(uc×uc) -= L(uc×nc)·U(nc×uc)`, (M,N,K) = (uc, uc, nc), FLOP = 2·uc²·nc.

## 2. fp32 factorize wall 분포 (nsys, batch당 median)

| 커널 | 8387 B=1 | 8387 B=64 | USA B=1 | USA B=64 |
|---|---:|---:|---:|---:|
| factor_mid | **85.0%** (30 μs) | **59.9%** (944 μs) | 19.0% (573 μs) | **46.7%** (14.1 ms) |
| factor_small | 15.0% (5 μs) | **40.1%** (632 μs) | 4.6% (138 μs) | 20.9% (6.3 ms) |
| factor_big | — | — | **76.4%** (2305 μs) | 32.4% (9.8 ms) |
| **per-system** | 35 μs | 24.6 μs | 3.0 ms | 473 μs |

`scatter_values`(FP64→FP32 변환·front 배치)는 별도 wall 232 μs (8387 B=64), factor의 ~15% 추가 비용.

ROI 절대 우선순위: 8387 B=1 → mid(85%); 8387 B=64 → mid+small; USA B=1 → big(76%); USA B=64 → mid(47%) > big(32%) > small(21%).

## 3. 이론 FLOP 분해 (`analyze_flops.py`)

### case8387pegase (fronts=7427, max_fsz=79, total=2.6 MFLOPs)

| phase | small (≤32) | mid (33–128) | 전체 |
|---|---:|---:|---:|
| panel LU | 24.3% | 11.0% | 18.3% |
| U-solve | 0.0% | 3.3% | 1.5% |
| **trailing GEMM** | 64.5% | 80.0% | **71.5%** |
| extend-add | 11.2% | 5.8% | 8.8% |
| tier별 FLOP 합 | 54.7% | 45.3% | – |

### case_SyntheticUSA (fronts=74196, max_fsz=235, total=155 MFLOPs)

| phase | small | mid | big (>128) | 전체 |
|---|---:|---:|---:|---:|
| panel LU | 20.3% | 15.2% | 6.2% | 12.5% |
| U-solve | 0.0% | 9.2% | 5.4% | 6.9% |
| **trailing GEMM** | 66.2% | 73.3% | **86.0%** | **77.1%** |
| extend-add | 13.6% | 2.4% | 2.3% | 3.5% |
| tier별 FLOP 합 | 10.1% | 54.5% | 35.4% | – |

→ trailing이 작업의 71–77%, big tier일수록 비율 높음 (uc²·nc dominant).

## 4. 실측 GEMM wall (skip-trailing 차분)

nsys median per-instance, 풀빌드 vs skip-trailing:

| 커널 | 풀빌드 μs | skip μs | **trailing wall %** |
|---|---:|---:|---:|
| 8387 mid B=1 | 12.77 | 10.75 | **16%** |
| 8387 mid B=64 | 26.83 | 22.05 | **18%** |
| USA mid B=1 | 41.07 | 31.62 | **23%** |
| USA mid B=64 | 1114 | 1008 | **10%** |
| USA big B=1 | 107.3 | 55.3 | **48%** |
| USA big B=64 | 311.2 | 165.7 | **47%** |

가중 평균 factor 전체 trailing wall: 8387 B=1 ~14%, 8387 B=64 ~11%, USA B=1 **~41%**, USA B=64 ~20%.

### 4.1 이론 vs 실측 격차 (핵심 발견)

| | 이론 GEMM FLOP % | 실측 GEMM wall % | wall/FLOP 비효율 비 |
|---|---:|---:|---:|
| 8387 B=1 | 71.5% | 14% | **5.1×** |
| 8387 B=64 | 71.5% | 11% | **6.5×** |
| USA B=1 | 77.1% | 41% | 1.9× |
| USA B=64 | 77.1% | 20% | **3.9×** |

→ **trailing은 작업의 71–77%지만 wall은 11–41%만.** 단순히 "GEMM이 안 빠르다"가 아니라 **비-GEMM phase가 per-FLOP 3–7배 느림**. 주범은 panel LU의 직렬성 (`nc`개 step 직렬 + 매 step `__syncthreads`).

## 5. 비-GEMM 병목 (ncu)

### 5.1 SOL / occupancy / stall

| 커널 | SOL_SM% | DRAM% | Occ% | 1순위 stall |
|---|---:|---:|---:|---|
| 8387 mid B=1 | 5.7 | 2.5 | 28.6 | scoreboard 31% (mem latency) |
| 8387 mid B=64 | 49.7 | 32.6 | 77.7 | **barrier 41%** (`__syncthreads`) |
| 8387 small B=64 | 55.2 | 23.6 | 42.7 | sub-threshold (분산) |
| USA mid B=1 | 22.3 | 13.2 | 41.0 | scoreboard 36% |
| USA mid B=64 | 38.9 | 31.4 | 39.0 | scoreboard 32% + fixed 30% |
| USA big B=1 | 5.1 | 2.2 | 66.3 | barrier 36% |
| USA big B=64 | 40.7 | 28.4 | 66.2 | barrier 37% |

→ **B=1: scoreboard (메모리 latency) 30–36%**; **B=64: CTA barrier (sync) 37–41%**.

### 5.2 sync 비용의 출처

per-front phase 구조 (`factorize_front`): stage_in 1 + panel LU ~2·nc + U-solve ~nc + trailing 2~3 + writeback 1 → **per-front sync ≈ 3·nc + 4~5**. 8387 mid(nc=8) → ~29회/front, USA mid(nc=20) → ~65회/front. sync 1개당 ~6 ns.

| tier | nc median | per-front sync | per-front FLOP | sync/FLOP |
|---|---:|---:|---:|---:|
| small (8387/USA) | 2 | 11 | 64 (4×4×2) | **0.17** |
| mid 8387 | 8 | 29 | 8192 | 0.0035 |
| USA mid | 14 | 47 | 17298 | 0.0027 |
| USA big | 20 | 65 | 353780 | 0.00018 |

→ **small은 sync-dominated** (그러나 factor_small이 `__syncwarp` 사용해 비용 작음), **mid/big은 FLOP-dominated**. mid/big에서 sync를 더 줄여도 wall 효과 작음을 정량 확인 (sync 다 제거해도 mid factor wall −11% 한계, 실측은 그 절반 −5~6%).

## 6. front 크기 분포 상세

### 6.1 case8387pegase (fronts=7427)

| tier | count | 비율 | trailing FLOP 비율 |
|---|---:|---:|---:|
| small | 7372 | 99.26% | 49.36% |
| mid | 55 | 0.74% | **50.64%** |
| big | 0 | 0% | – |

**small** (median): fsz=6, nc=**2**, uc=4, uc²·nc=32. nc 분포: nc=1 (19.4%), **nc=2 (75.5%)**, nc=8 (1.4%). Top GEMM = (4×4×2) = 32 FMAs (30.5%); 상위 5개 (fsz,nc)가 69.3%.

**mid** (55 fronts, median): fsz=43, nc=**8**, uc=37, uc²·nc=8192. nc 분포: **nc=8 (78.2%)** — panel_cap=8 상한에 sat. fsz 절반이 [32,48).

### 6.2 case_SyntheticUSA (fronts=74196)

| tier | count | 비율 | trailing FLOP 비율 |
|---|---:|---:|---:|
| small | 73247 | 98.72% | 8.65% |
| mid | 886 | 1.19% | **51.81%** |
| big | 63 | 0.08% | **39.54%** |

→ **886+63 = 949개 front가 91% FLOP.** small은 count 98.7%지만 FLOP 8.7%.

**small**: 8387과 거의 동일 (fsz=6, nc=2, uc=4 median; **nc=2 비율 88.1%** — 8387의 76%보다 높음). Top GEMM (4×4×2) 33.1%, 상위 5개 88.8%.

**mid** (886, median): fsz=51, nc=**14**, uc=38, uc²·nc=17298. nc 분포 long-tail에 **nc=20 sharp peak (35.4%)** (separator hat). nc max=20.

**big** (63, median): fsz=155, nc=**20**, uc=137, uc²·nc=353780. **nc=20 (85.7%)** 절대 다수. fsz∈[129,235].

→ count당 FLOP: 8387 mid 22.9k vs USA mid 69.7k (**3배**) vs USA big 748k.

### 6.3 표준 GEMM 형태

| 케이스 | 가장 흔한 (M,N,K) | 빈도 |
|---|---|---|
| 8387 small / USA small | (4, 4, 2) | 30.5% / 33.1% |
| 8387 mid | nc=8, (25..71)² × 8 | 78.2% nc=8 |
| USA mid | 평균 (38, 38, 13~14) | 35.4% nc=20 |
| USA big | 평균 (137, 137, 20) | 85.7% nc=20 |

## 7. WMMA padding — TF32가 정량적으로 옳음

WMMA 단위 = 16×16×K (FP16 K=16, TF32 K=8).

| tier | metric | FP16 (K=16) | TF32 (K=8) | M/N (16) |
|---|---|---:|---:|---:|
| 8387 mid | mean | 56.8% | **13.6%** | 14.4% |
| 8387 mid | median | 50.0% | **0.0%** | 12.5% |
| USA mid | mean / median | 41.8% / 37.5% | **24.1% / 16.7%** | 14.9% / 14.1% |
| USA big | mean / median | 39.3% / 37.5% | **18.5% / 16.7%** | 4.9% |

- **8387 mid**: nc=8이 K=8 tile에 완벽 fit → TF32 padding 0% (78%의 front). FP16은 K=16에 nc=8 → 50% 손실.
- **USA mid/big**: nc=20 dominant → TF32 24-tile (20/24 = 16.7%) vs FP16 32-tile (20/32 = 37.5%) → TF32가 **2.2배 좋은 padding 효율**.

front당 WMMA tile (ceil(M/16)·ceil(N/16)·ceil(K/Ktile)):

| 케이스 | 대표 GEMM | FP16 tiles | TF32 tiles |
|---|---|---:|---:|
| 8387 mid | (37,37,8) | 9 | 9 (동일) |
| USA mid | (38,38,14) | 9 | 18 (K 2개, padding 적음) |
| USA big | (137,137,20) | 162 | 243 (K 3개) |

→ TF32는 K-tile이 작아 tile 수는 많지만 padding이 작아 **실효 FLOP 활용률 ~25–30% 높음**.

## 8. GEMM 자체 효율과 tile geometry 한계

trailing wall만 추출한 실효 throughput (RTX 3090 fp32 peak 35.6 TFLOPs, TF32 tensor 71 TFLOPs):

| 케이스 | 달성 TF/s | fp32 peak% | TF32 peak% |
|---|---:|---:|---:|
| 8387 B=1 / B=64 | 0.41 / 0.74 | 1.15% / 2.08% | 0.57% / 1.04% |
| USA B=1 / B=64 | 0.10 / 1.26 | 0.27% / 3.54% | 0.14% / 1.77% |

→ trailing이 가장 효율적 phase인데도 **fp32 peak의 1–4%만** 사용. power-grid 행렬은 root separator가 ~90–250에서 cap (METIS ND 기하 한계, 평균 degree 낮아 separator 못 키움). 기존 FP16 시도(`../03-optimization-notes/01-kernel-engineering.md`): case8387 +25~49% 느림 (작은 tile setup 비용 amortize 실패), USA 15× NaN 발산 (relres 2e-3 → 2.9e-2).

## 9. 공격적 TC 활용 — T1~T4

### T1. TF32 WMMA 경로 (구현됨)

FP16 K=16 → TF32 **K=8** 로 줄여 8387 nc=8에서 K-padding 0%, USA nc=20에서 16.7%(FP16 37.5% 대비). TF32 input은 fp32 mantissa 10비트 유지 → FP16의 정밀도 손실/NaN 위험 제거, 정확도 사실상 fp32 동급. Ampere TF32 = 71 TFLOPs.

```cpp
// 현재: wmma::fragment<matrix_a, 16,16,16, __half, row_major>
// 제안: wmma::fragment<matrix_a, 16,16,8, wmma::precision::tf32, row_major>
```

**기대**: trailing wall ~50% 단축 → 8387 11–14%→6–7%, USA 20–41%→10–20%. **구현 상태**: `Precision::TF32` + `--precision tf32` 추가, FP32 front/solve 공유 + trailing만 TF32 WMMA(16×16×8, FP32 accum). sm80+ 전용 (CMake에서 arch<80 fail).

### T2. B-방향 grouped GEMM (B=64 전용)
같은 (M,N,K)의 B 시스템 trailing을 strided-batched WMMA 한 번으로 launch. (a) M-방향 stack, (b) `cublasGemmStridedBatchedEx(TF32_FAST)` 위임. **기대**: trailing throughput 3–5×. **위험**: 중 — mid의 fused stage/trailing/writeback 구조 분해 필요.

### T3. panel_cap 상향 sweep
cap=8 (nc≤8 보장) → 16/24/32 sweep. T1과 시너지: TF32 K=8 단위에서 nc=24면 3 tile 모두 full. **위험**: 낮음 (config 파라미터). 단 과거 cap=8이 최적, amalgamation 회귀(cap≥16 +72%) 주의 → TF32 도입 후 재측정 필요.

### T4. sync-fused panel LU (장기)
barrier stall 41% 주범인 `2·nc`개 `__syncthreads`. (a) warp-specialized panel LU (`__syncthreads`→`__syncwarp`), (b) `cp.async`+`cuda::barrier` overlap, (c) fine-grained partial barrier. **기대**: barrier 41%→20% 시 mid B=64 wall ~25% 단축. **위험**: 큼, 2–4주 R&D.

**우선순위**: T1 (1–2일, 위험 낮음, 즉시 측정) → T3 (1일) → T2 (1주, B=64) → T4 (2–4주). **T1 (TF32 trailing)이 모든 케이스 걸쳐 핵심 lever** — 이미 ship, 실측 평가만 남음.

## 10. tier별 factorize 구조 (요약 ASCII)

### small — warp-per-front, fused right-looking LU
```
block 256t = 8 warps, warp i ↔ front i (per-warp shared Fs ≈ fsz²·4 ≈ 144 B)
  stage_in → __syncwarp
  ╔ FUSED LU: for k in 0..nc-1: divide col k; __syncwarp;
  ║           Fs[i][j] -= Fs[i][k]·Fs[k][j]  (LU+U+GEMM merged); __syncwarp ╝
  writeback L/U → F;  extend-add CB → parent (atomicAdd)
  sync = nc (median 2);  warp barrier가 block barrier보다 ~8배 쌈
```
nc=2인 75–88% front에서 phase 분리 의미 없어 fused가 유리.

### mid — block-per-front, 3-phase + shared-staged trailing
```
block 256t, gridDim=(level_size, B), 한 block ↔ (front,batch)
  shared: Fs[fsz²] + sh_L[uc·nc] + sh_U[nc·uc] (~30 KB)
  stage_in (cp.async) → __syncthreads
  Phase1 lu_panel_factor: nc-loop {divide; sync; rank-1; sync} → 2·nc syncs
  Phase2 u_panel_solve:   (nc-1)-loop {row solve; sync}        → nc-1 syncs
  Phase3 trailing(staged): stage L/U→shared; sync; GEMM(uc²); sync → 2 syncs
  sync ≈ 3·nc+5 (8387 nc=8 → 29, USA nc=20 → 65)
```
핵심: panel LU(직렬)와 trailing GEMM(병렬 uc²)을 분리 → T1 TF32 WMMA가 trailing에만 적용.

### big — block-per-front, global-memory direct (USA만)
```
block 1024t. fsz²·4 > 96 KB shared 불가 (USA median fsz=155 → 96 KB) → global RW 직접
  Phase1/2/3 모두 global F. trailing: TF32 1024t WMMA + shared fragment scratch
  sync ≈ 3·nc+ (USA nc=20 → 65+); global traffic O(nc·fsz²)
```
8387은 max_fsz=79라 mid에 머물러 big 미사용.

## 11. ncu 병목 패턴 요약

- **B=1 모든 tier**: `long_scoreboard` (global mem latency) 지배, occ 16–40%로 hide 불가 → cp.async 타깃.
- **B=64 small**: `wait` (fused LU 짧은 dep chain), barrier=0 (warp-only).
- **B=64 mid**: barrier 지배(8387) 또는 wait 지배(USA nc=20 GEMM이 길어).
- **모든 big**: barrier 압도 (1024t block sync 비용 큼).

전체 factor wall 단축 잠재: 8387 B=1 −5~10% (T1), 8387 B=64 −10~20% (T1+small packing), USA B=1 −15~20% (TF32 big + shared staging), USA B=64 −10~15% (T1+T2).

## 12. 관련 문서

- `../02-design-analysis/04-gemm-fraction-tc-ceiling.md` — GEMM 분율 이론/실측 격차의 design 분석
- `../03-optimization-notes/01-kernel-engineering.md` — front 크기별 커널 라우팅 + FP16 negative result
- `04-multistream-impact.md` — subtree multi-stream tier별 임팩트
- `02-strumpack-vs-custom-case8387.md`, `../main-report.md`

원본: `/home/claude/prof/fronts_{8387,USA}.csv`, `analyze_flops.py`, `front_gemm_hist.py`, `ng_*.nsys-rep`, `skip_*.nsys-rep`, `ncu2_*_clean.csv`, `stall_*_clean.csv`.
