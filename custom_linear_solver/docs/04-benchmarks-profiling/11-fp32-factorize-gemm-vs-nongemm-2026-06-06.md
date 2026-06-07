# FP32 factorize 병목: GEMM vs 비-GEMM 분리와 공격적 TC 활용 방안

> **2026-06-06 update — 파일 경로 정정**: 본 문서가 인용하는 `src/factorize/primitives.cuh` / `mid.cuh` / `big.cuh` 는 리팩토링으로 통합됨. device 빌딩 블록 (lu_*, u_panel_solve, trailing_update_*, factorize_front 등) 은 **`src/factorize/phases.cuh`** 로, kernel 들 (factor_small, factor_mid, factor_big, +_tc/_tf32 변종) 은 **`src/factorize/kernels.cuh`** 로 이동. 자세한 매핑은 [`01-orientation/06-factorize-file-layout-2026-06-06.md`](../01-orientation/06-factorize-file-layout-2026-06-06.md) §7.

**작성일**: 2026-06-06
**대상**: refactored `custom_linear_solver` (commit `41897cd`)
**대상 케이스**: `case8387pegase`, `case_SyntheticUSA`
**측정 환경**: RTX 3090, CUDA 12.8, panel_cap=8 (기본)
**재현 빌드**: `-DCLS_INTERNAL_GRAPH=OFF` (개별 커널 nsys/ncu 측정 가능하도록)

## 0. TL;DR

- **이론**: trailing GEMM은 factor 작업의 71–77% FLOP을 차지.
- **실측 wall (trailing 제거 변종과의 차분)**: trailing은 wall의 **11–41%**뿐. 4–7배 격차.
- **B=1**: 모든 커널 SOL 5–22%, 주된 stall은 scoreboard (global memory latency).
- **B=64**: SOL 40–55%, 주된 stall은 **CTA barrier 37–41%** (`__syncthreads`).
- **GEMM 자체도 비효율**: trailing이 가장 효율 좋은 phase인데도 fp32 peak의 1–4%만 활용. tile geometry (nc≤8 → K=16에 50% padding) 때문.
- **권장 우선순위**: T1) TF32 WMMA 경로 추가 → T2) panel_cap sweep → T3) B-방향 grouped GEMM → T4) sync-fused panel LU.

## 1. 측정 방법론

### 1.1 빌드 변형

| 빌드 | 위치 | 목적 |
|------|------|------|
| baseline | `/home/claude/build_cls_baseline/` (914c917) | refactor 회귀 비교용 |
| refactor + graph | `/home/claude/build_cls_refactor/` (41897cd, default ON) | 정확도 검증, 통합 wall |
| refactor + nograph | `/home/claude/build_cls_nograph/` (41897cd, `CLS_INTERNAL_GRAPH=OFF`) | **nsys/ncu가 개별 커널을 보게 함** |
| refactor + skip-trailing | `/home/claude/build_cls_skip/` (`CLS_SKIP_TRAILING=1`) | trailing 호출만 stub해 차분 |

기본 빌드에서는 factor/solve가 내부 CUDA Graph로 캡처되어 nsys가 개별 커널을 한 graph node로 묶어버린다. 본 분석에서는 graph capture를 끄고 커널별 시간을 측정했다.

### 1.2 wall 분해 차분

`src/factorize/primitives.cuh`의 `trailing_update_scalar`, `src/factorize/mid.cuh`의 `trailing_update_staged`를 `#ifndef CLS_SKIP_TRAILING` 가드로 noop화 (`__syncthreads()` 한 번만 남김). 정확성은 깨지지만 wall은 측정 가능 — 풀빌드 wall과 차분하여 trailing wall 추출.

- `factor_small` (`lu_small_front`)은 panel LU와 trailing이 한 rank-1 update에 합쳐져 있어 깔끔히 분리 불가. 따라서 `factor_small`의 trailing 분해는 측정하지 않음.

### 1.3 이론 FLOP

`src/solver.cpp`의 `analyze()`에 `CLS_DUMP_FRONTS` env-var hook을 추가, analyze 후 per-front (fsz, nc, level)를 CSV로 dump. `analyze_flops.py`가 phase별 FLOP을 다음 공식으로 계산:

| phase | FLOP 공식 | 메모리 traffic (32-bit elts) |
|-------|-----------|------------------------------|
| 1. panel LU       | `Σ_{k=0..nc-1} (fsz-k-1) + 2·(fsz-k-1)(nc-k-1)` | – |
| 2. U-panel solve  | `uc · nc · (nc-1)` | – |
| 3. trailing       | `2 · uc² · nc` | – |
| 4. extend-add     | `uc²` (atomic add 1개) | `2·uc²` (load Fs + atomic) |
| stage-in          | – | `fsz²` |
| writeback         | – | `nc·fsz + uc·nc` |

소형 front (`fsz≤48`)는 `lu_small_front`로 panel LU와 trailing이 fused — 그 경우 trailing 등가 FLOP = `2·uc²·nc`로 정의하고 나머지는 panel LU로 귀속.

## 2. fp32 factorize 커널별 wall 분포 (nsys)

`run_custom_solver_run ... --precision fp32 --batch B --batch-only --repeat 10` 기준, batch당 wall (median):

| 커널 | 8387 B=1 | 8387 B=64 | USA B=1 | USA B=64 |
|------|---------:|----------:|--------:|---------:|
| factor_mid<float>   | **85.0%** (30 μs)  | **59.9%** (944 μs)  | 19.0% (573 μs)   | **46.7%** (14.1 ms) |
| factor_small<float> | 15.0% (5 μs)       | **40.1%** (632 μs)  | 4.6% (138 μs)    | 20.9% (6.3 ms) |
| factor_big<float>   | —                  | —                   | **76.4%** (2305 μs) | 32.4% (9.8 ms) |
| factor 합계         | 35 μs/call         | 1.58 ms/call        | 3.0 ms/call      | 30.3 ms/call |
| **per-system**      | 35 μs              | 24.6 μs             | 3.0 ms           | 473 μs |

`scatter_values` (factorize 입구의 FP64→FP32 변환·front 배치) 는 별도 wall로 232 μs (case8387 B=64) — factor의 ~15% 추가 비용으로, factor 커널 합과는 별개. 본 분석은 factor 커널 내부만 다룬다.

## 3. 이론 FLOP 분해

`/home/claude/prof/fronts_*.csv` + `analyze_flops.py` 결과.

### case8387pegase (fronts = 7427, max_fsz = 79, total = 2.6 MFLOPs)

| phase | tier=small (≤32) | tier=mid (33–128) | 전체 |
|-------|-----------------:|------------------:|-----:|
| panel LU         | 24.3% | 11.0% | 18.3% |
| U-solve          |  0.0% |  3.3% |  1.5% |
| **trailing GEMM**| 64.5% | 80.0% | **71.5%** |
| extend-add       | 11.2% |  5.8% |  8.8% |
| tier별 FLOP 합   | 54.7% | 45.3% | – |
| GEMM-only AI     | 0.31  | 0.93  | – flop/byte |

### case_SyntheticUSA (fronts = 74196, max_fsz = 235, total = 155 MFLOPs)

| phase | tier=small | tier=mid (33–128) | tier=big (>128) | 전체 |
|-------|-----------:|-------------------:|----------------:|-----:|
| panel LU         | 20.3% | 15.2% |  6.2% | 12.5% |
| U-solve          |  0.0% |  9.2% |  5.4% |  6.9% |
| **trailing GEMM**| 66.2% | 73.3% | 86.0% | **77.1%** |
| extend-add       | 13.6% |  2.4% |  2.3% |  3.5% |
| tier별 FLOP 합   | 10.1% | 54.5% | 35.4% | – |
| GEMM-only AI     | 0.27  | 1.77  | 2.60  | – flop/byte |

→ 두 케이스 모두 trailing이 작업의 **71–77%**이고, big tier 일수록 trailing 비율이 높다 (uc²·nc가 더 dominant).

## 4. 실측 GEMM wall 비율 (skip-trailing 차분)

nsys median per-instance, 풀빌드 vs skip-trailing 비교:

| 커널 | 풀빌드 (μs/inst) | skip (μs/inst) | **trailing wall %** |
|------|----:|----:|----:|
| 8387 mid B=1   | 12.77 | 10.75 | **16%** |
| 8387 mid B=64  | 26.83 | 22.05 | **18%** |
| USA mid B=1    | 41.07 | 31.62 | **23%** |
| USA mid B=64   | 1114  | 1008  | **10%** |
| USA big B=1    | 107.3 | 55.3  | **48%** |
| USA big B=64   | 311.2 | 165.7 | **47%** |

커널 wall 비율로 가중 평균한 **factor 단계 전체 trailing wall**:

| | 8387 B=1 | 8387 B=64 | USA B=1 | USA B=64 |
|---|---:|---:|---:|---:|
| trailing wall %     | ~14% | ~11% | **~41%** | ~20% |
| non-trailing wall % | ~86% | ~89% | ~59% | ~80% |

### 4.1 이론 vs 실측 격차 (핵심 발견)

| | 이론 GEMM FLOP % | 실측 GEMM wall % | wall/FLOP 비효율 비 |
|---|---:|---:|---:|
| 8387 B=1   | 71.5% | 14% | **5.1×** |
| 8387 B=64  | 71.5% | 11% | **6.5×** |
| USA B=1    | 77.1% | 41% | 1.9× |
| USA B=64   | 77.1% | 20% | **3.9×** |

→ **trailing은 작업의 71–77%지만 wall은 11–41%만 차지**. 즉 *trailing이 효율적이라 비-GEMM이 wall을 빼앗는 것*. 단순히 "GEMM이 안 빠르다"가 아니라 **비-GEMM phase가 per-FLOP 기준 3–7배 더 느리다**.

## 5. 비-GEMM 병목 원인 (ncu)

### 5.1 SOL throughput / occupancy

| 커널 | SOL SM% | SOL Mem% | DRAM% | Achieved Occ% | 분류 |
|------|--------:|---------:|------:|--------------:|------|
| 8387 mid B=1   |  5.7 |  4.0 |  2.5 | 28.6 | latency-bound, 점유 부족 |
| 8387 mid B=64  | 49.7 | 37.7 | 32.6 | 77.7 | sync로 cap된 균형 |
| 8387 small B=64| 55.2 | 30.2 | 23.6 | 42.7 | sync-bound |
| USA mid B=1    | 22.3 | 18.4 | 13.2 | 41.0 | latency-bound |
| USA mid B=64   | 38.9 | 37.1 | 31.4 | 39.0 | mem + sync |
| USA big B=1    |  5.1 |  4.0 |  2.2 | 66.3 | latency, 1024 block의 sync 비용 |
| USA big B=64   | 40.7 | 31.9 | 28.4 | 66.2 | 균형, sync로 cap |

### 5.2 Warp stall 1순위 (ncu rule descriptions, % of issue cycles)

| 커널/조건 | 1순위 | 2순위 |
|-----------|------|-------|
| 8387 mid B=1   | scoreboard 31% (global mem latency) | barrier 31% (`__syncthreads`) |
| 8387 mid B=64  | **barrier 41%** | – |
| 8387 small B=64| sub-threshold (다양한 작은 stall 분산) | wait/latency |
| USA mid B=1    | scoreboard 36% | – |
| USA mid B=64   | scoreboard 32%, fixed_latency 30% | – |
| USA small B=1/64 | scoreboard 36–38% | – |
| USA big B=1/64 | **barrier 36–37%** | scoreboard |

→ **B=1: scoreboard (메모리 latency) 30–36%**, **B=64: CTA barrier (sync) 37–41%**.

### 5.3 sync 비용의 출처 (코드 추적)

per-front phase 구조 (`src/factorize/primitives.cuh:factorize_front`):

```
stage_in:   1 sync
phase 1 (lu_panel_factor): nc-loop × { divide; sync; rank-1 update on panel; sync }  → ~2·nc syncs
phase 2 (u_panel_solve):   (nc-1)-loop × { row-update; sync }                          → ~nc syncs
phase 3 (trailing):        1 sync 전, staged 시 stage-L/U 후 1 sync, GEMM 후 1 sync   → 2~3 syncs
phase 4 (extend-add):      atomic, sync 없음
writeback:                 1 sync (전후)
```

per-front 총 `__syncthreads` 횟수 ≈ **3·nc + 4**. case8387 (nc 평균 ~6) → 22번/front. USA mid (nc ≤ 20) → 최대 64번/front.

각 sync는 256-thread block의 warp 정렬을 기다림. 실측 barrier 대기 평균 6–10 cyc → factor_mid B=64 19 cyc/issue 중 41% (=~8 cyc) 가 barrier 대기.

### 5.4 비-GEMM wall의 구성요소 추정

`trailing` 제거 차분으로 얻은 비-trailing wall 안에서 (factor_mid B=64 기준 80–90%):

| 구성요소 | 추정 비율 | 출처 |
|---------|---------:|------|
| panel LU (phase 1, 2·nc syncs) | ~30% | 직렬 rank-1, 매 step sync |
| U-panel solve (phase 2)        | ~15% | 직렬 row 의존 |
| stage-in / writeback (global)  | ~20% | DRAM bound |
| extend-add (atomicAdd to parent)| ~15% | parent cache line contention |
| `__syncthreads` 대기 자체       | ~20% | wait/barrier stall, 위 모두에 분산 |

(정확한 분해는 추가 instrumentation 필요. 위는 ncu stall % + 코드 구조 + skip-trailing 차분의 cross-check.)

→ **핵심 원인은 panel LU의 직렬성**: trailing은 `uc²` 출력을 모두 병렬, panel LU는 `nc`개 step을 직렬 + 매 step sync. wall에서 panel LU가 trailing보다 비대해지는 이유.

## 6. GEMM 병목

### 6.1 trailing의 실효 throughput

trailing wall만 추출해 실제 GEMM 처리량 계산 (RTX 3090 fp32 peak = 35.6 TFLOPs, TF32 tensor = 71 TFLOPs):

| 케이스 | trailing FLOPs/sys | trailing μs/sys | 달성 TF/s | fp32 peak% | TF32 peak% |
|--------|--------------------:|-----------------:|----------:|-----------:|-----------:|
| 8387 B=1   |   2.0 MF |    4.9 μs | 0.41 | 1.15% | 0.57% |
| 8387 B=64  |   2.0 MF |    2.7 μs | 0.74 | 2.08% | 1.04% |
| USA B=1    | 119.0 MF | 1237 μs   | 0.10 | 0.27% | 0.14% |
| USA B=64   | 119.0 MF |   95 μs   | 1.26 | 3.54% | 1.77% |

→ trailing이 가장 효율적인 phase인데도 **fp32 peak의 1–4%만 사용**. GEMM 자체도 강하게 under-utilized.

### 6.2 tile geometry 한계

WMMA 단위 = 16×16×16 (M, N, K). 두 케이스의 front 분포:

| | nc 분포 | uc 분포 | K-padding (FP16 WMMA K=16) | M/N-padding |
|---|--------|--------|-----------------------------|-------------|
| 8387 mid    | ≤ 8    | ≤ 71   | **K=16에 50% padding** | 1.4–11% |
| USA mid     | ≤ 20   | ≤ 111  | K=32에 37% padding     | 11–28% |
| USA big     | ≤ 20   | ≤ 215  | K=32에 37% padding     | 7–13% |

power-grid 행렬은 root separator가 ~90–250 수준에서 cap된다 (`02-design-analysis/04`, `03-optimization-notes/06`). 이는 METIS nested dissection의 기하학적 한계 — 그래프의 평균 degree가 낮아 separator를 키울 수 없다.

기존 시도 (`03-optimization-notes/archive/06-tc-dedicated-path-study.md`):
- FP16 WMMA + FP32 accumulate → case8387에서 +25–49% 느림 (작은 tile의 setup 비용 amortize 실패).
- FP16 inputs → USA에서 15× NaN 발산 (relres 2e-3 → 2.9e-2).

## 7. 공격적 TC 활용 방안

우선순위순 4가지 제안. 각 항목 = 변경 규모 + 기대 효과 + 위험.

### T1. TF32 WMMA 경로 추가 (구현됨)

**변경**: 기존 `--precision tc` FP16 WMMA 경로는 비교용으로 유지하고, 새 `--precision tf32` 경로를 추가한다. `src/factorize/mid.cuh` / `big.cuh`에 TF32 전용 trailing update를 두고, `nvcuda::wmma::precision::tf32` fragment를 사용한다.

```cpp
// 현재
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];

// 제안
wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> af[2];
```

- WMMA K가 16 → **8**로 줄어, **nc ≤ 8인 case8387에서 K-padding 0%**, nc ≤ 20인 USA에서 25%로 감소 (FP16의 37%보다 개선).
- TF32 input은 fp32 mantissa 10비트 유지 → FP16의 정밀도 손실 / NaN 위험 제거. 정확도는 사실상 fp32와 동급.
- Ampere TF32 throughput = 71 TFLOPs (FP16+FP32 accum과 동일 peak).

**기대 효과**: trailing wall ~50% 단축 → factor 전체 wall 단축:
- case8387: trailing 11–14% → 6–7% 단축
- USA: trailing 20–41% → 10–20% 단축

**위험**: 낮음. 정확도는 doc 06의 FP16 path와 달리 fp32 mantissa 유지. setup 비용은 FP16 path와 동일하므로 작은 front에서는 여전히 amortize 어려움 — 그래서 T3와 결합 필요.

**측정 계획**:
1. TF32 변형 빌드, case8387/USA × B=1/64 × `--precision tf32` 실행
2. batch_relres가 fp32(2e-5 ~ 3e-5) 수준 유지하는지 확인
3. nsys median으로 mid 커널 wall 단축률 측정
4. 단축 ≥ 30%면 PR. 아니면 T2/T3로 이동.

**구현 상태 (2026-06-06)**:
- `Precision::TF32`와 CLI `--precision tf32`가 추가되었다.
- TF32 path는 FP32 front/solve를 공유하고 trailing GEMM만 TF32 WMMA (`16×16×8`, FP32 accumulate)로 수행한다.
- TF32 WMMA가 sm80+ 전용이므로 `CLS_CUDA_ARCHITECTURES < 80`은 CMake configure 단계에서 실패한다.

### T2. B-방향 grouped GEMM (B=64에서 큰 이득 기대)

B=64는 같은 sparsity 패턴의 64개 시스템 → trailing은 B개 독립 `(uc × uc × nc)` GEMM.

**변경**: 같은 (M, N, K)를 가진 B 시스템의 trailing GEMM을 strided-batched WMMA grouped GEMM 한 번으로 launch.

- 두 가지 구체 방식:
  - **(a) M-방향 stack**: B 시스템의 L 패널을 M 차원에 쌓아 `(uc·B × nc)` 좌측 곱 + `(nc × uc)` 우측 = `(uc·B × uc)` 결과. WMMA에서 M=16 tile이 uc<16일 때 padding 큰 case8387에는 이득 (사실상 M-padding 0%).
  - **(b) cuBLAS `cublasGemmBatchedEx` / `cublasGemmStridedBatchedEx` 호출 (TF32_FAST)**: 외부 라이브러리에 위임, host overhead만 신경 쓰면 됨. 단 mid 커널의 fused stage/trailing/writeback 구조와 충돌하므로 mid 분기를 cuBLAS path로 별도 분리해야 함.

**기대 효과**:
- B=64에서 trailing tile당 amortization 32–64× 증가 → trailing 달성 throughput 3–5× 개선 가능.
- 단, mid 커널 내부의 stage/writeback과 fuse가 깨지면 stage/writeback wall이 다시 가산되므로 순 이득은 측정해야 함.

**위험**: 중. mid의 fused 구조를 풀어야 함. cuBLAS 호출은 host-side launch overhead 발생 — `factor_mid` 한 번의 wall이 mid level별 launch가 10–20개라 launch 누적 비용 위험.

### T3. panel_cap 상향 sweep

현재 `SolverConfig.panel_cap=8`로 nc ≤ 8 보장 → K가 항상 작음.

**변경**: cap을 16, 24, 32로 sweep해 다음 측정:
- 변화 후 max_fsz (case8387/USA 각각)
- tier 분포 (small/mid/big front 수 변동)
- factor 전체 wall (per-system)
- batch_relres (정확도)

**기대 효과**:
- cap=24 가정: nc 평균 1.5–2배 증가 → K-tile 활용률 증가, syncs 횟수 (3·nc+4) 증가 vs front 수 감소의 trade-off.
- T1과 결합 시 시너지: TF32 K=8 단위에서 nc=24면 K=24를 3개 tile로 분할, 모두 full.

**위험**: 낮음. cap은 이미 노출된 config 파라미터. 측정만 하면 됨. 단 docs(`03-optimization-notes/01`, doc 06)에 의하면 과거 sweep에서 cap=8이 최적이었음 — TF32 도입 후 재측정 필요.

### T4. sync-fused panel LU (장기, 비-GEMM 병목 해소)

barrier stall 41% (factor_mid B=64) 의 주범은 panel LU의 `2·nc`개 `__syncthreads`.

**대안**:
- **(a) Warp-specialized panel LU**: panel LU를 단일 warp (32 thread)에 할당해 `__syncthreads` → `__syncwarp` 로 격하. 다른 warp들은 동시에 U-solve / trailing의 stage-in을 준비 (producer-consumer).
- **(b) `cp.async` + `cuda::barrier`로 단계 overlap**: 다음 front의 stage-in을 현재 front의 panel LU와 겹침.
- **(c) Fine-grained partial barriers**: column k 완료 후 그 column만 의존하는 다음 work는 즉시 진행. Ampere의 `cuda::pipeline` API.

**기대 효과**: barrier stall 41% → 20% 미만으로 절감 시 factor_mid B=64 wall 약 25% 단축. case8387에서 panel LU가 wall의 30%인 점을 고려하면 동급 효과.

**위험**: 큼. warp specialization은 코드 구조 변경이 크고, 정확성 검증 부담. 2–4주 R&D.

### 우선순위 권장

1. **T1 (TF32 path)** — 1–2일 작업, 위험 낮음, 즉시 측정 가능. 효과가 작아도 코드는 깨끗.
2. **T3 (cap sweep)** — 1일 측정, 위험 낮음. T1과 같은 빌드에서 측정.
3. **T2 (grouped GEMM)** — 1주 작업, fused mid 구조 분해 필요. B=64 시나리오 전용.
4. **T4 (sync 재설계)** — 2–4주 R&D, 가장 큰 잠재 효과지만 가장 큰 위험.

## 8. 측정 재현 자료

`/home/claude/prof/` 에 다음 보존:

| 파일 | 내용 |
|------|------|
| `fronts_8387.csv`, `fronts_USA.csv` | per-front (p, fsz, nc, uc, level) — `CLS_DUMP_FRONTS` env로 생성 |
| `analyze_flops.py` | phase별 FLOP/메모리 분해 |
| `ng_*.nsys-rep` | nograph 빌드 nsys profile (4개 시나리오) |
| `skip_*.nsys-rep` | skip-trailing 빌드 nsys profile |
| `ncu2_*_clean.csv` | SOL/occupancy/IPC 등 ncu metric raw |
| `stall_*_clean.csv` | warp-stall raw |
| `ncu_extract.py`, `stall_table.py`, `extract_stalls.py` | 파싱 스크립트 |

`/home/claude/cls_skip_trailing/` 에 skip-trailing 변형 소스 보존. `/workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver/src/solver.cpp` 에는 `CLS_DUMP_FRONTS` env-var hook 추가됨 (production은 영향 없음, 환경변수 없으면 noop).

## 9. 기존 문서와의 관계

- `02-design-analysis/05-gemm-fraction-analysis.md` 는 trailing의 이론/실측 격차를 처음 보고. 본 문서는 (a) refactored 코드 기준으로 재측정, (b) skip-trailing 차분이라는 깔끔한 wall 분해 방법론을 추가, (c) case8387과 USA를 양쪽 다 분석.
- `04-benchmarks-profiling/08`, `10` 은 FP32 batched 측정의 canonical source. 본 문서는 그 측정 위에 GEMM/비-GEMM 분해를 얹음.
- `03-optimization-notes/archive/06-tc-dedicated-path-study.md` 는 FP16 TC가 실패한 이유를 정리. 본 문서의 T1(TF32) 제안은 doc 06의 negative result를 우회하는 별도 경로.

## 10. 다음 단계

1. T1 (`--precision tf32`) batch_relres + wall 측정.
2. T3 (cap sweep) — T1과 동일 빌드에서 cap=8/16/24/32 측정.
3. T1+T3 결과 따라 T2/T4 진행 여부 결정.
4. 본 문서의 `nsys`/`ncu` 측정은 `CLS_INTERNAL_GRAPH=OFF`에서 진행됨 — production 빌드(graph ON)에서는 launch overhead 다름. 최종 회귀 검증은 graph ON에서 재측정.
