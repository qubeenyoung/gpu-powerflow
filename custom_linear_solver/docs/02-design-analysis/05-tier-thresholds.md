# tier 임계값의 근거 — SMALL_THRESH = 32, MID_THRESH = 128

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: SMALL_THRESH=32 는 warp=32 lane HW alignment, MID_THRESH=128 은 sm_86 의 99 KB shared 에 FP32 fsz² 가 staging 여유까지 fit 하는 상한 — 둘 다 power-grid 분포 위 sweet spot 이고 sm_90 (228 KB) 에서만 MID 상향 여지.

**관련 코드**: `src/multifrontal.cu` 의 `SMALL_THRESH = 32`, `MID_THRESH = 128`, `MID_SHARED_BUDGET = 96 KB`
**선행 분석**: [`../04-benchmarks-profiling/03-gemm-fraction-front-distribution.md`](../04-benchmarks-profiling/03-gemm-fraction-front-distribution.md) (tier별 wall, 시각화) — 본 문서에서 `docs/12 §N` 으로 인용
**대상**: RTX 3090 (sm_86), CUDA 12.x

## 0. TL;DR

| 임계값 | 값 | 1차 근거 | 부수 근거 |
|--------|--:|---------|----------|
| **SMALL_THRESH** | 32 | warp = 32 lane HW alignment | sync cost vs work 균형 점 |
| **MID_THRESH** | 128 | sm_86 의 99 KB shared / FP32 fsz² fit | block size & thread-per-work 적정성 |

두 임계 모두 power-grid Jacobian 의 분포 (small dominant, nc≤20) 위에서 sweep / sweet spot 으로 안정. sm_90 Hopper(228 KB shared) 면 MID_THRESH 상향 여지.

## 1. SMALL_THRESH = 32 의 근거

### 1.1 dispatch 코드

```cpp
// src/multifrontal.cu
if (max_fsz <= SMALL_THRESH) {
    // factor_small<T> with 8 warps/block, 1 warp = 1 (front, batch)
}
```

`SMALL_THRESH=32` 이면 **fsz ≤ 32 인 레벨** 은 `factor_small` (warp-per-front) 로 dispatch. 디자인 의도 (`src/factorize/kernels.cuh`, 옛 `small.cuh`; 2026-06-06 리팩토링으로 통합):

> "the bottom etree levels... pays a full BLOCK `__syncthreads` barrier on every one of its nc rank-1 passes -> latency/occupancy bound, not compute bound. Fix: **one WARP per (front, batch)**, many warps per block. The dense no-pivot LU runs lane-parallel with `__syncwarp()`"

### 1.2 1차 근거: warp = 32 lane HW alignment

CUDA warp 는 32 lane 고정. fsz ≤ 32 일 때:
- divide step at k=0: fsz-1 entries ≤ 31 → **단일 lane 1회 sweep 으로 끝남**
- stage-in: fsz² ≤ 1024 / 32 lane = 32 iter/lane 이내
- LU trailing update at k=0: (fsz-1)² ≤ 961 / 32 lane = 30 iter/lane

fsz=33 이 되면 divide step 이 32 + 1 → 2회 iter, lane 0 이 2 entries 처리. 자연스러운 모듈로 32 정렬이 깨짐.

### 1.3 2차 근거: per-warp shared budget

per-warp shared = `fsz² × sizeof(T)`. 8 warps/block × 96 KB total → per-warp 12 KB 한도.

| fsz | fsz²·4 / warp (FP32) | 8 warp total | 한도 (96 KB) |
|----:|---------------------:|-------------:|:------------|
| 32 | 4 KB | 32 KB | ✓ 여유 |
| 48 | 9 KB | 72 KB | ✓ 마지노선 |
| 64 | 16 KB | 128 KB | ✗ 초과 |

→ warp-per-front 패턴의 **HW 상한은 fsz~48**. 그러나 SMALL_THRESH=32 로 더 일찍 끊는 이유는 §1.4.

### 1.4 3차 근거: barrier cost vs parallelism 교차점

| pattern | lanes/front | sync 종류 | sync cost |
|---------|------------:|----------|----------:|
| small (warp-per-front) | 32 | `__syncwarp` | ~1 cyc |
| mid (block-per-front) | 256 | `__syncthreads` | ~8 cyc (8 warp 정렬) |

fsz=32, nc=8 (case8387 small 의 upper bound):

| pattern | compute (iter × cyc) | sync (count × cyc) | 합 |
|---------|---------------------:|------------------:|----------------:|
| small | 8 iter/lane × 1 = 8 cyc | 2·nc × 1 = **16 cyc** | **24 cyc** |
| mid | 1 iter/lane × 1 = 1 cyc | 3·nc × 8 = **192 cyc** | **193 cyc** |

→ fsz=32 에서 small 이 mid 대비 **약 8배 효율**. sync-bound 영역. fsz=64 (가상 확장) 에서는 small 80 cyc vs mid 208 cyc — 여전히 small 우세지만 compute 비중이 sync 따라잡음. fsz=128 에서는 compute 가 sync 압도 → block 패턴 우위.

**fsz>32 에서 small 패턴이 못 쓰이는 실제 이유: §1.2 shared budget 한도 (fsz=48 마지노선) + load imbalance 위험** (docs/11 packing 실험 메타-결론: same-warp 안 fsz variance 가 SM occupancy 를 해침).

### 1.5 SMALL_THRESH 선택 과정 (역사적 합의)

`docs/03-optimization-notes/01, 06, 08`:
- THRESH=16: 너무 보수적 → mid tier 가 dispatch 폭주
- **THRESH=32**: warp HW alignment + sync sweet spot 으로 채택
- THRESH=48: `docs/03-optimization-notes/11` packing 실험에서 mid_warp 확장 시도 → variance gate 없이는 회귀

→ **32가 (1) HW 자연, (2) sync 우세 영역, (3) variance 안전한 임계로 합의**.

### 1.6 dispatch 결과 (case8387 / USA)

`docs/12 §10.1.1`:
- case8387: 29 levels 중 7개 (24%) 가 순수 small → factor_small dispatched
- USA: 40 levels 중 5개 (12.5%) 가 순수 small → factor_small dispatched

→ 임계 변경 시 (예: 48) 더 많은 mixed 레벨이 small 로 흡수되지만, mixed 안 큰 front 의 efficient routing 손실. 현 임계가 trade-off 균형.

## 2. MID_THRESH = 128 의 근거

### 2.1 dispatch 코드

```cpp
if (max_fsz <= MID_THRESH) {
    const size_t shb_tiled = fsz_cap² * elt + 2 * level_max_nc * level_max_uc * elt;
    if (shb_tiled <= MID_SHARED_BUDGET) {
        // factor_mid<T> with 256 threads/block, 1 block = 1 (front, batch)
        return;
    }
    // fall-through to big
}
// factor_big<T> with 1024 threads/block, global memory direct
```

`MID_THRESH=128` 은 **mid kernel 시도의 max fsz**. 그 안에서도 shared budget 동적 체크가 추가 가드.

### 2.2 1차 근거: 96 KB shared budget / FP32 front 전체 stage-in

mid kernel 은 **front 전체를 shared 에 복사** (`docs/12 §9.2`):
- `Fs[fsz_cap²]` ← 전체 front (지배적)
- `sh_L[uc·nc]`, `sh_U[nc·uc]` ← trailing GEMM staging (작은 부분)

sm_86 dynamic shared per block 한도: **99 KB** (opt-in via `cudaFuncSetAttribute(..., 99 * 1024)`). 코드 가드: 96 KB (`MID_SHARED_BUDGET`).

| fsz | fsz²·4 (front, FP32) | + 2·nc·uc·4 (USA worst nc=20, uc=108) | 합 | 96 KB 한도 |
|----:|---------------------:|--------------------------------------:|---:|----------:|
| 96 | 36 KB | 17 KB | 53 KB | ✓ 여유 큼 |
| 112 | 49 KB | 17 KB | 66 KB | ✓ |
| **128** | **64 KB** | **17 KB** | **81 KB** | ✓ (sweet spot, 81% 사용) |
| 144 | 81 KB | 22 KB | 103 KB | ✗ 초과 |
| 160 | 100 KB | – | – | ✗ |

→ **128² × 4 = 64 KB ≈ 96 KB 의 2/3**. staging panel (~17 KB) 더해 81 KB. 나머지 ~15 KB 는 register spill 여유. **128이 staging 여유 확보하는 최대 fsz**.

### 2.3 2차 근거: FP64 자동 fall-through

FP64 elt = 8 byte. fsz=128 일 때 128²·8 = **128 KB** → 한도 초과. 코드의 `if (shb_tiled <= MID_SHARED_BUDGET)` 가드가 자동으로 잡아 → big tier fall-through.

→ **MID_THRESH=128 은 FP32 기준 상한, FP64 는 동적으로 더 일찍 (fsz~96) 빠짐**.

### 2.4 3차 근거: 256 thread block 의 적정 thread당 work

mid kernel = 256 thread / 8 warp. work 분석 (fsz=128 max):

| 단계 | total work | per-thread (256 lane) |
|------|-----------:|---------------------:|
| stage-in | fsz² = 16384 | **64 iter/thread** |
| panel LU | nc·fsz ≈ 1024 | 4 iter/thread |
| U-solve | nc²·uc ≈ 11k | 43 iter/thread |
| trailing | uc²·nc = 108²·20 ≈ 234k | **914 ops/thread** |
| writeback | nc·fsz + uc·nc ≈ 2k | 8 ops |
| extend-add | uc² ≈ 11k atomic | 43 ops/thread |

256 thread 가 ~914 trailing ops/thread → register pressure / occupancy 적정. fsz=160 으로 가면 trailing 140²·20 ≈ 392k / 256 = **1531 ops/thread** (over-subscribed) + stage-in 160²/256 = 100 iter/thread 만으로 wall 큰 비중.

→ fsz=160 부근부터는 **block size 키우거나 (1024 thread big kernel) shared 포기하고 global memory direct 가 더 효율**.

### 2.5 WMMA tile 제약 (non-binding for power-grid)

factor_mid_tc / factor_mid_tf32 활성 조건:

```cpp
if (nc <= 32 && uc <= 256) {
    trailing_update_wmma_*(...);
}
```

- **nc ≤ 32**: FP16 WMMA K=16 × 2 fragments (`af[2]`) → 32 K limit, 또는 TF32 K=8 × 4 fragments
- **uc ≤ 256**: shared staging stride (32 × 32 fragment 정렬)

power-grid 실측 (docs/12 §3.3): USA mid nc ≤ 20, uc ≤ 111 → **두 조건 모두 만족**; USA big nc ≤ 20, uc ≤ 215 → 만족.

→ **WMMA tile 제약은 본 워크로드에서 binding 아님**. 3D PDE 등 nc > 32 워크로드에서는 nc 가 먼저 binding 될 수 있음.

### 2.6 big tier (fsz > 128) 의 디자인

```cpp
constexpr int bigT = 1024;  // 32 warp
factor_big<float><<<grid, bigT, 0, stream>>>(...)  // no shared!
```

- **1024 thread / 32 warp**: fsz=235 (USA big max) trailing 234²·20 ≈ 1100k ops 처리, thread당 ~1100 ops 적정
- **global memory direct**: shared 부족 → staging 포기. L1 cache + 32-warp scheduler 의 ILP 로 latency hide
- per-front parallelism 4x (256 → 1024) → 큰 uc² trailing throughput ↑

trade-off: 잃는 것 = shared cache locality (mid stage-in 효율), 얻는 것 = 더 큰 block parallelism + global memory 거대 working set. fsz>128 에서 후자 우위 — MID_THRESH=128 직후 big 전환 이유.

### 2.7 MID_THRESH 선택 과정

`docs/03-optimization-notes/01, 02`:
- THRESH=96: 너무 보수적, USA mid 큰 front 들이 big 으로 빠져 wall ↑
- **THRESH=128**: shared budget sweet spot. 81% 사용. staging 여유 확보
- THRESH=159: 초기 시도 (`docs/06-tc-dedicated-path-study.md` 의 `MID_THRESH=159`) — TF32 K=8 padding 분석 후 128 로 표준화

→ **128이 (1) shared budget 한도, (2) block size 효율, (3) TC fragment 정렬 의 종합 sweet spot**.

## 3. 향후 하드웨어 변화 시

| HW | dynamic shared / block | 예상 MID_THRESH 상한 |
|----|----------------------:|---------------------:|
| sm_86 (RTX 3090, A6000) | 99 KB | **128** (현재) |
| sm_89 (RTX 4090) | 99 KB | 128 (동일) |
| sm_90 (Hopper H100) | 228 KB | 192 (192²·4 = 144 KB < 228) |

sm_90 시:
- MID_THRESH=192 로 상향 가능 → USA big fronts (max fsz=235) 일부 흡수
- 단 block size (현 256) 도 검토 필요 — fsz=192 trailing 1500 ops/thread 라면 thread 늘리거나 register pressure 조정
- 단 **block 크기 1024 가 big tier 의 정체성** — block size 변동은 dispatch 모델 전체 재설계 필요

본 솔버는 현재 sm_86 가정. sm_90 대응은 별도 SP 에서.

## 4. SMALL_THRESH 변경 가능성

SMALL_THRESH 는 HW (warp=32) 에 묶여 **변경 거의 불가**:
- 16 으로 낮추면: warp 패턴은 동작하지만 underutilize, mid 가 너무 많이 처리 → 회귀
- 48 등으로 올리면: docs/03-optimization-notes/11 packing 실험의 mid_warp 회귀와 같은 문제 (load imbalance 위험)

→ **32가 HW 동기화 + sync trade-off 둘 다의 sweet spot** — sm_90 에서도 동일.

## 5. 정리: 두 임계의 dimensional 비교

| 측면 | SMALL_THRESH=32 | MID_THRESH=128 |
|------|----------------|---------------|
| binding HW constraint | warp lane count | dynamic shared budget |
| 1차 trade-off | sync cost vs parallelism | shared usage vs block size 효율 |
| FP64 영향 | 동일 (lane count 무관) | 자동 fall-through (96/64=1.5x) |
| sm_90 영향 | 동일 (warp 변동 없음) | 상향 여지 (192) |
| 본 솔버에서의 안정성 | 매우 안정 | 매우 안정 (FP32 위주) |

→ 두 임계 모두 power-grid + RTX 3090 가정 위에서 **정합한 design choice**. 변경 여지는 sm_90 + nc>20 워크로드에서 등장.

---

## 관련 문서
- `../main-report.md` — 전체 서사 맥락
- [`04-gemm-fraction-tc-ceiling.md`](04-gemm-fraction-tc-ceiling.md) — tier별 trailing 비중 + TC ceiling (본 임계 위 분석)
- [`01-why-custom-fast.md`](01-why-custom-fast.md) D2 — 3-tier 커널 라우팅 (본 임계의 상위 맥락)
- [`../03-optimization-notes/03-tensor-core-investigation.md`](../03-optimization-notes/03-tensor-core-investigation.md) — WMMA tile 제약 (§2.5) 의 상세
- [`../04-benchmarks-profiling/03-gemm-fraction-front-distribution.md`](../04-benchmarks-profiling/03-gemm-fraction-front-distribution.md) — tier별 wall 분포 (docs/12)
