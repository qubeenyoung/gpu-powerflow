# 비-GEMM (sync) 병목 해소 계획 — T4 상세 설계

> **2026-06-06 update — 파일 경로 정정**: 본 문서가 참조하는 `src/factorize/primitives.cuh` / `small.cuh` / `mid.cuh` / `mid_warp.cuh` 는 모두 리팩토링으로 위치 변경됨. primitives/small/mid 의 device 빌딩 블록은 **`src/factorize/phases.cuh`** 로, kernel 들은 **`src/factorize/kernels.cuh`** 로 통합. T4.1 의 `mid_warp.cuh` 는 net loss 결정으로 **`deprecated/mid_warp/`** 로 이동. 자세한 매핑은 [`01-orientation/06-factorize-file-layout-2026-06-06.md`](../01-orientation/06-factorize-file-layout-2026-06-06.md). 측정 결과는 [`docs/10`](10-t4.1-t4.3-results-2026-06-06.md) (T4.3 / T4.2.A 는 default ON 유지).

**작성일**: 2026-06-06
**대상 베이스**: refactored `custom_linear_solver` (`41897cd`)
**선행 문서**: [`04-benchmarks-profiling/11-fp32-factorize-gemm-vs-nongemm-2026-06-06.md`](../04-benchmarks-profiling/11-fp32-factorize-gemm-vs-nongemm-2026-06-06.md)
**핵심 문제**: factor_mid/big의 FP32 path는 panel LU 단계의 `__syncthreads()`가 wall의 37–41% (B=64, ncu barrier stall)를 차지. 이 비-GEMM 비용을 줄이는 게 목표.

## 0. TL;DR

- **이미 처리된 영역**: factor_small은 이미 `__syncwarp` + warp-per-front 패턴 사용 (B=64에서 barrier stall 미관측). 본 계획은 **factor_mid (33≤fsz≤128) + factor_big (>128)** 의 FP32 path 전용.
- **재시도하지 말 것**: spine 메가커널 (+19% 회귀), multi-stream race, sibling amalgamation (+72% 회귀), Σ.5 amalgamate (cap≥16에서 +72%).
- **시도된 적 없는 영역**: cp.async (`cuda::memcpy_async`), `cuda::barrier::arrive/wait` split, mid tier의 warp-per-front, persistent block per level.
- **계획 4 phase**:
  - **T4.1**: factor_small의 warp-per-front 패턴을 mid tier (fsz ≤ 64)로 확장. *우선 구현*.
  - **T4.2**: 남는 mid (64<fsz≤128)와 big에서 sub-block 분할 + `__syncwarp`. T4.1의 fallback.
  - **T4.3**: `cuda::memcpy_async` (cp.async)로 stage-in 비동기화 → B=1 scoreboard stall 완화.
  - **T4.4** (stretch): `cuda::barrier::arrive/wait` split + cross-front persistent block. R&D, T4.1–3 후 결정.

## 1. 현 상태 정확히 파악

### 1.1 `__syncthreads` 인벤토리 (refactored 코드, `src/factorize/`)

```
small.cuh        : __syncthreads 0개, __syncwarp 4개         (이미 최적)
primitives.cuh   : __syncthreads 5개 (lu_panel_factor, u_panel_solve 내부)
mid.cuh          : __syncthreads 8개 (factor_mid/factor_mid_tc 본체), __syncwarp 2개 (WMMA 내부)
big.cuh          : __syncthreads 2개 (extend-add 직전 + 내부)
                   + primitives.cuh의 5개를 factorize_front로 inherit
```

`factor_mid<float>` 한 호출당 (FP32 path) total syncs:
- stage-in 1
- `lu_panel_factor`: 2·nc (각 k의 divide→update 사이 1, k 끝 1)
- `u_panel_solve`: nc-1 (각 k의 row update 후 1)
- `trailing_update_staged`: 2 (stage L/U 후, GEMM 후)
- writeback 전후 2
- extend-add 직전 1

→ **per-front 합계 ≈ 3·nc + 5**. case8387 (nc 평균 6): **23번/front**. USA mid (nc≤20): **65번/front**.

### 1.2 factor_small이 이미 푸는 문제

`src/factorize/small.cuh:9-13` 의 주석이 분명히 말함:

> "the bottom etree levels... pays a full BLOCK `__syncthreads` barrier on every one of its nc rank-1 passes -> latency/occupancy bound, not compute bound.
> Fix: **one WARP per (front, batch), many warps per block**. The dense no-pivot LU runs lane-parallel with `__syncwarp()` (a warp barrier, far cheaper than a 4-warp block barrier)"

핵심 패턴 (`lu_small_warp`, small.cuh:27-44):
```cpp
for (int k = 0; k < nc; ++k) {
    FT piv = F[k*fsz+k];
    for (int i = k+1+lane; i < fsz; i += 32) F[i*fsz+k] /= piv;
    __syncwarp();                     // ← BLOCK 대신 WARP barrier
    const int m = fsz - k - 1;
    for (int e = lane; e < m*m; e += 32) {
        F[ii*fsz+jj] -= F[ii*fsz+k] * F[k*fsz+jj];
    }
    __syncwarp();
}
```

**한 warp가 한 front 전체를 담당** → 32 lane이 협력. front이 작아서 32 lane으로 충분. block당 W warps로 W개 front를 동시 처리.

### 1.3 측정으로 본 효과 증거

[`04-benchmarks-profiling/11`](../04-benchmarks-profiling/11-fp32-factorize-gemm-vs-nongemm-2026-06-06.md) §5.2의 stall 1순위:

| 커널 | B=1 1순위 stall | B=64 1순위 stall |
|------|----------------|------------------|
| factor_small (이미 warp-pattern) | scoreboard 36% | (threshold 미달, 다양한 stall 분산) |
| factor_mid (block-pattern) | scoreboard 31% | **barrier 41%** |
| factor_big (block-pattern) | barrier 36% | **barrier 37%** |

→ warp-per-front 패턴을 mid/big에 적용하면 barrier stall 37–41% → 10–15% 수준 회수 가능 (small이 그 증거).

## 2. 이전 시도 검토 (재시도 회피)

| 시도 | 결과 | 출처 | 회피해야 할 함정 |
|-----|------|------|-----------------|
| Spine 메가커널 (Phase 4) | +19% 회귀 (41→49 μs) | `08-tree-restructuring-research-plan.md` §9.2 | 큰 persistent kernel은 register pressure / barrier overhead로 dispatch 절감 압도. **그러나 "level 내부 persistent"는 다른 scope이고 미시도** |
| Multi-stream subtree (Phase 3) | wall -44% but relres NaN (1e+28) | 같은 문서 §10 | CUDA Graph fork/join이 memory ordering 보장 X. **Race-free한 fine-grained sync 도구가 필요** |
| Sibling amalgamation (Phase 2) | +72% 회귀 (cap=16) | `07-symbolic-gemm-research.md` §10.5 | nc를 키우면 per-front O(fsz³)가 sync 절감을 압도. **nc는 그대로 두고 sync 비용만 낮춰야 함** |
| FP16 WMMA TC | +25–49% 회귀 + NaN | `03-optimization-notes/archive/06-tc-dedicated-path-study.md` | TC는 별도 (`docs/11` T1 TF32 제안에서 다룸). 본 계획은 sync만 |
| Σ.5 etree amalgamate (cap=16) | +72% 회귀 | `07-symbolic-gemm-research.md` §10.5 | 동일: nc 키우기 금지 |
| Σ.6–7 cuBLAS grouped-batched | wall ±1–2% | `07-symbolic-gemm-research.md` §12.7 | cuBLAS는 wall lever 아님 |

**결론**: "더 크게 / 더 합치게 / 더 fuse하게"는 다 실패. 본 계획은 정반대 방향 — **현재 dispatch 구조를 유지하면서 barrier 종류를 block→warp로 격하 + cp.async로 memory latency hide**.

## 3. T4 4-Phase 계획

### Phase T4.1 — Mid tier의 warp-per-front (fsz ≤ 64)

**대상**: case8387의 mid level (max fsz ≤ 79, 55개 front) + USA mid의 하단부 (fsz ≤ 64, 추정 ~700 front).

**변경 내용**:
- 새 kernel `factor_mid_warp<T>` 작성 (small.cuh의 패턴 그대로 확장)
- block당 W warps, 각 warp가 한 (front, batch) 담당
- panel LU + U-solve + trailing 모두 `__syncwarp` 만 사용
- multifrontal.cu의 dispatch에 `max_fsz <= WARP_MID_THRESH` 분기 추가 (제안: 48 또는 64)
- shared budget: W warps × fsz²·4 byte. fsz=64면 W warp당 16 KB → 96 KB / 16 = **6 fronts/block 동시 처리**

**구현 위치**:
- 새 파일: `src/factorize/mid_warp.cuh`
- multifrontal.cu의 `issue_factor_level_range` 분기 수정 (line 104 근처 `SMALL_THRESH` 분기 위에 `WARP_MID_THRESH` 분기 추가)

**예상 효과**:
- factor_mid B=64 barrier stall 41% → ~15% (small 수준)
- factor_mid B=64 wall 47 μs/inst → **~32–35 μs/inst** (25–30% 단축)
- factor 전체 wall에서 mid가 60% (case8387) / 47% (USA mid 부분) → **case8387 B=64 −16–18%**, USA B=64 −9–10%
- B=1에서는 occupancy 증가도 추가 이득 (현재 28%) → wall 추가 단축 기대 (정량 측정 필요)

**위험**:
- 낮음. factor_small에서 이미 검증된 패턴 (`02-design-analysis/01-why-custom-fast-on-power-grid.md` 의 D2).
- 단, fsz=64에서 한 warp(32 lane)가 64² = 4096 entry를 담당 → 128 entry/lane. nc=8이면 LU 패스가 8개 → 동작은 작지만 launch overhead 늘 수 있음. **임계값 sweep 필수** (32 / 48 / 64 / 80).

**측정 (Go/No-Go 기준)**:
1. case8387/USA × B=1/64 × fp32 nsys: factor_mid wall median
2. ncu barrier stall %: WARP_MID 적용 level에서 < 20% 확인
3. batch_relres: 현 fp32 대비 동등 (단, no-pivot LU 수치는 동일하므로 변동 없을 것)
4. **Go**: mid wall ≥ 15% 단축 + 정확도 유지
5. **No-Go**: 단축 < 10% → T4.2로 이동

### Phase T4.2 — Sub-block 분할 (64 < fsz ≤ 128, factor_big의 일부)

**대상**: T4.1이 커버 못 하는 mid 상단 + factor_big 의 작은 fsz 영역.

**변경 내용**:
- factor_mid<float> 본체 유지 (1 block per (front, batch))
- 단, **2개 sub-block 사용**: 256 thread block을 2개 128-thread "half-block"으로 논리적 분할
- 각 half가 front의 row 절반씩 담당
- panel LU 의 column k divide는 row를 절반씩 나누어 두 half가 동시에 처리
- half 내부는 `__syncwarp` (왜냐하면 128 thread = 4 warp인데, 같은 half의 warp 4개끼리만 sync 필요한 경우)
- half 간 sync는 `__syncthreads` 유지 (어쩔 수 없음)

**핵심**: panel LU의 rank-1 update는 (fsz-k-1) × (nc-k-1) 출력을 가짐. 이를 half-block 단위로 row 분할하면 **column k의 piv broadcast 외에는 cross-half 의존성 없음**. 그러면 rank-1 update 내부는 `__syncwarp` 가능, k 끝의 sync만 `__syncthreads`.

**기대**: per-front sync 23회 중 12회를 `__syncwarp`으로 격하 → barrier stall ~40% → ~25%.

**위험**:
- 중. half-block 분할은 컴파일러 최적화에 의존. CUDA Cooperative Groups API (`thread_block_tile<128>`) 사용 권장.
- factor_mid 의 staged trailing이 256-thread 전체로 동작하므로 half-block 분할이 trailing phase와 충돌 가능 — half 모드는 panel-LU 단계에만 적용, trailing은 풀 block.

**측정**:
- 같은 ncu 비교. 단축 ≥ 10%면 채택, 미만이면 폐기.

**대안 (덜 침습적)**: panel LU 내부의 sync 일부만 `__syncwarp(activemask())`로 격하 — 정합성 확인 필요. CUDA programming guide 권장 패턴 아니라 보수적 접근.

### Phase T4.3 — cp.async stage-in (B=1 scoreboard 완화)

**대상**: B=1의 scoreboard stall 31–36% (factor_mid/big 전부). B=64에서도 stage-in이 차지하는 wall 약 10% 회수 기대.

**현 코드** (`mid.cuh:145-147`):
```cpp
// Stage the front into shared.
for (int e = t; e < fsz2; e += nt) Fs[e] = F[e];
__syncthreads();
```

이는 global → shared 복사. 모든 thread가 `Fs[e] = F[e]`로 load-store하며, `__syncthreads`에서 모두 대기. B=1에서 global memory latency가 cover 안 됨 (다음 phase가 stage-in 결과를 즉시 사용).

**변경**:
```cpp
#include <cuda/barrier>
#include <cuda/pipeline>
namespace cg = cooperative_groups;

__shared__ cuda::barrier<cuda::thread_scope_block> bar;
auto block = cg::this_thread_block();
if (block.thread_rank() == 0) init(&bar, block.size());
block.sync();

// Asynchronous copy: cp.async loads global directly into shared, no register round-trip
cuda::memcpy_async(block, Fs, F, sizeof(T) * fsz2, bar);
bar.arrive_and_wait();
```

`cuda::memcpy_async` 는 Ampere의 `cp.async.cg` instruction으로 컴파일되어:
- **register file을 거치지 않고** global→shared 직접 복사
- load latency 동안 다른 instruction 발행 가능 (compiler가 자동 schedule)
- L1 cache pollution 감소

**기대**:
- B=1 factor_mid SOL 5.7% → 추정 12–15% (occupancy/issue rate 증가)
- B=1 factor_mid wall 12.8 μs → ~10 μs/inst (15–20% 단축)
- B=64 영향 작음 (~5%, stage-in이 이미 짧음)

**위험**:
- 낮음. CUDA 11+의 표준 API, 잘 문서화됨.
- 단, Ampere(sm_80+) 필요. RTX 3090(sm_86)은 지원. 빌드에서 sm_60 fallback path는 유지 (`#if __CUDA_ARCH__ >= 800`).

**구현 위치**:
- `src/factorize/primitives.cuh` 에 `stage_in_async<T>()` helper 추가
- mid.cuh / big.cuh에서 stage-in 부분만 교체
- writeback도 동일한 패턴 적용 가능 (shared→global, but `cp.async`는 단방향 — writeback은 일반 store 유지)

### Phase T4.4 — `cuda::barrier::arrive/wait` split + persistent within level (stretch)

**조건부**: T4.1–T4.3 결과가 좋은 경우에만 진행. R&D 성격.

**아이디어 (split barrier)**:
```cpp
auto tok = bar.arrive();   // 신호만 보내고 대기 X
// ... 다른 일 수행 (예: 다음 front 의 stage-in 예약)
bar.wait(std::move(tok));  // 이제 대기
```

panel LU의 sync 사이에 "다른 일"이 거의 없으므로 단순 적용은 효과 미미. 의미 있으려면:

**Cross-front persistent 패턴**:
- 한 block이 **한 level 내의 여러 (front, batch)** 를 순차 처리
- front i의 trailing GEMM 동안 front i+1의 stage-in을 `cp.async`로 시작
- front i+1의 panel LU 동안 front i+2의 stage-in 시작
- 진정한 latency hiding

**위험**: 큼. `08-tree-restructuring-research-plan.md` 의 spine 메가커널 (+19% 회귀) 와 구조적으로 유사하지만 scope가 작음 (level 내부, level 간이 아님). 그래도 다음 위험 존재:
- (a) register pressure 증가
- (b) work queue (atomic counter)의 contention
- (c) level 내 front 수가 적으면 (case8387 big level은 0–3개) 효과 없음

**Decision**: T4.1+T4.3 적용 후 USA B=1 factor_big의 barrier stall이 여전히 30%+ 면 시도.

## 4. 작업 순서와 측정 계획

### Step 1 (1–2일): T4.1 prototype

1. `src/factorize/mid_warp.cuh` 작성 — small.cuh를 fsz≤64로 확장
2. `multifrontal.cu` dispatch 분기 추가, `WARP_MID_THRESH=64` 시작
3. 빌드 + 정확도 회귀 (case30 case118 case8387 batch_relres)
4. nsys: case8387/USA × B=1/64 × fp32
5. ncu: factor_mid_warp의 barrier stall < 20% 확인
6. THRESH sweep: 32/48/56/64/80에서 mid 전체 wall 측정 → 최적값 결정

### Step 2 (1일): T4.3 cp.async

1. `cuda::memcpy_async` helper 추가
2. factor_mid (T4.1 후 fsz>WARP_MID_THRESH path) 와 factor_big에 적용
3. 측정: B=1 factor_mid/big scoreboard stall 추적, wall 단축률
4. T4.1 + T4.3 합성 효과 측정

### Step 3 (Go/No-Go 결정): 결과 평가

- case8387 B=64 factor wall: 기준 24.6 μs/sys, 목표 ≤ 20 μs/sys (18% 단축)
- USA B=64 factor wall: 기준 473 μs/sys, 목표 ≤ 400 μs/sys (15% 단축)
- USA B=1 factor wall: 기준 3.0 ms/sys, 목표 ≤ 2.5 ms/sys (17% 단축)
- 정확도: batch_relres 변동 ≤ 5%
- 위 목표 미달 시 **T4.2 (sub-block 분할)** 진행
- T4.2 후에도 목표 미달이면 **T4.4 (persistent within level)** 진행

### Step 4 (조건부, 2주+): T4.4

- T4.1+T4.3+T4.2가 목표 미달일 때만
- 별도 brach에서 prototype, USA B=1 case 위주
- spine 메가커널의 +19% 회귀와 구조적 차이 (level 내부 vs level 간) 를 ncu로 확인 — register pressure / barrier overhead 비교

## 5. 구현 시 주의사항

### 5.1 정확성

- no-pivot LU의 정확성은 입력 ordering(METIS ND)에 의존. T4.x는 numerical algorithm을 바꾸지 않음 — 동일한 연산을 동일한 순서로 수행, sync 구조만 변경.
- 단, warp-per-front 패턴에서 한 warp가 column k의 모든 divide를 마치기 전에 다른 lane이 rank-1 update를 시작하면 안 됨 → `__syncwarp` 필수 보장.
- `cuda::memcpy_async` 후 반드시 `bar.arrive_and_wait()` (또는 `__pipeline_wait_prior`) — 없으면 stage-in이 끝나기 전에 LU 시작 가능.

### 5.2 architecture compatibility

- `cuda::memcpy_async`: sm_80+ (Ampere). `CLS_CUDA_ARCHITECTURES` 기본 86 (3090). 빌드 옵션에 sm_60 같이 낮은 arch 포함되어 있으면 fallback 필요:
  ```cpp
  #if __CUDA_ARCH__ >= 800
      // cp.async path
  #else
      // 기존 synchronous load path
  #endif
  ```
- T4.1 (warp-per-front)은 architecture 무관 (`__syncwarp`은 sm_30+).

### 5.3 측정 인프라

- nsys/ncu 측정을 위해 `CLS_INTERNAL_GRAPH=OFF` 빌드 활용 (`docs/11`의 측정 환경)
- skip-trailing 빌드 (`CLS_SKIP_TRAILING=1`) 으로 비-GEMM wall 차분 측정 가능
- `CLS_DUMP_FRONTS` env로 front 분포 검증 (`docs/11` §1.3)

### 5.4 회귀 방지

- production 빌드 (`CLS_INTERNAL_GRAPH=ON`) 에서 wall 회귀 없는지 별도 측정. T4.1은 dispatch path만 추가하므로 영향 작아야 함.
- 모든 변경은 default OFF 환경변수로 토글 가능하게 (`CLS_USE_MID_WARP=1`, `CLS_USE_CP_ASYNC=1`) — 회귀 시 즉시 off.
- 정확도 회귀 테스트: `tests/run_custom_solver` 의 batch_relres를 case30/case118/case8387/case_ACTIVSg2000/case_SyntheticUSA 5개 케이스 × fp32 × B=1/64 에서 측정. 임계값: 변동 ≤ 5%.

## 6. 미해결 / 후속 과제

- **B=1의 scoreboard stall** (35%): cp.async로 일부 회수 기대 (~15%). 남은 20%는 occupancy 부족 (factor_mid B=1: 28%). occupancy 증가는 (a) 더 많은 block 동시 launch — 어차피 grid가 작아 한계 (b) register 줄이기 — 컴파일러 의존 (c) warp-per-front (T4.1) 가 부분적으로 해소.
- **factor_big의 barrier stall 37% (USA)**: T4.1은 fsz≤64만 커버 → big tier (fsz>128)는 T4.2/T4.4 의 대상. USA의 big front 25개가 USA B=1 wall의 76% → T4.4 진행 시 USA B=1 우선.
- **U-panel solve의 직렬성** (nc-1 sync): 본 계획은 sync 종류를 바꾸지만 횟수는 그대로. nc=8/20에서 nc-1=7/19 sync. 횟수 자체를 줄이려면 알고리즘 변경 (예: column-block U-solve, lookahead) — `08-tree-restructuring-research-plan.md` 의 *block U-solve* 미시도 영역.

## 7. 측정 재현 자료 위치

`docs/11`과 동일 자료 활용:
- `/home/claude/prof/ng_*.nsys-rep` — refactored + nograph 빌드의 baseline nsys
- `/home/claude/prof/ncu2_*_clean.csv` — SOL/occupancy ncu
- `/home/claude/prof/stall_*_clean.csv` — warp-stall raw
- `/home/claude/build_cls_nograph/`, `/home/claude/build_cls_skip/` — 측정용 빌드

T4 진행 시 새 측정은 `/home/claude/prof/t4_*` 명명규약으로 저장.

## 8. 관련 docs

- 같은 측정 기반 GEMM 분석: [`04-benchmarks-profiling/11`](../04-benchmarks-profiling/11-fp32-factorize-gemm-vs-nongemm-2026-06-06.md)
- factor_small의 warp-pattern 근거: [`02-design-analysis/01`](../02-design-analysis/01-why-custom-fast-on-power-grid.md), [`03-optimization-notes/01`](01-fp32-batched-kernel-optimization.md)
- 재시도 회피 reference (모두 `archive/` 로 이동된 negative-result 문서):
  - [`archive/08-tree-restructuring-research-plan.md`](archive/08-tree-restructuring-research-plan.md) §9.2 — spine fusion / multi-stream race / sibling amalgamation 결과
  - [`archive/07-symbolic-gemm-research.md`](archive/07-symbolic-gemm-research.md) §10.5 — Σ.5 amalgamate +72% 회귀
  - [`archive/06-tc-dedicated-path-study.md`](archive/06-tc-dedicated-path-study.md) — FP16 TC 회귀
