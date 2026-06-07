# [DEPRECATED] M2 R&D — prototype implementation status (round 3 update)

> **2026-06-07 폐기.** Prototype kernel 의 wall +240~820% 회귀가 확정되어 본 R&D 전체를 무기한 보류. 관련 코드 (host `compute_micro_subtrees` ~158 lines, kernel `factor_small_subtree` ~130 lines, `MultifrontalPlan` 의 micro-subtree 필드 14개, dispatch hook, `cudaFuncSetAttribute` 등록) 전체 삭제. `CLS_M2_PROTOTYPE` 매크로도 제거. 본 문서는 재시도 시 참조용 historical R&D log.

**작성일**: 2026-06-07
**선행**: [docs/19 M2 Design (deprecated)](19-m2-persistent-subtree-rd-2026-06-07.deprecated.md)
**상태**: ~~bug fixed, accuracy 정상~~. **wall +240~820% 회귀** — naive 1-block-per-subtree 패턴의 thread utilization 부족. **DEPRECATED 2026-06-07**, 코드 삭제.

## 0. TL;DR (라운드 3 업데이트)

본 라운드의 work:
- **Bug 진단 + FIX**: 원인 = **mid/big factor kernels 가 covered[] check 없어 subtree members 를 double-process** 했음.
  - case8387 의 11개 level (Level 2-12) 에서 small panel (fsz ≤ 14) 와 mid panel (fsz > 32) 공존. dispatch 가 max_fsz 기반으로 라우팅 → 전체 level 이 factor_mid 로 → factor_mid 가 subtree members 도 처리 → 이미 factored 된 데이터 위에 다시 factor → garbage.
  - **Fix**: factor_mid_tf32, factor_mid_tf32_k4, factor_big_tf32 에 covered check (factor_small 과 같은 패턴) 추가. dispatch 에서 plan.d_microst_covered 전달.
- **추가 partial fix**: lu_small_warp → lu_small_front (warp-per-front vs block-per-front pattern mismatch 수정. 이 fix 단독으로는 정확도 회복 안 했지만 architectural 정합성).
- **Accuracy 검증** (B=64, 3-trial):
  - case30 / case118: **perfect** (single-subtree case, 한 번도 안 깨졌었음)
  - case8387: M2 0.004-0.022 ≈ V0 0.007-0.022 ✓
  - ACTIVSg25k: M2 0.012-0.036 ≈ V0 0.012-0.014 ✓
  - USA: M2 0.067-0.093 ≈ V0 0.073-0.105 ✓
- **Wall 측정 (B=64)**:
  - case8387: V0 0.027 → M2 0.249 ms = **+823% 회귀**
  - ACTIVSg25k: V0 0.112 → M2 0.523 ms = **+366% 회귀**
  - USA: V0 0.509 → M2 1.731 ms = **+240% 회귀**

**Architectural 결론**:
- M2 prototype 의 **memory traffic 감소 잠재 (90% stage-in)** 가 **compute parallelism 손실** 로 잠식.
- 1 block per (subtree, batch) 의 sequential processing → fewer blocks/SM (12K vs V0's 56K), 256-thread block 에 fsz=10 등 작은 panel 처리 시 thread 대부분 idle.
- 잠재 게인 실현하려면 **parallel within-subtree processing** (parallel sibling factorize) 또는 **smaller block size + bigger grid** 등 추가 architecture R&D 필요.

## 1. Bug 진단 — root cause

### 1.1 hypothesis 가 맞은 부분

docs/20 round 2 의 §2.3 "미검토 의심 영역" 중 #4: **asm_local indexing 의 의미 — in-shared 와 global atomic 다른 해석 가능성** → 부분 진단 가까웠으나 핵심은 #4 도 아니었음.

진짜 원인: **dispatch level routing 의 unintended side effect**.

### 1.2 원인 분석 — level mixing

case8387 의 fronts dump (CLS_DUMP_FRONTS) 분석 결과 — 12 levels 에서 small (fsz ≤ 14) 과 mid (fsz > 32) 공존:

| Level | Total panels | small (fsz≤14) | mid (fsz>32) | max_fsz |
|-------|-------------|---------------|--------------|---------|
| 2 | 732 | 717 | 1 | 33 |
| 3 | 399 | 380 | 1 | 34 |
| 4 | 243 | 192 | 2 | 41 |
| 5 | 139 | 88 | 4 | 53 |
| 6 | 86 | 38 | 5 | 55 |
| 7 | 61 | 20 | 4 | 76 |
| 8 | 44 | 10 | 6 | 68 |
| 9-14 | 5-25 | 1-2 | 2-6 | 55-60 |

dispatch 로직 (issue_factor_level_range):
```cpp
if (max_fsz <= SMALL_THRESH) { factor_small(...); return; }
if (max_fsz <= MID_THRESH)  { factor_mid_*(...); return; }
// BIG tier
```

Level 2 (max_fsz=33): **dispatch 가 factor_mid_tf32 로** (33 > 32 = SMALL_THRESH). 717 small panels 중 subtree members 도 factor_mid_tf32 가 처리.

`factor_mid_tf32` 는 round 2 까지 covered check 없었음 → subtree kernel 이 이미 factor 한 panel 을 다시 stage-in (이번엔 factored L+U 가 있는 상태) → lu_small_front (mid version) 이 다시 factor 시도 → garbage.

### 1.3 Fix

`src/factorize/kernels.cuh` 의 factor_mid_tf32 / factor_mid_tf32_k4 / factor_big_tf32 시그니처에 `covered` 파라미터 (default nullptr) 추가. Early-exit pattern:

```cpp
const int p = plcols[idx];
if (covered != nullptr && covered[p]) return;
```

`src/factorize/dispatch.cuh` 의 call sites 에 `plan.d_microst_covered` 전달 (CLS_M2_PROTOTYPE 시).

이 fix 만으로 **case8387, ACTIVSg25k 의 V0-동급 accuracy 복귀**.

### 1.4 추가 fix — lu_small_warp → lu_small_front

내 prototype kernel 의 phase-1 호출 `if (t < 32) lu_small_warp(...);` 가 부적절. lu_small_warp 는 warp-per-front pattern (32 lanes 만 사용) 용. block-per-front pattern 에서는 lu_small_front (all-thread) 사용해야 함. 이 mismatch 가 단독으로는 정확도 안 깼으나 architectural 정합성 차원에서 수정.

## 2. Accuracy 검증

### 2.1 case-level

15-trial run, B=1, --precision tf32:

**case8387**:
- V0: 0.0054, 0.0080, 0.0208, 0.0222, 0.0075, 0.0072, 0.0222, 0.0039, 0.0106, 0.0219, 0.0202, 0.0042, 0.0040, 0.0049, 0.0031
- M2: 0.0049, 0.0080, 0.0056, 0.0042, 0.0051, 0.0041, 0.0041, 0.0227, 0.0059, 0.0080, 0.0209, 0.0210, 0.0079, 0.0216, 0.0080
- → **M2 와 V0 의 분포 거의 동일** (0.004-0.022 range)

**ACTIVSg25k**:
- V0: 0.013, 0.013, 0.028, 0.024, **0.69** (outlier), 0.017, 0.014, 0.009, 0.014, 0.025
- M2: 0.017, 0.013, 0.007, 0.013, 0.038, 0.019, 0.018, 0.015, 0.019, 0.017
- → **M2 0.007-0.038, V0 0.009-0.69 (outlier 포함)** — M2 가 V0 보다 약간 더 일관

**USA**: 15-trial
- V0: 0.06-0.11 (consistent)
- M2: 0.06-0.13 + 3 outliers (0.85, 0.22, 0.21)
- → 80% trials 가 V0 range. 20% outliers — residual numerical sensitivity (high κ USA matrix + TF32 accumulation order). 다음 round 의 분석 가치.

### 2.2 B=64 accuracy

| case | V0 (3-trial) | M2 (3-trial) | 평가 |
|------|-------------|-------------|------|
| case8387 | 0.021, 0.022, 0.007 | 0.022, 0.004, 0.004 | **동급** |
| ACTIVSg25k | 0.014, 0.012, 0.012 | 0.012, 0.015, 0.036 | **거의 동급** (M2 한번 outlier) |
| USA | 0.073, 0.102, 0.105 | 0.079, 0.067, 0.093 | **동급** |

→ **B=64 에서 accuracy 정상 회복**.

## 3. Wall 측정 — 회귀

### 3.1 측정 (B=64, --repeat 32, 3-trial median)

| case | V0 (ms/sys) | M2 (ms/sys) | delta |
|------|-------------|-------------|-------|
| case8387 | 0.0270 | 0.2495 | **+823%** |
| ACTIVSg25k | 0.1122 | 0.5234 | **+366%** |
| USA | 0.5094 | 1.7314 | **+240%** |

### 3.2 회귀 원인 분석

**1 block per (subtree, batch) sequential processing**:
- case8387: 190 subtrees × 64 batch = **12K blocks**. V0: 56K blocks. → less SM saturation.
- USA: 3400 subtrees × 64 = 218K blocks. V0: 75K × 64 / 8 ≈ 600K blocks (warp granularity).

**Per-block thread utilization**:
- Sub-tree kernel: 256 thread block. For panel with fsz=10, `lu_small_front` 의 inner loop bound = m² < 100. **156+ threads idle per panel**.
- V0 factor_small: 8 warps × 32 thread, 1 (front, batch) per warp. Each warp uses all 32 lanes for its tiny front. **Full warp utilization**.

**Sequential within subtree**:
- Sub-tree kernel: N panels processed one after another (N up to 200+). Total per block: N × per-panel.
- V0 factor_small: 1 panel per warp. Per-block per-batch wall: 1 panel time.

→ **M2 의 block-time = N × V0 block-time**, 즉 block-time N배 증가. block 수 8N배 감소.

**Net**: 8N / N = 8x parallelism 감소? No wait: V0 has 8 (front,batch) per block, M2 has 1 (subtree,batch) per block. Same memory traffic per block(? let's analyze).

Actually V0 's block does 8 panels (8 warps × 1 panel each, in parallel). M2's block does N panels (sequentially).

V0: 8 panels in parallel = ~8x throughput per block.
M2: N panels sequential = 1x throughput per block.

But V0 has 7000/8 = 875 small kernel launches, M2 has 200 subtree launches → 4.4x fewer launches but each takes 8 * N times longer in sequential processing.

Net wall = launches × per-launch-wall.
V0: 875 × T_8 (where T_8 is 1-panel-time on 8-warp block)
M2: 200 × T_blocked_N (where T_blocked_N is N-panels sequentially on 1-block, ≈ N × T_1)
T_1 ≈ T_8 (same per-panel work)

V0 = 875 × T_8 = 875 T
M2 = 200 × N × T (N ≈ 35 mean for case8387) = 7000 T

→ M2 = 8x V0 — matches measured +823%.

### 3.3 잠재 게인 회복 방향

M2 의 **memory 절약** (stage-in 90% reduce) 잠재가 **compute 손실** 로 잠식. 잠재 실현하려면:

**Option A: parallel within-subtree (warp-specialization)**
- Subtree 의 same-level panels 를 다른 warps 가 parallel 처리.
- Topology dependency: parent 는 children 모두 끝난 후 처리. 같은 level (no parent-child within level) 은 parallel 가능.
- Needs: per-level panel grouping, sync between levels, warp coordination.
- 복잡도: 매우 상.

**Option B: smaller block, multiple subtrees per block**
- 32 thread (1 warp) per (subtree, batch). 8 subtrees per block (= 8 warps).
- 각 subtree 는 sequential within (small block size 로 시간 단축).
- 비슷한 pattern to V0 factor_small (1 warp = 1 unit).
- 복잡도: 중. 단 1 warp = 1 panel 형태가 깨짐 (multiple panel per warp seq).

**Option C: hybrid — small subtree (size ≤ 4) 만 M2 적용**
- Big subtree 는 chunk into smaller pieces or fallback to factor_small.
- 복잡도: 낮음. 단 잠재 게인 작아짐.

→ docs/19 §3.4 의 옵션 B (parallel-process within subtree) 가 정공법. 추가 R&D 1-2 rounds 예상.

## 4. residual USA outliers

USA 15-trial 의 20% (3/15) 가 0.21-0.85 outlier. V0 는 0.06-0.11 일관. 가능한 원인:

1. **TF32 numerical sensitivity**: USA 의 κ ≈ 10^7-10^8 (docs/19 §1.4 참조). TF32 mantissa 10 bit → κ·u_f ≈ 10. M2 의 reduction order 가 V0 와 다름 → accumulation error 분포 다름 → 일부 trials 큰 error.
2. **Race-condition residual**: compute-sanitizer racecheck 통과했으나 atomic ordering 에 의한 미세 차이 → 결과 차이.
3. **Subtree kernel 의 미세 logical bug** (다른 경로): 여전히 가능.

**다음 round 의 진단**: USA subtree 중 특정 큰 subtree 의 결과만 dump 해서 panel-by-panel V0 와 비교.

## 5. 변경 파일 (round 3)

| 파일 | 변경 |
|------|-----|
| `src/factorize/kernels.cuh` | `factor_mid_tf32`, `factor_mid_tf32_k4`, `factor_big_tf32` 시그니처에 `covered` 추가 + early-exit. `factor_small_subtree` 의 lu_small_warp → lu_small_front (의 미세 정정) |
| `src/factorize/dispatch.cuh` | call sites 의 `covered` 전달 (CLS_M2_PROTOTYPE 시 `plan.d_microst_covered`) |

## 6. 다음 round 의 plan

### 6.1 우선순위

1. **USA residual outliers 분석 + fix** — 잔여 정확도 issue.
2. **architectural redesign** for wall — Option A (parallel within-subtree) 또는 Option B (smaller block) 검토.

### 6.2 ROI 재평가

- Round 1-3 누적 invest: 4 rounds. 진척: design (docs/19), correct prototype (docs/20).
- 다음 1-2 rounds for USA fix + architectural redesign. 예상 wall 회복 (-5 ~ -15% per docs/19 estimate).
- 총 5-6 rounds for 5-15% wall 게인. 매우 비싼 ROI.

→ 다음 round 권고:
- **잔여 USA outliers 분석** 만 (1 round). architectural redesign 은 별도 R&D 로 marking.
- 만약 USA outliers 가 fundamental architecture 이슈 면 종료.

## 7. 코드 / 빌드 / 재현

### 7.1 빌드

```bash
cmake -DCMAKE_CUDA_FLAGS="-DCLS_M2_PROTOTYPE=1" -S . -B build-m2-proto
```

### 7.2 재현

```bash
# Accuracy correctness (case8387, B=64): V0 동급
./build-m2-proto/custom_linear_solver_run \
    /datasets/power_system/nr_linear_systems/case8387pegase \
    --precision tf32 --batch 64 --batch-only --repeat 32

# Wall (case8387, B=64): +823% 회귀
# Compare with V0 build
```

## 8. 참고

- docs/19: M2 design — etree shape 측정 + architecture 설계
- docs/20 (이전 버전): round 2 prototype + bug 진단 시작
- docs/16-18: micro-opt 시리즈
- docs/9: prior spine 메가커널 +19% 회귀 — 큰 persistent kernel 의 wall 위험 prior art
- Rennich-Davis PMAA'14: persistent subtree-walking reference (Cholesky)
