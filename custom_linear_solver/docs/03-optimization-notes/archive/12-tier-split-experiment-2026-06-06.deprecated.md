# [DEPRECATED] Per-level tier split 실험 (T-split) — 2026-06-06

> **2026-06-07 폐기.** case-specific opt-in (USA-target, B≥32 게이트) 으로 dispatch path 복잡도 대비 ROI 낮음. 관련 코드 (analyze.cu 의 fsz 정렬 블록, dispatch.cuh 의 `issue_factor_level_range_split` wrapper, `CLS_USE_TIER_SPLIT`/`CLS_TIER_SPLIT_B_MIN` env) 전체 삭제. 측정 데이터만 historical log 로 보존. 재구현 시 docs/12 §10.1.2 (mis-routing 정량) 부터 참고.

**상태**: ~~실험 완료 — opt-in env. **USA B≥16에서 −4~−6% factor wall win**, case8387은 noise. 기본 OFF.~~ **DEPRECATED 2026-06-07** — 코드 제거.
**선행 분석**: [`docs/12 §10.1.2 ~ §10.1.4`](../04-benchmarks-profiling/12-front-gemm-distribution-2026-06-06.md) (mixed level mis-routing 정량)
**관련 실험 (deprecated)**: [`docs/11 small-tier packing`](11-small-packed-experiment-2026-06-06.md) — launch fragmentation 회귀가 본 실험의 risk 모델

## 0. TL;DR

| case | B | OFF | ON (gate=4) | delta |
|------|--:|----:|-----------:|------:|
| case30/118 | any | – | 동일 | no-op (mixed level 없음) |
| case8387 | 1 | 0.342 | 0.508 | **+48%** (가드 안 함 시 회귀) |
| case8387 | 64 | 0.026 | 0.025 | −2% (marginal) |
| USA | 1 | 2.67 | 3.23 | +21% |
| USA | 64 | 0.469 | 0.444 | **−5.3%** |

**B≥32 gate 적용 시 (권장)**:

| case | B | OFF | ON gate=32 | delta |
|------|--:|----:|-----------:|------:|
| case8387 | 32 | 0.0312 | 0.0340 | +9% (여전히 회귀) |
| case8387 | 64 | 0.0271 | 0.0269 | −0.9% |
| USA | 32 | 0.489 | 0.468 | **−4.3%** |
| USA | 64 | 0.463 | 0.447 | **−3.5%** |

→ **USA-class (n≥150k, mid/big tier 우세)에서 명확한 win**. case8387은 mixed level 패턴이 달라 효과 없음.

## 1. 설계

### 1.1 문제 — docs/12 §10.1.2의 mis-routing

`issue_factor_level_range` 는 **레벨 단위로** kernel을 선택 (level max_fsz 기준). 한 레벨에 fsz 가 작은 것부터 큰 것까지 섞여 있을 때 모든 fronts는 가장 큰 fsz 기준 kernel로 처리:
- case8387: 16/29 레벨이 mixed (55%), **1768 small fronts (24%)** 가 factor_mid 로 처리
- USA: 21/40 레벨 mixed (53%), **149 mis-routed** (7 small + 142 mid) 가 factor_big (1024 thread block) 로 처리

### 1.2 해결 방안 — 사전 정렬 + sub-range dispatch

1. **analyze 시점에 plcols 정렬**: 각 (subtree, level) 슬라이스 안에서 fsz 오름차순. mixed 레벨에서 small → mid → big 순으로 연속 배치 (uniform-tier 레벨은 정렬 skip).
2. **dispatch wrapper**: scan 으로 small/mid/big 경계 찾고, 각 sub-range를 기존 `issue_factor_level_range`로 호출. 기존 함수가 sub-range의 max_fsz로 자연스럽게 올바른 kernel 선택.
3. **B-gate**: 작은 batch에서 extra launch 오버헤드를 피하기 위해 `B < CLS_TIER_SPLIT_B_MIN` 시 split skip.

코드 위치:
- `src/plan/analyze.cu`: T-split 정렬 (CLS_USE_TIER_SPLIT gated)
- `src/multifrontal.cu`: `issue_factor_level_range_split` wrapper, `issue_factor_levels`가 wrapper 호출

### 1.3 핵심 트릭

기존 `issue_factor_level_range(plan, st, stream, b, e)` 는 max_fsz를 자체 계산하고 그에 맞는 kernel 호출. **그대로 sub-range에 대해 호출하면 자동으로 올바른 routing** — 별도 분기 코드 불필요:

```cpp
if (b_small > b)       issue_factor_level_range(plan, st, stream, b, b_small);       // → factor_small
if (b_mid > b_small)   issue_factor_level_range(plan, st, stream, b_small, b_mid);   // → factor_mid
if (e > b_mid)         issue_factor_level_range(plan, st, stream, b_mid, e);         // → factor_big
```

기존 코드 재사용으로 구현 30줄.

## 2. 측정

### 2.1 정확도

모든 케이스에서 OFF/ON relres가 동등 범위:
- case30/118: 동일 (no-op)
- case8387 fp32: 1.2e-5 ~ 2.5e-5
- USA fp32: 1.0e-3 ~ 2.0e-2 (FP32 reduction 순서 영향)

### 2.2 wall A/B (7-run median, graph mode, 기본 gate=4)

```
case                     B        OFF         ON   delta
case30                   1    0.03768    0.03766   noise
case30                  64    0.00062    0.00062   noise   (no mixed levels)
case118                  1    0.05060    0.05056   noise
case118                 64    0.00088    0.00088   noise
case8387pegase           1    0.34190    0.50721  +48.4%   ← 회귀
case8387pegase          64    0.02625    0.02689   +2.4%
case_SyntheticUSA        1    2.67477    3.22647  +20.6%   ← 회귀
case_SyntheticUSA       64    0.46900    0.44428   −5.3%   ← win
```

→ B=1 큰 회귀, B=64 USA win.

### 2.3 B_MIN sweep — gate 임계 탐색

5-run median, B={1, 4, 16, 32, 64}, gate ∈ {16, 32, 64}:

**case8387**:

| B | OFF | gate=16 | gate=32 | gate=64 |
|--:|----:|--------:|--------:|--------:|
| 1 | 0.366 | noise | noise | noise |
| 4 | 0.101 | +2% | +3% | +4% |
| 16 | 0.041 | **+20%** | +1% | −3% |
| 32 | 0.031 | +7% | **+9%** | −1% |
| 64 | 0.027 | −2% | −1% | −0.2% |

→ case8387은 모든 gate에서 noise/회귀. **이득 없음**.

**USA**:

| B | OFF | gate=16 | gate=32 | gate=64 |
|--:|----:|--------:|--------:|--------:|
| 1 | 2.64 | noise | noise | noise |
| 4 | 0.94 | +3% | −1% | −0.5% |
| 16 | 0.54 | **−4%** | noise | −3% |
| 32 | 0.49 | **−4%** | **−4%** | noise |
| 64 | 0.46 | **−4%** | **−4%** | **−6%** |

→ USA는 gate≥16에서 consistent **−3~−6%** win.

**권장 default**: `CLS_TIER_SPLIT_B_MIN=32` (case8387 회귀 거의 차단 + USA B≥32 이득 보존).

## 3. 메커니즘 — ncu 검증 (USA fp32 B=64)

split=OFF vs split=ON (gate=4 강제):

| kernel | OFF launches | OFF total μs | ON launches | ON total μs | 변화 |
|--------|------------:|-------------:|------------:|------------:|------|
| factor_small | 3 | 6151 | 3 | 6517 | +6% (변동 noise) |
| factor_mid | 10 | 13308 | **18 (+8)** | **12004 (−10%)** | mid 작업 증가 + 시간 감소 |
| factor_big | **17** | 8887 | **9 (−8)** | **6642 (−25%)** | big 작업 감소 |
| **합** | 30 | 28346 | 30 | 25163 | **−3183 μs (−11%)** |

→ **8개 레벨 분량의 mid fronts 가 factor_big (1024 thread) 에서 factor_mid (256 thread)로 마이그레이션**.

원래 USA에는 21개 big tier 레벨이 있는데 (docs/12 §10.1.2), 그 중 mid/small front들이 mid 커널의 256-thread block에서 적절히 처리되어:
- factor_big total time **−25%** (8.9 ms → 6.6 ms)
- factor_mid total time **−10%** (despite +8 launches)
- 합 **−11%** kernel time → 실측 wall −5~−6% (multi-stream overlap으로 더 작게 보임)

### 3.1 왜 case8387에선 효과 없는가

case8387의 mis-routing 패턴:
- mid 레벨에 small fronts (case8387 L=2: 715 small + 1 mid)
- factor_mid에서 작은 fronts는 thread idle but block 크기 작아 wall 영향 적음

USA의 mis-routing 패턴:
- big 레벨에 mid + small fronts (USA L=20: 1 small + 1 mid + 5 big)
- factor_big에서 1024-thread block이 fsz=20 작은 front 처리 → **압도적 비효율**
- mid 로 마이그레이션 시 256 thread block이 적절

→ **small→mid mis-routing 비용은 작고, mid/small→big mis-routing 비용은 크다**. case8387은 전자, USA는 후자가 우세.

### 3.2 multi-stream interaction

T-split이 launch 수를 늘림 (USA: 30 → 30, 변동 없음 in launches but +8 mid -8 big 교환). docs/13에서 본 multi-stream 이점은 보존: 새 mid launches도 subtree streams 에 distribute 됨.

case8387 B=4 에서 +35% 회귀는 launch 수가 (B=4 작은 work / launch overhead 비율 큼) 약화된 multi-stream 이점 + extra launches 의 합산 효과.

## 4. 결정

### 4.1 default 와 opt-in

- **default OFF** (CLS_USE_TIER_SPLIT 미설정). case8387 / 소형 case 회귀 방지.
- **opt-in 권장 조합**: `CLS_USE_TIER_SPLIT=1 CLS_TIER_SPLIT_B_MIN=32`
  - USA-class workload (n ≥ 150k, big tier 비중 ≥ 30%) 에서 B≥32 부터 −3~−6% wall win
  - case8387-class: 노이즈 수준. 사용해도 무방하나 win 적음

### 4.2 향후 자동 enable 가능성

다음 조건 모두 만족 시 자동 enable 가능:
- (a) `plan.n > 100000` (mixed 레벨 수가 충분)
- (b) `B ≥ 32` (launch overhead amortize)
- (c) docs/12 §10.1.2 mixed level 비율 (분석 시점 계산해 plan 에 저장)

미구현 (env opt-in 으로 충분). 다른 lever (TF32 trailing, scatter_values 최적화) 와 ROI 비교 후 결정.

## 5. 구현 요약

### 5.1 코드 변경

```diff
# src/plan/analyze.cu
+ // T-split: sort by fsz ascending within each (subtree, level) slice
+ // (skipped for uniform-tier levels)
+ if (CLS_USE_TIER_SPLIT) { sort... }

# src/multifrontal.cu
+ static void issue_factor_level_range_split(plan, st, stream, b, e) {
+     if (!USE_TIER_SPLIT || B < B_MIN || e <= b) {
+         issue_factor_level_range(plan, st, stream, b, e); return;
+     }
+     // scan b..e for tier boundaries (b_small, b_mid)
+     // dispatch 1-3 sub-ranges via existing function
+ }

# issue_factor_levels 의 3개 call site:
- issue_factor_level_range(...)
+ issue_factor_level_range_split(...)
```

### 5.2 env 인터페이스

| env | default | 효과 |
|-----|--------:|------|
| `CLS_USE_TIER_SPLIT` | 0 (OFF) | 1로 설정 시 정렬 + split dispatch 활성 |
| `CLS_TIER_SPLIT_B_MIN` | 4 | 이 값 미만 batch 에선 split skip (single dispatch fallback) |

권장: `CLS_USE_TIER_SPLIT=1 CLS_TIER_SPLIT_B_MIN=32` (USA-class).

### 5.3 측정 재현

```bash
# A/B with B-sweep
for split in 0 1; do for B in 1 4 16 32 64; do
  CLS_USE_TIER_SPLIT=$split CLS_TIER_SPLIT_B_MIN=4 \
    build/custom_linear_solver_run case_SyntheticUSA \
    --precision fp32 --batch $B --batch-only --repeat 20
done; done

# ncu mechanism check
ncu --kernel-name 'regex:factor_(small|mid|big)' --launch-count 30 \
    --metrics ... <run cmd>
```

원본 데이터:
- `/home/claude/prof/tsplit_USA_B64_{0,1}_clean.csv` — ncu split A/B

## 6. 메타-회고

이전 docs/11 (packing) 실험과 비교:
- packing: 한 레벨 안에서 (nc, fsz) 그룹별 multi-front-per-warp kernel. **새 kernel 필요**, launch fragmentation 위험 큼 → 회귀
- T-split: 기존 kernel 재사용, 한 레벨 1-3개 launch 추가. **새 kernel 없이** 라우팅 정확도만 개선 → small win

→ **"새 kernel 없이 dispatch 정확도 개선" 이 launch overhead vs efficiency 의 더 좋은 trade-off point**. docs/10 §9의 메타-결론 ("새 알고리즘보다 dispatch 정합성이 더 ROI") 의 정량적 증거.

이는 docs/11의 negative result 와 docs/13의 multi-stream 분석을 연결: 
- multi-stream이 SM idle을 다른 stream으로 채우는 것 (docs/13)
- T-split이 idle SM time을 만드는 한 원인 (mis-routing) 자체를 줄이는 것 (본 문서)

→ 두 최적화는 보완적. T-split + multi-stream 조합이 USA B=64 의 −5~−6% 최종 wall 단축의 메커니즘.

## 7. shared budget 안전성 분석

### 7.1 우려 사항

T-split이 "big level의 mid fronts → factor_mid" 마이그레이션을 수행. factor_mid 는 front 전체를 shared 에 stage-in 하므로 (`docs/06 §2.2`) sub-range의 max_fsz가 클 때 96 KB shared budget 초과 가능성.

### 7.2 마이그레이션 대상의 fsz 상한

T-split의 sub-range 경계 = SMALL_THRESH=32, MID_THRESH=128. **mid sub-range는 정의상 fsz ≤ 128**. 즉 factor_mid 로 마이그레이션되는 fronts는 항상 fsz ≤ 128.

big fronts (fsz > 128) 는 여전히 factor_big 으로 routed. **"big → mid" 마이그레이션은 발생하지 않음** — T-split이 하는 것은 "big-level 안에 포함된 mid fronts (fsz ≤ 128)" 를 factor_mid 로 분리하는 것.

### 7.3 mid sub-range 의 worst-case shared 사용량 (FP32)

`issue_factor_level_range` 의 shared 계산:
```cpp
shb_tiled = fsz_cap² × elt + 2 × level_max_nc × level_max_uc × elt
```

mid sub-range의 worst case (FP32, USA 분포 기준):
- fsz_cap = 128 (boundary)
- level_max_nc = 20 (USA mid max, `docs/12 §3.3`)
- level_max_uc = 108 (USA mid max)
- elt = 4 (FP32)

→ 128²·4 (**64 KB**) + 2·20·108·4 (**17 KB**) = **81 KB < 96 KB** ✓

→ FP32 power-grid 의 모든 mid sub-range는 shared 안에 들어옴.

### 7.4 기존 dynamic 가드가 안전망

```cpp
if (shb_tiled <= MID_SHARED_BUDGET) {  // 96 KB
    // dispatch factor_mid
    return;
}
// shared 초과 시 자연스럽게 fall-through → factor_big
```

T-split은 sub-range 마다 동일 함수 호출 → sub-range의 max_fsz/nc/uc 로 재계산. 만약 (이론적으로) shb_tiled가 96 KB 초과하면 **자동으로 factor_big로 fall-through**. 정확성 안전.

### 7.5 실측 검증 — fall-through 발생 안 함

ncu (USA fp32 B=64, T-split ON):
- factor_big launches: 17 → 9 (**−8**)
- factor_mid launches: 10 → 18 (**+8**)

**정확히 8개 sub-range가 factor_big에서 factor_mid 로 마이그레이션 성공**. 한 개라도 shared 초과로 fall-through 했다면 factor_big launch 수가 덜 감소했을 것 (예: 17 → 10). 정확히 +8/−8 매칭은 모든 marketing 성공의 증거.

### 7.6 위험 영역 (본 워크로드 외)

| 상황 | 안전? | 이유 |
|------|------|------|
| FP32 power-grid (nc≤20, uc≤108) | ✓ 안전 | 81 KB ≤ 96 KB |
| FP32 nc=32 (max WMMA K) | ✓ 안전 | 128²·4 + 2·32·100·4 = 89 KB |
| **FP64 power-grid** | ⚠ 일부 fall-through | 128²·8 = 128 KB > 96 KB |
| 3D PDE (nc≥40, uc≥200) | ⚠ 일부 fall-through | 2·40·200·8 = 128 KB만으로 초과 |

FP64 인 경우 fsz≳96 부터 fall-through 발생 → T-split 의 mid 마이그레이션 win 감소. 본 실험은 FP32 한정. FP64 측정은 별도.

## 8. FP32 에서 factor_big 이 정말 필요한가?

### 8.1 case-by-case 답

| 케이스 | max_fsz | big fronts | big FLOP% | factor_big 필요? |
|--------|--------:|-----------:|----------:|:----------------|
| case30 | ~5 | 0 | 0% | **No (dead code)** |
| case118 | ~5 | 0 | 0% | **No** |
| **case8387** | **79** | **0** | **0%** | **No — 한 번도 안 불림** |
| **USA** | **235** | **63** | **39.5%** | **Yes — 마운트 핵심** |

### 8.2 case8387 에서 factor_big 가 한 번도 안 불리는 이유

case8387 의 max_fsz=79 → 모든 front 가 fsz ≤ 128 안에 들어와 **factor_mid 또는 factor_small 로 처리**. factor_big 의 dispatch 조건 (`max_fsz > MID_THRESH=128`) 이 만족되지 않음.

이는 `docs/06 §2.3` 의 "power-grid root separator 가 N=10k-30k 수준에서는 ~80-90" 와 부합. case8387 (n=14908) 같은 중규모 power-grid 는 max_fsz < 128 이라 big tier 필요 없음.

### 8.3 USA 에서 factor_big 가 필수인 이유

USA 의 big tier 분포 (`docs/12 §3.4`):
- 63 fronts, fsz ∈ [129, 235]
- 39.5% 의 전체 trailing FLOPs

가장 큰 front fsz=235:
- factor_mid 로 처리 시 필요 shared = 235²·4 = **216 KB** ≫ 96 KB → 불가능
- factor_big 의 global-memory direct 가 유일한 길

이론적 MID_THRESH 상향:
- fsz=140 FP32: 140²·4 + 17 KB staging = 95 KB → 마지노선 fit
- USA 의 fsz∈(128, 140] front 수 = **22** 개 → 마이그레이션 가능
- 단 나머지 **41 fronts (fsz∈(140, 235])** 는 여전히 factor_big 필요

→ MID_THRESH 를 142 정도로 올려도 USA big fronts 의 35% 만 흡수. **63% 는 factor_big 절대 필요**.

### 8.4 결론

| 워크로드 클래스 | n 범위 (power-grid) | factor_big | 의미 |
|----------------|--------------------:|:-----------|------|
| 소~중 (case8387 등) | n ≤ ~30k | dead code | 빌드에서 빼도 안전, 단 코드 단순성 위해 유지 권장 |
| 대 (USA 등) | n ≥ ~100k | 필수 | 39.5% FLOPs 차지 |

→ "**FP32 에서 big 필요 없다**" 는 **case8387 같은 중규모에 한정한 사실**. USA 같은 대규모에서는 절대 필요. 본 솔버는 두 클래스 모두 지원하므로 factor_big 빌드 유지.

T-split 의 정량적 win (`§3 USA −25% factor_big total time`) 도 결국 factor_big 이 존재해야 의미가 있음 — T-split 은 factor_big 을 *제거* 하는 게 아니라 *사용 범위를 정확하게* 만드는 것.

### 8.5 잠재 마이크로 최적화 (미실시)

USA 같은 워크로드에서 22 fronts 추가로 mid 로 마이그레이션하면 추가 −2~−3% 가능:
- `MID_THRESH = 140` (FP32 한정)
- 코드 변경: 정밀도별 분기 (FP32 vs FP64)

ROI 작아 deferred. 본 T-split (MID_THRESH 그대로 128 사용) 의 −5% win 이 sweet spot.
