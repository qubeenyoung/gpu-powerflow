# Fsz-band-split factor dispatch — 추상 정리

**작성일**: 2026-06-08
**출처 round**: `perf/factorize-b1-10pct-codex` worktree, doc `05-reports/09-factorize-non-gemm-30pct-2026-06-08.md` 의 setup-time band-split 변경.
**범위**: factor dispatch 의 Launch SHAPE 변경. kernel internal / plan / etree 무수정.

## 문제 추상화

### 한 문장

GPU 의 kernel 자원 (shared memory, fsz_cap, tier kernel 선택) 은 **launch 단위로 결정**된다. 그런데 실제 일감 (front) 은 **panel 단위로 다양**하다. 한 launch 안에 이질적 front 가 섞여 있으면 그 launch 의 모든 panel 이 *worst-case* 비용을 지불한다.

### 비대칭의 구체 비용

| Dispatch 결정 (launch 단위) | 영향 받는 단위 |
|---|---|
| shared memory `fsz_cap²` 할당 | 모든 panel 이 그 size 점유 |
| tier kernel 선택 (small / mid / big) | 모든 panel 이 그 kernel 실행 |
| block size, launch_bounds | 모든 panel 이 그 occupancy 제약 |

**결과**: 같은 level 안 fsz=20 panel 과 fsz=120 panel 이 섞이면 → tier 결정은 `max_fsz=120` 기준 → fsz=20 panel 80개도 *block-per-panel mid kernel* 로 처리. tiny front 인데 warp-packing 의 효율 손실.

## 기존 방식의 추상 모델

```
for each level L (canonical etree 순서):
    max_fsz = max over panels in L
    tier   = pick_tier(max_fsz)            ── 1 회 결정
    launch tier_kernel <<<one launch>>>     ── 모든 panel 처리
```

특성:
- **panel-level 일감 모두 1 launch 로 묶음** → launch 수 ∝ levels
- **tier 분기 1 회** → "level 의 worst panel 이 모두를 대변"
- mixed-level 의 작은 panel 은 큰 panel 의 tier 에 *징발*

이게 작동하던 가정: *"각 level 의 panel 들이 size 적으로 균질"*. PF Jacobian 의 일부 case 에서는 이 가정이 깨짐 → 손실.

## 개선안의 추상 모델

```
─── analyze / setup 단계 (1 회) ─────────────────────
band-band of(fsz) = {0:≤32, 1:≤48, 2:≤64, 3:≤96, 4:≤128, 5:>128}

for each (subtree, level) range:
    sort panels by band  (stable, subtree 내부에서만)

→ alternate plcols_band 메타데이터 1 회 생성

─── dispatch 단계 (매 factorize) ────────────────────
for each (subtree, level) sub-range [b, e) in plcols_band:
    walk b..e, 같은 band 가 연속되는 (sub_b, sub_e) 구간 식별
    for each band-contiguous sub-range:
        tier = pick_tier(band 안의 worst fsz)
        launch tier_kernel <<<one launch>>>
```

특성:
- **panel-level 일감을 band 별로 *재정렬*** (level 경계는 그대로, 순서만 reshuffle)
- **한 level 의 launch 수 1 → 그 level 안 *서로 다른 band* 수만큼** (보통 2~3)
- **각 launch 가 band 안 worst 만 책임** → tier 도, fsz_cap 도 그 band 에 딱 맞게

### 본질적 변화

| | 기존 | 개선 |
|---|---|---|
| `pick_tier` 입력 | level 의 max_fsz | band 의 max_fsz |
| 1 launch 의 일감 균질도 | 낮음 (level 안 fsz 분산) | 높음 (band 안 fsz 좁음) |
| 한 level → kernel function 종류 | **1 가지** | **여러 가지 가능** (small + mid 동시) |
| launch 수 (per level) | 1 | band 수 (≤ 6) |
| 사용 메타데이터 | plan.plcols | State.h_plcols_band (alternate) |

핵심 추상: **"같은 일감 묶음을 어떻게 *재분류* 해서 GPU 에 줄지" 만 바꾼 것**. kernel 자체는 무수정.

## 예상 효과 (이론)

다음 두 요인이 곱해진다.

### (E1) Shared memory 정합성 향상

각 launch 의 `fsz_cap` 이 그 band 의 worst (예: band 1 = fsz ≤ 48) 가 되므로:
- mid kernel 의 `Fs[fsz_cap²]` shared 할당이 작아짐 → SM occupancy ↑
- WMMA / TC tile padding 손실 ↓

기여 예상: 단독으로 **5-10%** wall.

### (E2) Tier 재라우팅 (더 큰 lever)

mixed level 의 fsz≤32 panel 이 mid kernel 에 묻혀 있다가 → small kernel (8-warp-packed) 로 풀려나옴:
- block-per-panel → warp-per-panel 로 격하 → 같은 work 를 1/8 launch grid 로 처리
- per-front overhead 회복

기여 예상: **10-20%** wall (mixed level 비율 의존).

### 합산 예상

n × B 가 클수록 두 효과 다 큼. 예상 wall 감소:
- 25K B=64 / 256: **−10~15%** wall
- 70K B=64 / 256: **−10~15%** wall
- 13K B=256: **−5~10%** wall
- B=1 또는 작은 n: 영향 없음 (gate 차단)

## 실제 효과 (측정)

doc 9 의 direct no-trailing 측정 (codex pre-band → post-band 비교):

| Case | B | full factor Δ | direct non-GEMM Δ |
|---|---|---|---|
| 13659pegase | 256 | −7.4% | **−30.1%** ★ |
| ACTIVSg25k (regen) | 64 | −10.4% | −8.1% |
| ACTIVSg25k (regen) | 256 | −11.8% | −12.4% |
| ACTIVSg70k | 64 | −11.4% | −18.4% |
| ACTIVSg70k | 256 | −12.7% | −16.3% |
| SyntheticUSA | 64 | −15.1% | −13.9% |
| SyntheticUSA | 256 | −13.1% | −16.3% |
| B=1 (any case) | 1 | gate 차단 (변경 없음) | — |

실측 vs 예상:
- 큰 case (70K, USA) wall **−11~15%**: 예상 범위 안.
- 25K wall **−10~12%**: 예상 범위 안.
- 13K direct non-GEMM **−30%**: 예상 상한 + thin margin (repeat-301 필요), 유일한 30% 직접 달성 점.
- 25K direct non-GEMM **−8~12%**: 예상보다 작음. doc 9 의 분석에 따르면 25K 는 mid front packing 이 다음 lever.

### 회귀

- B=1 4 case: noise 안 (gate 차단).
- 솔브 (이전 spine + multi-stream): 동일 유지.
- 같은 round 안 fp16 PTX default: 동일 유지.

## 효력의 한계

doc 9 의 14 건 reverted experiment 가 *같은 layer 의 미세 변형* 시도:

| 시도 | 결과 |
|---|---|
| fine band split (`33..40` / `41..48` / `49..56` / ...) | 회귀 (launch 수 ↑ 가 손실 더 큼) |
| 25K small_warp 16/4 packing | 회귀 (shared 압박 / scheduling 손실) |
| 25K mid block size 64/128/256 fsz-scale | 회귀 |
| scatter 512-thread block | 회귀 |
| `__launch_bounds__(256, 2)` on factor_small/mid | 회귀 |
| `__ldg` metadata, contiguous `asm_local` fast path | 회귀 |
| 70K big tier launch 512 thread | 회귀 |

→ **현재 6-bucket coarse band 분할이 *local optimum***. 이 layer 안에선 더 미세하게 못 쪼개고, 더 거칠게 못 묶는다.

doc 9 의 명시적 다음 round 후보 (이 layer 아닌 곳):
- 25K mid-front packing (kernel internal 재설계 — Layer 0)
- 70K big-tier / front-memory layout (plan/etree 재설계 — Layer 4)

## 추상화: 어디에 해당하는 lever 인가

```
Layer 0  kernel internals     ── 손 안 댐
Layer 1  launch content       ── 손 안 댐
Layer 2  launch count         ── 약간 늘어남 (band 수 만큼)
Layer 3  launch SHAPE         ── ★ 여기. 한 launch 의 일감 동질화
Layer 4  plan / etree         ── 손 안 댐 (canonical 순서 보존)
```

핵심 통찰: **GPU 일감 자체는 그대로, *분류 / 정렬 / 묶음 단위* 만 바꿔서 launch 가 더 적합한 자원과 더 적합한 kernel 함수를 고르도록 한다**.

## 코드 책임 분담 (SRP)

5 함수가 한 결정의 5 가지 책임:

| 함수 | 책임 | 호출 빈도 |
|---|---|---|
| `dispatch_fsz_band` | "이 panel 의 equivalence class 는?" | per-panel, 매번 |
| `wants_factor_band_order` | "이 (n, B) 에서 활성?" | analyze 후 1 회 |
| `build_factor_band_order` | "활성이면 reorder + 메타데이터 생성" | setup(B) 1 회 |
| `use_factor_band_order` | "데이터 준비됐나" | 런타임 가드 |
| dispatcher 안 split-loop | "band 경계마다 sub-launch" | 매 dispatch |

각 함수 단독으로는 작음 (≤ 40 lines). 합쳐서 **한 추상 결정 ("homogeneous bucket per launch") 의 정책 + 분류 + 데이터 + 게이트 + 실행** 을 구성.

## 결론

| 항목 | 평가 |
|---|---|
| **개념적 정당성** | 강함 — GPU dispatch 의 본질적 비대칭 비용을 정확히 겨눔 |
| **예상 vs 실측** | 큰 case 일치 (E2 효과 dominant 확인), 25K direct non-GEMM 만 미달 |
| **이 layer 의 local 최적성** | doc 9 의 14 회귀 실험으로 확정. 더 미세 / 거친 분할 다 손해 |
| **B=1 영향** | 0 (gate 보호) |
| **확장 가능성** | 이 layer 안에서는 limit. 추가 30% 는 Layer 0 (kernel 재설계) 또는 Layer 4 (plan/etree) |
| **회귀 위험** | 낮음 (가드 다중, alternate 데이터만 사용, canonical analyze 무수정) |

**한 줄**: launch SHAPE 의 *균질화* 가 multi-batch factor wall 의 5-15% 를 안전하게 회수하는 lever. 그 다음 30% 는 이 layer 밖에 있다.
