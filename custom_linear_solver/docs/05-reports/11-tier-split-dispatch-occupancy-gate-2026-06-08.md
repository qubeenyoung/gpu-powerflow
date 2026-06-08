# Per-front tier-split factorize dispatch (occupancy-gated)

**작성일**: 2026-06-08
**선행**: [`docs/05-reports/10`](10-single-stream-dispatch-8000-2026-06-08/) (single-stream dispatch 분석 데이터), [`docs/04/14`](../04-benchmarks-profiling/14-fp32-singlestream-baseline-2026-06-08.md) (fp32 baseline)
**대상**: `src/plan/analyze.cu`, `src/plan/multifrontal_plan.{hpp,cu}`, `src/factorize/dispatch.cuh`, `src/solve/dispatch.cuh`
**환경**: RTX 3090 (sm_86, 82 SM), CUDA 12.8, fp32, single-stream. 클럭 미고정 → 모든 A/B는 **interleaved + median**(doc 14 §4).

## 0. TL;DR

현재 factorize dispatch는 tier(small/mid/big 커널)를 **level의 max_fsz로** 정해, 한 level에 큰 front가 하나만 있어도 그 level의 작은 front 전부가 큰 커널로 **승격**된다(case8387에서 fronts의 **23.68%**가 small→mid). 이를 **front별 tier로 분해**해 같은 level을 tier-동질 sub-range로 나눠 각각 right-sized 커널에 dispatch한다. 단, 이 분해는 **B(배치 병렬도)에 의존**해 득실이 갈려서(저 B는 latency, 고 B는 throughput), warp 공급이 GPU를 채울 때만 split하는 **occupancy 게이트**를 단다.

| case | B=1 | B=16 | B=64 | B=256 |
|------|----:|-----:|-----:|------:|
| case8387 factor | +1.9%* | −1.4% | **−11.2%** | **−13.0%** |
| ACTIVSg25k factor | −4.4%* | +0.1% | **−3.6%** | **−3.9%** |

`*` B=1은 게이트가 split을 끄므로 BASE와 동일 커널 → 순수 noise. 정확성 보존(relres 불변, fp64/fp32/multistream 경로 모두 확인).

## 1. 문제 (doc 10 데이터)

분포(`front_summary.tsv`, `front_fsz_histogram_fine.tsv`): n=14908, fronts=7431, levels=28.
- **fsz ≤ 8이 전체의 87%** (1–4: 29.7%, 5–8: 57.4%), 9–32: ~13%, mid(33–80): **50개(0.67%)**, big: 0.
- nc_p50=2, uc_p50=4 — 극히 작은 front가 압도적.

dispatch 동작(`factor_level_dispatch_*.tsv`): tier는 `issue_factor_level_range`가 받은 range의 **max_fsz**로 결정.
- 예: level 2는 708 fronts = 707 small + **mid 1개(fsz=33)**. max_fsz=33 → **전체 level이 mid 커널**(grid=(708,B), front당 1블록·256스레드). 707개 작은 front가 mid 커널에서 처리됨(`small_to_mid=707`).
- 전체: **small→mid 승격 = 1760 fronts = 23.68%** (`dispatch_mismatch_*.tsv`).

mid 커널은 front당 256-스레드 블록을 쓰는데, fsz≤8(64 elements) front엔 과한 스레드. small 커널은 8 warp/block로 front당 1 warp 패킹 → 블록 수가 ~8× 적다.

## 2. 설계 — front별 tier 분해 (일반화)

**핵심 아이디어**: tier를 level-max가 아니라 **front별**로. 같은 level의 front는 etree상 서로 독립(부모는 항상 상위 level)이므로, level을 tier별 sub-range로 쪼개 순차 launch해도 정확. extend-add는 상위 level 부모로 atomicAdd → 교환법칙·same-stream 직렬화로 안전.

구현(전부 analyze 1회 비용, front 크기는 값-독립):
- `analyze.cu`: 각 level 내에서 panel을 tier(0: fsz≤32 small, 1: ≤128 mid, 2: big)로 stable 그룹핑 → `h_plcols_tier`(level-major, tier-contiguous) + `h_level_tier_off`(per-(level,tier) CSR 경계). tier cut은 dispatch의 SMALL_THRESH=32/MID_THRESH=128와 일치.
- `dispatch.cuh`: 단일-스트림 walk가 level마다 tier sub-range를 각각 `issue_factor_level_range`로 dispatch. 기존 커널/시그니처는 `plcols` 포인터만 파라미터화(d_plcols ↔ d_plcols_tier).

case 특정 상수 없음 — tier 경계는 커널 능력 자체. 따라서 임의 행렬에 적용.

## 3. 함정과 일반화: B 의존성 (occupancy 게이트)

순진하게 "항상 split"하면 **B에 따라 정반대**:

| case | B=1 | B=64 | B=256 |
|------|----:|-----:|------:|
| 8387 factor (ungated) | **+82%** | −7% | −18% |
| 25k factor (ungated) | **+22%** | −3% | −4% |

**왜 B=1에서 +82%인가**: B=1에선 mid 커널(front당 256스레드)이 작은 front를 **더 빨리** 처리한다 — front가 707개라 GPU가 이미 차고, front당 warp(32)보다 8× 많은 스레드를 받기 때문. doc 10이 "낭비"로 본 승격이 **latency 영역(저 B)에선 이득**. warp-packing(small 커널)은 warp 공급이 많아 throughput-bound일 때만 이긴다.

즉 분해 입도는 front 크기만이 아니라 **occupancy(=front수×B)에 의존**. 이건 case가 아니라 **runtime 병렬도/하드웨어로 파라미터화**되는 일반 규칙이다.

**게이트**(`dispatch.cuh`): mixed level에서 small 영역의 warp 공급이 GPU warp 슬롯을 채울 때만 split.
```
warp_fill = num_SMs × (max_threads_per_SM / 32)   // 순수 HW값 (device attr), 3090=3936
split  ⇔  level이 mixed  ∧  small_cnt × B ≥ warp_fill
else   →  whole-level dispatch (= 승격, pre-split 동작)
```
- B=1: mixed level의 small_cnt(≤707) × 1 < 3936 → split 안 함 → BASE와 동일 → regression 0.
- 고 B: small_cnt × B ≫ 3936 → split → throughput win.
- pure-small/pure-mid level은 게이트와 무관하게 whole dispatch(기존 잘 튜닝된 경로 불변).

게이트 적용 결과가 §0 표. B=1 regression 소멸 + 고 B win 유지, 양쪽 케이스 일반화.

## 4. 기각한 실험 — finer 5-class 분해

"fsz≤8이 87%"라, split된 small sub-range도 그 안의 max_fsz로 per-warp shared(`fsz²cap`)를 잡는 **같은 max-끌어올림 문제**(shared 사이징 차원)가 남는다. small을 [1–8]/[9–16]/[17–32] 3 class로 더 쪼개봤다(kNumTiers=5).
- 결과: 고 B에서 3-class 대비 **+2% 수준**의 미미한 추가 이득, 반면 **25k B=1을 +6.5% 악화**(가장 큰 케이스의 mixed level이 B=1에서도 게이트를 통과해 split → launch가 3개로 쪼개짐).
- 판정: 이득 대비 복잡도·B=1 위험이 커서 **기각**. 3-tier split 유지. ("미세 최적화/숫자놀이 금지" 기준에도 부합.)

## 5. 측정 방법론
- BASE = single-stream walk를 whole-level로 강제한 빌드(게이트 분기 `if(false)`), TIER = 본 변경. 두 바이너리를 **interleaved**로 번갈아, 각 config median(5 trial × repeat 20).
- 정확성 게이트: relres 불변(8387 ~2e-5, 25k ~1.3e-4, fp64 3e-14). multistream(default) 경로도 회귀 없음(시그니처 변경분 검증).

## 6. solve로 확장

solve dispatch(`src/solve/dispatch.cuh`)도 `fwd_level`/`bwd_level`이 동일한 level-max tiering을 써서 같은 승격이 일어난다(solve small 커널의 per-warp shared는 고정 64라, 핵심 이득은 작은 front를 block-per-front에서 warp-packed로 옮기는 것). 같은 plan tier 배열(`h_plcols_tier`/`h_level_tier_off`)과 **동일 occupancy 게이트**를 forward·backward 두 sweep에 적용. 같은 level 내 front 독립 → 두 sweep 모두 sub-range order-free·correct.

solve 시간 delta (median of 5, interleaved; factor는 양쪽 다 tier-split ON):

| case | B=1 | B=16 | B=64 | B=256 |
|------|----:|-----:|-----:|------:|
| case8387 solve | −0.4%* | −1.1% | **−8.9%** | **−15.8%** |
| ACTIVSg25k solve | −0.1%* | −1.3% | **−2.5%** | **−4.1%** |

`*` B=1은 split 비활성(중립). factor와 동일 패턴. 정확성 불변(relres 8387 ~2e-5, 25k ~1.3e-4, multistream 1.9e-5).

→ 고 B에서 factor+solve 합산 per-system이 case8387 B=256 기준 **~−14%**(factor −13% + solve −16%).

## 7. multistream으로 확장

multistream은 subtree마다 별도 스트림에서 자기 panel slice를 sweep한다(factor: `issue_factor_levels`, solve: forward/backward fork). 분해가 의미있는 단위는 **(subtree, level) cell**이다(spine은 cnt=1 chain이라 split 대상 아님). analyze의 subtree bucketing에서 각 cell 내부를 **tier-major로 정렬**하고 per-cell tier 경계(`h_subtree_level_tier_off`)를 기록 → `d_plcols`를 그대로 재사용해(cell이 in-place tier-정렬됨) dispatch가 single-stream과 **동일한 occupancy 게이트**로 cell을 tier-split. cell 경계(`h_subtree_level_off/cnt`)·fork/join은 불변, cell 내부 순서만 바뀌므로(같은 level 내 독립) 정확.

multistream factor+solve 합산 delta (default 경로, median of 5, interleaved):

| case | B=1 | B=64 | B=256 |
|------|----:|-----:|------:|
| case8387 | −0.4%* | **−2.6%** | **−4.3%** |
| ACTIVSg25k | −1.1%* | **−3.4%** | **−3.5%** |

`*` B=1 중립. 단일-스트림(factor −11~13%)보다 이득이 작은 이유: cell이 ~1/K 크기라 per-cell small front 수가 적어 게이트 발동·warp-packing 이득이 옅고, 이미 K 스트림으로 병렬화돼 있음. 그래도 양쪽 케이스 무회귀 win. 정확성 불변(multistream relres 8387 ~2e-5, 25k ~1.6e-4).

## 8. 한계 / 후속
- 25k 이득이 작은 건 panel_cap=12(n≥16k auto-bump)로 front가 커서 tiny-front 비중이 작기 때문 — 일반 성질(큰 행렬일수록 amalgamation↑).
- 게이트 임계(warp_fill)는 1-wave 충전 기준. 매우 큰 행렬의 cell이 B=1에서도 이를 넘으면 미세 launch 오버헤드 가능(3-tier에선 관측 안 됨).
- factor·solve, single·multistream 네 경로 모두 동일 분해+게이트로 통일됨(dispatch의 `issue_factor_tiered` / solve의 `dispatch_tiered` 헬퍼 공유).
- 게이트 임계(warp_fill)는 1-wave 충전 기준. 매우 큰 행렬의 mixed level이 B=1에서도 이 값을 넘으면 미세 launch 오버헤드가 날 수 있으나(5-class에서 관찰), 3-tier에서는 B=1 영향이 noise 내.
