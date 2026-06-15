# B=1 factorize 체제 — under-fill 천장, 그리고 두 개의 레버(ordering · 텐서코어)

> **상태**: reference   **갱신**: 2026-06-13
> **한 줄**: B=1(단일 시스템) fp32 factorize 는 **소거트리 깊은 레벨의 under-fill(occupancy) 바운드**다 — scheduling/tiling/sync/amalgamation/thread-width 는 모두 1 block/SM·per-front latency 벽에 막힌다. 천장을 회수하는 레버는 단 둘: **ND ordering 선택**(→ [`08-ordering-best-of-k`](08-ordering-best-of-k-2026-06-13.md)) 과 **B=1 텐서코어 trailing**(Ozaki 로 fp32 정확도 유지). B=64 와는 정반대 체제.

exp_260612 (RTX 3090, sm_86, 82 SM) 의 B=1 분석을 압축한 노트. 원자료: `../exp_260612/` (notes 01·03·04·05·09·11),
배치 체제 구조 최적화는 [`07-batch-factorize-structural`](07-batch-factorize-structural-2026-06-13.md). 케이스 사다리:
13K=`case8387pegase`, 25K=`case_ACTIVSg25k`, 70K=`case_SyntheticUSA`. 모든 수치 median(repeat 25–30, warmup 8–10).

---

## 1. 진단 — 깊은 레벨이 GPU 를 비운다

소거트리를 leaf→root 로 레벨 단위 factorize 할 때, 깊은(root 근처) 레벨에는 독립 front 가 **물리적으로 2–66 개**뿐이다.
B=1 이면 레벨당 블록 수 = `cnt_L·B = cnt_L < 82(SM)` → 대다수 SM 유휴, 큰 front 는 **1 block/SM** 로 돌아 occupancy 3–30%.
factorize wall 의 대부분이 이 under-filled tail 에서 나온다(상위 well-filled 레벨은 throughput-bound, ordering 무관).

**정밀도 정정([exp 03]).** 초기 측정은 `--single-precision fp32`(= fp32 입력 + **fp64 factor**, ncu 가 `<double>` 노출)였다.
진짜 fp32 factor 는 `--precision fp32`(`<float>`). 진짜 fp32 는 대형 case 의 factor 시간을 크게 줄여(25K 1.85→0.73ms)
ordering 천장을 좁힌다. occupancy 패턴은 두 정밀도가 동일(둘 다 per-front latency-bound).

---

## 2. 불가능성 — ordering 외 레버는 벽에 막힌다 ([exp 04])

work–span 관점: B=1 critical path = 직렬 spine(레벨 직렬 의존)이고, scheduling/occupancy/kernel-tuning 으로는
이 span 을 줄일 수 없다. **소거트리 구조를 바꾸는 ordering 만**이 span 을 바꾼다. 실험·문헌이 일치:

| 레버 | 결과 | 근거 |
|---|---|---|
| **ND ordering 선택**(best-of-k) | ✅ +12–15% (fp64 factor) / +4–14% (true fp32) | fp32 seed 간 factor 시간 15–67% 변동 |
| subtree stream 8→16 | ✗ 무효 | deep level 에 독립 front 부족(2–66개) |
| amalgamation(deep-K) | ✗ 회귀 | thin-K 천장 — [`07`](07-batch-factorize-structural-2026-06-13.md) §neg, `deprecated/amalgamation/` |
| big-front tiling | ✗ 회귀 0.62–0.72× | re-staging — [`07`](07-batch-factorize-structural-2026-06-13.md) §neg, `deprecated/tiled_trailing/` |
| mid sync 감소(fewsync) | ✗ 중립 | barrier 는 증상, `deprecated/mid_fewsync/` |

문헌(STRUMPACK/CHOLMOD/cuDSS)은 *큰* root 를 가정 — B=1 작은-root under-fill 은 literature 의 빈 구멍이다([exp 02]).

---

## 3. 레버 ① ND ordering 선택 (best-of-k)

deterministic serial-ND 를 K 시드에 돌려 가장 빠른 plan 을 고르면 천장을 회수하고, parallel-ND 의 run-to-run
비결정성(±8–24%)도 제거된다(relres 불변 — 같은 valid permutation). proxy 기반(`CLS_ORDER_K`, tail-cube):

| case | parallel-ND median | best-of-8 (`CLS_ORDER_K=8`) | speedup | 천장(26-seed oracle) |
|---|---:|---:|---:|---:|
| 13K | 0.572 | 0.509 (seed 49) | **1.124×** | 1.121× |
| 25K | 1.896 | 1.646 (seed 47) | **1.152×** | 1.175× |
| 70K | 8.476 | 7.480 (seed 44) | **1.133×** | 1.133× |

단 tail-cube proxy 는 fp32 에서 **anti-informative**(1순위 픽이 default 보다 8% 느릴 수 있음). 실제 레버는 시드별
**factorize 시간 측정** 선택(`CLS_ORDER_MEASURE_K`, B=1 −6~13%) — 상세·구현은 [`08-ordering-best-of-k`](08-ordering-best-of-k-2026-06-13.md).

---

## 4. 레버 ② B=1 텐서코어 trailing — 체제 반전 ([exp 11])

B=64 에서는 TC trailing 이 occupancy 를 깎아 손해다(panel-resident 가 정답, [`07`](07-batch-factorize-structural-2026-06-13.md)).
**B=1 은 정반대**: occupancy 가 이미 무의미(80+ SM 유휴, block-starved)하므로 trailing 의 mma 가 **critical path 를 단축** → 이득.

| 정밀도 | 25K | 70K(USA) | relres |
|---|---:|---:|---|
| TF32 TC (`--precision tf32`) | −17.4% | −17.4% | tf32 수준 |
| **Ozaki-TC** (fp32 정확도 유지) | **−8.5%** | **−17.5%** | 1.4e-4–4.6e-3 (fp32 보존) |

panel-residency 는 B=1 에선 중립/무해. → **두 체제 최적 커널이 정반대**: B≥16 = panel-scalar, B=1 = whole-front-TC.
Ozaki-TC 빌드: `-DCLS_TF32_OZAKI_TC2`(+`_FIRST_ORDER` 로 tail·tail 생략 가속), 런타임 `--precision tf32`.

---

## 5. 배치가 under-fill 을 메운다 ([exp 05])

B=1→B>1 은 `P_L ≤ cnt_L·B` 로 레벨 병렬성을 연다(B=1 의 `P_L ≤ cnt_L` 천장 해제). per-system 가속:

| case | B=1→B=64 | knee |
|---|---:|---|
| 13K | 13.1× | — |
| 25K | 7.3× | — |
| 70K | 5.2× | ~B=16 |

배치에선 ordering 이 씻겨나가고(under-fill 소멸), 구조 레버가 panel-residency 로 바뀐다([`07`](07-batch-factorize-structural-2026-06-13.md)).

---

## 결론

- B=1 factorize 의 1.2× 목표는 어느 정밀도에서도 세 case 일괄 미달(under-fill 벽 + ordering 천장이 대형 case 에서 <1.2×).
- 닫는 레버는 둘뿐: **measured best-of-k ordering**([`08`](08-ordering-best-of-k-2026-06-13.md)) 과 **B=1 Ozaki-TC**(이 노트 §4).
- worst-seed 대비로는 1.2×+ 를 만들 수 있으나 cap-inflation 비교는 금지(honest baseline = parallel-ND median).
