# Tier 통합: 4-tier → 3-tier (panel-resident "big" 제거, large→big 개명)

> **상태**: 구현·측정 완료(25k·USA·parabolic, B=1/64)   **갱신**: 2026-06-18
> **한 줄**: 옛 4-tier(small/mid/big=panel-resident/large=global)를 **3-tier(small/mid/big=global multi-block)**
> 로 통합. A/B 측정으로 panel-resident "big" 티어가 무용함을 확인 — 배치에서 ~1.2%뿐(옛 −9.3% 재현 안 됨),
> B=1 에선 65–111 front 를 global multi-block 에 보내는 게 **16% 더 빠름**. 라우팅이 size→커널 3갈래로 일관됨.

선행: [§07 일반화](07-generalization-suitesparse-2026-06-16.md)(large-tier multi-block), [§09 기전](09-strumpack-mechanism-ncu-2026-06-17.md).

---

## 1. 옛 4-tier 와 문제의식

| 옛 tier | front | 커널 |
|---|---|---|
| small ≤32 | sub-group packing | factor_small |
| mid 33–64 | whole-front shared, 1 block/front | factor_mid |
| **big 65–111** | **panel-resident**(L/U 패널만 shared, CB global) — B=1 은 mid 로 위임, B>1 만 panel-resident | factor_big |
| large >111 | global 상주, 타일 staging, **한 front 를 여러 블록에 분산** | factor_large |

문제: **big 과 large 는 같은 "크기 클래스"(>64)이고 차이는 batch 변형뿐**이다. big 의 64 경계는 구조적(용량·
paradigm)이 아니라 occupancy 휴리스틱. 그리고 big 은 B=1 에서 어차피 mid 로 위임 → 실효 tier 가 아니었다.

## 2. A/B 측정 (25k·USA, FP64; classify_front_tier 에서 65–111 라우팅 변경, analyze grouping 까지 일관)

**big 흡수 비교 (factor):**
| | B=1 (USA) | B=64 (USA) |
|---|---|---|
| baseline (panel-resident big) | 2.28 ms | 0.238 ms/sys |
| big→mid (whole-front) | 2.08 (−9%) | 0.239 (동률) |
| **big→large (global multi-block)** | **1.90 (−16%)** | 0.238 (동률) |

**panel-resident 게이트 비교 (USA B=64, panel-resident 가 정말 이득인가):**
| 정책 | factor/sys | vs whole-front |
|---|---|---|
| 옛 게이트(항상 panel-resident, §07 동작) | 0.2365 | −1.2% |
| never PR (whole-front) | 0.2394 | 기준 |
| big→large | 0.2378 | −0.7% |

→ **옛 §07 "panel-resident USA B=64 −9.3%"는 더 이상 성립 안 함 — 현재 ~1.2%**(그 사이 whole-front 경로가
panel_width 16→8·포화 게이트·커널 개선으로 빨라져 격차가 닫힘). 네 정책 전부 1.2% 안쪽 = 노이즈.

**왜 B=1 에서 big→large 가 16% 빠른가:** 65–111 은 few-but-large front. whole-front 1 block/front 는 그 큰 front
하나를 한 블록에 가둬 occupancy 낮음. global multi-block 은 한 front 의 pivot·패널·trailing 을 여러 블록에 분산해
GPU 를 채운다 — §09 의 "front 적을 때 thread/block 을 늘려 채운다"와 정확히 일치(fill 이론).

## 3. 결정 — 3-tier 통합

- **small ≤32** (sub-group packing, factor_small) — 변화 없음.
- **mid 33–64** (whole-front shared, factor_mid) — 변화 없음.
- **big >64** = **옛 large 커널**(global-resident, 타일 staging, multi-block). 옛 panel-resident big 은 삭제,
  65–111 은 여기로 흡수. "large"는 "big"으로 개명.

`enum FrontTier{kSmall,kMid,kBig}`, `kNumFrontBuckets 4→3`, `classify_front_tier`: `>64 → kBig`.
파일: `large.cuh → big.cuh`(factor_large→factor_big 등), 옛 `big.cuh`(panel-resident) 삭제. `CLS_BIG_GATE`
진단 토글도 제거.

## 4. 검증 (빌드 OK, 잔차 정상)
| 케이스 | B=1 factor | 잔차 |
|---|---|---|
| case3120sp | — | 6.8e-14 |
| ACTIVSg25k | 0.908 (불변) | 2.3e-14 |
| **SyntheticUSA** | **1.90 (옛 2.28, −16%)** | 2.4e-14 |
| parabolic_fem (>111 front, big 멀티블록) | 57 ms | 1.1e-14 |

→ **단순화 + B=1 USA −16% + B=64 무손실 + FEM(>111) 정상.**

## 5. 의미 (정책 일관성)
이전엔 batch 결정이 size tier(mid/big)에 섞여 있었다(big=panel-resident-at-B>1). 이제:
- **size → 커널** (small/mid/big), 결정적.
- big 의 multi-block 은 batch 와 무관하게 항상 GPU 를 채우는 방향(few-front 면 B=1 도 채움) → batch-only 분기 불필요.
- 남은 batch 게이트는 small 의 packing(`factor_saturates`, front×batch 결합) 하나로 통일됨.

## 6. 한계
- big(=옛 large) 커널을 65–111 에 쓰는 게 B=1 에서 16% 이득이지만, 그 이상의 multi-block 오버헤드가 *매우 작은*
  big front(65 근처, 많은 수)에서 손해일 가능성은 미검증(측정상 USA·FEM 에선 이득/중립).
- panel-resident 가 ~1.2% 라도 줄 수 있는 특정 batch 레짐(매우 큰 big-front 수 × 큰 B)은 이론상 존재하나, 측정
  케이스에선 무의미해 제거함.
</content>
