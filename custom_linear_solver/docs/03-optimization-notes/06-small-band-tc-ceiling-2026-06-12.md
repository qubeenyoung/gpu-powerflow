# small-tier 상단 band(fsz 17–32) TC 경로 — ceiling-first 기각

> **상태**: negative result   **갱신**: 2026-06-12
> **한 줄**: small 의 상단 sub-band(fsz 17–32)에 TC trailing 을 얹는 [`260612_goal.md`](../260612_goal.md) 목표 1 은, 그 band 이 trailing FLOP 의 2.7–5.4%(large case)뿐이고 fronts 의 65–89% 가 **thin-K(nc<8)** 라 TF32 mma 타일을 K 차원에서 못 채워 — ceiling 이 factorize 의 <1% 라 기각한다. small-tier-no-TC 결론을 상단 band 까지 정량 확장하며, 불가 원인을 "front 가 작다"에서 "**front 가 커져도 K=nc 가 작다**"로 날카롭게 한다.

**방법**: 빌드/실행 없이 기존 per-front 덤프(`20260612_lab_meeting/data/fronts_*.csv`, 컬럼 `q,p,fsz,nc,uc,level,...`) 위에서 회계. 스크립트 [`20260612_lab_meeting/scripts/small_band_ceiling.py`](../20260612_lab_meeting/scripts/small_band_ceiling.py). trailing-GEMM FLOP = `2·uc²·nc`, Ampere TF32 mma 타일 = 16×8×8(M×N×K). 대상 7 케이스.

---

## 1. ceiling — band 의 FLOP 비중 (단계 1)

| case | fronts | band#(17–32) | band% | **band FLOP%** | small≤32 FLOP% | mid FLOP% | big FLOP% |
|---|---:|---:|---:|---:|---:|---:|---:|
| case3012wp | 2,780 | 57 | 2.1% | **30.9%** | 62.0% | 38.0% | 0% |
| case6468rte | 5,902 | 125 | 2.1% | **21.4%** | 43.0% | 57.0% | 0% |
| case8387 | 7,406 | 240 | 3.2% | **27.7%** | 49.3% | 50.7% | 0% |
| case13659 | 12,388 | 228 | 1.8% | **14.4%** | 31.3% | 68.7% | 0% |
| ACTIVSg25k | 22,724 | 482 | 2.1% | **5.4%** | 13.7% | 86.3% | 0% |
| ACTIVSg70k | 63,838 | 1,254 | 2.0% | **2.7%** | 7.2% | 62.5% | 30.3% |
| SyntheticUSA | 74,231 | 1,713 | 2.3% | **3.4%** | 8.2% | 60.4% | 31.4% |

전체 189,269 front 중 band = 4,099 (2.2%), trailing FLOP 비중 = **3.81%** (small 전체 9.3%).

**두 갈래**:
- **large case (25k/70K/USA — factorize 가 비싼 곳)**: band = FLOP 의 **2.7–5.4%**. TC 를 무한히 가속해도 factorize ceiling < 5.4%, 실효 < 1%.
- **small low-fill case (3012wp/6468rte/8387)**: band FLOP% 가 21–31% 로 커 보이나, 이는 **big tier 가 없고(big%=0) 전체 FLOP 이 작아서**(0.6–2.5 M vs USA 130 M) 생긴 상대 비중일 뿐 — 절대 시간 이득이 작다. 게다가 fill 이 나쁘다(§2). 이 영역은 이미 기각된 front-coarsening(storyline §3B2, net≈0)과 겹친다.

---

## 2. TC 타일 충족 — K=nc 가 binding (단계 2)

trailing GEMM 은 M=N=uc, **K=nc**. band 의 (uc, nc) 분포로 16×8×8 타일 충족도:

| case | band# | **nc≥8 %** (K 충족) | uc≥16 % (M=N 충족) | nc med | uc med | **K<8 인 band 비중** | median 타일 fill |
|---|---:|---:|---:|---:|---:|---:|---:|
| case8387 | 240 | **34.6%** | 62.9% | 5 | 17 | **65%** | 34.7% |
| ACTIVSg70k | 1,254 | **11.0%** | 86.2% | 2 | 18 | **89%** | 25.0% |
| SyntheticUSA | 1,713 | **11.2%** | 84.3% | 2 | 18 | **89%** | 25.0% |

**핵심**: band 에서 **M=N=uc 는 충분**(uc median 18, ≥16 이 63–86%)한데 **K=nc 가 작다**(median 2–5, nc≥8 은 11–35% 뿐). TF32 mma 의 K=8 을 못 채우는 front 가 large case 에서 **89%**. median 타일 fill 25–35% → 타일의 65–75% 가 K 방향 zero-padding.

→ [`20260612_lab_meeting/small-tier-no-tensorcore.md`](../20260612_lab_meeting/small-tier-no-tensorcore.md) 는 "median front(4×4×2)가 작아 M·N 을 못 채운다"고 했지만, **상단 band 에서는 M·N 은 충분하고 K 가 binding**이다. 즉 불가 원인은 "front 크기"가 아니라 **전력망 Jacobian 의 thin-K 성질이 fsz 와 무관하게 유지**되는 데 있다(storyline §0 thesis 의 thin-K 와 동일 뿌리). front 를 키워도(coarsening) K 는 안 커지므로 이 결론은 front-coarsening 으로도 안 뒤집힌다.

---

## 3. occupancy 손익 — 채택 시 잃는 것

band(fsz 17–32)은 `factor_small_sg` 가 sub_group_size=32(warp 1개/front)로 처리([`src/factorize/small.cuh:135`](../src/factorize/small.cuh)). small tier 는 warp-packing 으로 **occupancy 48–69%** — 전 tier 중 최고([`factorize-bottleneck-ncu.md`](../20260612_lab_meeting/factorize-bottleneck-ncu.md)). TC 를 위해 block-per-front 로 가면 mid/big 과 같은 **1 block/SM(occ 25%)** 로 떨어진다. 즉 band TC 는 *small 의 유일한 강점(occupancy)을 3.8% FLOP·25% 타일 fill 짜리 TC 와 맞바꾸는* 거래다.

---

## 결론 — 기각

1. band 은 large case 에서 trailing FLOP 의 **2.7–5.4%** → TC ceiling < factorize 의 1%.
2. band 의 **65–89% 가 thin-K(nc<8)** → TF32 mma 타일을 K 에서 못 채움(median fill 25–35%). 불가 원인은 front 크기가 아니라 **fsz 와 무관한 thin-K**.
3. 채택 시 small 의 occupancy(48–69%, 최고)를 1 block/SM(25%)로 희생.
4. small low-fill case 의 큰 FLOP% 는 전체 FLOP 이 작은 데서 온 상대값이고, 이미 net≈0 인 front-coarsening(§3B2)과 겹침.

→ **목표 1 종료(reject)**. ceiling-first 원칙대로 구현 없이 마감. 다음 레버는 [`260612_goal.md`](../260612_goal.md) 우선순위대로 **목표 3(big front tiling, 시간 64%·1-block/SM)** 와 목표 2(경계 근거화).

> 편입: storyline §4(negative result)에 "small-band TC — thin-K ceiling" 한 줄 추가 대상. small-tier-no-TC 의 상단-band 정량 확장으로 본 노트를 인용.
