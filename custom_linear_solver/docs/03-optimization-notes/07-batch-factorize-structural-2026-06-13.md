# 배치 factorize 구조 최적화 — panel-resident 커널이 whole-front shared 천장을 깬다

> **상태**: SUPERSEDED (panel-resident tier 제거됨, 2026-06-18)   **갱신**: 2026-06-13
> **정정(2026-06-18, [§10 tier 통합](../05-reports/10-tier-consolidation-2026-06-18.md)):** 이 문서의 핵심 결과인
> "panel-resident USA B=64 −9.3%"는 **더 이상 재현되지 않는다 — 현재 ~1.2%**(그 사이 whole-front 경로가
> panel_width 16→8·포화 게이트·커널 개선으로 빨라져 격차가 닫힘). A/B 결과 65–111 front 는 panel-resident보다
> **global multi-block(옛 large) 커널이 B=1 에서 16% 빠르다.** 그래서 panel-resident "big" 티어는 **제거**되고
> 65–111 은 global 커널에 흡수됨(4-tier→3-tier). 아래 내용은 역사적 기록.
> **한 줄**: B=64 factorize 는 memory-latency 바운드(AI≈2 ≪ roofline ridge 10) + thin-K(nc median ~15)라 bandwidth/compute 어느 쪽도 못 채운다. **STRUCTURAL 승리는 panel-resident mid 커널**: L/U 패널만 shared(CB 는 global)로 두어 shared 를 fsz²→nc(fsz+uc) ≈ 3× 축소 → blocks/SM 3–4× → **DRAM 2–32%→55–65%, SyntheticUSA B=64 −9.3%**(default-on). register-block trailing(+4%)·scatter 미세최적화(+2–3%)는 미세 레버. gather/tiled/fewsync/sysblk/small-band-TC 는 전부 회귀(→ §negative).

exp_260612 (RTX 3090, sm_86) 배치 체제. 원자료: `../exp_260612/`(notes 06·09·10), B=1 체제는 [`06-b1-factorize-regime`](06-b1-factorize-regime-2026-06-13.md).

---

## 1. 진단 — 배치는 latency·thin-K 바운드

B=64 면 under-fill(B=1 의 병목)은 사라진다 — factorize 시간의 84% 가 꽉 찬(waves≫1) launch 에서 나온다.
그런데도 큰 mid front 는 **whole-front(fsz²) shared 가 1–2 block/SM 로 occupancy 를 묶어** DRAM 이용률이 큰 front 에서
2–3%(ncu)까지 떨어진다. roofline: AI≈2 FLOP/byte ≪ ridge 10 → latency-bound. 게다가 nc median ~15 의 **thin-K** 라
trailing GEMM 이 K-thin → occupancy 로 latency 를 숨길 여지도 좁다.

---

## 2. STRUCTURAL 승리 — panel-resident mid 커널 (`factor_mid_panel`)

whole-front 대신 **L/U 패널만 shared** 에 둔다(CB(uc², 대부분)는 global 유지):
- `Lpan` = rows[0,fsz)×cols[0,nc), `Upan` = rows[0,nc)×cols[nc,fsz). shared = nc(fsz+uc) ≈ fsz²/3.
- Phase 1/2(panel LU + sync-free U-solve)는 shared, Phase 3(single-pass trailing)는 global CB 를 읽어
  Schur = assembled_CB − L·U 를 계산하고 extend-add 를 부모로 직접 fuse(global CB 트래픽은 baseline stage-in 과 동일 1-pass).
- shared 3× 축소 → **blocks/SM 3–4× → DRAM 2–32%→55–65%**.

| case | 정밀도 | B | Δfactorize | 게이트 |
|---|---|---:|---:|---|
| 70K(USA) | fp32 | 64 | **−9.3%** | big(fsz≥112, ≥2·SM blocks) |
| 70K(USA) | fp32 | 16 | −2.1% | |
| 25K | fp32 | 64 | −1.4~4.3% | med(fsz≥64, ≥16·SM blocks) |
| 25K/70K | fp64 | 64 | −7~8% | FP64 가 더 shared-starved |

default-on(`CLS_MID_PANEL=1`), 게이트: `CLS_MID_PANEL_MIN=112`(big fsz), `_MED=64`(med fsz), `_MED_BLK=16`(med occupancy).
TF32 panel-TC trailing(`CLS_MID_PANEL_TC=1`)은 측정상 중립이라 scalar 기본(`=0`). 모든 정밀도 적용, 회귀 없음.
06-10 cuDSS 스윕 대비 custom factorize 우위가 usa B=64 4.4×→5.05× 로 확대.

> 재측정 주의: 본 환경의 run-to-run factorize 분산(~±10%)이 큰 case 에서 9% 효과를 가릴 수 있다. 게이트가 닫히는
> 케이스(작은 front·낮은 occupancy)에선 panel 경로가 자동 우회되어 회귀가 없다.

---

## 3. 미세 레버 (default-on, ~1%씩)

- **register-blocked trailing**(`trailing_update_rb`, `CLS_TRAIL_RB`={0,2,4,8}, 기본 4): 스칼라 trailing 의 FMA:load
  를 1:2→MR:(1+MR) 로 끌어올림. **+4%**([exp 06]).
- **sync-free U-solve**(`u_panel_solve_fewsync`): nc block barrier → 1. ~1%.
- **conditional row-fused panel LU**(nc≤12 항상, nc≤16 & fsz≤96 까지): pivot 당 barrier 2→1. 25K ~3%, USA 회귀 회피.
- gather 코드를 hot path 에서 컴파일 제외 → 레지스터 mid 60→51·big 48→40 → occupancy +1.3%.
- 합산 8387 −1.3% / 25K −2.7% / USA −2.5% @B64([exp 09]). 정직한 천장: micro-lever 당 ~1%.

---

## negative — 회귀로 결론난 실험들 (코드는 `deprecated/`)

| 실험 | 결과 | 기전 | 위치 |
|---|---|---|---|
| **gather assembly** (5 모드) | ✗ fused −15%, phase-batched +78~100% | scatter 의 구조적 우위(streaming memset + atomic-free unique scatter + factor-fused extend-add); fused-assembly 천장 1.2× 도 도달 불가 | `deprecated/gather_assembly/` |
| **tiled-trailing** (2-커널) | ✗ B=1 0.66–0.72×, B=64 0.62–0.68× | L/U DRAM 재staging +28%·launch +57% 가 occupancy 이득 잠식; thin-K 에 tiling 비효율 | `deprecated/tiled_trailing/` |
| **mid fewsync / blocked-fp32** | ✗ barrier 5.36→5.12, wall 중립/+2% | barrier 는 증상, 묶이는 건 whole-front shared 1 block/SM | `deprecated/mid_fewsync/` |
| **sysblk** (systems-per-block) | ✗ 중립 | double-buffer cp.async 로드 hide 가 이득 없음 | `deprecated/mid_sysblk/` |
| **small-band TC** (fsz 17–32) | ✗ 천장 <1% | band trailing FLOP 2–5.4%, 65–89% thin-K(mma K<8) → small-tier occupancy peak 희생 | (분석만, `../20260612_lab_meeting/scripts/small_band_ceiling.py`) |

→ 공통 교훈: 전력망+ND front 의 **thin-K 구조**가 compute-bound 전환을 막는다(amalgamation 으로 nc 를 키워도 work-wt nc
~4.6 plateau — [`06`](06-b1-factorize-regime-2026-06-13.md) §2, `deprecated/amalgamation/`). 배치의 진짜 레버는 occupancy =
**panel-residency(이 노트 §2) + tf32(별도)**. 정확한 통합 형태는 git 커밋 `7fe15a7`.
