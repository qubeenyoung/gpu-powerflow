# Research Storyline — Tiny-Front 전력망 Jacobian 을 위한 배치 GPU Multifrontal 기여

**작성일**: 2026-06-10
**구성**: 하나의 근원(tiny-front regime)에서 내려오는 논리. cuDSS 와 공유하는 substrate 를 분리하고,
그 위의 **연구 기여 4개**를 체계적으로 서술한다. 각 기여는 ablation 토글에 매핑된다.
(빌드/실행 설정은 [`optimal-configuration.md`](optimal-configuration.md).)

---

## 0. Thesis

> 전력망 Jacobian 인수분해의 비효율은 **front 가 작다(tiny-front, thin-K)** 는 단일 성질에서 나온다.
> 배치 정적 multifrontal(cuDSS 와 공유하는 바닥) 위에, tiny-front 를 GPU 에 맞추는 **front-execution
> specialization** 를 얹는다: (A) front 를 **어디로(tier routing)·언제(dispatch scheduling)** 보낼지,
> (B) trailing 을 **텐서코어로 태우고(TC trailing)·작은 front 를 키워(front coarsening)** TC 에 맞춘다.

---

## 1. 근원과 병목 — tiny-front (thin-K) regime

**base multifrontal**: METIS nested-dissection → supernode/multifrontal assembly → front 별 dense LU
→ extend-add (모든 ablation 의 바닥).

전력망 Jacobian 의 정의적 성질: **front 가 작다**(대부분 `fsz≤16`, panel width `nc=1..20`),
**etree 가 깊고 좁다**. 이 하나가 모든 병목의 근원이다:

| 입력 성질 | 귀결 병목 |
|---|---|
| front 가 작다 | dense 커널이 GPU 를 못 채움(occupancy↓); trailing GEMM 이 **thin-K(K=nc) memory-bound** → 텐서코어 underfill |
| 크기가 이질적 | 한 커널로 다 돌리면 자원 부정합; mixed-size launch 의 shared 낭비 |
| etree 깊고 좁다 | level 마다 launch+barrier, level 내 동시성 부족 |

요약: big-kernel 시간의 **61% 가 panel-LU+U-solve barrier, 19.5% 가 trailing**(`03-optimization-notes/29`).
가속은 FLOP 이 아니라 **점유·지연·동기화·thin-K**에서 나온다. 아래 4개 연구 기여는 각각 이 귀결을 친다.

---

## 2. Substrate — cuDSS 와 공유하는 바닥 (연구 기여 아님)

다음은 우리만의 기여가 아니라 **전제·엔지니어링**이다. 연구 축에서 제외한다.

- **배치 정적 multifrontal**: 사전 확정 symbolic plan(analyze-once / refactorize), B-시스템 uniform
  batch, device-resident solve. — cuDSS 도 `CUDSS_CONFIG_UBATCH_SIZE` + analyze/refactorize 로 동일하게
  한다. 차별점이 아니라 **공통 바닥**.
- **No-pivot 정적 구조**: power-grid 대각우세로 부분 pivoting 없이 정확(`02-design-analysis/03`).
  정적 파이프라인의 **전제**.
- **CUDA graph capture**: 반복 인수분해의 호출 오버헤드를 줄이는 표준 엔지니어링.
- **Kernel optimization** *(구 "execution shaping")*: 각 커널 내부의 on-chip staging·occupancy·
  thread/layout 튜닝(direct-shared staging, blocked, launch config, cp.async). **이것이 실측상 가장 큰
  속도 덩어리**다 — experiment #3 에서 tf32 이득의 **~62% 가 여기**(텐서코어 없이 정확도 보존만으로
  +11~15%). **그러나 이는 숙련된 GPU 엔지니어링이지 연구 novelty 가 아니므로 기여 축에서 제외**한다.
  (연구 기여의 크기를 정직하게 보려면 이 부분을 분리해서 봐야 한다 — §5.)

---

## 3. 연구 기여 — Front-execution specialization (4개)

*고정·배치된 front 연산을 tiny-front regime 에 맞게 매핑하는 아이디어.* 각 기여:
**아이디어 → 메커니즘 → 공격 병목 → ablation 토글 → 실측**.

### A. Execution mapping — front 를 어디로/언제

#### A1. Tier routing *(3단 커널 라우팅)*
- **아이디어**: front 크기로 전용 커널 선택 — **small**=warp-packed, **mid**=shared-resident,
  **big**=global + 1024-thread. 이질적 tiny front 에 *어느 커널*인지를 결정.
- **메커니즘**: tier 별로 occupancy / shared 예산 / thread 수를 front 모양에 맞춤.
- **공격 병목**: 단일 커널로 모든 front 를 돌릴 때의 자원 부정합.
- **ablation**: `SolverConfig.tier_split` (false = whole-level 단일 커널).
- **근거**: 임계값 `small≤32, mid≤128` HW 근거 `02-design-analysis/06`. tier 순기여는 whole-level vs
  tiered ablation 으로 측정 가능(예상 양성).

#### A2. Dispatch scheduling *(동형 디스패치 + 서브트리 멀티스트림)*
- **아이디어**: front 를 *언제·어느 launch/stream* 으로 보낼지. 두 메커니즘:
  - **동형 디스패치**(구 fsz-band-split): 한 (subtree, level) 의 panel 을 front 크기 band 로 stable-sort
    후 같은 크기끼리만 sub-launch → launch 일감 동형화 + tier 경계 가로지르는 tiny front 재라우팅.
  - **서브트리 멀티스트림**: 독립 subtree 를 별도 CUDA stream 으로 → Hyper-Q 가 한 stream 의 SM idle
    cycle 을 다른 stream 의 block 으로 채움.
- **공격 병목**: mixed-size launch 의 shared 낭비/오배정 + level 내 occupancy 부족.
- **ablation**: 동형 정책 on/off; `--no-multistream`.
- **실측**: 동형 → 70K/USA B=64/256 wall **−11~15%**(`23`); 멀티스트림 → case8387 B=64 **−22%**,
  USA B=64 **−14%**(`13`).

### B. Numeric specialization — trailing 을 텐서코어에 맞춘다

> 진단 전제: trailing 은 `M=N=uc≈140, K=nc≈20` 의 **thin-K GEMM** → naive 하게는 텐서코어가 안 먹힘
> (mma 기준 K=2.5 tile, staging 이 mma 압도; `28`). 그래서 이 축은 *TC 를 그냥 켜는 것이 아니라 먹히게
> 만드는 것*이다.

#### B1. TC trailing *(TF32 텐서코어 trailing + Ozaki 정확도 회복)*
- **아이디어**: trailing 업데이트 `C ← C − L·U` 를 **TF32 `mma` 로** 수행(mid=direct-shared,
  big=blocked). 저정밀 round-off 는 **Ozaki 2-성분 분할**(`x0=cvt.rna.tf32(x)`, `x1=cvt.rna.tf32(x−x0)`,
  `L0U0+L1U0+L0U1` 를 텐서코어 누산기에서 누적)로 **FP32 정확도 회복**.
- **메커니즘**: front 는 FP32, L/U panel 만 TF32 변환 후 mma; Ozaki 는 trailing mma 내부에만 작용해
  어떤 routing/scheduling 위에도 직교 스택. memory-bound trailing 의 **놀던 mma 헤드룸**을 쓰므로 보정이
  거의 공짜.
- **공격 병목**: thin-K trailing 의 scalar 연산 + 저정밀.
- **ablation**: `CLS_MID_TF32_TC`, `CLS_BIG_TF32_BLOCKED_TC`, `CLS_TF32_OZAKI_TC2[_FIRST_ORDER]`.
- **실측(본 세션 재현)**:
  - **텐서코어 순기여 = +6~9%** (large case; tf32 이득 중 ~38%. 나머지 62% 는 kernel optimization §2 —
    experiment #3 분해).
  - **Ozaki**: 8387 tf32 relres **3.97e-2 → 4.77e-5**(≈FP32), 속도 거의 불변. → *저정밀 TC 를 쓰되
    FP32 정확도 유지* 가 이 기여의 핵심(속도보다 방법론).

#### B2. TC-routable front coarsening *(패널 융합 — 저-fill 전용)*
- **문제**: 저-fill 케이스(8387)는 front 가 너무 작아(`nc≈1–2`) 텐서코어 tile(K=8)을 못 채운다 — tile
  효율 1.5–13%(`37`). 아무리 좋은 mma 도 K=1 일감엔 무용.
- **아이디어**: elimination tree 에서 **인접 panel 을 융합해 front 를 키우되**, ① multifrontal 구조가
  깨지지 않고(부모-자식 containment) ② 융합 결과가 실제로 **TC-routable 범위(`fsz>32, uc≥16,
  4≤nc≤32`)** 일 때만. *"TC 가 회수할 때만 fill 을 더 쓴다."* 커진 front 가 mid TC 로 가도록 tier 경계도
  조정(`kSmallFrontMax 32→16`).
- **공격 병목**: tiny-K front 의 텐서코어 underfill (구조적).
- **ablation**: `CLS_TC_CLOSURE_PANEL_AMALGAMATE(_CAP)`, `CLS_SMALL_FRONT_MAX_16`.
- **실측**: 8387/13K 가 처음 TC 목표권 진입(8387 B64 **1.24×** raw; `50`). 마진 얇고 seed 민감 →
  large 정책과 분리.

---

## 4. 안 통한 것 (negative result)

- **Column-owned U-solve** (`CLS_TF32_COLUMN_USOLVE`): 61% 병목(panel-LU+U-solve barrier, `29`)을 겨눠
  U-panel 을 column 단위 병렬화. **재현 결과 효과 ≈0%**(fp32 에 동일 적용해도 −0.4~+0.9%, 노이즈).
  코드 비대칭 ≠ 성능 비대칭. 코드엔 토글로만 잔존, 기여로 세지 않는다.
- 기타 폐기(cuBLAS grouped mid, fused trail+extend, sibling/chain/validated amalgamation, FP16 force-all,
  batch-dim packing)는 회귀/무효라 master 에서 제거(`33/39/41/43/47/48`).

---

## 5. Ablation 설계 + experiment #3 의 정직한 분해

**baseline = base multifrontal**(`--precision fp64`, 토글 OFF). 각 기여를 leave-one-out 으로 끄고
`factor_ms/sys` + `relres` delta 측정.

| 끄는 기여 | 토글 | 측정 의도 |
|---|---|---|
| A1 Tier routing | `tier_split=false` | 단일 커널 대비 tier 순기여 |
| A2 Dispatch scheduling | 동형 off / `--no-multistream` | launch 균질화·동시성 (8387 +22%) |
| B1 TC trailing (mma) | `--precision fp32` | TC + kernel-opt 합산 |
| B1 Ozaki | `OZAKI_..._FIRST_ORDER` off | **relres 5e-2 ↔ 1e-4**(정확도 회수), 속도 +~5% |
| B2 Front coarsening | `TC_CLOSURE_*` off | 8387/13K TC 진입 여부 |
| (음성) column-U-solve | `TF32_COLUMN_USOLVE` off | **≈0% 확인** |

**Experiment #3 (TC trailing 의 정직한 분해)** — tf32 커널의 형태는 그대로 두고 mma 만 scalar 로 치환해
측정한 large-case 결과:

| | fp32 → tf32 총 이득 | = kernel optimization(§2, 제외) | × **TC trailing mma(B1, 기여)** |
|---|---|---|---|
| 25K | 1.254× | 1.149× | **1.092×** |
| USA | 1.186× | 1.114× | **1.064×** |
| 70K | 1.207× | 1.124× | **1.074×** |

→ **연구 기여로서의 텐서코어는 +6~9%(이득의 ~38%)**; 큰 덩어리(~62%)는 kernel optimization(엔지니어링,
연구 축 제외). 이 분해를 **정직성의 근거**로 전면에 둔다 — TC 를 과장하지 않는 것이 신뢰를 만든다.

---

## 6. 한 장 요약

| 축 | 연구 기여 | 공격 병목 | 토글 | 효과 |
|---|---|---|---|---|
| A | **Tier routing** | 크기 이질성 / 자원 부정합 | `tier_split` | 우 커널 per tier |
| A | **Dispatch scheduling** | launch 낭비 / level 동시성 | 동형 정책, `--no-multistream` | 70K/USA −11~15%; 8387 −22% |
| B | **TC trailing** | thin-K 저정밀 trailing | `MID_TF32_TC`,`BIG_TF32_BLOCKED_TC`,`OZAKI_*` | mma +6~9%; relres 5e-2→1e-4 |
| B | **TC-routable front coarsening** | tiny-K underfill | `TC_CLOSURE_*`,`SMALL_FRONT_MAX_16` | 8387/13K TC 진입 |
| — | (substrate) 배치·정적 plan·kernel optimization | — | — | cuDSS 공유 / 엔지니어링(exp#3 62%) |
| — | (음성) column-U-solve | panel barrier | `TF32_COLUMN_USOLVE` | ≈0% |

**메시지**: 근원은 tiny-front regime. cuDSS 와 공유하는 배치 정적 multifrontal 위에, **front 를 어디로/
언제 보낼지(tier routing, dispatch scheduling)** 정하고 **trailing 을 텐서코어에 태우되 작은 front 를
키워(TC trailing, front coarsening)** thin-K 를 푼다. 정직한 분해(exp#3)상 텐서코어 자체는 +6~9% 의
*방법론적* 기여(정확도 보존이 핵심)이고, 가장 큰 raw 속도 덩어리는 연구 축 밖의 kernel optimization 이다.
