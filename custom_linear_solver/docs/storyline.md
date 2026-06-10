# Research Storyline — Power-Grid Jacobian 배치 인수분해를 위한 GPU Multifrontal 최적화 메소드

**작성일**: 2026-06-10
**목적**: base multifrontal 에 어떤 **메소드**로 최적화를 쌓았는지 체계적으로 서술한다. 각 메소드는
하나의 아이디어 — 공격하는 병목 — ablation 토글 — 실측 효과로 정의한다. (최적 빌드/실행 설정은
별도 문서 [`optimal-configuration.md`](optimal-configuration.md) 참조.)

---

## 0. Thesis

> 전력망 Newton–Raphson Jacobian 의 배치 인수분해는 **연산(FLOP)이 아니라 launch/sync latency,
> occupancy, memory-latency 에 묶인다.** 따라서 우리의 메소드는 (Ⅰ) **디스패치·스케줄링**으로 작은
> front 들이 GPU 를 채우게 만들고, (Ⅱ) trailing 업데이트를 **TF32 텐서코어**로 돌리되 작은 front 를
> **융합해 TC 가 먹히게** 하고, (Ⅲ) **Ozaki 오차보정**으로 FP32 정확도를 회수한다.

---

## 1. 출발점과 병목

**base multifrontal**: METIS nested-dissection → supernode/multifrontal assembly → front 별 dense LU
→ extend-add. 이것이 모든 ablation 의 바닥(reference)이다.

**전력망 특이성과 그 결과** — 이것이 모든 메소드가 겨누는 표적이다:

| 입력 성질 | 병목으로의 귀결 |
|---|---|
| front 가 작다 (대부분 `fsz≤16`, panel width `nc=1..20`) | dense kernel 이 GPU 를 못 채움(occupancy↓); trailing GEMM 이 **thin-K(K=nc) memory-bound** 라 텐서코어 underfill |
| etree 가 깊고 좁다 (spine + 2-way subtree) | level 마다 launch+barrier, level 내 동시성 부족 |
| 동일 sparsity 로 반복 인수분해 | per-call/per-level launch latency 누적 |

요약: **big-kernel 시간의 61% 가 panel-LU+U-solve barrier, 19.5% 가 trailing, 19.2% 가 extend**
(`03-optimization-notes/29`). 즉 가속은 GEMM 이 아니라 **점유·지연·동기화**에서 나온다.

### 전제(substrate) — 메소드가 아님

다음은 기여가 아니라 정적 파이프라인을 가능케 하는 **전제·엔지니어링**이다. 본 문서는 이들을
메소드로 세지 않는다:
- **No-pivot 가정**: power-grid Jacobian 의 대각우세로 부분 pivoting 없이 정확(`02-design-analysis/03`).
  구조를 analyze 에서 1회 확정 → factorize 는 numeric 만. *(가속 메소드가 아니라 정적화의 전제.)*
- **CUDA graph capture / device-resident solve**: 반복 인수분해의 호출 오버헤드를 줄이는 표준 엔지니어링.
- **FP32 front 저장**: 우리가 작업하는 **정밀도 regime**(연산을 TF32 텐서코어로 보내기 위한 substrate).
  정확도는 Ⅲ(Ozaki)에서 회수한다.

---

## 2. 메소드 (3개 축, 6개 메소드)

각 메소드: **핵심 아이디어 → 메커니즘 → 공격 병목 → ablation 토글 → 실측**.

### 축 Ⅰ — 디스패치 & 스케줄링 (작은 front 로 GPU 를 채운다)

#### M1. 배치 front-major 인수분해 *(Batched front-major factorization)*
- **아이디어**: B 개 시스템이 같은 symbolic plan 을 공유하므로, front arena 를 `B×front_total` 로 잡고
  `blockIdx.y=batch` 로 두면 **한 launch 가 한 level 의 모든 B front 를 처리**한다.
- **메커니즘**: per-level launch·sync latency 를 B 로 분할상환; occupancy-starved level 을
  `B×level_size` block 으로 채움.
- **공격 병목**: launch/sync latency, occupancy.
- **ablation**: `--batch 1` vs `64/256`.
- **실측**: throughput saturation point — small B≈256, mid B≈64, big B≈16 (`04-benchmarks-profiling/16`).
  이후 모든 텐서코어 메소드의 전제(배치가 없으면 TC tile 도 못 채움).

#### M2. 3단 커널 라우팅 *(3-tier kernel routing)*
- **아이디어**: front 크기로 전용 커널 선택 — **small**=warp-packed, **mid**=shared-resident,
  **big**=global + 1024-thread.
- **메커니즘**: tier 별로 occupancy / shared-memory 예산 / thread 수를 front 모양에 맞춤.
- **공격 병목**: 단일 커널로 모든 front 를 돌릴 때의 자원 부정합.
- **ablation**: `SolverConfig.tier_split` (false = whole-level 단일 커널).
- **실측**: 임계값 `small≤32, mid≤128` 의 HW 근거는 `02-design-analysis/06`.

#### M3. 동형 디스패치 *(Shape-homogeneous dispatch — 구 "fsz-band-split")*
- **아이디어**: 한 (subtree, level) 안의 panel 들을 **front 크기 band 로 stable-sort 한 뒤 같은 크기끼리만
  모아 sub-launch** 한다. *한 launch 의 일감을 동형(同形)으로 만든다.*
- **메커니즘**: launch 마다 shared `fsz_cap²` 가 front 크기에 정합 → 점유↑; 또한 tier 경계를 가로지르는
  tiny front 가 큰 tier 의 커널에서 작은 tier 커널로 **재라우팅**됨.
- **공격 병목**: mixed-size launch 의 shared 낭비 + 잘못된 tier 배정.
- **ablation**: 동형 디스패치 정책 on/off (gate 예: `n≥40k→B>1`; `03-optimization-notes/23`).
- **실측**: 70K/USA B=64/256 wall **−11~15%**.

#### M4. 서브트리 병렬 멀티스트림 *(Subtree-parallel multistream)*
- **아이디어**: etree 의 독립 subtree 를 **서로 다른 CUDA stream** 으로 발사 → Hyper-Q 가 한 stream 의
  SM idle cycle 을 다른 stream 의 block 으로 채움.
- **메커니즘**: 좁은 etree 의 level 내 부족한 동시성을 subtree 간 동시성으로 보충(fork/join).
- **공격 병목**: level 내 occupancy 부족.
- **ablation**: `--no-multistream`.
- **실측**: case8387 B=64 **−22%**, USA B=64 **−14%** (`04-benchmarks-profiling/13`).

### 축 Ⅱ — 텐서코어 trailing (작은 front 에 TC 를 먹인다)

> 진단 전제: trailing 은 `M=N=uc≈140, K=nc≈20` 의 **thin-K GEMM** 이라 naive 하게는 텐서코어가
> 안 먹힌다(mma 기준 K=2.5 tile, staging 이 mma 압도; `03-optimization-notes/28`). 그래서 이 축의
> 메소드는 *TC 를 그냥 켜는 것이 아니라 TC 가 먹히도록 만드는 것*이다.

#### M5. TF32 텐서코어 trailing *(TF32 tensor-core trailing update — 구 C1+C2 통합)*
- **아이디어**: front 의 trailing 업데이트 `C ← C − L·U` 를 **TF32 `mma` 로** 수행. mid 는 shared-resident
  direct-shared 형태로(base 가 비워둔 슬롯에 신규 추가), big 은 L/U 를 shared 에 한 번 staging 하는
  blocked 형태로(기존 scalar/PTX 경로 대신).
- **메커니즘**: front 를 FP32 로 두고 L/U panel 만 TF32 로 변환해 `mma.m16n8k8` 누적. mid·big 양 tier 를
  하나의 텐서코어 trailing 경로로 통일.
- **공격 병목**: trailing 의 scalar 연산 + (big) uncoalesced C-drain.
- **ablation**: `CLS_MID_TF32_TC`, `CLS_BIG_TF32_BLOCKED_TC` (+ dispatch 보조 `*_LOW_SPLIT`,
  `*_SHARED_THREADS_512`). 끄면 `--precision fp32` scalar trailing.
- **실측**: 대형 배치(25K/70K/USA, B64/256)에서 FP32 대비 **1.20–1.27×** (본 세션 재현).
  **단 caveat**: 이득이 텐서코어 mma 자체인지 새 커널 launch-shape 인지 아직 미분리(§4 open question).

#### M6. TC 적합 패널 융합 *(TC-routable panel amalgamation — 저-fill 전용, 구 C5+C6)*
- **문제**: 저-fill 케이스(8387)는 front 가 너무 작아(`nc≈1–2`) 텐서코어 tile(K=8)을 못 채운다 — 아무리
  좋은 mma 도 K=1 짜리 일감엔 무용. tile 효율 1.5–13% (`03-optimization-notes/37`).
- **아이디어**: elimination tree 에서 **인접한 panel 을 융합해 front 를 키우되**, ① multifrontal 구조가
  깨지지 않고(부모-자식 containment 유지) ② 융합 결과가 실제로 **TC-routable 범위(`fsz>32, uc≥16,
  4≤nc≤32`)** 에 들 때만 병합한다. 즉 *"TC 가 회수할 수 있을 때만 fill 을 더 쓴다."*
- **메커니즘(2단)**: (a) analyze 단계에서 위 조건을 만족하는 consecutive panel group 만 amalgamate;
  (b) 융합으로 커진 중간 크기 front 가 mid 텐서코어 경로로 가도록 **tier 경계를 조정**
  (`kSmallFrontMax 32→16`) + mid TC launch shape 를 일감(49..64 dominant)에 맞춰 128-thread 로.
- **공격 병목**: tiny-K front 의 텐서코어 underfill (구조적).
- **ablation**: `CLS_TC_CLOSURE_PANEL_AMALGAMATE(_CAP)`, `CLS_SMALL_FRONT_MAX_16`,
  `CLS_MID_TF32_TC_THREADS_128`.
- **실측**: 8387/13K 가 처음으로 TC 목표권 진입 — 8387 B64 **1.24×** (`03-optimization-notes/50`).
  마진이 얇아 large-case 정책과 분리해야 한다(seed/cap 민감).

### 축 Ⅲ — 정확도 회수

#### M7. Ozaki 오차보정 TF32 *(Ozaki error-corrected TF32)*
- **아이디어**: TF32 의 10-bit 가수 round-off(raw tf32 relres ~5e-2)를, FP32 입력을 두 TF32 성분으로
  분할해 보정한다: `x0=cvt.rna.tf32(x)`, `x1=cvt.rna.tf32(x−x0)` → `L0U0+L1U0+L0U1(+L1U1)` 를
  **텐서코어 FP32 누산기에서 체인 누적**.
- **메커니즘**: M5 의 trailing mma 내부에만 작용하므로 어떤 디스패치/타일링/오더링 위에도 **직교 스택**.
  전 tier(mid/big/blocked/small-warp) 적용.
- **공격 병목**: TF32 정밀도 한계 (정확도 축, 속도 축 아님).
- **ablation**: `CLS_TF32_OZAKI_TC2`(4항) / `CLS_TF32_OZAKI_TC2_FIRST_ORDER`(3항).
- **실측(본 세션 재현)**: 8387 tf32 relres **3.97e-2 → 4.77e-5**(≈FP32 3.3e-5), 13K **6.47e-3 → 1.3e-4**.
  **속도는 거의 불변**(8387 B64 1.24×→1.19×) — trailing 이 memory-bound 라 놀던 mma 헤드룸을 쓰는 것이라
  보정 mma 를 더 돌려도 wall 이 안 늘어난다. *정확도를 거의 공짜로 회수.*

---

## 3. 안 통한 것 (negative results — 메소드 아님)

정직성을 위해 보존한다. 이들은 ablation 표에 **음성**으로 남는다.

- **Column-owned U-solve** (`CLS_TF32_COLUMN_USOLVE`): big-kernel 의 61% 병목(panel-LU+U-solve
  barrier, `29`)을 겨눠 U-panel 을 column 단위로 병렬화. **재현 결과 효과 ≈0%** — fp32 에 동일 적용해도
  −0.4~+0.9%(노이즈). *코드 비대칭 ≠ 성능 비대칭.* 코드엔 토글로 남아있으나 메소드로 세지 않는다.
- **그 외 폐기**: cuBLAS grouped TF32 mid, fused trail+extend(tf32), sibling/plain/chain/validated
  amalgamation, FP16 mid TC force-all, batch-dim packing 등 — 모두 회귀/무효(`33/39/41/42/43/47/48`).
  master 코드에서 제거됨.

---

## 4. Ablation study 설계

**baseline = base multifrontal** (`--precision fp64 --batch 1`, 모든 토글 OFF). 각 메소드를 켜며
`factor_ms/sys` + `relres` delta 측정 (`--batch-only --repeat 61 --warmup 8`).

**Leave-one-out (optimal 에서 메소드별 −1):**

| 끄는 메소드 | 토글 | 측정 의도 |
|---|---|---|
| M1 배치 | `--batch 1` | latency 분할상환 기여 |
| M3 동형 디스패치 | 정책 off | launch 균질화 기여 |
| M4 멀티스트림 | `--no-multistream` | 동시성 기여 (8387 +22%) |
| M5 TF32 TC trailing | `--precision fp32` | **TC 순이득** (대형 ~−17~21% factor) |
| M6 패널 융합 (저-fill) | `CLS_TC_CLOSURE_*` off | 8387/13K TC 진입 여부 |
| M7 Ozaki | `OZAKI_TC2_FIRST_ORDER` off | **relres 5e-2 ↔ 1e-4** (정확도 회수분), 속도 +~5% |
| (음성) column-U-solve | `CLS_TF32_COLUMN_USOLVE` off | **≈0% 확인** |

**정밀도×정확도 격자(핵심 결과표)**: `--precision {fp64, fp32, tf32, tf32+Ozaki}` × `--batch {1,64,256}`
× case 로 **속도와 relres 동시 보고** → 메시지: *tf32+Ozaki 가 fp32 보다 빠르면서 relres 동급.*

> 측정 규율(본 세션 학습): USA 는 fp32 자체가 conditioning floor(~1e-3, `27`)라 baseline 대조 없이
> 정확도 판정 금지; ordering(seed) 변동을 분리해 보고.

---

## 5. 열린 질문 (다음 실험)

- **M5 의 순이득 출처 미분리**: 대형 TC 이득이 **텐서코어 mma 자체**인지 **새 tf32 커널 launch-shape
  (512/128-thread, direct-shared)** 인지 아직 분리 안 됨. *Experiment #3*: tf32 커널에서 TC-trailing 만
  scalar 로 치환(shape 유지)해 mma 순기여 격리.
- **M6 의 일반성**: 저-fill 융합이 8387 B256 에선 Ozaki 와 결합 시 1.2× 미달 — 구조적 일반화 또는
  외부 정제로 보완 필요.
- **정확도 한계의 본질**: USA 부정확은 TF32 가 아니라 데이터셋 conditioning — GEMM 내부(Ozaki)가 아니라
  외부 IR/GMRES(`53` 레버 B)가 정답.

---

## 6. 한 장 요약

| 축 | 메소드 | 공격 병목 | 토글 | 효과 |
|---|---|---|---|---|
| Ⅰ | M1 배치 front-major 인수분해 | launch/occupancy | `--batch` | throughput saturation |
| Ⅰ | M2 3단 커널 라우팅 | 자원 부정합 | `tier_split` | tier 정합 |
| Ⅰ | M3 동형 디스패치 | launch 균질화 | 정책 | 70K/USA −11~15% |
| Ⅰ | M4 서브트리 멀티스트림 | level 동시성 | `--no-multistream` | 8387 B64 −22% |
| Ⅱ | M5 TF32 텐서코어 trailing | thin-K scalar trailing | `MID_TF32_TC`+`BIG_TF32_BLOCKED_TC` | 대형 1.2–1.27× |
| Ⅱ | M6 TC 적합 패널 융합 | tiny-K underfill | `TC_CLOSURE_*`+`SMALL_FRONT_MAX_16` | 8387/13K TC 진입 |
| Ⅲ | M7 Ozaki 오차보정 TF32 | TF32 round-off | `OZAKI_TC2_FIRST_ORDER` | relres 5e-2→1e-4, 속도 불변 |
| — | (음성) column-U-solve | panel barrier | `TF32_COLUMN_USOLVE` | ≈0% |

**메시지**: base multifrontal 의 진짜 병목(launch/occupancy/memory-latency)을 디스패치·스케줄링(Ⅰ)으로
분할상환하고, **작은 front 를 융합해 TF32 텐서코어를 먹게(Ⅱ) 만든 뒤, Ozaki 로 정확도를 공짜로
회수(Ⅲ)** 해, 대형 power-grid Jacobian 배치 인수분해에서 **FP32 대비 1.2–1.27× + FP32 동급 정확도**를
달성한다.
