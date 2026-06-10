# Research Storyline — Power-Grid Jacobian 을 위한 배치 GPU Multifrontal + Tensor-Core 가속

**작성일**: 2026-06-10
**목적**: master 기본값/Codex 실험을 구분하지 않고, **현재 최적 경로를 단일 기본값**으로 세우고,
base multifrontal 위에 **어떤 아이디어로 최적화했는지**를 한 줄로 잇는 연구 서사. 각 기여는
**ablation 가능**하도록 토글(플래그/런타임 knob)에 매핑한다.

본 문서는 narrative + ablation 설계서다. 세부 측정·실패 로그는 `03-optimization-notes/` 를,
설계 분해는 `02-design-analysis/` 를 참조한다.

---

## 0. 한 줄 thesis

> 전력망 Newton–Raphson Jacobian 은 **(i) 고정 sparsity 로 반복 인수분해**되고 **(ii) pivoting
> 없이 안정**하며 **(iii) front 가 작다**. 이 세 성질을 이용해, 표준 GPU multifrontal 을 ① 정적
> 구조·그래프·배치로 **launch/sync 를 분할상환**하고 ② front 크기별 kernel 로 **자원을 맞추고**
> ③ trailing GEMM 을 **TF32 텐서코어**로 돌리되 **Ozaki 오차보정**으로 FP32 정확도를 회복한다.
> 결과: 대형 케이스 배치 인수분해에서 **FP32 대비 1.2–1.27× 가속 + FP32 동급 정확도**.

---

## 1. 문제와 base — 왜 표준 multifrontal 이 여기서 GPU-비친화적인가

**워크로드**: cuPF 의 NR 루프는 동일 sparsity 의 Jacobian 을 매 iteration·매 배치마다 다시
인수분해한다(`set_data → analyze(1회) → factorize(반복) → solve`). B 개 시스템이 같은 패턴을
공유한다.

**base multifrontal** (참조점, 모든 ablation 의 바닥): METIS nested-dissection ordering →
supernode/multifrontal assembly → front 별 dense LU → extend-add 로 부모에 기여 누적.

**왜 base 가 GPU 에서 느린가** (power-grid 특이성):

| 성질 | 결과 | 공격 대상 |
|---|---|---|
| front 가 작다 (대부분 `fsz≤16`, `nc=1..20`) | dense kernel 이 GPU 를 못 채움(occupancy↓), trailing GEMM 이 **thin-K(K=nc) memory-bound** | tiering, batching, TC |
| etree 가 깊고 좁다 (spine, 2-way subtree) | level 마다 launch+barrier, 동시성 부족 | graph, multistream |
| 반복 인수분해 | per-call launch overhead 가 누적 | CUDA graph, batching |
| pivoting 불필요 (대각우세) | 정적 구조 가능 | no-pivot 전반의 전제 |

→ 즉 base 의 병목은 FLOP 이 아니라 **launch/sync latency + occupancy + memory latency**. 최적화는
전부 이 셋을 겨냥한다.

---

## 2. 최적 경로 (현재 기본값으로 권장)

regime 에 따라 두 갈래지만, 공통 토대 위에 얹는다.

**공통 토대** (항상 ON): no-pivot · CUDA graph capture · METIS-ND · 3-tier routing ·
multistream subtree · fsz-band-split · uniform-batch front-major.

**정밀도/정확도**: `--precision tf32` (front=FP32, trailing=TF32 텐서코어) **+ Ozaki first-order
오차보정** → relres 가 FP32 band(~1e-4..1e-5).

**TC dispatch 정책**:
- mid: 텐서코어 trailing(신규) · big: blocked-TC trailing · column-owned U-solve · respect-cap.
- **low-fill regime** (n≲16k, 8387/13K): + TC-closure panel amalgamation + `small16` + `mid128`.
- **large regime** (n≳25k, 25K/70K/USA): 위 TC 정책 그대로(amalgamation/small16 불필요).

**배치**: B=64–256 (throughput saturation).

권장 빌드(대형, 정확도 보정 포함):
```bash
cmake -S custom_linear_solver -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  -DCLS_MID_TF32_TC=ON -DCLS_MID_TF32_DIRECT_SHARED=ON -DCLS_MID_TF32_LOW_TC=ON \
  -DCLS_BIG_LOW_SPLIT=ON -DCLS_MID_LOW_SPLIT=ON -DCLS_BIG_TF32_BLOCKED_TC=ON \
  -DCLS_BIG_TF32_SHARED_THREADS_512=ON -DCLS_TF32_COLUMN_USOLVE=ON -DCLS_RESPECT_PANEL_CAP=ON \
  -DCLS_MID_TF32_MIN_FSZ=48 -DCLS_TF32_OZAKI_TC2_FIRST_ORDER=ON
build/custom_linear_solver_run <case> --precision tf32 --batch 64 --repeat 61 --single-precision fp64
```
low-fill 은 여기에 `-DCLS_TC_CLOSURE_PANEL_AMALGAMATE=ON -DCLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP=32
-DCLS_SMALL_FRONT_MAX_16=ON -DCLS_MID_TF32_TC_THREADS_128=ON -DCLS_MID_TF32_LOW_TC_FORCE_ALL=ON` 추가.

---

## 3. 기여 스택 — base 위에 쌓은 아이디어 (각각 ablation 토글)

base multifrontal 에서 출발해 병목별로 한 겹씩 얹는다. 각 항목은 **아이디어 → 메커니즘 →
공격한 병목 → ablation 토글 → 실측 효과** 로 적는다.

### 토대 레이어 (launch/sync/occupancy 분할상환)

**A1. No-pivot 정적 구조**
- 아이디어: power-grid Jacobian 은 대각우세 → 부분 pivoting 없이 정확(`02-design-analysis/03`).
- 메커니즘: 구조를 analyze 에서 1회 확정, factorize 는 numeric 만. 런타임 분기 제거.
- 병목: pivot search 의 데이터 의존 제어흐름.
- ablation: `-DCLS_USE_PIVOTING=ON` (켜면 비교).
- 효과: 전체 정적 파이프라인(그래프/배치)의 **전제**. 실패 경계는 `03`.

**A2. CUDA graph capture**
- 아이디어: 반복 인수분해의 per-level kernel launch 를 1회 캡처 후 replay.
- 병목: per-call launch latency (B=1 deep etree 지배).
- ablation: `-DCLS_INTERNAL_GRAPH=OFF` (external capture).
- 효과: B=1 spine-bound 케이스의 launch overhead 제거(`02-design-analysis/01`).

**A3. Uniform-batch front-major + level batching**
- 아이디어: B 시스템이 같은 plan 공유 → front arena `B×front_total`, blockIdx.y=batch 로 한 launch 가 B fronts 처리.
- 병목: occupancy-starved level + per-level sync 를 B 로 분할상환.
- ablation: `--batch 1` vs `--batch 64/256`.
- 효과: throughput saturation (small B≈256, mid B≈64, big B≈16; `16-large-batch-bottleneck`).

**A4. 3-tier kernel routing (small/mid/big)**
- 아이디어: front 크기로 kernel 선택 — small=warp-packed, mid=shared-resident, big=global+1024-thread.
- 병목: 한 kernel 로 모든 front 를 돌릴 때의 자원 부정합(`02-design-analysis/06` 임계값 근거).
- ablation: `SolverConfig.tier_split` (false=whole-level dispatch).
- 효과: tier 별 occupancy/shared 예산 정합.

**A5. Multistream subtree dispatch (동시성)**
- 아이디어: 독립 subtree 를 별도 CUDA stream 으로 → Hyper-Q 가 SM idle cycle 을 다른 stream 으로 채움.
- 병목: 좁은 etree 의 level 내 occupancy 부족.
- ablation: `--no-multistream`.
- 효과: case8387 B=64 **−22%**, USA B=64 **−14%** (`04-benchmarks-profiling/13`).

**A6. fsz-band-split dispatch**
- 아이디어: 한 (subtree, level) 의 panel 을 fsz-band 로 stable-sort 후 같은 band 끼리 sub-launch — launch 일감 균질화 + tier 경계 가로지르는 tiny front 재라우팅.
- 병목: mixed-level launch 의 shared `fsz_cap²` 부정합.
- ablation: `front_bucket()`/band-order 정책(현재 baked, gate `n≥40k→B>1` 등; `03-optimization-notes/23`).
- 효과: 70K/USA B=64/256 wall **−11~15%**.

**A7. 정밀도 강하 (FP32 front)**
- 아이디어: front 를 FP64→FP32 로 (front storage 절반, 연산 2×).
- 병목: FP64 memory 대역/연산.
- ablation: `--precision fp64|fp32`.
- 효과: ~2× vs FP64 (정확도 ~1e-4).

### 텐서코어 레이어 (trailing GEMM 을 TC 로) — 본 연구의 핵심 기여

여기서부터가 "과제 목표(배치 TC 가속)"를 직접 겨눈 기여다. **먼저 진단**(아래 §5)이 말한다:
trailing GEMM 은 전체의 ~19.5% 이고 **thin-K(M=N=uc≈140, K=nc≈20) memory-bound** 라 텐서코어가
naive 하게는 안 먹힌다(`28-tensor-core-ceiling`). 그래서 기여는 *TC 를 그냥 켜는 것*이 아니라
**TC 가 먹히도록 dispatch/구조를 바꾸는 것**이다.

**C1. mid-tier TF32 텐서코어 trailing (신규 슬롯)**
- 아이디어: base 는 mid 를 scalar 로 비워뒀다(과거 mid-TC 가 더 느림). direct-shared staging 으로 mid front 의 trailing 을 TF32 mma 로.
- 병목: mid trailing 의 scalar 연산.
- ablation: `-DCLS_MID_TF32_TC` (off=scalar mid). `MID_TF32_DIRECT_SHARED`, `MID_TF32_LOW_TC`, `MID_TF32_MIN_FSZ` 로 적용 범위 조절.
- 효과: 대형 mid-dominant(25K)에서 TC dispatch 정책의 일부로 기여.

**C2. big-tier blocked-TF32 텐서코어 trailing (교체)**
- 아이디어: shared-resident blocked 형태로 big trailing 의 L/U 재사용 + TC mma.
- 병목: big trailing 의 uncoalesced C-drain store(2.5×, `30-concurrency-c-drain`) + thin-K.
- ablation: `-DCLS_BIG_TF32_BLOCKED_TC` (off=기존 PTX big-TC), `BIG_TF32_SHARED_THREADS_512`.
- 효과: 대형 big-dominant(USA/70K) 가속의 핵심 경로.

**C3. low-split dispatch bucketing**
- 아이디어: mid/big tier 를 fsz 로 더 잘게 쪼개 TC 적격 front 를 tighter shared 예산으로.
- 병목: heterogeneous mid level 의 shared `fsz_cap²` 점유.
- ablation: `-DCLS_MID_LOW_SPLIT`, `-DCLS_BIG_LOW_SPLIT` (`front_bucket()` 재정의 — A6 band-split 과 공유).
- 효과: TC dispatch 정책의 enabler.

**C4. column-owned U-solve** *(정직한 negative)*
- 아이디어: note 29 의 61% 병목(panel-LU+U-solve barrier)을 풀고자 U-panel 을 column 단위 병렬화.
- ablation: `-DCLS_TF32_COLUMN_USOLVE`.
- **실측(본 세션 재현): 효과 ≈ 0** (fp32 에 동일 적용 시 −0.4~+0.9%, 노이즈). 정책에 포함돼 있으나 **속도 기여 없음** — ablation 표에 음성 결과로 보존.

### 저-fill 구조 레이어 (8387/13K 를 TC-routable 로)

tiny front 지배 케이스(8387)는 TC tile 효율이 1.5~13% 라 per-front TC 가 구조적으로 안 됨
(`36/37/48`). 그래서 **symbolic 구조를 TC 에 맞게 바꾼다.**

**C5. TC-closure panel amalgamation**
- 아이디어: parent-front containment + multifrontal invariant 를 통과하면서 `fsz>32, uc≥16, 4≤nc≤32` 로 **TC-routable 해지는** consecutive panel 만 병합. fill 을 *TC 가 회수할 때만* 지불.
- 병목: tiny front 의 K=nc 가 너무 작아 mma tile 못 채움.
- ablation: `-DCLS_TC_CLOSURE_PANEL_AMALGAMATE(_CAP=32)`.
- 효과: 8387/13K 가 처음으로 TC 목표권 진입.

**C6. small16 재-threshold + mid128 launch shape**
- 아이디어: `kSmallFrontMax 32→16` 으로 17..32 front 를 mid TC dispatch 에 노출 + mid TC 를 128-thread 로(49..64 dominant 일감에 맞춤).
- ablation: `-DCLS_SMALL_FRONT_MAX_16`, `-DCLS_MID_TF32_TC_THREADS_128`, `-DCLS_MID_TF32_LOW_TC_FORCE_ALL`.
- 효과: 8387 B256 을 1.2× 위로 미는 마지막 한 끗(`50-tc-closure`).

### 정확도 레이어 (TC 정확도 회복) — 직교 기여

**C7. Ozaki TC2 / first-order 오차보정**
- 아이디어: FP32 입력을 `x0=cvt.rna.tf32(x)`, `x1=cvt.rna.tf32(x−x0)` 로 분할, `L0U0+L1U0+L0U1(+L1U1)` 를 **텐서코어 FP32 누산기에서 체인 누적**. 전 tier(mid/big/blocked/small-warp) 적용.
- 병목: TF32 10-bit 가수 round-off (raw tf32 relres ~5e-2).
- ablation: `-DCLS_TF32_OZAKI_TC2`(full 4항) / `-DCLS_TF32_OZAKI_TC2_FIRST_ORDER`(3항).
- **실측(본 세션 재현): 8387 tf32 relres 3.97e-2 → 4.77e-5 (≈FP32 3.3e-5), 13K 6.47e-3 → 1.3e-4.**
  속도 비용은 **memory-bound trailing 의 놀던 mma 헤드룸**을 쓰는 것이라 작음(8387 B64 1.24×→1.19×).
- 직교성: trailing mma 내부에만 작용 → 어떤 dispatch/tiering/ordering 위에도 **스택**.

---

## 4. Ablation study 설계

**baseline = base multifrontal** (모든 토글 OFF, `--precision fp64`). 여기서 한 레이어씩 켜며
`factor_ms/sys` 와 `relres` 의 delta 를 측정한다(`--batch-only --repeat 61 --warmup 8`).

### 4.1 누적 ablation (base → optimal, 각 +1 레이어)

| 단계 | +레이어 | 토글 | 측정 축 |
|---|---|---|---|
| 0 | base multifrontal (fp64, single, no-graph) | `--precision fp64 --batch 1 -DCLS_INTERNAL_GRAPH=OFF` | factor_ms |
| 1 | +graph | `INTERNAL_GRAPH=ON` | launch overhead↓ |
| 2 | +batch | `--batch 64` | throughput |
| 3 | +multistream | (default; `--no-multistream` 로 off 비교) | 동시성 |
| 4 | +band-split | (band-order 정책) | launch 균질화 |
| 5 | +fp32 | `--precision fp32` | 2× |
| 6 | +TC dispatch | `--precision tf32` + `MID_TF32_TC,BIG_TF32_BLOCKED_TC,*_LOW_SPLIT` | TC 가속 |
| 7 | +Ozaki | `TF32_OZAKI_TC2_FIRST_ORDER` | 정확도 회복 |
| (low-fill) | +amalgamation/small16/mid128 | `TC_CLOSURE_*,SMALL_FRONT_MAX_16,MID_TF32_TC_THREADS_128` | 8387/13K TC |

### 4.2 leave-one-out ablation (optimal 에서 −1 레이어)

각 기여의 **순기여**를 보려면 최적 빌드에서 해당 토글만 끄고 재측정:

| 끄는 토글 | 기대 delta (가설/실측) |
|---|---|
| `MID_TF32_TC`+`BIG_TF32_BLOCKED_TC` (→fp32) | TC 순이득 (대형 ~−17~21% factor; **단 mma 자체 vs launch-shape 미분리 — open**) |
| `TF32_OZAKI_TC2_FIRST_ORDER` | 속도 +~5%, **relres 5e-2 ↔ 1e-4 (정확도 회복분)** |
| `TC_CLOSURE_PANEL_AMALGAMATE`(low-fill) | 8387/13K TC 진입 여부 |
| `TF32_COLUMN_USOLVE` | **≈0% (실측 무효 — negative result)** |
| `--no-multistream` | case8387 B64 +22% (동시성 기여) |

### 4.3 정밀도×정확도 격자 (핵심 표)

`--precision {fp64, fp32, tf32, tf32+ozaki}` × `--batch {1,64,256}` × case 로 **속도와 relres 를
동시에** 보고. 메시지: **tf32+Ozaki 가 fp32 대비 빠르면서 relres 동급.**

> 측정 지침(본 세션 학습): flaky 케이스(USA)는 fp32 자체가 conditioning floor(~1e-3, `docs/27`)라
> baseline 대조 없이 정확도 판정 금지. ordering(seed) 변동도 분리해 보고.

---

## 5. 핵심 통찰 / 정직한 caveat (연구의 뼈대)

1. **병목은 FLOP 이 아니다.** big kernel: panel-LU+U-solve **61% (barrier-bound)**, trailing 19.5%,
   extend 19.2% (`29`). 그래서 가속의 대부분은 GEMM 이 아니라 **launch/sync/occupancy**(토대 레이어)에서 나온다.
2. **텐서코어 ceiling 은 thin-K memory-bound.** trailing 은 `M=N=uc≈140, K=nc≈20` → mma 기준
   K=2.5 tile 뿐 → staging 이 mma 를 압도. amalgamation 으로 front 를 키워도 uc 만 커지고 K=nc 는
   안 커진다(`28`). TC 가 *그나마* 먹히는 건 dispatch/구조를 맞춰줬을 때.
3. **그 memory-bound 성질이 Ozaki 를 거의 공짜로 만든다.** mma 유닛이 놀고 있으니 보정 mma 3–4회를
   더 돌려도 wall 거의 불변 → **정확도를 공짜로 회수**(가장 깔끔한 기여).
4. **column-U-solve 는 무효였다(정직한 음성).** 61% 병목을 겨눴지만 재현 결과 ≈0%. 코드 비대칭이
   성능 비대칭을 뜻하지 않음 — ablation 에 음성으로 남긴다.
5. **저-fill 정책은 case-specific·seed 민감.** 8387/13K 는 amalgamation+small16 로 진입하나 마진이
   얇고(8387 B256 은 Ozaki 켜면 1.2× 미달), 70K 는 seed/cap 민감(`44/50`). large 정책과 분리 필요.
6. **USA 의 부정확은 TF32 가 아니라 데이터셋 conditioning floor**(`27`) — 정확도는 Ozaki(GEMM 내부)가
   아니라 외부 IR/GMRES(`53` 레버 B)로 푸는 게 정답.
7. **열린 질문(다음 실험):** 대형 TC 순이득이 **텐서코어 mma 자체**인지 **tf32 커널 launch-shape
   (512/128-thread, direct-shared)**인지 미분리(experiment #3: tf32 커널에서 TC-trailing 만 scalar 로
   치환해 격리).

---

## 6. 한 장 요약 (idea → 병목 → 토글 → 효과)

| 레이어 | 아이디어 | 공격 병목 | ablation 토글 | 효과(실측/근거) |
|---|---|---|---|---|
| A1 | no-pivot 정적 구조 | pivot 제어흐름 | `CLS_USE_PIVOTING` | 정적 파이프라인 전제 |
| A2 | CUDA graph | launch latency | `CLS_INTERNAL_GRAPH` | B=1 overhead↓ |
| A3 | uniform-batch front-major | occupancy/sync | `--batch` | throughput saturation |
| A4 | 3-tier routing | 자원 부정합 | `tier_split` | tier 정합 |
| A5 | multistream subtree | level 동시성 | `--no-multistream` | 8387 B64 −22% |
| A6 | fsz-band-split | launch 균질화 | band-order | 70K/USA −11~15% |
| A7 | FP32 front | FP64 대역 | `--precision fp32` | ~2× |
| C1 | mid TF32 TC (add) | mid scalar trailing | `CLS_MID_TF32_TC` | TC dispatch enabler |
| C2 | big blocked-TF32 TC (교체) | C-drain/thin-K | `CLS_BIG_TF32_BLOCKED_TC` | 대형 가속 핵심 |
| C3 | low-split bucketing | shared 점유 | `CLS_*_LOW_SPLIT` | TC enabler |
| C4 | column-U-solve | panel barrier 61% | `CLS_TF32_COLUMN_USOLVE` | **≈0% (음성)** |
| C5 | TC-closure amalgamation | tiny-K front | `CLS_TC_CLOSURE_PANEL_AMALGAMATE` | 8387/13K TC 진입 |
| C6 | small16 + mid128 | tier 경계/launch shape | `CLS_SMALL_FRONT_MAX_16`,`CLS_MID_TF32_TC_THREADS_128` | 8387 B256 >1.2× |
| C7 | **Ozaki 오차보정** | TF32 round-off | `CLS_TF32_OZAKI_TC2[_FIRST_ORDER]` | **relres 5e-2→1e-4, 속도 거의 불변** |
| — | (외부) IR/GMRES 정제 | conditioning floor | `--ir`/`--gmres` (`53`) | USA 정확도 정답 |

**최종 메시지**: 표준 GPU multifrontal 의 진짜 병목(launch/sync/occupancy/memory-latency)을
토대 레이어로 분할상환한 뒤, **TF32 텐서코어를 dispatch·구조로 먹게 만들고 Ozaki 로 정확도를
공짜로 회수**해, 대형 power-grid Jacobian 배치 인수분해에서 **FP32 대비 1.2–1.27× + FP32 동급
정확도**를 달성한다 — 각 아이디어는 위 토글로 독립 ablation 가능하다.
