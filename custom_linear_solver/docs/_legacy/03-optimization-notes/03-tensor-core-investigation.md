# Tensor-Core Trailing 조사 — large-case 성공·low-fill dead-end·정직한 ~1.1× 결론

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: TF32/FP16 텐서코어로 trailing GEMM 을 가속하려는 26개 일별 로그(notes 30–55)의 통합 — 무엇이 통했고, 헤드라인 1.2× 가 왜 ~1.1× 로 정정됐는가.

이 문서는 하나의 조사("power-grid Jacobian 의 trailing GEMM 을 텐서코어로 가속할 수 있는가")를
다룬 notes 30–55 를 통합한다. 중간의 낙관적 결과를 모두 기록하되, **notes 54·55(2026-06-10)의
자기정정**으로 끝난다: 헤드라인 "1.2×/1.24× TC 가속"은 대부분 **cap-inflation 아티팩트**였고,
best-vs-best 공정 통제하에선 large case 가 ~**1.1×**, TC mma 순기여는 **+6~9%** 에 불과하다.

---

## 1. 목표와 진단 전제 — thin-K trailing 은 왜 naive TC 를 안 먹는가

**목표**: B64/B256 batched factorize 에서 `case_ACTIVSg25k`(25K) / `case_ACTIVSg70k`(70K) /
`case_SyntheticUSA`(USA) / `case8387pegase`(8387) / `case13659pegase`(13K) 중 하나 이상이 FP32
대비 `1.2..1.4×` 빨라지고, **텐서코어가 그 enabler** 임을 보인다.

multifrontal 의 trailing update 는 front 마다

```text
C[uc x uc] -= L[uc x nc] * U[nc x uc]
```

여기서 K 차원 = `nc`. power-grid Jacobian 은 low-fill 이라 대다수 front 의 `nc`(=K)가 1~2 다.
TF32 `mma.m16n8k8` 는 K=8 타일을 amortize 하려 하므로, `nc=1/2` 는 단순히 "작다"가 아니라
**TC 의 K granularity 미달**이다. 이 thin-K memory-bound 천장은 본 조사의 전제이고
[../02-design-analysis/04-gemm-fraction-tc-ceiling.md](../02-design-analysis/04-gemm-fraction-tc-ceiling.md)
및 [01-kernel-engineering.md](01-kernel-engineering.md),
[01-kernel-engineering.md](01-kernel-engineering.md) 에서 확립됐다.

따라서 조사는 두 갈래로 갈린다:

- **dense-update 가 충분한 large case(25K/70K/USA)**: mid/big front 가 TC-routable shape 를 가지므로
  TF32 TC trailing 이 가시적 이득을 줄 여지가 있다.
- **small-front 지배 low-fill case(8387/13K)**: front 의 97% 가 `fsz<=16`, K=1/2 → 어떤 local
  per-front TC 레버도 구조적으로 막힌다.

---

## 2. Large-case (25K/70K/USA) 결과

### 2.1 big multi-block 제거가 TC 를 가시화 (note 30)

기본 구성에선 big underfill front 가 multi-block(MB) scalar trailing 경로로 빠져서 TF32 가
underfill 에서 **TC 를 아예 안 썼다**(scalar MB). `CLS_NO_BIG_MB` 로 모든 big front 을 fused 단일
커널(`factor_big_staged`/`factor_big_tf32_ptx`/`factor_big_fp16_ptx`)로 강제하면 비로소 TC trailing
이 돌고, 같은 no-MB 구성 안에서 TF32/FP16 이 FP32 를 −1~−12% 이긴다(USA B=1 −11.3%).

**단, no-MB 는 절대적으로 느린 구성**이다. 진짜 default 인 fp32(MB)와 비교하면 B=1 에선 fp32(MB)가
최速이라 TC 가 진다. note 30 의 교훈: no-MB 는 TC 를 *보이게* 만들 뿐, thin-K 천장(note 28)을 깨서
*근본적으로 빠르게* 만들진 않는다.

### 2.2 large-case 정책과 raw repeat=61 수치

수십 번의 토글 sweep(note 33) 끝에 정착한 **stable large-case 정책**:

```text
CLS_BIG_LOW_SPLIT=ON              # big bucket 을 shared-resident low / global high 로 분할
CLS_MID_LOW_SPLIT=ON
CLS_BIG_TF32_BLOCKED_TC=ON        # shared-resident big-low blocked TF32
CLS_BIG_TF32_SHARED_THREADS_512=ON
CLS_MID_TF32_TC=ON
CLS_MID_TF32_DIRECT_SHARED=ON     # mid front 가 Fs 에서 L/U 직접 읽음
CLS_MID_TF32_LOW_TC=ON
CLS_TF32_COLUMN_USOLVE=ON
CLS_RESPECT_PANEL_CAP=ON          # n-기반 cap override 해제 → cap 이 fp32/tf32 양쪽에 보임
CLS_FUSE_TF32_TRAIL_EXTEND=OFF    # TF32 fused extend 는 불안정
CLS_MID_TF32_DIRECT_FUSE_EXTEND=OFF
CLS_MID_TF32_MIN_FSZ=48
```

핵심 형태 레버: mid direct-shared TF32 TC, shared-resident big-low blocked TF32(512-thread),
global big-high TF32 fallback(512-thread). 이 정책의 **accepted repeat=61** 수치(`fp32/tf32`,
`batch_factor_per_sys_ms`, warmup=8):

| case | B | cap | FP32 ms/sys | TF32 ms/sys | speedup | TF32 relres |
|---|---:|---:|---:|---:|---:|---:|
| 25K | 64 | 31 | 0.111079 | 0.090940 | **1.222×** | 5.42e-02 |
| 25K | 256 | 32 | 0.110488 | 0.086665 | **1.275×** | 5.54e-02 |
| USA | 64 | 31 | 0.470502 | 0.378995 | **1.241×** | 4.75e-02 |
| USA | 256 | 32 | 0.442969 | 0.359998 | **1.231×** | 5.26e-02 |
| 70K | 64 | 29 | 0.408891 | 0.325282 | **1.257×** | 5.04e-02 |
| 70K | 256 | 29 | 0.407432 | 0.324091 | **1.257×** | 5.71e-02 |

(70K 는 seed99/cap29; seed42/cap32·seed7/cap29 는 repeat=31 만 통과하고 repeat=61 미달 — seed/cap
민감.) TF32 relres 는 모두 `~5e-2` large-case band 로, 정확도 중립 교체가 아니라 speed-enabler 결과다.

이 수치들이 곧 헤드라인 "1.2× TC 가속"의 출처였다. **§7 의 정정이 이 수치의 해석을 뒤집는다 —
숫자 자체는 raw 측정이 맞지만, fp32 baseline 이 같은 큰 cap 에 묶여 있어 비율이 부풀려졌다.**

note 33 의 부수 발견: BK=16 / NTJ8_16 등 K·N 타일 확대는 register/accumulator 압력으로 USA 를
크게 회귀시켰고, cuBLAS grouped TF32 trailing 은 USA correctness 실패 또는 +2% 미미.
big front 의 global right-looking blocked TC(`CLS_BIG_TF32_BLOCKED_TC` 를 big-high 까지)는
반복 global L/U traffic 으로 USA B64 0.75× 로 망함 → shared/full-front residency 필수.

---

## 3. 8387 / 13K dead-end 매트릭스

low-fill case 에 대한 모든 local·구조 레버는 paired B64/B256 `1.2×` 를 안정적으로 만들지 못했다.
8387 front 분포: cap28 에서 `fsz<=16` 가 7122/7334, 지배 shape 가
`(fsz,nc,uc)=(6,2,4),(4,2,2),(5,2,3),(3,1,2)` — 즉 K=`nc`=1/2.

| 레버 (note) | 메커니즘 | 결과 (repeat=31, paired) | 왜 실패 |
|---|---|---|---|
| upper-bound (36) | mid-tier·extend-add 제거 timing 천장 | small제거 B256 ~1.12×; small+extend제거 B256 ~1.16× | 두 wall 제거해도 B256 1.2× 미달; tile eff ~13% |
| many-front packed TC (37) | 독립 small front 들을 1 MMA 타일에 block-diagonal pack | `small<=16` packed eff **6.5%**, useful share 21~22% | `(6,2,4)`·`(4,2,2)` 가 N차원 미충전; 4×~8× peak × 6.5% = 0.26~0.52× FP32 |
| ordering seed sweep (38) | METIS seed×cap sweep | best paired seed7/cap28 **1.078/1.118×** | seed 변화는 residual·noise 만 바꿈, TC-covered work 불변 |
| sibling amalgamation (39) | 같은 parent sibling 패널 병합(cap8/16) | serial cap32 **1.037/1.015×**; parallel B256-only 1.312 | 공통 FP32 work·fill·schedule variance 증가, TC work 증폭 안 됨 |
| low-mid force-all (41) | `CLS_MID_TF32_LOW_TC_FORCE_ALL` 로 size-gate 해제 | best seed42/cap32 **1.131/0.985×** | 새로 열린 front 가 수십 개뿐, `fsz<=16` 지배 불변 |
| extend-add uc-bucket (42) | `uc<=16` / `uc>16` extend 선택 skip(timing) | skip`uc<=16` best **~1.10×** | parent assembly 제거해도 hidden TC win 없음 |
| panel-chain amalgamation (46) | parent-child 체인 패널 병합(cap32) | best paired 13K seed99/cap31 **1.042×**, 8387 0.941× | 패널 수 −1~2% 만, 양 case 97% small 잔존 |
| sibling+chain combo (47) | sibling 후 chain, 가장 강한 conservative 병합 | 8387 0.982~1.009×, 13K 0.976~1.028× | cap32 후도 >96% small; cap40/48 은 FP32 relres 발산(invalid) |
| batch-dim packed TC (48) | B 차원으로 TC 타일 채우기 | 구현 안 함(불가 증명) | batch-as-GEMM 은 cross-batch `L_b·U_{b'}` invalid → block-diagonal 로 환원, eff 동일 6.5% |
| nonlocal panelization / parND threshold (49) | 임의 consecutive-run 병합 + parND base-threshold | validated-run 병합 cap2 도 **invalid fallback**; parND 13K best 1.112/1.188× B64 미달 | 임의 run 은 assembly containment 위반; threshold 변화는 분포 못 옮김 |
| 13K stable·force-all 외 variants (43) | mid direct-fuse / blocked mid / small16 / cuBLAS mid / fp16 force-all / no-multistream | 전부 paired target 미달 또는 회귀 | mid direct-fuse 0.965/0.956×; cuBLAS factorize 실패; fp16 O(1) relres |

**누적 결론**: 8387/13K 는 KLU/circuit 같은 low-fill региме 에 가깝고, per-front TC 는 구조적으로
부적합하다. cap40/48 로 fronts 를 키우려는 시도는 FP32 residual 을 먼저 깨뜨린다.

### 3.1 note 50 의 two-policy raw pass (기록)

예외적으로, note 50 의 **low-fill structural 정책**은 repeat=61 raw pass 를 만들었다:

```text
CLS_TC_CLOSURE_PANEL_AMALGAMATE=ON   # TC-routable(fsz>32,uc>=16,4<=nc<=32) group 만 병합
CLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP=32
CLS_MID_TF32_TC_THREADS_128=ON       # 49..64-지배 mid-TC 에 128-thread
CLS_SMALL_FRONT_MAX_16=ON            # 17..32 front 를 small 에서 빼서 low-mid 가 보게
+ stable large-case 정책, deterministic --serial-nd
```

| case | seed | cap | B | fp32 ms/sys | tf32 ms/sys | fp32/tf32 | tf32 relres |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8387 | 7 | 30 | 64 | 0.0331602 | 0.0266655 | **1.244** | 3.97e-02 |
| 8387 | 7 | 30 | 256 | 0.0289548 | 0.0239879 | **1.207** | 3.97e-02 |
| 13K | 99 | 32 | 64 | 0.0679609 | 0.0541622 | **1.255** | 6.47e-03 |
| 13K | 99 | 32 | 256 | 0.0572868 | 0.0461665 | **1.241** | 6.47e-03 |

구조 shift: 8387 패널 7322→6354, `49..64` f2 = 33.8%; 13K 12339→10391, `49..64` f2 = 42.7%.
이것이 four-flag bundle 의 출처이고, **§7.3 에서 cap-inflated·fragile 로 정정된다.** small16
없이는 8387 serial 이 0.3% 차로 fail 했고, 이 build 는 25K 를 불안정하게(B64 1.192×) 만들어
**universal 정책이 아니다.** note 44 는 이를 25K/70K/USA(stable) + 8387/13K(low-fill)의 two-policy
pass/fail 매트릭스로 정리했다.

---

## 4. 정확도 회복 (Ozaki) — notes 51/52/53

large-case TF32 relres `~5e-2`, 8387 raw `3.97e-2` 는 두 원인이 섞인다(note 53 §0):
**(i) TF32 10-bit 가수 round-off**(보정 가능), **(ii) USA 데이터셋 conditioning floor**(fp32 로도
cuDSS ~1e-3, [01-kernel-engineering.md](01-kernel-engineering.md)).
(i)은 GEMM 내부 오차보정, (ii)는 바깥 반복 정제로만 풀린다 — 상보적.

### 4.1 레버 A — Ozaki TC2 (GEMM 내부 오차보정, note 51)

`CLS_TF32_OZAKI_TC2` 는 TF32 MMA 입력을 2개의 TF32-표현가능 성분으로 분할:

```text
x0 = cvt.rna.tf32.f32(x);  x1 = cvt.rna.tf32.f32(x - x0)
```

각 곱을 `L0·U0 + L1·U0 + L0·U1 + L1·U1` 로 TC 안에서 `.f32` 누산기로 체인 누적. global trailing,
mid direct-shared, blocked, small-warp 전 tier 에 적용. 결과:

| case | B | FP32 relres | Ozaki TF32 relres | (raw TF32 relres) | speedup |
|---|---:|---:|---:|---:|---:|
| 8387 | 64 | 3.33e-05 | **4.77e-05** | 3.97e-02 | 1.181× |
| 8387 | 256 | 3.41e-05 | **4.77e-05** | 3.97e-02 | 1.139× |
| 13K | 64 | 1.24e-04 | 1.33e-04 | 6.47e-03 | 1.214× |
| 25K | 256 | 1.62e-04 | 2.45e-04 | — | 1.347× |

→ **8387 relres 3.97e-2 → 4.77e-5** (10²~10³× 개선, FP32 band 회복). 비용은 추가 3 MMA pass +
head/tail 변환으로 **low-fill speed margin 감소**(특히 8387 B256). 이는 note 53 의 옛 TF32x3
(`CLS_TF32X3`)를 대체한다 — TF32x3 는 big(UCP>64) fall-through 경로에만 적용돼 small/mid 지배
8387 에선 보정이 **실행조차 안 됐다**(커버리지 버그). Ozaki 는 전 tier 적용.

### 4.2 first-order Ozaki (note 52)

`CLS_TF32_OZAKI_TC2_FIRST_ORDER` 는 `L1·U1` 을 버리고 `L0·U0 + L1·U0 + L0·U1`. 정확도는 동일하게
FP32 band 유지(8387 4.77e-5), 속도는 full TC2 대비 **~1% 더 빠름**(8387 B256 −1.0%). 즉 `L1·U1`
은 정확도에도 속도에도 핵심이 아니고, 주 overhead 는 2-성분 변환 + 2 correction pass. large case
는 first-order 로도 좋은 속도 유지(USA B256 1.299×) → accuracy-aware 측정의 better default.
`CLS_TF32_OZAKI_STAGE_DIRECT`(operand 사전 staging)는 shared/occupancy 비용으로 회귀, reject.

8387 B256 은 Ozaki 보정 후 raw 1.207× margin 이 너무 얇아져 1.2× 유지 불가(best ~1.15×).

### 4.3 레버 B — 바깥 반복 정제 (note 53)

저정밀 factor 를 preconditioner M≈A⁻¹ 로 쓰고 바깥에서 FP64 잔차로 회복(factor 1회, solve 반복):

- **`--ir <steps>`**: mixed-precision iterative refinement — x 는 FP64, `r=b−A·x`(FP64 SpMV) →
  `d=M⁻¹r`(저정밀 solve) → `x+=d`.
- **`--gmres <m>`**: right-preconditioned GMRES, 참 잔차 `‖b−A·x‖` FP64 최소화 — ill-conditioned 에
  IR 보다 강함.

**USA 처럼 conditioning floor(ii)에 걸린 case 의 올바른 해법은 레버 B** — 레버 A 는 GEMM round-off
만 흡수한다. 두 레버는 stackable(저정밀+TC 로 빠른 factor → 바깥 IR/GMRES 로 정확도). 정량
in-solver sweep 은 후속 과제로 남음.

영구 학습(note 53): 직교 발견으로 `factor_big_tf32_ptx` 의 ncu(70K B=64) 가 trailing 의 진짜
병목이 thin-K mma 가 아니라 **uncoalesced C-drain global store(9.98 sector/req, 2.5×, occupancy
33%, DRAM 8% latency-bound)** 임을 보였다 — note 28 ceiling 의 메모리측 보강, §5 의 동기.

---

## 5. fused C-drain / fp16 TC — notes 31/32/33/55

**fused trail+extend**: big-front trailing 결과를 C block 에 쓰지 않고 부모로 직접 atomicAdd →
uncoalesced C-drain 왕복 제거.

- **fp16 fused**(`CLS_FUSE_FP16_TRAIL_EXTEND=ON` default, note 31): USA B=1 factor 1.274×
  (factor+solve 1.179×). 그러나 B≥64 무효~손해.
- **mid fp16 TC**(`CLS_MID_FP16_TC`, note 32): 25K default parallel-ND B=1 평균 **1.195×**. 단
  size-gate(`20000<=n<80000`)로 8387 은 nc>=10 gate 에서 제외, USA 는 row-gate 제외(forcing 시
  relres ~5.86 spike). 1024-thread setup 실패 → **512-thread 고정**.
- **large-batch fp16**(note 33): 25K/USA B=64/256 전부 0.76~1.05× — 배치에서 staging/padding/drain
  overhead 가 이미 saturated 된 scalar 경로를 못 이김.

**note 55 재현**(RTX 3090, median of 10): fused C-drain 은 fp16/tf32 둘 다 **B=1 만 이득, 배치
무효~손해**:

| fp16 fused ON/OFF | B=1 | B=64 | B=256 |
|---|--:|--:|--:|
| 70K | **1.33×** | 1.04× | 0.96× |
| USA | 1.01× | 1.00× | **0.90×** |

배치 무효 이유: (1) 절감 트래픽이 SM 포화 시 다른 batch-block 으로 overlap 되어 숨겨짐, (2) fused
atomic 은 mma 누산기 레이아웃(laneC=0,2,4,6)으로 흩어진 atomicAdd 라 배치 atomic 병목에서 손해.
note 53 이 보류 사유로 든 "tf32 큰 uc front correctness 버그"는 fp16 패턴 미러링 구현에선 재현 안
됨 — tf32 fused 는 **정확하나 register spill**(STACK 24B, fp16 은 half-packing 으로 STACK 0).
결론: **fused C-drain 은 B=1/underfill 게이트가 맞다.** 진짜 배치 미개척 lever 는 fp16 *저장*
(half CB/arena, 바이트 반감)이나 정확도(range) trade-off 동반, 미구현.

---

## 6. 문헌 (notes 34/40/45 압축)

광범위 문헌 조사(NVIDIA 공식 문서 + peer-reviewed)는 현재 방향을 **뒤집지 않고 좁힌다**.
multifrontal/supernodal GPU solver(STRUMPACK GPU/BLR, SuperLU_DIST batched, cuDSS)는 모두 dense
region 노출·analysis 재사용·batching·vendor dense kernel 로 이득을 얻으므로 large-case
mid/big TF32 경로는 문헌 정합적이다. 반대로 8387/13K 는 KLU·Basker·PanguLU·GLU3.0·Caracal 의
**circuit/low-fill sparse-LU regime** 에 가깝고, 이 line 은 dense supernodal aggregation 에서
의도적으로 멀어진다. cuDSS 는 small/medium matrix 에 hybrid host/device 를 권하고, 소형 batched-GEMM
TC 문헌(IPDPS 2019)은 packing/aggregation 후에만 TC 가 의미있다고 본다 — 본 조사의 packed-TC
6.5% eff 부정 결과와 정합. cuTeSpMM 의 "TCU synergy" 개념은 `useful_mma_flop/padded_mma_flop`,
`nc/uc` 를 TC 추가 전 go/no-go gate 로 쓸 근거를 준다. 핵심 인용:

- **SuperLU_DIST batched** ([10.1177/10943420241268200](https://doi.org/10.1177/10943420241268200)):
  elimination-tree level batching + 전용 batched Scatter kernel — `extend_add`/scatter 를 first-class
  로 취급할 근거.
- **cuDSS blog**: batching 은 개별 시스템이 GPU 를 못 채울 때 enabler; small matrix 는 hybrid/host.
- **PTX ISA / A100 whitepaper**: TF32 `mma.m16n8k8` 의 K granularity 는 hard constraint —
  `nc=1/2` 는 underfill.
- **Ootomo–Yokota** ([2203.03341](https://arxiv.org/abs/2203.03341)) / TC iterative refinement
  ([PMC7735315](https://pmc.ncbi.nlm.nih.gov/articles/PMC7735315/)): §4 의 Ozaki·IR/GMRES 의 학술 근거.
- **ACOPF GPU** ([2302.08656](https://arxiv.org/abs/2302.08656)) / **SABLE**
  ([2606.07099](https://arxiv.org/abs/2606.07099)): 25K/70K 타깃과 B64/B256 batched·mixed-precision
  방향이 application-realistic 임을 확인.

---

## 7. ⚠️ 정직한 결론 (notes 54·55 self-correction)

§2·§3.1 의 낙관적 수치(25K/USA 1.22~1.28×, 8387/13K 1.24×)는 **best-vs-best 공정 통제하에서 재현되지
않는다.** notes 54·55(2026-06-10)는 이를 자기정정한다.

### 7.1 cap-inflation 해명

note 51/52 의 "큰 케이스 1.2~1.3×"는 **fp32 baseline 을 tf32 와 같은 큰 panel-cap(31)에 묶어 fp32 를
느리게** 만든 비교였다(25K fp32 cap16=0.139 vs cap31=0.152). fp32 도 자기 최적 cap 으로 튜닝하면
1.05~1.07× 로 내려온다. note 55 의 same-build·same-cap·`--serial-nd` 통제 재현:

| case | cap | B=1 | B=64 | B=256 |
|---|--:|--:|--:|--:|
| 25K | 31 | 1.18× | 1.12× | 1.14× |
| 70K | 24 | 1.11× | 1.09× | 1.10× |
| USA | 31 | 1.11× | 1.08× | 1.08× |
| 8387 | 30 | 1.00× | 1.10× | 1.07× |

→ note 51/52 의 1.24~1.30× **재현 안 됨, 실측 ~1.1×.**

### 7.2 honest ~1.1× 와 TC mma 순기여 +6~9%

best-vs-best(각자 최적 cap, parallel ND, factor+solve `tf32/fp32`, note 55 §1):

| case | B=1 | B=16 | B=64 | B=256 |
|---|--:|--:|--:|--:|
| case1197 (1K~3K) | 1.00× | 1.00× | 1.00× | 1.00× |
| case3012wp (3K~6K) | 1.03× | 0.94× | 0.97× | 1.00× |
| case8387 (6K~10K) | 0.96× | 1.02× | 1.03× | 1.00× |
| 25K | 1.05× | 1.02× | 1.06× | 1.07× |
| 70K | 1.16× | 1.03× | 1.10× | 1.06× |
| USA | 1.15× | 1.09× | 1.11× | 1.13× |

**≤10K 는 무가속**(0.94~1.03× 노이즈), 25K +5~7%, 70K/USA +10~16%(B=1 피크). **어디서도 1.2× 미달.**

note 54 §3.1 의 분해(같은 staging/thread, mma 만 scalar 치환):

| case (B64) | 총(fp32/tf32) | = kernel-shape | × **TC mma** |
|---|---|---|---|
| 25K | 1.254× | 1.149× | **1.092×** |
| USA | 1.186× | 1.114× | **1.064×** |
| 70K | 1.207× | 1.124× | **1.074×** |

→ **honest TC mma 순기여는 +6~9%(이득의 ~38%); 나머지 ~62%는 kernel 형태(엔지니어링).** 추가로
solve(삼각대입)는 fp32·tf32 동일 scalar 커널이라 TC 무관이고 wall 의 30~46% 를 차지 → factor 가
1.2× 빨라도 전체는 `1/(0.6/1.2+0.4)≈1.11×`. 이것이 **factor+solve 1.1× 천장의 구조적 원인**이다
(Amdahl). note 29 가 지목한 panel-LU+U-solve barrier 를 겨눈 `CLS_TF32_COLUMN_USOLVE` 는 효과 ≈0%
(fp32 에 적용해도 −0.4~+0.9% 노이즈) — negative result.

### 7.3 low-fill fragility — net 이득 거의 0

note 50 의 8387 1.24× 는 **cap30 artifact**다. cap30 은 amalgamation fill 을 키워 절대 성능을
악화시킨다. **자연 cap** 에서(note 54 §3.2):

- 8387: low-fill tf32 0.0220 vs fp32 0.0246 → 진짜 ratio **1.12×**(목표 미달).
- 13K: low-fill tf32 0.0341 vs large 0.0377 → **1.09×.**

게다가 low-fill 은 amalgamation + small16 + mid128 + low_tc_force_all 을 **전부 함께** 켜야 win 하는
**fragile 4-플래그 번들**이고, 부분집합은 large 정책보다 느리다. serial-ND·고정 cap·특정 seed 의
fragile 조합을 벗어나 **각 구성을 공정 cap 튜닝(parallel ND)하면 우위가 소멸**한다(note 55 §3):
case3012wp 는 번들이 5~9% 느리고, 8387 은 tie, case1197 무차별 → **<10K 에서 low-fill 번들 net
이득 0.** small16 무조건 적용은 25K 도 악화시킨다(tf32 0.103 vs 0.092). 즉 low-fill 의 "진입"은
헤드라인이 아니라 측정 아티팩트 + 깨지기 쉬운 번들이다.

### 7.4 다음 방향 — front-distribution 스케줄링 정책 레이어

cleanup(3-정밀도 bake-in)은 **중단**했다(note 54). 이유: 케이스별 최적 알고리즘이 갈리고, 입력 크기
`n` 이 아니라 **post-symbolic front 분포**로 경로를 정해야 하기 때문. 다음 방향:

```text
reorder → symbolic → [front 분포 feature 추출] → policy(features, B, precision) → config → 커널 스케줄링
```

정책 입력 feature: tier별 work(FLOP) 비중, `nc(=K)` 분포 + TC-routable work%(nc≥4,uc≥16),
trailing vs panel-LU/extend 비중, etree depth/spine, fill ratio. **배치 크기 B 는 1차 축**
(B=1 launch/occupancy-bound, B≥64 throughput-bound, saturation 이 tier 마다 다름). 정책 도출 sweep
은 78 MATPOWER 케이스의 NR 선형시스템 생성(현재 ~9개)을 선행 과제로 둔다. low-fill 은 삭제하지 말고
(작은 케이스 절대시간 컨텍스트), "언제 켤지"를 정책이 결정한다.

### 7.5 한 줄 정정

> 헤드라인 "1.2×/1.24× TC 가속"은 **cap-inflation 아티팩트**였다. 공정 best-vs-best 에서 large
> case 는 ~**1.1×**(≤10K 무가속), 정직한 TC mma 순기여는 **+6~9%**(나머지는 kernel engineering),
> low-fill 8387/13K "진입"은 fragile 번들 + cap-inflation 으로 **net 이득 거의 0**. 텐서코어는
> large case 에서 의미있는 보조 lever 이되 단독 1.2× enabler 는 아니며, 다음 작업은 front-분포
> 기반 스케줄링 정책 레이어다.

> **독립 재현(2026-06-10)**: 대표 5케이스 × B{1,4,16,64,256} 정확도-매칭 best-vs-best 에서 factor
> speedup median **1.010×**, 신뢰 최대 ~1.12×, 1.2× 셀 없음 — 위 정정을 확증.
> [`../05-reports/05-tf32-reproduction-2026-06-10.md`](../05-reports/05-tf32-reproduction-2026-06-10.md).

## 8. 긍정 단서 — blocked 구조가 thin-K underfill 을 우회한다 (mid → shared-blocked)

§7 의 "thin-K(K=nc) 라 TC underfill" 은 *naive 단일 trailing GEMM* 전제다. 그런데 `factor_big_shared_tf32_blocked`
의 **blocked right-looking** trailing(`factorize_front_blocked_tf32`, **BK=8 column block**)은 trailing 을
K=8 풀-타일로 쪼개 누적하므로 각 block update 의 K = kb ≤ 8 = **mma.m16n8k8 의 K-tile 을 정확히 채운다**
— thin-K underfill 을 *구조로* 피한다.

이 커널을 TF32 mid tier 전체에 적용(`CLS_MID_AS_BIG_SHARED`)하고, **같은 blocked 구조에서 inner
trailing 만 mma↔scalar 로 교체**(`CLS_BLOCKED_SCALAR_UPDATE`)해 **순수 mma 기여**를 분리(25K, factor/sys):

| B | thr | blocked-MMA | blocked-SCALAR | 순수 mma 기여 |
|---:|---:|---:|---:|---:|
| 1 | 512 | 0.6004 | 0.7272 | **+21.1%** |
| 1 | 128 | 0.6883 | 1.0291 | **+49.5%** |
| 256 | 128 | 0.08140 | 0.09034 | **+11.0%** |
| 256 | 512 | 0.10152 | 0.08723 | −14.1% |

- **B=1 win 의 본체는 mma**: 구조 고정 시 mma 가 scalar 보다 +21~+49%. blocked-**scalar**(0.7272) ≈
  base `factor_mid`(0.7256) → **blocked 구조 자체는 ~0, 전부 mma 기여.** base mid 는 B<64 라 scalar
  (mid TC 가 B≥64 gate)인데 이 구조가 **B=1 에 TC 노출**.
- **정확도 무손실**: blocked-mma(TF32+Ozaki) relres 2.28e-4 = blocked-scalar(FP32) 2.28e-4.
- batch 는 occupancy 민감: 128 thr mma +11%, 512 thr 에선 Ozaki register 압박으로 mma 가 짐(−14%).

**의미**: §7 의 "TC 는 작은 보조 lever" 정정은 *naive thin-K* 한정이다. **적절한 blocked 구조(K=8 풀-타일)
+ B-적응 thread** 면 mid 에서 TC 가 실효적으로 작동한다(특히 B=1). 정량·thread sweep 은
[`../05-reports/05-tf32-reproduction-2026-06-10.md`](../05-reports/05-tf32-reproduction-2026-06-10.md) §11.
