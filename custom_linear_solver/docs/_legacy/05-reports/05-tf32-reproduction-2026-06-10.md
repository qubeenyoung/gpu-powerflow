# TF32 vs FP32 재현 — 정확도-매칭 best-vs-best (2026-06-10)

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: 계통 크기별 대표 5케이스에서 fp32 대비 tf32(+Ozaki) factor 속도를 *각자 최적 cap·정확도 매칭*으로 재측정 — notes 54/55 의 honest ~1.1× 천장을 독립 재현.

## 0. 목적

[`03-optimization-notes/03-tensor-core-investigation.md`](../03-optimization-notes/03-tensor-core-investigation.md)
§7(notes 54/55)의 자기정정 — *헤드라인 "1.2~1.3× TC 가속"은 cap-inflation 아티팩트이고, 공정 비교 시
~1.1×* — 을 새 환경에서 독립적으로 재현한다. 핵심은 **fp32 에도 자기 최적 cap 을 주는 best-vs-best**
통제로 cap-inflation 을 제거하고, **정확도(relres)를 FP32 band 로 매칭**해 "빠르지만 부정확한" cap 을
배제하는 것이다.

## 1. 환경 · 설정

- **HW/SW**: RTX 3090 (sm_86), CUDA 12.8, driver 580.
- **빌드**: `build-codex-large` (large-regime 정책). 주요 플래그: `CLS_MID_TF32_TC=ON`,
  `CLS_BIG_TF32_BLOCKED_TC=ON`, `CLS_BIG_TF32_SHARED_THREADS_512=ON`, `CLS_MID/BIG_LOW_SPLIT=ON`,
  `CLS_MID_TF32_TC_THREADS_128=ON`, `CLS_RESPECT_PANEL_CAP=ON`, `CLS_MID_TF32_MIN_FSZ=48`,
  `CLS_TF32_OZAKI_TC2(_FIRST_ORDER)=ON`.
  - ⚠️ **low-fill 번들(`CLS_TC_CLOSURE_PANEL_AMALGAMATE` 등 B2 front coarsening)은 OFF** — 본 재현에는
    포함되지 않는다. low-fill 정책 대조는 별도(`build-codex-lowfill`)로 한다.
  - tier routing(`tier_split=true`)과 small-tier warp-packing 은 기본 ON(프로덕션 경로).
- **정밀도**: `fp32` vs `tf32`(=FP32 front + TF32 텐서코어 trailing + Ozaki first-order).
- **런타임 공통**: `--batch-only --repeat 61 --warmup 8 --single-precision fp64`.

### 1.1 결정성(reproducibility) 프로토콜

두 단계로 측정했다 — (a) §4 의 best-vs-best 탐색은 parallel-ND(seed 42), (b) **CAP 확정용 재측정**은
아래 결정적 프로토콜로 다시 돌렸다(§8):

- **순서 고정**: `--serial-nd --metis-seed 1588` — parallel-ND 의 순서 무작위성을 제거. 효과 실측: 동일
  셀 3회 factor 변동이 **1.1% → 0.16%**, relres 는 결정적(예: case8387 B=64 relres 5.62e-5 고정).
- **GPU 클럭 고정**: 호스트에서 클럭 고정 + persistence mode ON (관측 SM clock **1695 MHz** 고정,
  boost/thermal jitter 억제). *(컨테이너 내부 권한으로는 lock 불가 — 호스트에서 설정.)*
- 결과: CAP 확정 sweep 200런 **FAIL 0**, 셀 변동 ~0.2%.

### 1.2 확정 CAP (이 보고서의 측정에 사용)

`--panel-cap` 을 케이스별로 아래 값으로 **고정**한다(serial-ND seed 1588, throughput B≥16 기준, §8 근거):

| case | n | 확정 cap | 근거 |
|---|---:|:--:|---|
| case1197 | 2,392 | **8** | 전 B cap8 (큰 cap은 +4% 손해) |
| case3012wp | 5,725 | **8** | fp32 cap8 / tf32 cap16 동률(<0.3%) → cap8 채택 |
| case8387pegase | 14,908 | **8** | 전 B cap8 |
| case_ACTIVSg25k | 47,246 | **16** | B≥16 cap16 명확(cap8 +1%, cap32 +19%); B=1 latency만 cap8 |
| case_SyntheticUSA | 156,255 | **16** | cap16(cap24 <1% 동률, cap8 +18%) |

규칙: **n ≲ 15K → cap8, n ≳ 47K → cap16.** fp32·tf32 확정 cap 일치(3012wp만 동률 차).

## 2. 대표 케이스 (계통 크기 n = Jacobian 차원)

| regime | case | n |
|---|---|---:|
| small | case1197 | 2,392 |
| ≤10K | case3012wp | 5,725 |
| mid / low-fill | case8387pegase | 14,908 |
| large | case_ACTIVSg25k | 47,246 |
| very large | case_SyntheticUSA | 156,255 |

(toy case30/118, 근사-중복 case_ACTIVSg2000/case6468rte 제외.)

## 3. 방법 — best-vs-best + 정확도 게이트

각 `(case, B, precision)` 에 대해 **cap ∈ {auto, 8, 16, 24, 32}** 를 스윕(총 5×5×2×5 = 250 run @
repeat 61). 그 중 **relres ≤ 5e-3 (FP32 band)** 를 통과한 cap 들에서 `factor_per_sys_ms` 최소값을 그
정밀도의 "best" 로 택한다. speedup = best(fp32) / best(tf32).

> 정확도 게이트가 핵심이다. USA 는 cap 선택에 따라 relres 가 1e-3(floor)~6e-2 로 흔들리는데, 게이트
> 없이 최속 cap 만 고르면 *부정확한 tf32* 가 빠른 것처럼 보여 speedup 이 과대평가된다.

## 4. 결과 (정확도-매칭 best-vs-best, factor/sys)

| case | n | B | fp32 cap | fp32 f/sys | tf32 cap | tf32 f/sys | speedup | tf32 relres |
|---|---:|---:|:--:|---:|:--:|---:|---:|---:|
| case1197 | 2392 | 1 | auto | 0.07179 | 16 | 0.07186 | **0.999×** | 2e-04 |
| case1197 | 2392 | 4 | 8 | 0.01902 | 16 | 0.01901 | **1.000×** | 2e-04 |
| case1197 | 2392 | 16 | 8 | 0.00570 | auto | 0.00571 | **1.000×** | 2e-04 |
| case1197 | 2392 | 64 | auto | 0.00286 | 8 | 0.00286 | **1.001×** | 2e-04 |
| case1197 | 2392 | 256 | auto | 0.00152 | 24 | 0.00152 | **1.001×** | 2e-04 |
| case3012wp | 5725 | 1 | auto | 0.17922 | 32 | 0.20198 | **0.887×** | 2e-04 |
| case3012wp | 5725 | 4 | 24 | 0.05323 | 24 | 0.05243 | **1.015×** | 2e-04 |
| case3012wp | 5725 | 16 | 8 | 0.01754 | 16 | 0.01867 | **0.940×** | 2e-04 |
| case3012wp | 5725 | 64 | auto | 0.00871 | 8 | 0.00880 | **0.990×** | 2e-04 |
| case3012wp | 5725 | 256 | auto | 0.00627 | auto | 0.00620 | **1.012×** | 2e-04 |
| case8387pegase | 14908 | 1 | 16 | 0.33374 | auto | 0.33787 | **0.988×** | 2e-05 |
| case8387pegase | 14908 | 4 | 8 | 0.10427 | 16 | 0.09933 | **1.050×** | 3e-05 |
| case8387pegase | 14908 | 16 | 24 | 0.04025 | 8 | 0.03943 | **1.021×** | 2e-05 |
| case8387pegase | 14908 | 64 | auto | 0.02382 | 24 | 0.02262 | **1.053×** | 2e-05 |
| case8387pegase | 14908 | 256 | auto | 0.02054 | 24 | 0.01997 | **1.029×** | 3e-05 |
| case_ACTIVSg25k | 47246 | 1 | 8 | 0.78008 | 16 | 0.77245 | **1.010×** | 1e-04 |
| case_ACTIVSg25k | 47246 | 4 | 16 | 0.24361 | 16 | 0.22731 | **1.072×** | 2e-04 |
| case_ACTIVSg25k | 47246 | 16 | 16 | 0.12101 | auto | 0.12259 | **0.987×** | 3e-04 |
| case_ACTIVSg25k | 47246 | 64 | 16 | 0.10337 | 16 | 0.09538 | **1.084×** | 2e-04 |
| case_ACTIVSg25k | 47246 | 256 | auto | 0.09575 | 16 | 0.08557 | **1.119×** | 2e-04 |
| case_SyntheticUSA | 156255 | 1 | auto | 2.56987 | 16 | 1.70297 | **1.509×** ⚠️ | 1e-03 |
| case_SyntheticUSA | 156255 | 4 | 16 | 0.82281 | 8 | 0.87605 | **0.939×** | 1e-03 |
| case_SyntheticUSA | 156255 | 16 | 16 | 0.49129 | 16 | 0.43967 | **1.117×** | 1e-03 |
| case_SyntheticUSA | 156255 | 64 | 16 | 0.40965 | 8 | 0.48955 | **0.837×** | 1e-03 |
| case_SyntheticUSA | 156255 | 256 | 16 | 0.40172 | 16 | 0.39648 | **1.013×** | 1e-03 |

**집계**: factor speedup median **1.010×** (USA B=1 제외 시 1.006×), 신뢰 가능 최대 **~1.12×**
(25K B=256 1.119×, USA B=16 1.117×), **≥1.10× 는 3/25 셀**뿐, 일부는 회귀. 1.2× 도달 셀 없음.

원자료: [`05-tf32-reproduction-2026-06-10/sweep_cap_best_vs_best.tsv`](05-tf32-reproduction-2026-06-10/sweep_cap_best_vs_best.tsv)
(250 run 전체), 동일-cap 참고: [`.../sweep_samecap_b64_b256.tsv`](05-tf32-reproduction-2026-06-10/sweep_samecap_b64_b256.tsv).

## 5. 해석

1. **cap-inflation 제거 확인**: best-vs-best(median 1.010×)는 동일-default-cap 1차 측정(~1.05×)보다 *더
   낮다*. fp32 에도 최적 cap 을 주니 tf32 우위가 사라진다 — notes 54/55 의 핵심 주장을 그대로 재현.
2. **계통 크기별**: ≤10K(1197/3012wp) 무가속~경미한 회귀, low-fill 8387 ~1.05×, large 25K/USA ~1.1×.
   = storyline §5 의 정직한 분해(≤10K 0.94~1.03×, 25K/USA +5~16%)와 정합.
3. **정확도**: 게이트 통과 셀 모두 FP32 band 유지(8387 ~2e-5, 25K ~2e-4, USA ~1e-3 floor). tf32+Ozaki
   의 정확도 보존 주장 확인.

## 6. Caveats

- **USA B=1 = 1.509× ⚠️ 는 노이즈**로 본다: 같은 USA 의 B=4(0.94×)·B=64(0.84×)가 회귀하는데 B=1 만
  1.5× 는 비정합. tf32 cap16 B=1 = 1.70 ms 가 이웃 cap(2.3~2.5)보다 비정상적으로 빠른 단일-실행
  outlier(B=1 USA 는 latency-bound, 변동 큼). 프로세스 다중-실행 median 으로 확정 필요.
- 각 셀은 프로세스 1회 실행(내부 repeat=61 median). 경계 셀(±3%)은 실행 간 변동 범위.
- **low-fill 정책(closure amalgamation 등 B2) 미포함** — 본 재현은 `build-codex-large` 기준이다.
  closure amalgamation 이 8387/13K 에 주는 효과는 `build-codex-lowfill` 대조로 별도 측정해야 한다.

## 7. 결론

계통 크기 전 구간에서 텐서코어의 *정확도-공정* 순속도 이득은 **median ~1.01×, 신뢰 최대 ~1.12×**로,
[`03-optimization-notes/03-tensor-core-investigation.md`](../03-optimization-notes/03-tensor-core-investigation.md)
§7 의 **honest ~1.1× 천장**을 독립적으로 확증한다. 헤드라인 1.2~1.3× 는 baseline cap inflation 산물이라는
정정이 재현으로 뒷받침된다.

## 8. CAP 확정 — 결정적 재측정 (serial-ND seed 1588 + 클럭 고정)

§1.1 프로토콜로 cap ∈ {8, 16, 24, 32} 를 재스윕(200 run, parallel-ND/auto 제거). 정확도 게이트는
케이스 floor 기준(USA 는 fp32·tf32 가 같은 conditioning floor ~1.5e-2 라 상대 비교). throughput
(B≥16) factor 의 cap별 상대값(1.000=최선):

| case | cap8 | cap16 | cap24 | cap32 | 확정 |
|---|---:|---:|---:|---:|:--:|
| case1197 | **1.017** | 1.039 | 1.039 | 1.038 | cap8 |
| case3012wp | **1.001** | 1.001 | 1.022 | 1.063 | cap8 |
| case8387pegase | **1.004** | 1.014 | 1.051 | 1.042 | cap8 |
| case_ACTIVSg25k | 1.010 | **1.000** | 1.055 | 1.185 | cap16 |
| case_SyntheticUSA | 1.179 | **1.000** | 1.008 | 1.066 | cap16 |

- cap24/32(큰 cap)은 padded fill 만 키워 전 케이스 손해 → 탈락. 작은 cap(8↔16)은 ≤15K 에서 평탄(<1.5%).
- **B 의존성**: cap 은 analyze(구조) 선택이라 케이스당 하나지만, 25K 는 B=1(latency)에선 cap8 이 더 빠름.
  위 확정값은 throughput(B≥16) 기준.
- **USA 정확도**: seed 1588 순서는 USA 의 conditioning 이 다소 나빠 fp32·tf32 모두 relres ~1.5e-2
  (데이터셋 floor). cap 선택은 두 정밀도가 같은 floor 라 유효하나, USA 정확도 자체가 중요하면 seed 를
  몇 개 더 확인하는 것이 좋다.

원자료: [`05-tf32-reproduction-2026-06-10/cap_sweep_serial-nd_seed1588.tsv`](05-tf32-reproduction-2026-06-10/cap_sweep_serial-nd_seed1588.tsv) (200 run, 결정적).

## 9. 파생 측정 — fused trail+extend 를 scalar 에 적용

§1 의 fp32 베이스라인이 얼마나 최적화돼 있는지 보던 중, scalar trailing 의 남은 병목인 **CB(contribution
block) 의 global write→read 왕복**(note 30/53 의 "uncoalesced C-drain 2.5×")을 제거하는 lever 를 측정했다.
non-fused 는 trailing 이 CB 를 front 에 쓰고 `extend_add` 가 도로 읽어 부모에 atomicAdd 하는데, **fused** 는
trailing 누산기를 **부모에 직접 atomicAdd** 하고 CB 왕복을 생략한다.

- **구현**: `trailing_update_staged<T, true>` (기존 `FuseExtend` 분기) 를 `factor_big_staged`·`factor_mid`
  의 `fsz > 48` front 에 배선. `CLS_FUSE_SCALAR_TRAIL_EXTEND` (기본 OFF, `build-scalar-fuse`).
  (`fsz ≤ 48` 은 `lu_small_front` 가 trailing functor 를 호출하지 않으므로 제외 — 누락 시 CB 유실.)
- **검증**: 잘-conditioned 케이스 relres 가 base 와 일치(8387 ~2.5e-5, 25K ~1.6e-4) → 정확성 확인.

**결과** (fp32, serial-ND seed 1588, 확정 cap, factor/sys, base=`build-codex-large`):

| case | n | B=1 (base→fuse) | Δ | B=256 | Δ |
|---|---:|---|---:|---|---:|
| case1197 | 2,392 | 0.0677→0.0679 | +0.3% | 0.00162→0.00163 | +0.2% |
| case3012wp | 5,725 | 0.2090→0.2086 | −0.2% | 0.00614→0.00613 | −0.1% |
| case8387pegase | 14,908 | 0.3287→0.3298 | +0.3% | 0.02159→0.02150 | −0.4% |
| **case_ACTIVSg25k** | 47,246 | 0.7862→0.7549 | **−4.0%** | 0.09222→0.09045 | **−1.9%** |
| **case_SyntheticUSA** | 156,255 | 2.6138→2.4092 | **−7.8%** | 0.43926→0.40695 | **−7.4%** |

**해석**: **large-case 전용 +4~8% factor lever, B=1·B=256 둘 다 유효.** 작은/low-fill 은 big front 가
적어 CB 트래픽 자체가 미미 → 무효(노이즈). tf32/fp16 fused(note 55: B=1 만 이득, 배치 무효~−10%)와
**반대** — scalar 는 trailing 이 느려 C-drain 이 wall 에 노출돼 있고 register spill 도 없어, 제거 이득이
배치에서도 남는다. 이는 substrate kernel-opt 개선이라 **fp32 베이스라인을 직접 끌어올려** best-vs-best
에서 tf32 우위를 더 좁힌다. 기록: [`../03-optimization-notes/01-kernel-engineering.md`](../03-optimization-notes/01-kernel-engineering.md) §3.6.

- **Caveat**: USA 만 atomicAdd 누적 순서로 relres 가 floor(~1e-2) 내에서 소폭 변동(B=256 base 1.5e-2 →
  fuse 2.5e-2). 정확도 손실이 아니라 ill-conditioned 데이터셋의 reduction-order 민감성. 구현은 staged
  경로(fsz>48)만 — `factor_big` 비-staged fallback 과 fsz≤48 은 baseline.

원자료: [`05-tf32-reproduction-2026-06-10/fused_scalar_trail_extend_b1_b256.tsv`](05-tf32-reproduction-2026-06-10/fused_scalar_trail_extend_b1_b256.tsv).

## 10. fused 가 scalar 는 가속·TC 는 감속 — launch_bounds×spill 분석

§9 의 fused 를 **tf32(텐서코어)** 에 적용하면 scalar 와 **반대로 batch 에서 감속**한다(USA B=256). 원인을
끝까지 추적했다.

**현상** (USA tf32, factor/sys, serial-ND seed 1588, cap16):

| 구성 | B=1 | B=256 |
|---|---:|---:|
| scalar fused (§9 비교) | −7.8% | **−7.4% (가속)** |
| tf32 fused | −8.1% | **+4.3% (감속)** |

**원인 = `__launch_bounds__(512,2)` 의 register cap → spill.** `(512,2)` 는 2 block/SM 을 위해 register 를
64 로 cap 하는데, fused drain 로직이 더해지면 커널이 ~106 reg 를 원해 **24-byte spill** 이 강제된다.
cuobjdump `factor_big_tf32_ptx`:

| 구성 | REG | STACK(spill) |
|---|---:|---:|
| non-fused, LB(512,2) | 64 | 0 |
| non-fused, no-LB | 80 | 0 |
| fused, LB(512,2) | 64 | **24** |
| fused, no-LB | 90 | 0 |

**launch_bounds 를 삭제하면 감속이 뒤집힌다** (USA tf32, factor/sys):

| 구성 | B=1 | B=256 |
|---|---:|---:|
| base-LB (non-fused) | 2.16643 | 0.41171 |
| no-LB (non-fused) | 2.15824 (−0.4%) | 0.40937 (**−0.6%**) |
| fused-LB (spill) | 1.98919 (−8.2%) | 0.42927 (**+4.3%**) |
| **fused-no-LB** | 1.97825 (−8.7%) | **0.40281 (−2.2%)** |

- **비-fused 기본 path 는 LB 삭제에 무영향**: 8387/25K/USA × B{1,256} 전부 **±0.5%(노이즈)**. EXP-B 의
  `(512,2)` −2.7~3.8% 이득은 현 V9h 커널에서 재현 안 됨(80 reg/1 block 으로도 big front warp 충분).
- **fused 는 LB 삭제로 spill 제거** → USA B=256 **+4.3% → −2.2%** 로 전환, B=1 −8.7%.

**해석 — fusion 손익을 가르는 것은 "trailing 이 memory-bound 이고 register 여유가 있나"**:
- scalar(`factor_big_staged`, REG:40): trailing 이 느린 memory-bound 라 CB 왕복 절감이 batch 에서도 크고,
  register 여유로 spill 없음 → **가속**.
- tf32 TC(`factor_big_tf32_ptx`, REG:64 cap): trailing 이 빨라 CB 절감이 작은데 `(512,2)` cap 이 spill 까지
  유발 → **batch 감속**. LB 삭제로 spill 제거하면 소폭 가속(절감폭은 scalar 보다 작음 — C-drain 이 TC
  에선 wall 비중이 작아서). fp16 fused 가 기본 ON 인 이유도 동일: fp16 은 mma A 가 2-reg 라 reg 여유로
  spill 이 안 남.

**결정**: `factor_big_tf32_ptx` 의 `__launch_bounds__` **삭제**(코드 반영). 양쪽(fused/non-fused) 안전.
상세: [`../03-optimization-notes/01-kernel-engineering.md`](../03-optimization-notes/01-kernel-engineering.md) §3.3.

원자료: [`05-tf32-reproduction-2026-06-10/tf32_fused_vs_base_b1_b256.tsv`](05-tf32-reproduction-2026-06-10/tf32_fused_vs_base_b1_b256.tsv).

## 11. mid → shared-blocked TC — B-적응 thread + thin-K 우회 (긍정 결과)

`factor_big_shared_tf32_blocked`(big-low 129~159 용, front-resident shared + blocked TC)를 **TF32 mid
tier 전체**에 적용해봤다(`CLS_MID_AS_BIG_SHARED`). mid-dominant 케이스(25K)에서 의미있는 이득.

### 11.1 thread 수가 효과를 가른다 (occupancy)

mid front 는 작아 기본 512-thread 가 batch 에서 점유율을 낭비한다. thread 수 sweep(25K tf32, vs base
`factor_mid`):

| case | B | 512 | 256 | 128 |
|---|---:|---:|---:|---:|
| case_ACTIVSg25k | 1 | **+17.0%** | +16.7% | +5.3% |
| case_ACTIVSg25k | 256 | −21.4% | −0.9% | **+2.7%** |
| case8387pegase | 1 | +4.5% | +3.2% | −9.4% |
| case8387pegase | 256 | −32.6% | −11.8% | −2.9% |

→ thread 최적이 **B 에 정반대**: B=1 은 많을수록(512, latency 병렬화), B=256 은 적을수록(128, block
packing). **B-적응(B=1→512, B=256→128)** 이면 25K 가 **B=1 +17% AND B=256 +2.7% 양쪽 승**.
감속 원인은 메모리가 아니라 occupancy — 커널은 연산이 전부 shared 라 global 트래픽은 stage-in/
writeback(coalesced)+extend(atomic)뿐(메모리 효율적).

### 11.2 이 구조는 TC-favorable 인가 — 예 (조건부)

**순수 mma 분리** — 같은 blocked 구조에서 inner trailing update 만 mma↔scalar 로 교체
(`CLS_BLOCKED_SCALAR_UPDATE`, `block_update_scalar_direct_shared`). panel-LU·U-solve·shared-residency 모두
동일, **오직 rank-kb trailing 만 TF32 mma vs FP32 FMA** (25K, factor/sys):

| B | thr | blocked-MMA | blocked-SCALAR | 순수 mma 기여 |
|---:|---:|---:|---:|---:|
| 1 | 512 | 0.6004 | 0.7272 | **+21.1%** |
| 1 | 128 | 0.6883 | 1.0291 | **+49.5%** |
| 256 | 512 | 0.10152 | 0.08723 | **−14.1%** (mma 짐) |
| 256 | 128 | 0.08140 | 0.09034 | **+11.0%** |

- **B=1: mma 가 win 의 본체** — 구조 고정 시 mma 가 scalar 보다 +21~+49%. 게다가 blocked-**scalar**
  (0.7272) ≈ nonblocked-scalar(0.7230) ≈ base `factor_mid`(0.7256) → **blocked 구조 자체는 B=1 에서 ~0
  기여, 전체 win 은 mma.** base mid 는 B<64 라 scalar(mid TC 가 B≥64 gate)인데 이 구조가 B=1 에 TC 노출.
- **왜 mma 가 먹히나**: `factorize_front_blocked_tf32` 는 **BK=8 column block** right-looking → 각 block
  update 의 K=kb≤8 = **mma.m16n8k8 K-tile 을 정확히 채움**. thin-K(K=nc) 를 한 번에 GEMM 하지 않고 K=8
  풀-타일로 쪼개 큰 trailing 에 누적 + L operand 를 N 축으로 재사용 + C 누산기 register 상주
  → "thin-K underfill" 을 **구조로 우회**(`../03-optimization-notes/03-tensor-core-investigation.md` §8).
- **B=256 은 occupancy-gated**: 128 thr 에서 mma +11%, 그러나 512 thr 에선 TF32+Ozaki 의 register 압박이
  점유율을 깎아 mma 가 scalar 에 −14%. thread 를 맞춰야 mma 이점이 드러난다.
- **정확도 무손실**: blocked-mma(TF32+Ozaki) relres 2.28e-4 = blocked-scalar(FP32) 2.28e-4 — Ozaki 가 FP32
  band 회복.

### 11.3 결론

**mid → shared-blocked + B-적응 thread 는 mid-dominant 케이스(25K)에서 양쪽-B universal win 후보이며,
B=1 이득은 명백한 TC win(blocked K=8 의 thin-K 우회).** 이는 storyline §7 의 "TC 는 작은 보조 lever"
정정에 대한 **긍정적 단서** — *적절한 blocked 구조 + B-적응 thread* 면 mid 에서 TC 가 실효적으로 작동한다.
다음: B-gate thread + case-크기(mid 비중) gate 를 단 운영 버전 확정, 8387(small-dominant)은 별도.

플래그: `CLS_MID_AS_BIG_SHARED`(기본 OFF), `CLS_MID_AS_BIG_THREADS`(기본 512), `CLS_BLOCKED_FORCE_SCALAR`
(진단용). 원자료: [`05-tf32-reproduction-2026-06-10/mid_as_big_shared_threads_tc_vs_scalar.tsv`](05-tf32-reproduction-2026-06-10/mid_as_big_shared_threads_tc_vs_scalar.tsv).

## 12. mid-TC thread 휴리스틱 + 운영 정책 (`CLS_MID_TC_POLICY`)

§11 의 mid→shared-blocked 를 7케이스(8387/9241/10k/13659/25k/70k/USA) × B{1,4,16,64,256} 로 thread
{64,128,256,512} 스윕(tf32, serial-ND 1588, auto-cap, warmup=16, repeat=61).

### 12.1 thread 최적은 occupancy 로 결정

best thread (vs base `factor_mid`): B↑ 일수록 단조 감소 — **B=1→512(n≲25k)/256(70k·USA), B=4→256,
B≥16→128**. 64 는 한 번도 최적 아님(thread 부족), 512 는 B=1 한정. 근본 변수는 **GPU 채움 =
level_size×B vs SM 수**:

```
blocks = level_size * B
threads = (blocks < num_SMs) ? 512 : (blocks < 4*num_SMs ? 256 : 128)
```

### 12.2 운영 정책 — B≤4 gate + occupancy thread

`dispatch_factor_mid` 에 hook(`CLS_MID_TC_POLICY`, 기본 OFF): **TF32 mid 를 B≤4 일 때만** shared-blocked
TC 로 라우팅(위 occupancy thread). base mid 는 B<64 에서 scalar 라(mid-TC gate B≥64), 이 정책이 **저배치에
TC 를 가져온다**. B≥16 은 gate 로 base 유지.

**정책 vs base** (factor/sys 가속률):

| case | B=1 | B=4 | B≥16 |
|---|---:|---:|---:|
| 8387 | +6.6% | +4.5% | ~0 |
| 10k | +9.4% | +8.5% | ~0 |
| 13659 | +8.1% | +4.2% | ~0 |
| **25k** | **+14.5%** | **+10.2%** | ~0 |
| 70k | +2.2% | +2.7% | ~0 |
| USA | +2.4% | +0.1% | ~0 |

→ B=1 평균 +6.9%, B=4 평균 +4.9%, **B≥16 정확히 ~0(최악 −0.1% 노이즈)**. 단일-512 가 만들던 B=256
−20~33% 회귀가 gate 로 완전 제거. 정확성 relres FP32 band.

### 12.3 fp32 vs tf32 vs policy — 분해

총이득(fp32→policy) = **① 텐서코어 정밀도(fp32→tf32) + ② 저배치 정책(tf32→policy)**:

| | ① fp32→tf32 (TC 정밀도) | ② tf32→policy (mid-TC 라우팅) |
|---|---|---|
| B=1 | ≤25k **+0.3~1.6%**(작음), 70k/USA +5~6% | **+4.9~14.7%** (25k 최대) |
| B=4 | ~0~+1.4% | +0.4~11.1% |
| B≥16 | +0~2%(16), +1~9.4%(64/256) | **0** (gate) |

핵심: **저배치 텐서코어 정밀도 자체는 거의 무이득**(storyline §7 정정과 일치)이고, **저배치 이득의 본체는
정책(②)** — base 가 B<64 mid 에서 TC 를 놓치던 것을 메운 것. 고배치는 정책 무관(=tf32), fp32 대비 +1~9%
는 순수 TC 정밀도.

플래그: `CLS_MID_TC_POLICY`(기본 OFF). 원자료:
[`05-tf32-reproduction-2026-06-10/mid_tc_thread_sweep.tsv`](05-tf32-reproduction-2026-06-10/mid_tc_thread_sweep.tsv),
[`.../mid_tc_policy_vs_base.tsv`](05-tf32-reproduction-2026-06-10/mid_tc_policy_vs_base.tsv),
[`.../fp32_vs_tf32_vs_policy.tsv`](05-tf32-reproduction-2026-06-10/fp32_vs_tf32_vs_policy.tsv).
