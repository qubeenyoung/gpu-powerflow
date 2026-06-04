# `perf/warm-cache-stack`(mysolver) 기법을 우리 솔버/ cuPF에 적용할 때의 예상

대상: `/workspace/cuDSS_reproduce_perf_warm_cache_stack` 의 `perf/warm-cache-stack` 브랜치
(연구 솔버 `mysolver-gpu`). 본 문서는 그 기법들을 `custom_linear_solver` 및 cuPF Mixed 경로에
적용했을 때의 **예상 효과**를, 실측으로 뒷받침해 정리한다.

## 0. 핵심 결론 먼저
- `custom_linear_solver`의 **단일-케이스 경로**(`src/factorize/multifrontal.cu`,
  `src/solve/multifrontal.cu`)는 사실상 **mysolver와 같은 lineage**다. cy85/147/169/333/338/346
  주석·코드(size-adaptive cap, big-front 멀티블록 split, single-warp spine, partitioned-inverse,
  PAR_ND_BASE 등)가 그대로 들어 있다. → **warm-cache-stack 기법 대부분이 이미 우리 단일-케이스
  경로에 존재**한다.
- 그런데 cuPF **Mixed 프로파일은 batched 경로**(`src/batched/*.cuh`)만 쓴다(B=1 포함). batched
  커널은 batch 처리량용으로 따로 작성돼 **단일-시스템 warm 최적화가 없다.**
- 그래서 "기법을 적용한다"는 것은 대부분 **새 기법 포팅이 아니라, Mixed/저배치가 이미 존재하는
  최적화된 단일-케이스 경로를 쓰게 만드는 라우팅 문제**다.

## 1. 측정 증거 (RTX 3090, cuPF, B=1, iteration당)

`cupf_batch_bench`(per-stage 타이밍) factorize, B=1:

| | compute | cuDSS | custom | 비고 |
|---|---|---|---|---|
| 9k (MIOHIN) | **FP64** | 0.99 ms | **0.90 ms** | custom=단일-케이스 경로 → **이김** |
| 25k | **FP64** | 1.45 ms | **1.10 ms** | custom=단일-케이스 경로 → **이김** |
| 9k | **Mixed** | 0.58 ms | 1.46 ms | custom=batched 경로 → 짐 |
| 25k | **Mixed** | 0.87 ms | 1.58 ms | custom=batched 경로 → 짐 |

읽는 법:
- **FP64 B=1에서 custom(단일-케이스)은 cuDSS를 이긴다** — warm-cache-stack이 우리 솔버에 실재하고
  효과적임을 확인.
- **Mixed B=1의 custom(1.46 ms)은 FP64 단일-케이스(0.90 ms)보다도 느리다.** FP32 산술이 더 싼데도
  느린 이유는 (a) Mixed가 batched 커널을 쓰고(단일-시스템 최적화 없음, B=1에서 GPU 굶음),
  (b) cuDSS-Mixed는 FP32로 factor 한다는 두 가지.
- 즉 cuPF Mixed B=1이 cuDSS에 지는 건 **기법 부재가 아니라 batched 라우팅** 때문.

### 1.1 path-only(같은 정밀도) batched-B1 페널티 — standalone, FP64

위 표는 path와 precision이 섞여 있어, path 효과만 분리하려고 standalone에서 **FP64 단일-케이스
vs FP64 batched-B1**(같은 행렬·같은 정밀도)을 측정:

| matrix | n | factor 단일 | factor batched-B1 | 페널티 |
|---|---|---|---|---|
| case3120sp | 5991 | 0.347 ms | 0.466 ms | **1.34×** |
| case6470rte | 12485 | 0.496 ms | 0.690 ms | **1.39×** |
| case9241pegase | 17036 | 0.846 ms | 1.217 ms | **1.44×** |

→ 같은 정밀도면 **batched-B1 factor는 단일-케이스보다 34~44% 느리다**(solve는 대등). batched의
B=1 점유 부족이 순수한 path 비용이다.

> **주의 — 이전 분석 정정**: 앞선 세션에서 "단일-케이스 ≈ batched-B1이라 라우팅해도 무의미"라고
> 했던 건 **confounded 비교**(FP64-단일 0.72 vs Mixed-FP32-batched 0.795, 한 행렬)였다. batched의
> path 페널티(~1.4×)를 FP32 산술 이득이 우연히 상쇄해 비슷해 보였을 뿐, **같은 정밀도로 보면
> path 비용은 실재**한다. 따라서 Mixed를 단일-케이스로 라우팅하는 레버는 유효하다(아래 §3A).

## 2. warm-cache-stack 기법별 — 우리 솔버 현황과 예상

| 기법 (cy) | mysolver 효과 | 우리 단일-케이스 | 우리 batched(=cuPF Mixed) | 적용 시 예상 |
|---|---|---|---|---|
| Partitioned-inverse pivot (cy335) | S 평탄화 돌파 | **있음** (`mf_invert_pivot`) | **있음** (`mf_invert_pivot_b`) | 이미 반영 |
| ncu micro-wins: register bwd / single-warp spine (cy332/333) | S −2~8% | **있음** (ts_spine/ts_sw) | 없음(batched bwd는 별도 설계) | batched 저배치에 포팅 시 소폭 |
| Size-adaptive amalgamation cap (cy338) | S −5~10% | **있음** (eff_cap 16/12/8) | 미적용(공유 plan의 amalg만) | batched 점유 개선 여지(중간) |
| Big-front 멀티블록 split (cy147) | 대형 front 점유↑ | **있음** (MF_BIGMULTI, opt-in) | 없음 | batched 대형 front에 유효할 수 있음 |
| Production plan reuse (cy342) | NR당 분석 생략 | — | — | **cuPF가 이미 함**(analyze 1회/initialize, factor 다회). 추가 이득 없음 |
| Shift-retry default-on (cy343) | 견고성 | **있음** (`enable_shift_retry`) | (단일과 공유) | 이미 반영 |
| Cold-A: GPU_ND/O(1) fidx/PAR_ND_BASE (cy346) | 분석 −13~29% | 부분(PAR_ND_BASE 있음) | n/a (analyze 공유) | cuPF `initialize()`(분석 60~170 ms) 단축 가능 |
| FP32 working LU (MYSOLVER_GPU_FP32) | warm E2E −3.5~12% | `analyze_multifrontal(...,fp32)` 인자 존재 | Mixed가 batched로 대체 | **Mixed 단일-케이스 FP32 경로**가 핵심 레버(아래) |
| 값-안정 캐시: cache_match/factor_cache (cy351/354) | warm E2E −56~82% | — | — | **NR에 부적합**(값 매 iter 변경 → MC64 재적응 필요). mysolver도 DEFAULT OFF |

## 3. regime별 예상

### (A) B=1 / 저배치 — 가장 큰 기회
현재 cuPF Mixed B=1: custom factor 1.46 ms (batched), cuDSS 0.58 ms.
- **레버**: Mixed/저배치를 batched-B1 대신 **단일-케이스 경로**로 라우팅. FP64 단일-케이스가 이미
  0.90 ms로 batched 1.46 ms보다 빠르므로, factor를 **1.46 → ~0.9 ms** 수준으로 즉시 개선 예상.
  여기에 단일-케이스의 **FP32 working LU**(이미 `fp32` 인자 존재)를 켜면 cuDSS-Mixed(0.58 ms)에
  근접/경쟁 가능.
- **필요 작업**: Mixed 어댑터(`CudaLinearSolveCustomMixed`)에 FP32 입력을 단일-케이스 경로로
  넣는 길이 현재 없다(값=nullptr로 batched만 사용). 단일-케이스 FP32 factor/solve 경로를
  Mixed 어댑터에 연결해야 함(신규 통합 작업; 알고리즘은 이미 존재).
- **주의**: 단일-케이스 경로는 CUDA 그래프 캡처 측면에서 batched와 다르다 — cuGraph 모드와의
  상호작용은 별도 검토 필요(현재 cuGraph는 batched만 캡처).

### (B) 배치 (B≥32) — 이미 우세, 한계 이득
custom batched가 cuDSS를 1.3~2.7× 앞선다(별도 벤치). warm-cache-stack의 **단일-시스템** 미세
최적화(single-warp spine 등)는 batch가 이미 GPU를 채우므로 이득이 작다. 다만 **size-adaptive
amalgamation cap / big-front split**의 사상을 batched 커널에 이식하면 대형 front 점유가 올라
**중간 규모(−수 %)** 개선 여지는 있다. 큰 기대는 금물.

### (C) `initialize()` / 분석(cold-A)
cuPF 분석은 custom 기준 ~60(3k)~170(70k) ms로 1회성이다. cy346(GPU_ND default-on, O(1) fidx
hash, PAR_ND_BASE=4000)은 분석을 −13~29% 줄였다. 반복 solve가 많은 워크로드에선 상각돼 의미가
작지만, **단발/소수 재사용** 시 체감 가능. 우리에 PAR_ND_BASE는 있으나 GPU_ND/ O(1) fidx hash는
확인 필요(없으면 포팅 후보).

## 4. 적용하면 안 되는 것 / 함정
- **cy351 cache_match / cy354 factor_cache**: 값-안정 워크로드 전용(같은 값 반복). cuPF NR은
  Jacobian 값이 매 iteration 바뀌므로 **부적합**(factor를 건너뛰면 틀린 답). mysolver도 NR에는
  DEFAULT OFF로 명시.
- **FP32+TC 1e-3 게이트(cy348)**: mysolver의 큰 E2E 이득은 berr 게이트를 1e-3으로 완화한 모드다.
  cuPF Mixed는 FP64 residual + 반복 정련으로 ~1e-9까지 수렴 중이므로, 정확도 요구가 다르면 그대로
  쓰면 안 된다(정확도/속도 트레이드오프를 cuPF tol에 맞춰 재설정 필요).
- **iteration 비결정성**: Mixed FP32 factor는 비결정적 extend-add atomic 때문에 tol 경계에서
  iteration 수가 흔들린다 — 어떤 최적화든 벤치 시 iteration 고정/노이즈 보고 필요.

## 5. 권장 우선순위 (기대 대비 노력)
1. **Mixed/저배치 → 단일-케이스(FP32) 경로 라우팅** (가장 큰 B=1 레버; 알고리즘은 존재, 통합만).
   목표: cuPF Mixed B=1 factor 1.46 → **~0.9 ms**(FP64 단일-케이스 실측값) 이하. FP32 단일-케이스는
   더 낮을 것으로 보이나 **미측정**(현재 Mixed 어댑터가 단일-케이스 FP32 경로를 안 타므로 직접
   측정 불가) — cuDSS-Mixed(0.58 ms)를 이긴다는 보장은 없고 행렬 의존적. 실측 후 확정 필요.
2. **size-adaptive cap / big-front split 사상을 batched 커널에 이식** (중간 규모 배치 점유 개선).
3. **cold-A(cy346) 포팅** (initialize 단축; 재사용 적은 워크로드에 한정).
4. 값-안정 캐시(cy351/354)는 **건드리지 말 것** (NR 부적합).

## 6. 근거/재현
- 단계별 측정: `cupf_batch_bench`(ENABLE_TIMING 빌드). `CUPF_BENCH_CUSTOM=1`로 custom 선택,
  compute=`fp64|mixed`. B=1에서 FP64는 단일-케이스, Mixed는 batched 경로를 탄다(코드:
  `cuda_custom_solver.cpp` 의 `CudaLinearSolveCustomFp64::factorize` 분기 vs
  `CudaLinearSolveCustomMixed::factorize` 항상 batched).
- mysolver 기법 원문: `/workspace/cuDSS_reproduce_perf_warm_cache_stack/docs/results-and-methodology.md` §4,
  `/workspace/cuDSS_reproduce_perf_warm_cache_stack/docs/cy348-recommended-config.md`.
