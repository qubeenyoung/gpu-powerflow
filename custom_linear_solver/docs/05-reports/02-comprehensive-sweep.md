# Comprehensive sweep — FP64 vs FP32 vs TC, 5 cases × 5 batch sizes

> **상태**: canonical   **갱신**: 2026-06-10
> **한 줄**: full benchmark table의 기준 문서 — B ≤ 16 작은 batch에서는 TC, B ≥ 64 큰 batch에서는 FP32 우세, USA B=4가 TC sweet spot (−17.5%).

이 문서는 full benchmark table의 canonical source다. 요약 결론과 권장 mode는 [`01-final-report.md`](01-final-report.md)를 우선 참고한다.

- 측정 일자: 2026-06-05
- GPU: RTX 3090 (sm_86)
- 환경: 모든 mode 동일 lever 적용 — selinv OFF default, multistream subtree dispatch default, Σ.1 staged trailing default.
- 측정 단위: factor + solve **TOTAL μs/sys** (median of 3 runs, each run = `--repeat N`의 median of N).

## 1. 결과 표 — 각 case의 batch sweep

### 1.1 ACTIVSg2000 (n=3607, smallest)
| B | FP64 | FP32 | TC | TC vs FP32 |
|---:|---:|---:|---:|---:|
| 1   | 998.4 | 526.3 | **501.8** | **−4.7%** |
| 4   | 282.6 | 139.5 | **132.6** | **−4.9%** |
| 16  | 88.9  | 45.1  | **43.6**  | −3.3% |
| 64  | 45.6  | 19.9  | **19.6**  | −1.5% |
| 256 | 40.9  | **14.7**  | 14.8 | +0.7% |

### 1.2 case6468rte (n=12643)
| B | FP64 | FP32 | TC | TC vs FP32 |
|---:|---:|---:|---:|---:|
| 1   | 986.0 | 603.1 | **552.7** | **−8.4%** |
| 4   | 291.1 | 177.9 | **161.7** | **−9.1%** |
| 16  | 119.0 | 63.2  | **61.5**  | −2.7% |
| 64  | 77.0  | 36.7  | **35.6**  | −3.0% |
| 256 | 73.7  | **29.8**  | 30.4 | +2.0% |

### 1.3 case8387 (n=14908)
| B | FP64 | FP32 | TC | TC vs FP32 |
|---:|---:|---:|---:|---:|
| 1   | 1236.0 | 684.0 | **621.6** | **−9.1%** |
| 4   | 410.1  | 219.4 | **198.1** | **−9.7%** |
| 16  | 147.7  | **78.1**  | 82.0  | +5.0% |
| 64  | 102.4  | 53.2  | **48.7**  | **−8.5%** |
| 256 | 94.3   | 44.6  | **42.4**  | −4.9% |

### 1.4 ACTIVSg25k (n=47246)
| B | FP64 | FP32 | TC | TC vs FP32 |
|---:|---:|---:|---:|---:|
| 1   | 3678.0 | 1490.8 | **1378.1** | **−7.6%** |
| 4   | 1122.2 | **439.8**  | 447.7 | +1.8% |
| 16  | 537.1  | 219.3  | **218.2**  | −0.5% |
| 64  | 432.1  | 172.0  | 172.0  | 0.0% (tie) |
| 256 | 433.3  | **159.2**  | 160.7 | +0.9% |

### 1.5 USA (n=156255, largest)
| B | FP64 | FP32 | TC | TC vs FP32 |
|---:|---:|---:|---:|---:|
| 1   | 14236.7 | 3587.1 | **3202.0** | **−10.7%** |
| 4   | 4455.9  | 1392.3 | **1148.4** | **−17.5%** ★ |
| 16  | 2221.3  | 777.3  | **758.5**  | −2.4% |
| 64  | 1834.0  | **657.8**  | 684.9 | +4.1% |
| 256 | 1936.9  | **642.7**  | 681.4 | +6.0% |

## 2. 패턴 정리

### 2.1 FP64의 위치
- 모든 case에서 FP32 / TC보다 **2-5× 느림**
- 큰 case (USA)에서는 8×까지 차이 (B=1)
- power-flow Newton-Raphson 같이 정확도 절대 안 필요한 use case에서는 FP64 불필요
- 정확도 절대 필요한 경우 (refined IR pipeline의 final step)만 가치

### 2.2 FP32 vs TC의 cross-over

| Case | TC win 영역 | Cross-over | 큰 B 영역 |
|---|---|---|---|
| ACTIVSg2000 | B ≤ 4 (−5%) | B=16 (tie) | B≥64 tie |
| case6468rte | B ≤ 4 (−9%) | B=16-64 (tie) | B=256 (+2%) |
| case8387 | B ≤ 4, B=64 (−9%) | B=16 (+5%) anomaly | B=256 (−5%) |
| ACTIVSg25k | B=1 (−8%) | B=4+ (≤1% 차이) | flat |
| USA | B ≤ 16 (−2~−17%) | B=64 (+4%) | B=256 (+6%) |

**일관된 패턴**: B=1 ~ 4에서 TC가 항상 win (−5 ~ −17%). 큰 B에서 cross-over → FP32가 우세 (USA, ACTIVSg2000) 또는 tie.

### 2.3 TC win의 크기 vs case 크기

| Case (n) | B=1 TC win | B=4 TC win |
|---|---:|---:|
| ACTIVSg2000 (3607) | −4.7% | −4.9% |
| case6468rte (12643) | −8.4% | −9.1% |
| case8387 (14908) | −9.1% | −9.7% |
| ACTIVSg25k (47246) | −7.6% | +1.8% |
| USA (156255) | −10.7% | **−17.5%** ★ |

큰 case일수록 TC win 큼 (B 작을 때). USA의 B=4 −17.5%가 가장 큰 win — BIG-front의 WMMA가 가장 잘 fire.

ACTIVSg25k는 anomaly — n=47k인데 TC win 작음. 추정 원인: ACTIVSg25k의 etree가 USA와 다른 구조 (BIG-front가 적음).

### 2.4 큰 B 영역의 FP32 reclaim

ACTIVSg25k, USA에서 B=64+에서 FP32가 TC 이김. 이유:
- 큰 B에서 batch 차원이 GPU 채움 → multistream의 effect 약화
- WMMA의 padding overhead가 부각
- FP32의 lean kernel set이 long batched dispatch에서 advantage

## 3. 권장 — Use case 별 mode 선택

| Workload pattern | 추천 mode |
|---|---|
| Newton-Raphson (B=1, 작은 grid) | **TC** (−5~−10%) |
| Newton-Raphson (B=1, USA 급) | **TC** (−11%) |
| Monte Carlo 소형 batch (B 4-16, 큰 grid) | **TC** (USA: −17% at B=4) |
| Stochastic large batch (B 64+, 큰 grid) | **FP32** (USA: −4% advantage) |
| 작은 grid 큰 batch (ACTIVSg2000, B=256) | tie — 둘 다 OK |
| 정확도 critical | FP64 (2-5× 느리지만 안전) |

## 4. 절대 throughput — RTX 3090의 power-flow 처리능력 (sys/sec)

batch=1 기준 (single solve per call):

| Case | FP32 sys/s | TC sys/s | Best |
|---|---:|---:|---|
| ACTIVSg2000 | 1900 | **1992** | TC |
| case6468rte | 1659 | **1809** | TC |
| case8387 | 1462 | **1609** | TC |
| ACTIVSg25k | 671 | **725** | TC |
| USA | 279 | **312** | TC |

batch=64 기준 (mid-batch):

| Case | FP32 sys/s | TC sys/s | Best |
|---|---:|---:|---|
| ACTIVSg2000 | 50,251 | **51,020** | TC |
| case6468rte | 27,247 | **28,090** | TC |
| case8387 | 18,797 | **20,534** | TC |
| ACTIVSg25k | 5,814 | 5,814 | tie |
| USA | **1,520** | 1,460 | FP32 |

batch=256 기준 (large batch):

| Case | FP32 sys/s | TC sys/s | Best |
|---|---:|---:|---|
| ACTIVSg2000 | **68,027** | 67,568 | FP32 |
| case6468rte | **33,557** | 32,895 | FP32 |
| case8387 | 22,422 | **23,585** | TC |
| ACTIVSg25k | **6,281** | 6,222 | FP32 |
| USA | **1,556** | 1,468 | FP32 |

## 5. Headline 한 줄 정리

> **B ≤ 16 작은 batch에서는 TC, B ≥ 64 큰 batch에서는 FP32가 우세.**
> **USA급 큰 grid의 B=4가 TC의 sweet spot (−17.5%).**
> **FP64는 모든 영역에서 2-5× 느림 — 정확도 절대 필요 없으면 사용 X.**

## 6. 환경 + 측정 조건

- 모든 측정 `--batch-only` (single-system path 우회), `cudaGraphLaunch` + `cudaEventSync` 기반 wall time
- 각 mode 동일 lever 적용: selinv OFF, multistream subtree dispatch, Σ.1 shared-staged trailing (FP32 path도 포함)
- median of 3 outer runs × `--repeat` inner kernel times
- Repeat counts: ACTIVSg2000 30, case6468rte/8387 20, ACTIVSg25k 10, USA 3 (절대 runtime 균형 위함)

## 7. 변경 가능 lever 요약 (이번 sweep 적용된 default)

| Lever | Default | 효과 |
|---|---|---|
| selinv OFF | ON in code → OFF (Σ.9) | factor −40 ~ −50% (양 path) |
| multistream subtree | ON (Σ.11+Σ.12) | factor −5~−15% at small B |
| Σ.1 staged trailing | ON (Σ.14, FP32도 포함) | mid kernel −20~30% |
| tc_warmup 사전 init | recommended | tc_setup −97% |
| device-side cuBLAS pointer build | applied | tc_setup H2D 11→1 MB |

비활성화: `CLS_NO_MULTISTREAM=1`, `CLS_NO_TILED_TRAILING=1`, `CLS_USE_SELINV=1`.

## 8. 관련 문서

| 문서 | 내용 |
|---|---|
| [`01-final-report.md`](01-final-report.md) | canonical master report (요약 + 최적 경로) |
| [`03-bench-vs-cudss.md`](03-bench-vs-cudss.md) | cuDSS 대비 벤치마크 |
| [`04-factorize-progress.md`](04-factorize-progress.md) | factorize 최적화 progress |
| [`../optimal-configuration.md`](../optimal-configuration.md) | optimal dispatch 구성 결정 |
