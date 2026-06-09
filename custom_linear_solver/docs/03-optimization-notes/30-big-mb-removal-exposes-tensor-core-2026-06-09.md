# Removing big multi-block (MB) exposes tensor-core acceleration

**작성일**: 2026-06-09
**범위**: `CLS_NO_BIG_MB`(big underfill 의 multi-block 경로 제거)가 tf32/fp16(TC)을 fp32 대비
이기게 만드는 측정. **이번 주 랩미팅은 no-MB 구성으로 TC 가속을 제시.** MB 포함 전체 그림(아래
caveat)은 교수 보고용으로 보존.

---

## 1. 왜 기본 구성에선 TC 가 안 보였나

big tier 의 underfill 레벨(few large fronts on a near-idle GPU)은 **MB 경로**(panel /
multi-block trailing / extend 3-커널)로 trailing 을 SM 에 fan-out 한다. 그런데:
- fp32 underfill → MB(scalar). tf32 **severe-underfill → MB(scalar)** — 즉 **tf32 가 underfill
  에서 TC 를 아예 안 쓴다**(scalar MB 로 빠짐).
- 그래서 B=1(전부 underfill) 비교는 "fp32-MB vs tf32-MB(scalar)+일부 TC" 라 TC 가 안 드러났다.

## 2. CLS_NO_BIG_MB: 모든 big 을 fused 단일 커널로

`big_underfill = false`(+ `tf32_severe_underfill = false`)로 강제 → **모든 big front 이 fused
단일 커널**(`factor_big_staged`(fp32) / `factor_big_tf32_ptx`(tf32) / `factor_big_fp16_ptx`)로,
1 block/front. 이제 **tf32/fp16 이 모든 big 에서 TC trailing** 을 쓴다.

## 3. 측정 (factor_per_sys_ms, **factorize만**, LDB bank-conflict fix 포함)

| case | B | fp32(MB) | fp32(noMB) | tf32(noMB) | fp16(noMB) |
|---|---|---|---|---|---|
| 70K | 1 | **1.98** | 2.39 | 2.31 | 2.25 |
| 70K | 64 | 0.409 | 0.419 | 0.394 | **0.390** |
| USA | 1 | **2.08** | 2.73 | 2.42 | 2.39 |
| USA | 64 | 0.471 | 0.467 | 0.457 | **0.463** |

**같은 no-MB 구성 안에서 (apples-to-apples):**
| | tf32/fp32(noMB) | fp16/fp32(noMB) |
|---|---|---|
| 70K B=1 | −3.3% | −5.8% |
| 70K B=64 | −5.8% | −6.9% |
| USA B=1 | **−11.3%** | −12.5% |
| USA B=64 | −2.2% | −0.9% |

→ **no-MB 구성에선 TC(tf32/fp16)가 fp32 를 일관되게 이긴다** (−1~−12%). fp16 ≈ tf32 (fp16 살짝 빠름).
이유: 모든 big 이 1-block fused 커널이라, 거기서 **TC trailing 이 scalar trailing 을 이긴다.**

## 4. 정직한 caveat (교수 보고용 — 빠뜨리면 안 됨)

- TC 가 이기는 건 **no-MB 끼리** 비교일 때. **진짜 default 인 fp32(MB)** 와 비교하면:
  - **B=1: fp32(MB)가 최速** (1.98/2.08). tf32(noMB)는 +16% 더 느림 → **TC 가 진다.**
  - B=64: tf32(noMB)가 fp32(MB)를 −3~−4% 이김 (marginal).
- 즉 **no-MB 는 절대적으로 느린 구성**(MB 가 fp32·tf32 둘 다 도와줌, 특히 B=1). TC 의 상대 우위는
  "MB 라는 fp32 의 무기를 뺀" 핸디캡 비교에서 나온다.
- 근본 원인(thin-K memory-bound trailing, doc 28)은 불변 — no-MB 는 TC 를 *보이게* 만들 뿐
  *근본적으로 빠르게* 만들진 않는다.

## 5. 결정

- **이번 주 랩미팅**: no-MB 구성(`CLS_NO_BIG_MB`)으로 "TC 가 fp32 대비 −1~−12% 가속" 제시.
- **교수 보고**: MB 포함 전체 표 + caveat (MB 가 B=1 에서 더 빠름, TC 우위는 no-MB 한정).
- 코드: `CLS_NO_BIG_MB` 토글로 전환 (default 변경 안 함). LDB bank-conflict fix(commit a12f059)는
  baked-in.
