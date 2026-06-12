# 결과 요약 — 케이스별 (공정 비교: 같은 정밀도 cuDSS)

> **상태**: reference   **갱신**: 2026-06-11
> **한 줄**: custom 은 fp32/tf32(조류계산은 mixed). **공정하게 cuDSS 도 같은 정밀도**(선형계 = cuDSS-fp32, 조류계산 = cuDSS-mixed)로 비교. cuDSS-fp64 는 참조(정확 baseline)지만, cuDSS 자체도 fp32/mixed 로 fp64 대비 **1.8×(factor)·1.4×(NR)** 빨라지므로 fp64 비교는 custom 가속을 부풀린다. 원자료 [`06`](../05-reports/06-cudss-vs-custom-sweep-2026-06-10.md)·[`07`](../05-reports/07-cupf-backend-comparison-2026-06-11.md).

## 1. 선형계 단독 (J·x=b) — B=64, per-system (ms)

| case | n | cuDSS-fp32 | custom-fp32 | custom-tf32 | **factor 가속(fp32/tf32)** |
|---|---:|---:|---:|---:|---:|
| case3012wp | 5,725 | 0.0685 | 0.00819 | 0.00825 | **8.4× / 8.3×** |
| case6468rte | 12,643 | 0.1255 | 0.01802 | 0.01880 | 6.97× / 6.68× |
| case8387pegase | 14,908 | 0.1577 | 0.02415 | 0.02562 | 6.53× / 6.16× |
| case13659pegase | 23,225 | 0.2259 | 0.03570 | 0.03864 | 6.33× / 5.85× |
| case_ACTIVSg25k | 47,246 | 0.5361 | 0.09298 | 0.08451 | 5.77× / 6.34× |
| case_SyntheticUSA | 156,255 | 1.785 | 0.4096 | 0.4052 | 4.36× / 4.41× |

- **factorize 4.4–8.4×, solve 6.1–7.4×** vs **cuDSS-fp32**(같은 정밀도).
- (참조: cuDSS-fp64 factor 는 fp32 보다 **1.8× 느림** → fp64 비교 시 7.7–13.4× 로 부풀려짐.)
- 정확도(relres, B=64): custom fp32/tf32 ~1e-4–1e-5 (usa 1e-2), cuDSS-fp32 도 비슷한 자릿수.

## 2. cuPF 전체 조류계산 (end-to-end NR) — per-system (ms) + 수렴 반복수

baseline = **cuDSS-mixed**(custom 과 같은 mixed 정밀도). 둘 다 inexact-Newton(FP32 step + FP64 state), fp64 와 동일 수렴.

**B=1**

| case | cuDSS-mixed | custom-mix-fp32 | custom-mix-tf32 | **가속(fp32/tf32)** |
|---|---:|---:|---:|---:|
| case3012wp | 1.764 (4it) | 1.396 | 1.417 | 1.26× / 1.24× |
| case6468rte | 2.057 (4it) | 1.820 | 1.839 | 1.13× / 1.12× |
| case8387pegase | 2.901 (4it) | 2.071 | 2.063 | 1.40× / 1.41× |
| case13659pegase | 6.749 (7it) | 4.492 | 4.393 | 1.50× / 1.54× |
| case_ACTIVSg25k | 6.553 (5it) | 5.222 | 4.686 | 1.25× / 1.40× |
| case_SyntheticUSA | 26.231 (7it) | 21.138 | 18.566 | 1.24× / 1.41× |

**B=64**

| case | cuDSS-mixed | custom-mix-fp32 | custom-mix-tf32 | **가속(fp32/tf32)** |
|---|---:|---:|---:|---:|
| case3012wp | 0.342 | 0.0765 | 0.0786 | **4.48× / 4.35×** |
| case6468rte | 0.570 | 0.156 | 0.144 | 3.65× / 3.96× |
| case8387pegase | 0.774 | 0.202 | 0.199 | 3.84× / 3.90× |
| case13659pegase | 2.114 | 0.488 | 0.491 | 4.33× / 4.30× |
| case_ACTIVSg25k | 3.318 | 0.812 | 0.797 | 4.08× / 4.16× |
| case_SyntheticUSA | 17.714 | 4.990 | 4.916 | 3.55× / 3.60× |

- **조류계산 가속: B=64 custom-mixed 3.6–4.5× / B=1 1.1–1.5×** vs **cuDSS-mixed**(같은 정밀도).
- (참조: cuDSS-mixed 자체가 cuDSS-fp64 보다 **1.4× 빠름** → fp64 비교 시 4.9–7.0× 로 부풀려짐.)
- **수렴 동일**: mixed 는 cuDSS·custom 모두 fp64 와 같은 반복수(4–8it). raw fp32/tf32 는 1e-8 미수렴.

## 핵심 (공정 비교)

- **선형계 단독 (vs cuDSS-fp32)**: custom factorize **4.4–8.4×** / solve 6.1–7.4×.
- **전체 조류계산 (vs cuDSS-mixed)**: custom-mixed **B=64 3.6–4.5× / B=1 1.1–1.5×** (선형해가 NR 의 일부라 배율↓).
- **tf32 ≈ fp32** — factorize tf32 이득(1.05–1.2×)이 전체 NR 에선 희석 (근거: [`small-tier-no-tensorcore.md`](small-tier-no-tensorcore.md)·[`factorize-bottleneck-ncu.md`](factorize-bottleneck-ncu.md)).
- **불공정 fp64 비교와의 차이**: cuDSS 도 fp32/mixed 로 1.4–1.8× 빨라지므로, custom 의 *진짜* 이득은 같은 정밀도 비교(위)가 정답. fp64 대비 수치(선형계 7.7–13.4×, 조류계산 4.9–7.0×)는 cuDSS 의 정밀도 하향 이득이 섞인 것.
