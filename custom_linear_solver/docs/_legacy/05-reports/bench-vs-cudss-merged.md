# cuDSS vs Custom — 통합 벤치 (raw B=1 · 공정 ubatch · 전체 NR)

> **상태**: reference (통합본)   **갱신**: 2026-06-18
> **한 줄**: cuDSS 대비 custom solver 비교를 세 측정 setup으로 통합 — (A) raw B=1 single-system은 5.8~44.7×(cuDSS batching 없어 **과대평가**), (B) 공정한 ubatch+mt-auto는 B=256 tf32 기준 **3.60~10.16×**, (C) 전체 NR 조류계산(graph off)에서 mixed custom이 cuDSS 대비 **~4~6×**.

이 문서는 서로 겹치는 세 개의 cuDSS 비교 리포트를 하나로 합친 것이다. **세 setup의 구분이 핵심**이므로 각각의 headline / accuracy / phase·operator breakdown / TSV 데이터 포인터를 분리해 보존한다.

- 원본 리포트: `03-bench-vs-cudss.md`(setup A·B), `06-cudss-vs-custom-sweep-2026-06-10.md`(setup C-linear), `07-cupf-backend-comparison-2026-06-11.md`(setup C-NR). 본 통합본이 이 세 문서를 대체한다.
- 공통 하드웨어: RTX 3090 (sm_86), CUDA 12.8, `Release` build, 클럭 고정.

## 0. 세 측정 setup 이 왜 따로 존재하는가

| setup | 무엇을 재나 | cuDSS 호출 방식 | 공정성 | headline (custom vs cuDSS) |
|---|---|---|---|---|
| **A. raw B=1 single-system** | factor+solve per-system, 선형해만 | single-system, B=1 1회 (native uniform-batching 없음) | **과대평가** — cuDSS가 system마다 full single-system launch 비용 지불 | 5.78× ~ 44.7× (B=256, tf32 vs cuDSS-fp32) |
| **B. ubatch + mt-auto** | analyze/factorize/solve per-system, 선형해만 | `CUDSS_CONFIG_UBATCH_SIZE=B`, batch-major value buffer, `cudssSetThreadingLayer(...gomp.so)` (cuPF production 패턴) | **공정** — cuDSS도 ubatch로 amortise + mt-auto | **3.60× ~ 10.16×** (B=256, tf32) — **방어 가능한 숫자** |
| **C. 전체 NR (graph off)** | 수렴 반복수 + init + solve/it + per-operator | ubatch + mt-auto, 정식 `NewtonOptions::custom` | **공정**, end-to-end | mixed custom ~4~6× (배치, 전 case) |

setup A와 B는 **같은 선형해 측정에서 cuDSS 호출 방식만 다르다** (custom 측은 동일). A는 cuDSS가 batching을 전혀 못 하는 경우로, 작은 case에서 45×까지 부풀려진다 — **인용 시 반드시 over-estimate임을 명시한다**. B가 방어 가능한 공정 비교다. C는 선형해를 넘어 전체 Newton-Raphson 조류계산에 묻어 측정한 것으로, factorize의 tf32 텐서코어 이득이 tri-solve·Jacobian 등에 희석되는 실제 효과까지 보여준다.

비교 solver / precision:
- **cuDSS fp32 / fp64** (`CUDA_R_32F` / `R_64F`); setup C는 `cudss-mixed`(FP32 step + FP64 state)도 포함.
- **custom fp32** (`Precision::FP32` — staged-scalar trailing)
- **custom fp16** (FP16 WMMA m16n16k16 trailing, FP32 accumulate) — setup A·B만
- **custom tf32** (V9h PTX `mma.m16n8k8/k4` hybrid + `__launch_bounds__(512,2)`; setup B·C는 Ozaki first-order TF32)

---

## TL;DR — case × setup → headline custom-vs-cuDSS factor

각 setup의 best-of-precision custom을 동일 case에 대해 비교한 headline 배수. (A·B = factor+solve per-system, B=256, custom-tf32 / cuDSS-fp32; C = solve per-system, B=256 best-mixed custom / cuDSS-fp64.)

| case | n | A. raw B=1 (과대평가) | B. ubatch+mt-auto (공정) | C. 전체 NR (B=256, vs cuDSS-fp64) |
|---|--:|--:|--:|--:|
| case1197 | 2,392 | 44.7× | 10.16× | — |
| case_ACTIVSg2000 | 3,607 | 35.5× | 7.96× | — |
| case3012wp (3xxx) | 5,725 | 28.7× | 6.96× | ~9.1× (501/55) |
| case6468rte (6xxx) | 12,643 | 17.6× | 5.94× | ~6.7× (840/126) |
| case8387pegase (8xxx) | 14,908 | 18.6× | 5.03× | ~5.1× (1167/229) |
| case9241pegase | 17,036 | — | 5.01× | — |
| case_ACTIVSg10k | 18,544 | — | 5.47× | — |
| case13659pegase (13K) | 23,225 | — | 4.73× | ~5.1× (2743/536) |
| case_ACTIVSg25k (25K) | 47,246 | 10.0× | 4.57× | ~5.3× (4996/936) |
| case_ACTIVSg70k | 134,104 | — | 3.60× | — |
| case_SyntheticUSA (usa) | 156,255 | 5.78× | 3.66× | ~5.2× (24289/4714) |

> A열은 cuDSS가 batching을 못 받는 over-estimate, **B열이 방어 가능한 공정 숫자**, C열은 전체 NR에서의 end-to-end 비율. 작은 case일수록 A↔B 격차가 큼(cuDSS per-call setup이 batching으로 줄지 않기 때문).

---

## Setup A — raw B=1 single-system cuDSS (과대평가)

cuDSS는 native uniform-batching semantics가 없어 B=1에서 1회만 측정한다. per-system time은 더 큰 B에서도 동일(각 system이 별도 call). **이 setup의 speedup은 over-estimate** — cuDSS가 batching 이점을 전혀 못 받기 때문이다. Setup B와 custom 측은 동일하고, cuDSS 호출 방식만 다르다.

데이터: [`04-bench-vs-cudss-2026-06-07/sweep_results.tsv`](04-bench-vs-cudss-2026-06-07/sweep_results.tsv) (113 rows). Median of 10, `--batch-only`, `CLS_INTERNAL_GRAPH=ON`, `use_multistream_subtrees=true`.

### A.1 Per-system factor+solve time (ms), median of 10

Format `factor+solve (per-system)`. cuDSS는 B=1 1회 측정값을 모든 B에 동일 적용.

| case | n | solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|--:|---|---:|---:|---:|---:|---:|
| case1197 | 2,392 | cuDSS/fp32 | **0.192** | – | – | – | – |
| | | custom/fp32 | 0.147 | 0.039 | 0.0076 | 0.0058 | **0.0043** |
| | | custom/fp16 | 0.147 | 0.039 | 0.0076 | 0.0058 | **0.0043** |
| | | custom/tf32 | 0.146 | 0.039 | 0.0076 | 0.0058 | **0.0043** |
| case_ACTIVSg2000 | 3,607 | cuDSS/fp32 | **0.510** | – | – | – | – |
| | | custom/fp32 | 0.471 | 0.126 | 0.027 | 0.019 | **0.0144** |
| | | custom/fp16 | 0.552 | 0.147 | 0.029 | 0.021 | 0.0154 |
| | | custom/tf32 | 0.545 | 0.145 | 0.028 | 0.020 | **0.0144** |
| case3012wp | 5,725 | cuDSS/fp32 | **0.400** | – | – | – | – |
| | | custom/fp32 | 0.416 | 0.101 | 0.024 | 0.018 | **0.0137** |
| | | custom/fp16 | 0.482 | 0.126 | 0.027 | 0.018 | 0.0136 |
| | | custom/tf32 | 0.489 | 0.123 | 0.026 | 0.019 | 0.0139 |
| case6468rte | 12,643 | cuDSS/fp32 | **0.528** | – | – | – | – |
| | | custom/fp32 | 0.479 | 0.161 | 0.042 | 0.035 | **0.0297** |
| | | custom/fp16 | 0.621 | 0.190 | 0.046 | 0.037 | 0.0305 |
| | | custom/tf32 | 0.577 | 0.186 | 0.045 | 0.036 | 0.0300 |
| case8387pegase | 14,908 | cuDSS/fp32 | **0.772** | – | – | – | – |
| | | custom/fp32 | 0.617 | 0.182 | 0.058 | 0.047 | **0.0416** |
| | | custom/fp16 | 0.712 | 0.230 | 0.060 | 0.054 | 0.0430 |
| | | custom/tf32 | 0.731 | 0.196 | 0.058 | 0.052 | 0.0414 |
| case_ACTIVSg25k | 47,246 | cuDSS/fp32 | **1.541** | – | – | – | – |
| | | custom/fp32 | 1.327 | 0.448 | 0.182 | 0.173 | 0.159 |
| | | custom/fp16 | 1.300 | 0.441 | 0.184 | 0.167 | 0.168 |
| | | custom/tf32 | 1.325 | **0.397** | **0.171** | **0.162** | **0.154** |
| case_SyntheticUSA | 156,255 | cuDSS/fp32 | **3.742** | – | – | – | – |
| | | custom/fp32 | 3.582 | 1.222 | 0.701 | 0.648 | 0.640 |
| | | custom/fp16 | 3.538 | 1.213 | 0.691 | 0.688 | 0.646 |
| | | custom/tf32 | **3.054** | 1.217 | **0.681** | **0.622** | 0.647 |

### A.2 Per-system speedup of custom/tf32 over cuDSS-fp32 (over-estimate)

| case | cuDSS f+s (ms) | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|---:|
| case1197 | 0.192 | 1.31× | 4.90× | 25.2× | 32.8× | **44.7×** |
| case_ACTIVSg2000 | 0.510 | 0.93× | 3.52× | 18.1× | 25.7× | **35.5×** |
| case3012wp | 0.400 | 0.82× | 3.24× | 15.3× | 21.6× | **28.7×** |
| case6468rte | 0.528 | 0.91× | 2.84× | 11.7× | 14.9× | **17.6×** |
| case8387pegase | 0.772 | 1.06× | 3.94× | 13.2× | 15.0× | **18.6×** |
| case_ACTIVSg25k | 1.541 | 1.16× | 3.89× | 9.0× | 9.5× | **10.0×** |
| case_SyntheticUSA | 3.742 | 1.23× | 3.07× | 5.49× | 6.02× | 5.78× |

**주의 (핵심 caveat)**: B=256의 45×/35×/29×(작은 case)는 cuDSS가 system마다 full single-system kernel-launch + analyze-fixed 비용을 지불하는 반면 custom은 하나의 dispatch sequence를 B개 system이 공유하기 때문이다. cuDSS에 batching이 없는 setup이라 **over-estimate**다. n ≳ 50k에서는 per-system work 자체가 dominant → B amortisation이 6~10×로 saturate. 인용 시 setup B의 공정 숫자(3.6~10.2×)를 쓸 것.

### A.3 Accuracy — max relative residual (모든 측정 B)

| case | cuDSS/fp32 | custom/fp32 | custom/fp16 | custom/tf32 |
|---|---:|---:|---:|---:|
| case1197 | 7.1e-05 | 2.1e-04 | 2.1e-04 | 2.1e-04 |
| case_ACTIVSg2000 | 1.3e-05 | 1.5e-05 | 2.8e-03 | 1.5e-02 |
| case3012wp | 1.6e-04 | 2.5e-04 | 4.7e-03 | 3.2e-02 |
| case6468rte | 4.2e-05 | 1.3e-04 | 1.1e-03 | 2.8e-03 |
| case8387pegase | 1.3e-05 | 3.5e-05 | 2.1e-02 | 5.0e-02 |
| case_ACTIVSg25k | 1.5e-04 | 2.7e-04 | 8.4e-02 | 5.4e-02 |
| case_SyntheticUSA | 8.9e-04 | 2.4e-02 | 1.2e-01 | 5.6e-02 |

- **fp32**(cuDSS & custom)은 모든 case ≲ 1e-3 — NR Jacobian residual로 acceptable.
- **fp16/tf32**는 큰 case에서 ~1e-1..1e-2 (trailing-GEMM rounding 누적). Newton loop + iterative refinement 안에서 유용, single-shot FP32 drop-in 대체는 아님.

### A.4 Multi-stream overlap (case8387 B=1 fp32)

NVTX 빌드(`CLS_ENABLE_NVTX=ON`), `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`. profile: [`04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_on.nsys-rep`](04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_on.nsys-rep), [`..._off.nsys-rep`](04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_off.nsys-rep).

**Multi-stream ON** (default) — 9 streams: main(13) 990 kernels/6.67 ms, subtree streams(23~29) 110~180 kernels/0.75~2.03 ms each, runtime(7) 0.36 ms. `factor_*` total 1,340 instances / 13.76 ms.
**Multi-stream OFF** — 2 streams: main(13) 930 kernels/5.75 ms, runtime(7) 0.36 ms. `factor_*` total 300 instances / 3.21 ms.

**해석**: 스트림은 실제로 사용되나 **B=1에서는 wall이 개선되지 않는다** (ON factor 0.340/solve 0.262 vs OFF factor 0.322/solve 0.245 ms — OFF가 ~5% 빠름). case8387은 spine(etree 최상단 chain)이 factor wall을 dominant하고 spine은 main stream에서만 돈다; etree leaf subtree work는 ~2 ms뿐. B=1에선 SM occupancy가 낮아 split해도 event/wait overhead만 추가. win은 B≥4에서 (각 subtree stream이 충분한 launch로 dispatch overhead를 숨김).

---

## Setup B — ubatch + mt-auto cuDSS (공정한 비교)

cuDSS를 `CUDSS_CONFIG_UBATCH_SIZE=B`로 실제 uniform-batching하고 batch-major value buffer(B × nnz)를 cuPF production과 동일하게 먹인다. `cudssSetThreadingLayer(handle, "...gomp.so")` mt-auto 적용. **이쪽이 방어 가능한 공정 speedup**이다.

데이터: [`05-bench-vs-cudss-ubatch-2026-06-07/sweep_v2.tsv`](05-bench-vs-cudss-ubatch-2026-06-07/sweep_v2.tsv) (220 rows: 11 cases × 4 paths × 5 batch). 추가 cases(case9241pegase, case_ACTIVSg10k, case13659pegase, case_ACTIVSg70k) 포함.

### B.1 Analyze (ms, 1회성)

| case | cuDSS ubatch+mt-auto / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 17.5 | **3.2** | 3.3 | 3.4 |
| case_ACTIVSg2000 | 95.0 | **7.4** | 7.4 | 7.4 |
| case3012wp | 27.6 | **8.8** | 8.7 | 9.0 |
| case6468rte | 57.2 | **16.6** | 17.5 | 17.3 |
| case8387pegase | 30.6 | **20.2** | 21.0 | 20.9 |
| case9241pegase | 32.3 | **22.7** | 21.6 | 22.7 |
| case_ACTIVSg10k | 33.2 | 23.3 | **22.7** | 23.1 |
| case13659pegase | **37.5** | 39.2 | 39.7 | 40.0 |
| case_ACTIVSg25k | 57.0 | **55.6** | 57.9 | 56.3 |
| case_ACTIVSg70k | **127.3** | 155.3 | 155.6 | 156.0 |
| case_SyntheticUSA | **149.8** | 178.0 | 183.2 | 179.3 |

custom analyze는 ~25K까지 더 빠르고, 최대 case(70K, USA)에서 cuDSS가 약간 우세. analyze는 많은 Newton iteration에 amortise되어 warm해지면 gap이 거의 무의미.

### B.2 Factorize per system (ms), B=1

| case | cuDSS / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.103 | **0.069** | 0.070 | 0.070 |
| case_ACTIVSg2000 | 0.345 | **0.274** | 0.355 | 0.349 |
| case3012wp | 0.253 | **0.233** | 0.300 | 0.300 |
| case6468rte | 0.338 | **0.274** | 0.451 | 0.427 |
| case8387pegase | 0.540 | **0.353** | 0.444 | 0.418 |
| case9241pegase | 0.564 | **0.385** | 0.480 | 0.431 |
| case_ACTIVSg10k | 0.467 | **0.370** | 0.459 | 0.431 |
| case13659pegase | 0.659 | **0.396** | 0.490 | 0.482 |
| case_ACTIVSg25k | 1.095 | **0.813** | 0.816 | 0.808 |
| case_ACTIVSg70k | 2.558 | 2.796 | **2.073** | 2.161 |
| case_SyntheticUSA | 2.792 | 2.622 | **2.166** | 2.490 |

custom/fp32가 B=1에서 cuDSS를 9/11 cases에서 이김. fp16/tf32는 두 최대 case에서 catch up (trailing-GEMM share가 충분히 커 WMMA/PTX 이득).

### B.3 Solve per system (ms), B=1

| case | cuDSS / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | **0.087** | 0.077 | 0.078 | 0.078 |
| case_ACTIVSg2000 | **0.166** | 0.197 | 0.197 | 0.197 |
| case3012wp | **0.147** | 0.189 | 0.180 | 0.161 |
| case6468rte | **0.187** | 0.214 | 0.233 | 0.236 |
| case8387pegase | **0.230** | 0.242 | 0.246 | 0.234 |
| case9241pegase | **0.229** | 0.257 | 0.267 | 0.255 |
| case_ACTIVSg10k | **0.237** | 0.251 | 0.256 | 0.248 |
| case13659pegase | **0.271** | 0.273 | 0.257 | 0.264 |
| case_ACTIVSg25k | **0.450** | 0.499 | 0.535 | 0.528 |
| case_ACTIVSg70k | **0.866** | 0.891 | 0.914 | 0.935 |
| case_SyntheticUSA | **0.957** | 1.017 | 0.947 | 1.043 |

cuDSS가 single-system solve를 소유(triangular-solve kernel이 B=1 throughput에 튜닝됨). custom의 `solve_small_warp`/`solve_level`은 batch amortisation에 최적화. gap은 작지만(~10K 이상 10~20% 이내) 일관됨.

### B.4 Relative residual at B=1

| case | cuDSS/fp32 | custom/fp32 | custom/fp16 | custom/tf32 |
|---|---:|---:|---:|---:|
| case1197 | 7.8e-05 | 1.6e-04 | 1.8e-04 | 1.7e-04 |
| case_ACTIVSg2000 | 1.4e-05 | 1.3e-05 | 2.8e-03 | 1.5e-02 |
| case3012wp | 1.9e-04 | 1.9e-04 | 4.1e-04 | 1.8e-04 |
| case6468rte | 4.6e-05 | 4.1e-05 | 4.9e-04 | 5.7e-03 |
| case8387pegase | 1.2e-05 | 2.0e-05 | 6.5e-03 | 3.6e-02 |
| case9241pegase | 2.7e-06 | 2.8e-06 | 6.7e-04 | 8.2e-04 |
| case_ACTIVSg10k | 6.3e-04 | 2.8e-04 | 1.2e-02 | 5.0e-02 |
| case13659pegase | 1.1e-04 | 1.1e-04 | 5.0e-02 | 1.1e-02 |
| case_ACTIVSg25k | 1.4e-04 | 2.0e-04 | 1.3e-02 | 7.4e-02 |
| case_ACTIVSg70k | 1.2e-03 | 3.4e-03 | 7.6e-02 | 5.5e-02 |
| case_SyntheticUSA | 9.2e-04 | 1.2e-03 | 6.1e-02 | 5.4e-02 |

custom/fp32 ≈ cuDSS/fp32 (동일 precision). fp16/tf32는 1~2 order 뒤짐 — iterative refinement 안에서만 적합.

### B.5 Multi batch — factor + solve per system (ms)

각 (case, B)는 factor+solve per-system wall, median of 10. cuDSS는 `UBATCH_SIZE=B`, batch-major value buffer.

| case | B | cuDSS ubatch | custom/fp32 | custom/fp16 | custom/tf32 |
|---|--:|---:|---:|---:|---:|
| case1197 | 4 | 0.0748 | **0.0387** | 0.0388 | 0.0386 |
| | 32 | 0.0448 | **0.0077** | 0.0076 | 0.0077 |
| | 64 | 0.0447 | **0.0059** | 0.0058 | 0.0058 |
| | 256 | 0.0436 | **0.0043** | 0.0043 | 0.0043 |
| case_ACTIVSg2000 | 4 | 0.200 | **0.126** | 0.146 | 0.145 |
| | 32 | 0.125 | **0.0265** | 0.0285 | 0.0282 |
| | 64 | 0.119 | **0.0191** | 0.0205 | 0.0197 |
| | 256 | 0.114 | 0.0144 | 0.0153 | **0.0144** |
| case3012wp | 4 | 0.163 | **0.106** | 0.123 | 0.117 |
| | 32 | 0.104 | **0.0248** | 0.0251 | 0.0243 |
| | 64 | 0.099 | 0.0178 | 0.0194 | 0.0188 |
| | 256 | 0.096 | 0.0139 | **0.0138** | 0.0138 |
| case6468rte | 4 | 0.251 | **0.159** | 0.180 | 0.177 |
| | 32 | 0.185 | **0.0430** | 0.0457 | 0.0463 |
| | 64 | 0.179 | **0.0342** | 0.0366 | 0.0367 |
| | 256 | 0.175 | 0.0301 | 0.0313 | **0.0295** |
| case8387pegase | 4 | 0.344 | **0.187** | 0.211 | 0.200 |
| | 32 | 0.229 | 0.0644 | 0.0580 | **0.0586** |
| | 64 | 0.220 | **0.0480** | 0.0503 | 0.0525 |
| | 256 | 0.213 | **0.0423** | 0.0426 | 0.0424 |
| case9241pegase | 4 | 0.374 | **0.197** | 0.221 | 0.224 |
| | 32 | 0.253 | **0.0617** | 0.0682 | 0.0653 |
| | 64 | 0.243 | 0.0558 | **0.0543** | 0.0565 |
| | 256 | 0.237 | 0.0483 | 0.0604 | **0.0473** |
| case_ACTIVSg10k | 4 | 0.387 | **0.197** | 0.227 | 0.233 |
| | 32 | 0.294 | **0.0656** | 0.0699 | 0.0656 |
| | 64 | 0.288 | 0.0563 | 0.0618 | **0.0572** |
| | 256 | 0.283 | **0.0512** | 0.0543 | 0.0517 |
| case13659pegase | 4 | 0.465 | **0.218** | 0.243 | 0.261 |
| | 32 | 0.329 | **0.0865** | 0.0823 | 0.0843 |
| | 64 | 0.319 | 0.0732 | 0.0753 | **0.0726** |
| | 256 | 0.311 | 0.0705 | 0.0766 | **0.0658** |
| case_ACTIVSg25k | 4 | 0.905 | 0.428 | 0.451 | **0.420** |
| | 32 | 0.731 | 0.178 | 0.183 | **0.172** |
| | 64 | 0.718 | 0.173 | 0.178 | **0.159** |
| | 256 | 0.707 | 0.160 | 0.167 | **0.155** |
| case_ACTIVSg70k | 4 | 2.377 | 1.204 | 1.082 | **1.022** |
| | 32 | 2.093 | 0.622 | 0.616 | **0.572** |
| | 64 | 2.071 | **0.564** | 0.579 | 0.580 |
| | 256 | 2.056 | **0.548** | 0.576 | 0.571 |
| case_SyntheticUSA | 4 | 2.681 | 1.348 | **1.221** | 1.286 |
| | 32 | 2.395 | 0.670 | 0.688 | **0.675** |
| | 64 | 2.373 | 0.663 | 0.664 | **0.656** |
| | 256 | 2.357 | 0.650 | 0.652 | **0.644** |

### B.6 custom/tf32 speedup over cuDSS ubatch+mt-auto / fp32 (공정)

(factor+solve per system; `>1.0×` = custom faster)

| case | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| case1197 | 1.29× | 1.94× | 5.83× | 7.64× | **10.16×** |
| case_ACTIVSg2000 | 0.94× | 1.38× | 4.43× | 6.03× | **7.96×** |
| case3012wp | 0.87× | 1.39× | 4.26× | 5.27× | **6.96×** |
| case6468rte | 0.79× | 1.42× | 3.99× | 4.89× | **5.94×** |
| case8387pegase | 1.18× | 1.73× | 3.91× | 4.19× | **5.03×** |
| case9241pegase | 1.16× | 1.67× | 3.87× | 4.31× | **5.01×** |
| case_ACTIVSg10k | 1.04× | 1.66× | 4.49× | 5.03× | **5.47×** |
| case13659pegase | 1.25× | 1.78× | 3.90× | 4.39× | **4.73×** |
| case_ACTIVSg25k | 1.16× | 2.15× | 4.24× | 4.51× | **4.57×** |
| case_ACTIVSg70k | 1.11× | 2.33× | 3.66× | 3.57× | **3.60×** |
| case_SyntheticUSA | 1.06× | 2.09× | 3.55× | 3.62× | **3.66×** |

공정한 B=256 tf32 speedup 범위 = **3.60× (USA) ~ 10.16× (case1197)**. Setup A의 5.78~44.7×는 cuDSS batching이 없어서 나온 over-estimate임을 다시 강조한다. cuDSS도 `UBATCH_SIZE`로 amortise되지만(B=256 per-system이 B=1의 ~25%), custom의 dispatch가 SM utilization을 더 saturate한다. 최대 case는 ~3.5~3.7×로 saturate (custom wall이 per-batch sequential spine에 dominate), 최소 case는 최고 비율(cuDSS per-call setup이 ubatch로 안 줄어듦).

---

## Setup C — 전체 NR 조류계산 (graph off): linear-only 스윕 + 전체 cuPF

setup C는 선형해를 넘어 **전체 Newton-Raphson 조류계산**에 묻어 측정한다. 두 하위 측정으로 구성:
- **C-linear** (2026-06-10): cuDSS(fp64/fp32) vs custom(fp64/fp32/tf32) linear-only 스윕, phase별(analyze/factorize/solve), 6 case × B{1,4,16,64,256}. 원본: `06-cudss-vs-custom-sweep-2026-06-10.md`.
- **C-NR** (2026-06-11): 전체 cuPF NR(graph off), fp64/mixed, 결정적(Ozaki TF32 + serial-ND seed 1588 + 클럭 고정 + 통제 배치 scale_step=0), 6 case × B{1,16,64,256}, 수렴 iters + init + solve/it + per-operator. 원본: `07-cupf-backend-comparison-2026-06-11.md`.

공통 결정성 설정(두 측정 일치): **Ozaki TF32 빌드 + serial-ND(seed 1588) + GPU 클럭 고정 + 통제 배치**. 기본 parallel-ND는 비결정적이라 fp32↔tf32 비교가 무효가 된다.

데이터: [`06-cudss-vs-custom-sweep-2026-06-10/sweep.tsv`](06-cudss-vs-custom-sweep-2026-06-10/sweep.tsv) (C-linear), [`07-cupf-backend-comparison-2026-06-11/b1_init_solve_perop.tsv`](07-cupf-backend-comparison-2026-06-11/b1_init_solve_perop.tsv) · [`batch_scaling.tsv`](07-cupf-backend-comparison-2026-06-11/batch_scaling.tsv) · [`operator_ms_full.tsv`](07-cupf-backend-comparison-2026-06-11/operator_ms_full.tsv) (C-NR).

> **참고 — analyze 수치 불일치**: C-linear(§C.1, linear-only, repeat 21)와 C-NR(§C.5, 전체 cuPF init, repeat 11~21)의 analyze/init은 측정 하네스가 달라 값이 다르다. 예: case3012wp custom-tf32 analyze는 C-linear 14.0 ms(2026-06-10) vs C-NR init 13.3 ms(2026-06-11); cuDSS-fp64는 75.2 ms vs 43.2 ms. 두 측정 모두 보존한다.

### C-linear — phase별 per-system 벽시계 (6 case × B 5단계)

cuDSS는 ubatch + mt-auto, repeat 21 median. custom은 serial-ND(seed 1588) + Ozaki first-order, warmup 5 / repeat 21, `--single-precision fp64`. solve는 04-solve-optimization 통합 후 재측정(factorize/analyze 불변).

#### C.1 ANALYZE (ms, 1회성)

| config | 3xxx | 6xxx | 8xxx | 13K | 25K | usa |
|---|---|---|---|---|---|---|
| `cudss-fp64` | 75.2 | 116 | 118 | 105 | 134 | 326 |
| `cudss-fp32` | 107 | 107 | 95.7 | 80.3 | 166 | 330 |
| `custom-fp64` | 14.1 | 44.9 | 51.8 | 89.3 | 165 | 628 |
| `custom-fp32` | 23 | 45.9 | 51.7 | 87.9 | 164 | 622 |
| `custom-tf32` | 14 | 52.2 | 51.9 | 93.1 | 162 | 619 |

#### C.2 FACTORIZE (per-system ms)

| case | config | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|---|
| 3xxx | cudss-fp64 | 0.384 | 0.17 | 0.123 | 0.11 | 0.107 |
| | cudss-fp32 | 0.26 | 0.109 | 0.0772 | 0.0685 | 0.0662 |
| | custom-fp64 | 0.283 | 0.0802 | 0.0288 | 0.015 | 0.012 |
| | custom-fp32 | 0.209 | 0.0493 | 0.017 | 0.00819 | 0.0061 |
| | custom-tf32 | 0.186 | 0.0493 | 0.0169 | 0.00825 | 0.00644 |
| 6xxx | cudss-fp64 | 0.536 | 0.281 | 0.218 | 0.202 | 0.198 |
| | cudss-fp32 | 0.345 | 0.174 | 0.136 | 0.126 | 0.123 |
| | custom-fp64 | 0.542 | 0.158 | 0.0594 | 0.0359 | 0.0321 |
| | custom-fp32 | 0.308 | 0.0892 | 0.0319 | 0.018 | 0.015 |
| | custom-tf32 | 0.301 | 0.0883 | 0.0328 | 0.0188 | 0.016 |
| 8xxx | cudss-fp64 | 0.87 | 0.399 | 0.283 | 0.253 | 0.246 |
| | cudss-fp32 | 0.546 | 0.249 | 0.177 | 0.158 | 0.153 |
| | custom-fp64 | 0.553 | 0.166 | 0.071 | 0.0492 | 0.0455 |
| | custom-fp32 | 0.311 | 0.0965 | 0.0381 | 0.0242 | 0.0211 |
| | custom-tf32 | 0.309 | 0.0969 | 0.041 | 0.0256 | 0.0221 |
| 13K | cudss-fp64 | 1.11 | 0.546 | 0.405 | 0.37 | 0.361 |
| | cudss-fp32 | 0.664 | 0.33 | 0.247 | 0.226 | 0.221 |
| | custom-fp64 | 0.738 | 0.227 | 0.101 | 0.0745 | 0.071 |
| | custom-fp32 | 0.394 | 0.123 | 0.0521 | 0.0357 | 0.0331 |
| | custom-tf32 | 0.378 | 0.118 | 0.053 | 0.0386 | 0.0347 |
| 25K | cudss-fp64 | 1.82 | 1.16 | 1.01 | 0.966 | 0.957 |
| | cudss-fp32 | 1.1 | 0.666 | 0.562 | 0.536 | 0.529 |
| | custom-fp64 | 1.87 | 0.573 | 0.275 | 0.226 | 0.218 |
| | custom-fp32 | 0.721 | 0.227 | 0.111 | 0.093 | 0.0907 |
| | custom-tf32 | 0.595 | 0.19 | 0.1 | 0.0845 | 0.0825 |
| usa | cudss-fp64 | 4.67 | 3.52 | 3.23 | 3.16 | 3.14 |
| | cudss-fp32 | 2.78 | 2.02 | 1.83 | 1.79 | 1.77 |
| | custom-fp64 | 9.2 | 2.91 | 1.34 | 1 | 0.978 |
| | custom-fp32 | 2.42 | 0.796 | 0.452 | 0.41 | 0.412 |
| | custom-tf32 | 1.89 | 0.714 | 0.445 | 0.405 | 0.404 |

#### C.3 SOLVE (per-system ms, 04-solve-optimization 통합 후 재측정)

| case | config | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|---|
| 3xxx | cudss-fp64 | 0.217 | 0.0803 | 0.0514 | 0.0442 | 0.0425 |
| | cudss-fp32 | 0.155 | 0.0561 | 0.0354 | 0.0307 | 0.0295 |
| | custom-fp64 | 0.207 | 0.0651 | 0.0226 | 0.00853 | 0.00581 |
| | custom-fp32 | 0.125 | 0.0386 | 0.0131 | 0.00448 | 0.00263 |
| | custom-tf32 | 0.123 | 0.0384 | 0.0131 | 0.00447 | 0.00263 |
| 6xxx | cudss-fp64 | 0.287 | 0.124 | 0.0873 | 0.0783 | 0.0761 |
| | cudss-fp32 | 0.195 | 0.0797 | 0.0596 | 0.054 | 0.0524 |
| | custom-fp64 | 0.325 | 0.125 | 0.038 | 0.0175 | 0.0129 |
| | custom-fp32 | 0.189 | 0.0775 | 0.0225 | 0.00862 | 0.00572 |
| | custom-tf32 | 0.189 | 0.0778 | 0.0225 | 0.00863 | 0.00571 |
| 8xxx | cudss-fp64 | 0.355 | 0.15 | 0.0997 | 0.0876 | 0.0846 |
| | cudss-fp32 | 0.238 | 0.0996 | 0.0701 | 0.0624 | 0.0602 |
| | custom-fp64 | 0.334 | 0.127 | 0.0423 | 0.021 | 0.0165 |
| | custom-fp32 | 0.188 | 0.0811 | 0.0239 | 0.0103 | 0.00734 |
| | custom-tf32 | 0.189 | 0.0811 | 0.0239 | 0.0103 | 0.00735 |
| 13K | cudss-fp64 | 0.443 | 0.201 | 0.144 | 0.13 | 0.127 |
| | cudss-fp32 | 0.278 | 0.139 | 0.102 | 0.0934 | 0.091 |
| | custom-fp64 | 0.467 | 0.137 | 0.0489 | 0.0289 | 0.025 |
| | custom-fp32 | 0.217 | 0.0793 | 0.0257 | 0.0138 | 0.0109 |
| | custom-tf32 | 0.217 | 0.0793 | 0.0256 | 0.0138 | 0.0109 |
| 25K | cudss-fp64 | 0.733 | 0.367 | 0.29 | 0.27 | 0.264 |
| | cudss-fp32 | 0.452 | 0.24 | 0.193 | 0.181 | 0.178 |
| | custom-fp64 | 0.711 | 0.215 | 0.0851 | 0.0565 | 0.0508 |
| | custom-fp32 | 0.379 | 0.126 | 0.0448 | 0.0261 | 0.0219 |
| | custom-tf32 | 0.379 | 0.126 | 0.0448 | 0.0261 | 0.0219 |
| usa | cudss-fp64 | 1.49 | 0.937 | 0.827 | 0.8 | 0.793 |
| | cudss-fp32 | 0.963 | 0.665 | 0.605 | 0.589 | 0.586 |
| | custom-fp64 | 1.34 | 0.435 | 0.226 | 0.183 | 0.176 |
| | custom-fp32 | 0.715 | 0.235 | 0.108 | 0.0802 | 0.0746 |
| | custom-tf32 | 0.715 | 0.235 | 0.108 | 0.0801 | 0.0746 |

**C-linear 핵심**: 배치 throughput(B≥16) — custom factorize 4~6×, solve 5~9×(재최적화 후). solve 재측정으로 통합 전 대비 1.4~1.7× 추가 가속(예: 25K B256 custom-tf32 0.034→0.0219). tf32=fp32(solve 동일). B=1은 custom-fp32/tf32 우세, analyze는 ≤25K custom 빠르고 USA 역전.

#### C.4 fp32 vs tf32 (B=1 factorize) — TC 가속과 계통별 front tier 분포

**B=1 factorize per-system (ms), custom fp32 vs tf32(Ozaki):**

| case | n | fp32 | tf32 | 가속비 (fp32/tf32) |
|---|---:|---:|---:|---:|
| case3012wp | 5,725 | 0.2090 | 0.1858 | **1.12×** |
| case6468rte | 12,643 | 0.3078 | 0.3008 | **1.02×** |
| case8387pegase | 14,908 | 0.3109 | 0.3089 | **1.01×** |
| case13659pegase | 23,225 | 0.3939 | 0.3782 | **1.04×** |
| case_ACTIVSg25k | 47,246 | 0.7210 | 0.5949 | **1.21×** |
| case_SyntheticUSA | 156,255 | 2.4230 | 1.8910 | **1.28×** |

> ⚠️ **Tier 경계 주의**: 아래 front 분포 표의 `mid(33–159)`·`big(≥160)` 비닝은 **측정 시점(2026-06-10)의 옛 4-tier** 기준(float 경로 `kFloatSharedFrontMax`=159)이다. **현재 코드의 tier 경계는 mid|big=64**(`kMidFrontMax`) — 4-tier→3-tier 통합 결과다. 상세: [`10-tier-consolidation-2026-06-18.md`](10-tier-consolidation-2026-06-18.md).

**계통별 front tier 분포** (serial-ND seed 1588, 측정 시점 옛 mid/big 경계 = 159):

| case | fronts | small(≤32) | mid(33–159) | big(≥160) | mid% | big% | fsz 최대 |
|---|---:|---:|---:|---:|---:|---:|---:|
| case3012wp | 2,780 | 2,765 | 15 | 0 | 0.54% | 0% | 50 |
| case6468rte | 5,902 | 5,848 | 54 | 0 | 0.91% | 0% | 65 |
| case8387pegase | 7,406 | 7,345 | 61 | 0 | 0.82% | 0% | 76 |
| case13659pegase | 12,388 | 12,280 | 108 | 0 | 0.87% | 0% | 90 |
| case_ACTIVSg25k | 22,724 | 22,416 | 308 | 0 | 1.36% | 0% | 137 |
| case_SyntheticUSA | 74,231 | 73,174 | 1,016 | **41** | 1.37% | 0.055% | 250 |

- 작은 계통일수록 mid/big이 거의 없다: case3012wp는 mid 15개·big 0개, big front(≥160)은 case_SyntheticUSA(41개)에만 존재. mid 개수는 계통 크기에 따라 15→308→1,016 증가.
- 텐서코어는 mid/big의 trailing GEMM만 가속 → mid/big이 많은 큰 계통(25K 1.21×, usa 1.28×)에서 tf32 이득이 크고, small 지배 계통(6468rte~13659pegase 1.01~1.04×)은 이득이 거의 없다. small tier의 TC 불가 근거: [`../20260612_lab_meeting/small-tier-no-tensorcore.md`](../20260612_lab_meeting/small-tier-no-tensorcore.md).

### C-NR — 전체 cuPF Newton-Raphson (graph off)

하네스: `cupf_cpp_evaluate`(B=1) + `cupf_batch_bench`(배치). 정식 `NewtonOptions::custom`, 환경변수 0. 결정성: serial-ND seed 1588 + 클럭 고정 + scale_step=0 + Ozaki TF32 빌드. 5 config: `cudss-fp64`/`custom-fp64`(FP64), `cudss-mixed`/`custom-mixed-fp32`/`custom-mixed-tf32`(Mixed=FP32 step+FP64 state). tol 1e-8, max-iter 30, warmup 3~5 / repeats 11~21 median.

#### C.5 수렴 반복수 (NR iters) & Initial(analyze, ms, B=1)

| config | 3xxx | 6xxx | 8xxx | 13K | 25K | usa | | init 3xxx | 6xxx | 8xxx | 13K | 25K | usa |
|---|--|--|--|--|--|--|--|--|--|--|--|--|--|
| `cudss-fp64` | 4 | 4 | 4 | 6 | 5 | 7 | | 43.2 | 73.5 | 97.3 | 127 | 220 | 785 |
| `custom-fp64` | 4 | 4 | 4 | 6 | 5 | 7 | | 13.2 | 49.3 | 53.4 | 94 | 283 | 619 |
| `cudss-mixed` | 4 | 4 | 4 | 7 | 5 | 7 | | 43.4 | 73.8 | 93 | 130 | 219 | 794 |
| `custom-mixed-fp32` | 4 | 4 | 4 | 7 | 5 | 8 | | 14 | 44.4 | 66.7 | 96.3 | 184 | 647 |
| `custom-mixed-tf32` | 4 | 4 | 4 | 7 | 5 | 8 | | 13.3 | 46.1 | 58.3 | 93.9 | 183 | 635 |

> 모든 정밀도 4~8회 동일 수렴(fp32≈tf32, Ozaki TF32가 fp32 정확도). custom analyze ~1.4~2× 빠름.

#### C.6 factorize per-iter (B=1, ms/iter)

| config | 3xxx | 6xxx | 8xxx | 13K | 25K | usa |
|---|---|---|---|---|---|---|
| `custom-mixed-fp32` | 0.208 | 0.309 | 0.365 | 0.412 | 0.664 | 1.98 |
| `custom-mixed-tf32` | 0.205 | 0.309 | 0.347 | 0.394 | 0.565 | 1.6 |

> tf32 factorize가 fp32 대비 ~1.05~1.2× 빠름(단독과 일치). tri-solve는 TC 미사용이라 fp32=tf32 → 전체 solve로는 ~1~3% 희석.

#### C.7 배치 스케일링 — solve **per-system** (μs, 괄호 iters)

| case | config | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| 3xxx | cudss-fp64 | 2249 (4) | 610 (4) | 534 (4) | 501 (4) |
| | custom-fp64 | 1978 (4) | 221 (4) | 118 (4) | 90 (4) |
| | cudss-mixed | 1764 (4) | 426 (4) | 342 (4) | 333 (4) |
| | custom-mixed-fp32 | 1396 (4) | 147 (4) | 77 (4) | 55 (4) |
| | custom-mixed-tf32 | 1417 (4) | 150 (4) | 79 (4) | 62 (4) |
| 6xxx | cudss-fp64 | 2906 (4) | 960 (4) | 866 (4) | 840 (4) |
| | custom-fp64 | 2805 (4) | 383 (4) | 233 (4) | 207 (4) |
| | cudss-mixed | 2057 (4) | 750 (4) | 570 (4) | 558 (4) |
| | custom-mixed-fp32 | 1820 (4) | 244 (4) | 156 (4) | 130 (4) |
| | custom-mixed-tf32 | 1839 (4) | 237 (4) | 144 (4) | 126 (4) |
| 8xxx | cudss-fp64 | 4238 (4) | 1285 (4) | 1145 (4) | 1167 (4) |
| | custom-fp64 | 3186 (4) | 484 (4) | 327 (4) | 359 (4) |
| | cudss-mixed | 2901 (4) | 859 (4) | 774 (4) | 792 (4) |
| | custom-mixed-fp32 | 2071 (4) | 321 (4) | 202 (4) | 229 (4) |
| | custom-mixed-tf32 | 2063 (4) | 317 (4) | 199 (4) | 236 (4) |
| 13K | cudss-fp64 | 8650 (6) | 3452 (6) | 2724 (6) | 2743 (6) |
| | custom-fp64 | 7086 (6) | 996 (6) | 738 (6) | 788 (6) |
| | cudss-mixed | 6749 (7) | 2342 (7) | 2114 (7) | 2140 (7) |
| | custom-mixed-fp32 | 4492 (7) | 803 (7) | 488 (7) | 536 (7) |
| | custom-mixed-tf32 | 4393 (7) | 684 (7) | 491 (7) | 536 (7) |
| 25K | cudss-fp64 | 10365 (5) | 5130 (5) | 4885 (5) | 4996 (5) |
| | custom-fp64 | 10878 (5) | 1869 (5) | 1539 (5) | 1645 (5) |
| | cudss-mixed | 6553 (5) | 3128 (5) | 3318 (5) | 3628 (5) |
| | custom-mixed-fp32 | 5222 (5) | 930 (5) | 812 (5) | 935 (5) |
| | custom-mixed-tf32 | 4686 (5) | 910 (5) | 797 (5) | 936 (6) |
| usa | cudss-fp64 | 39369 (7) | 25013 (7) | 24499 (7) | 24289 (7) |
| | custom-fp64 | 62918 (7) | 10430 (7) | 8931 (7) | 8495 (7) |
| | cudss-mixed | 26231 (7) | 17728 (8) | 17714 (8) | 17527 (8) |
| | custom-mixed-fp32 | 21138 (7) | 4974 (8) | 4990 (8) | 4845 (8) |
| | custom-mixed-tf32 | 18566 (7) | 4842 (8) | 4916 (8) | 4714 (8) |

> **배치서 custom(mixed)이 cuDSS 대비 ~4~6×**: 25K B=64 custom ~800μs vs cuDSS-fp64 4885(6×); 13K B=64 custom ~490 vs cuDSS-fp64 2724(5.6×); usa B=256 custom-tf32 4714 vs cuDSS-fp64 24289(5.2×). **tf32 ≥ fp32 (일관 근소 우위)** — factorize TC 이득이 NR(tri-solve 포함)에 희석돼 작지만 방향 일관. custom-fp64는 B=1에서 cuDSS보다 느림(배치 전용 이점).

#### C.8 연산자별 실행시간 (per-operator) — 5 config × 6 case × 4 B

`cupf_batch_bench` per-stage 타이머(ENABLE_TIMING), repeats 15 median, 결정적. 전체 raw(총합 μs): [`07-cupf-backend-comparison-2026-06-11/operator_ms_full.tsv`](07-cupf-backend-comparison-2026-06-11/operator_ms_full.tsv).

값 단위: `factorize`~`vupd`는 **반복당 μs**(solve 누적 ÷ iters); `upload`·`download`는 NR 1회당 1번뿐(count=1)이라 ÷iters 하지 않은 **solve당 μs**(`*` 표기). factorize·tri-solve = 선형해(백엔드/정밀도 차이), 나머지 = cuPF 공통. 검증: custom-mixed의 tri-solve(per-iter)는 fp32≈tf32(TC 미사용 동일 삼각해), factorize(per-iter)만 tf32가 ~1.05~1.2× 빠르다.

대표 발췌 — case_SyntheticUSA (usa), solve/it·factorize·tri-solve (μs):

| config | B | iters | solve/it | factorize | tri-solve |
|---|--:|--:|--:|--:|--:|
| cudss-fp64 | 256 | 7 | 889449 | 649964.3 | 172752.9 |
| custom-fp64 | 256 | 7 | 308766 | 204111.4 | 38667.3 |
| cudss-mixed | 256 | 8 | 569410 | 380083.8 | 130630.0 |
| custom-mixed-fp32 | 256 | 8 | 156602 | 88758.6 | 16594.5 |
| custom-mixed-tf32 | 256 | 8 | 151526 | 84382.9 | 16600.1 |

> 전체 6 case × 4 B의 per-operator(factorize/tri-solve/jacobian/ibus/mis+norm/vupd/upload/download)는 [`operator_ms_full.tsv`](07-cupf-backend-comparison-2026-06-11/operator_ms_full.tsv)에 보존. **반복당 factorize가 지배**(custom이 cuDSS보다 훨씬 작음), tri-solve가 다음. upload/download(solve당 1회)는 배치가 커질수록 절대값↑(특히 usa B=256 download).

---

## 종합 해석 & 정직성 caveat

1. **raw B=1(setup A)은 과대평가** — 5.78~44.7×는 cuDSS가 native batching이 없어 system마다 full single-system launch를 지불하기 때문. 인용 금지, 맥락 설명용으로만.
2. **공정한 숫자는 setup B의 B=256 tf32 = 3.60×(USA) ~ 10.16×(case1197)** — cuDSS도 ubatch+mt-auto로 amortise한 비교. 최대 case는 ~3.5×로 saturate, 최소 case가 최고 비율.
3. **전체 NR(setup C)에서 mixed custom이 cuDSS 대비 ~4~6×** — fp64와 동일 수렴 보장. custom analyze ~1.4~2× 빠름. custom-fp64는 B=1 단일 시스템에서 cuDSS보다 느림(배치 전용 이점).
4. **tf32-Ozaki ≈ fp32, 근소 우위** — factorize의 TC 이득(1.05~1.2×)이 tri-solve(TC 미사용)·Jacobian 등에 희석돼 전체 NR로는 ~1~3%. *느려지는 게 아니라* 방향만 일관되게 빠르다. small 지배 계통에서는 mid/big front가 적어 TC 이득이 거의 없다.
5. **정밀도 한계**: fp16/tf32 single-shot residual은 큰 case에서 1e-2~1e-1 (trailing-GEMM rounding) — Newton loop + iterative refinement 안에서만 권장, single-shot FP32 drop-in 대체 아님(`‖Ax-b‖ ≲ 1e-3` 요구 시).

## 재현

```bash
# Setup A: raw B=1 single-system cuDSS (custom은 5 B sizes × 3 precisions)
./build-bench/cudss_run <case-dir> --precision fp32 --repeat 10
./build-bench/custom_linear_solver_run <case-dir> --batch <B> --batch-only \
  --precision {fp32|fp16|tf32} --repeat 10

# Setup B: cuDSS ubatch + mt-auto (cuPF pattern)
./build-bench/cudss_run <case-dir> --precision fp32 --mt-auto --batch <B> --repeat 10
./build-bench/custom_linear_solver_run <case-dir> \
  --precision {fp32|fp16|tf32} --batch <B> --batch-only --repeat 10

# Setup C-NR: 전체 cuPF (Ozaki TF32 빌드 + serial-ND seed 1588 + 클럭 고정 + scale_step=0)
./build-bench/cupf_batch_bench <case-dir> --config {cudss-fp64|custom-fp64|cudss-mixed|custom-mixed-fp32|custom-mixed-tf32} --batch <B>
```

## 관련 문서

| 문서 | 내용 |
|---|---|
| [`main-report.md`](../main-report.md) | canonical master report |
| [`10-tier-consolidation-2026-06-18.md`](10-tier-consolidation-2026-06-18.md) | 4-tier→3-tier 통합 (mid\|big=64 경계) |
| [`02-comprehensive-sweep.md`](02-comprehensive-sweep.md) | 내부 FP64/FP32/TC full sweep |
| [`../optimal-configuration.md`](../optimal-configuration.md) | optimal dispatch 구성 결정 |
