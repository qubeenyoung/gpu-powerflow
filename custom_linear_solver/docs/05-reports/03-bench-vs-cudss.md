# Benchmark vs cuDSS — bus-count > 1K power-grid Jacobians, RTX 3090 (sm_86)

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: cuDSS 대비 두 측정 setup 병합 — raw B=1 single-system은 5.8~44.7×(cuDSS batching 없어 과대평가), 공정한 ubatch+mt-auto setup은 B=256 tf32 기준 3.6~10.2×.

- 작성일: 2026-06-07 (병합 2026-06-10)
- Setup 공통: RTX 3090 (sm_86), CUDA 12.8, `Release` build, `CLS_INTERNAL_GRAPH=ON`, `use_multistream_subtrees=true`. Median of 10 trials. 커스텀 측은 `--batch-only`.

이 문서는 두 cuDSS 비교 측정을 하나로 합친다. **두 setup의 차이가 핵심**이므로 각 setup의 숫자를 분리해 보존한다. canonical 요약/권장은 [`01-final-report.md`](01-final-report.md), full 내부 sweep은 [`02-comprehensive-sweep.md`](02-comprehensive-sweep.md)를 참조한다.

## 0. 두 측정 setup 요약

| | Setup A (raw B=1) | Setup B (ubatch + mt-auto) |
|---|---|---|
| cuDSS 호출 | single-system, B=1 1회 (native uniform-batching 없음) | `CUDSS_CONFIG_UBATCH_SIZE=B`, batch-major value buffer (B × nnz), `cudssSetThreadingLayer(handle, "<libcudss_mtlayer_gomp.so>")` (cuPF pattern) |
| speedup 범위 (custom/tf32 vs cuDSS-fp32) | **5.78× ~ 44.7×** (B=256) | **3.60× ~ 10.16×** (B=256) |
| 공정성 | **과대평가** — cuDSS가 batching을 못 해서 system마다 full single-system launch 비용 지불 | **공정** — cuDSS도 ubatch로 amortise, mt-auto 적용 |
| 데이터 파일 | [`04-bench-vs-cudss-2026-06-07/sweep_results.tsv`](04-bench-vs-cudss-2026-06-07/sweep_results.tsv) (113 rows) | [`05-bench-vs-cudss-ubatch-2026-06-07/sweep_v2.tsv`](05-bench-vs-cudss-ubatch-2026-06-07/sweep_v2.tsv) (220 rows) |

Setup A의 cuDSS는 B=1에서 1회만 측정한다 (각 system이 별도 call이므로 per-system time이 더 큰 B에서도 동일). Setup B에서 cuDSS는 ubatch로 실제 batching하므로 per-system time이 B 증가에 따라 감소한다. **두 setup 모두 custom 측은 동일** — 차이는 cuDSS 호출 방식뿐이다.

비교 solver / precision:
- **cuDSS fp32** (`CUDA_R_32F`)
- **custom fp32** (`Precision::FP32` — staged-scalar trailing)
- **custom fp16** (`Precision::FP16` — FP16 WMMA m16n16k16 trailing, FP32 accumulate)
- **custom tf32** (`Precision::TF32` — V9h PTX `mma.m16n8k8/k4` hybrid + `__launch_bounds__(512, 2)`)

## 1. Test cases

Setup A는 7 cases, Setup B는 11 cases (case9241~case_ACTIVSg70k 4개 추가). Jacobian dimension `n` 기준 정렬 (bus count ≈ n/2).

| case | n | nnz | scope | Setup A | Setup B |
|------|--:|--:|---|:--:|:--:|
| case1197 | 2,392 | 14,344 | ≈ 1.2k bus | ✓ | ✓ |
| case_ACTIVSg2000 | 3,607 | 21,529 | 2k bus | ✓ | ✓ |
| case3012wp | 5,725 | 33,983 | 3k bus | ✓ | ✓ |
| case6468rte | 12,643 | 75,917 | 6.5k bus | ✓ | ✓ |
| case8387pegase | 14,908 | 89,432 | 8.4k bus | ✓ | ✓ |
| case9241pegase | 17,036 | 102,189 | 9k bus | | ✓ |
| case_ACTIVSg10k | 18,544 | 110,855 | 10K bus | | ✓ |
| case13659pegase | 23,225 | 139,341 | 13K bus | | ✓ |
| case_ACTIVSg25k | 47,246 | 282,927 | 25k bus | ✓ | ✓ |
| case_ACTIVSg70k | 134,104 | 802,873 | 70K bus | | ✓ |
| case_SyntheticUSA | 156,255 | 936,533 | 78k bus | ✓ | ✓ |

---

## 2. Setup A — raw B=1 single-system cuDSS (과대평가)

cuDSS는 native uniform-batching semantics가 없어 B=1에서 1회 측정한다. per-system time은 더 큰 B에서도 동일하다 (각 system이 별도 call). **이 setup의 speedup은 over-estimate** — cuDSS가 batching의 이점을 전혀 못 받기 때문이다.

데이터: [`04-bench-vs-cudss-2026-06-07/sweep_results.tsv`](04-bench-vs-cudss-2026-06-07/sweep_results.tsv).

### 2.1 Per-system factor+solve time (ms), median of 10

Format `factor+solve (per-system)`.

#### case1197 (n=2,392)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.192** | – | – | – | – |
| custom/fp32 | 0.147 | 0.039 | 0.0076 | 0.0058 | **0.0043** |
| custom/fp16 | 0.147 | 0.039 | 0.0076 | 0.0058 | **0.0043** |
| custom/tf32 | 0.146 | 0.039 | 0.0076 | 0.0058 | **0.0043** |

#### case_ACTIVSg2000 (n=3,607)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.510** | – | – | – | – |
| custom/fp32 | 0.471 | 0.126 | 0.027 | 0.019 | **0.0144** |
| custom/fp16 | 0.552 | 0.147 | 0.029 | 0.021 | 0.0154 |
| custom/tf32 | 0.545 | 0.145 | 0.028 | 0.020 | **0.0144** |

#### case3012wp (n=5,725)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.400** | – | – | – | – |
| custom/fp32 | 0.416 | 0.101 | 0.024 | 0.018 | **0.0137** |
| custom/fp16 | 0.482 | 0.126 | 0.027 | 0.018 | 0.0136 |
| custom/tf32 | 0.489 | 0.123 | 0.026 | 0.019 | 0.0139 |

#### case6468rte (n=12,643)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.528** | – | – | – | – |
| custom/fp32 | 0.479 | 0.161 | 0.042 | 0.035 | **0.0297** |
| custom/fp16 | 0.621 | 0.190 | 0.046 | 0.037 | 0.0305 |
| custom/tf32 | 0.577 | 0.186 | 0.045 | 0.036 | 0.0300 |

#### case8387pegase (n=14,908)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **0.772** | – | – | – | – |
| custom/fp32 | 0.617 | 0.182 | 0.058 | 0.047 | **0.0416** |
| custom/fp16 | 0.712 | 0.230 | 0.060 | 0.054 | 0.0430 |
| custom/tf32 | 0.731 | 0.196 | 0.058 | 0.052 | 0.0414 |

#### case_ACTIVSg25k (n=47,246)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **1.541** | – | – | – | – |
| custom/fp32 | 1.327 | 0.448 | 0.182 | 0.173 | 0.159 |
| custom/fp16 | 1.300 | 0.441 | 0.184 | 0.167 | 0.168 |
| custom/tf32 | 1.325 | **0.397** | **0.171** | **0.162** | **0.154** |

#### case_SyntheticUSA (n=156,255)
| solver/prec | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|
| cuDSS/fp32 | **3.742** | – | – | – | – |
| custom/fp32 | 3.582 | 1.222 | 0.701 | 0.648 | 0.640 |
| custom/fp16 | 3.538 | 1.213 | 0.691 | 0.688 | 0.646 |
| custom/tf32 | **3.054** | 1.217 | **0.681** | **0.622** | 0.647 |

### 2.2 Per-system speedup of custom/tf32 over cuDSS-fp32 (over-estimate)

| case | cuDSS f+s (ms) | B=1 | B=4 | B=32 | B=64 | B=256 |
|---|---:|---:|---:|---:|---:|---:|
| case1197 | 0.192 | 1.31× | 4.90× | 25.2× | 32.8× | **44.7×** |
| case_ACTIVSg2000 | 0.510 | 0.93× | 3.52× | 18.1× | 25.7× | **35.5×** |
| case3012wp | 0.400 | 0.82× | 3.24× | 15.3× | 21.6× | **28.7×** |
| case6468rte | 0.528 | 0.91× | 2.84× | 11.7× | 14.9× | **17.6×** |
| case8387pegase | 0.772 | 1.06× | 3.94× | 13.2× | 15.0× | **18.6×** |
| case_ACTIVSg25k | 1.541 | 1.16× | 3.89× | 9.0× | 9.5× | **10.0×** |
| case_SyntheticUSA | 3.742 | 1.23× | 3.07× | 5.49× | 6.02× | 5.78× |

**주의**: B=256의 45× / 35× / 29× (작은 case)는 cuDSS가 system마다 full single-system kernel-launch + analyze-fixed 비용을 지불하는 반면 custom solver는 하나의 dispatch sequence를 B개 system이 공유하기 때문이다. cuDSS에 batching이 없는 setup이라 **over-estimate**다. n ≳ 50k cases에서는 per-system work 자체가 dominant → B amortisation이 6-10×로 saturate.

### 2.3 Accuracy (max relative residual across all measured B)

| case | cuDSS/fp32 | custom/fp32 | custom/fp16 | custom/tf32 |
|---|---:|---:|---:|---:|
| case1197 | 7.1e-05 | 2.1e-04 | 2.1e-04 | 2.1e-04 |
| case_ACTIVSg2000 | 1.3e-05 | 1.5e-05 | 2.8e-03 | 1.5e-02 |
| case3012wp | 1.6e-04 | 2.5e-04 | 4.7e-03 | 3.2e-02 |
| case6468rte | 4.2e-05 | 1.3e-04 | 1.1e-03 | 2.8e-03 |
| case8387pegase | 1.3e-05 | 3.5e-05 | 2.1e-02 | 5.0e-02 |
| case_ACTIVSg25k | 1.5e-04 | 2.7e-04 | 8.4e-02 | 5.4e-02 |
| case_SyntheticUSA | 8.9e-04 | 2.4e-02 | 1.2e-01 | 5.6e-02 |

- **fp32** (cuDSS & custom) 는 모든 case에서 ≲ 1e-3 — NR Jacobian residual로 acceptable.
- **fp16 / tf32** 는 큰 case에서 ~1e-1 .. 1e-2 (trailing-GEMM rounding 누적). Newton loop + iterative refinement 안에서 유용, single-shot FP32 drop-in 대체는 아님 (`‖Ax-b‖ ≲ 1e-3` 요구 시).

### 2.4 Multi-stream overlap (case8387 B=1 fp32)

NVTX-instrumented build (`CLS_ENABLE_NVTX=ON`), `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`. profile 파일:
- [`04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_on.nsys-rep`](04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_on.nsys-rep) (~185 KB)
- [`04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_off.nsys-rep`](04-bench-vs-cudss-2026-06-07/case8387_b1_fp32_multistream_off.nsys-rep) (~140 KB)

**Multi-stream ON** (`use_multistream_subtrees=true`, default) — 9 streams hit:

| stream | kernels | GPU-busy (ms) | role |
|---:|---:|---:|---|
| 13 (main) | 990 | 6.67 | analyze + spine + solve |
| 23 | 180 | 2.03 | subtree #1 |
| 24 | 180 | 1.88 | subtree #2 |
| 27 | 180 | 1.74 | subtree #3 |
| 26 | 120 | 1.25 | subtree #4 |
| 25 | 130 | 1.24 | subtree #5 |
| 29 | 120 | 0.96 | subtree #6 |
| 28 | 110 | 0.75 | subtree #7 |
| 7  | 48  | 0.36 | CUDA runtime (analyze copies) |

`factor_*` kernel total: **1,340 instances / 13.76 ms** across 8 subtree streams.

**Multi-stream OFF** (`--no-multistream`) — 2 streams hit:

| stream | kernels | GPU-busy (ms) | role |
|---:|---:|---:|---|
| 13 (main) | 930 | 5.75 | everything sequential |
| 7  | 48  | 0.36 | CUDA runtime |

`factor_*` kernel total: **300 instances / 3.21 ms** on the main stream only.

**해석**: 스트림은 실제로 사용된다 (ON profile에 8 subtree + 1 spine stream visible). 그러나 **B=1에서는 wall이 개선되지 않는다**:
- ON:  factor=0.340 ms, solve=0.262 ms (per-system)
- OFF: factor=0.322 ms, solve=0.245 ms (per-system) — ~5% faster

이유: case8387에서 spine (elimination tree 최상단 chain)이 factor wall을 dominant하고, spine은 어차피 main stream에서 돈다. etree leaf의 subtree work는 ~2 ms뿐. B=1에서는 SM occupancy가 이미 낮아 그 work을 stream에 split하면 event/wait overhead만 추가되고 spine을 더 일찍 돌릴 SM을 못 비운다. win은 B ≥ 4에서 나타난다 (각 subtree stream이 level당 충분한 launch를 가져 dispatch overhead를 숨김). main sweep이 확인: B=64 case8387에서 custom-tf32가 cuDSS-fp32 대비 15×.

---

## 3. Setup B — ubatch + mt-auto cuDSS (공정한 비교)

cuDSS를 `CUDSS_CONFIG_UBATCH_SIZE=B`로 실제 uniform-batching하고 batch-major value buffer (B × nnz)를 cuPF production과 동일하게 먹인다. `cudssSetThreadingLayer(handle, "<libcudss_mtlayer_gomp.so>")` mt-auto 적용. **이쪽이 공정한 speedup**이다.

데이터: [`05-bench-vs-cudss-ubatch-2026-06-07/sweep_v2.tsv`](05-bench-vs-cudss-ubatch-2026-06-07/sweep_v2.tsv) (220 rows: 11 cases × 4 paths × 5 batch sizes).

### 3.1 Single batch (B = 1) — analyze / factorize / solve per system (ms)

`analyze`는 one-time symbolic + plan/reordering 비용 (matrix sparsity pattern당, Newton iteration 간 재사용). `factorize` / `solve`는 per call, median of 10.

#### Analyze (ms)

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

custom analyze는 ~25K까지 더 빠르다; cuDSS가 최대 case (70K, USA)에서 약간 우세. analyze 비용은 많은 Newton iteration에 amortise되므로 solver가 warm해지면 gap이 거의 무의미.

#### Factorize per system (ms)

| case | cuDSS ubatch+mt-auto / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
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

custom/fp32가 B=1에서 cuDSS를 9/11 cases에서 이김. fp16/tf32 path는 두 최대 case (70K, USA)에서 catch up — trailing-GEMM share가 factor wall에서 충분히 커 B=1에서도 WMMA/PTX가 이득.

#### Solve per system (ms)

| case | cuDSS ubatch+mt-auto / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
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

cuDSS가 single-system solve를 소유한다: triangular-solve kernel이 B=1 throughput에 튜닝됨. custom의 `solve_small_warp` / `solve_level` path는 B=1 latency보다 batch amortisation에 최적화. gap은 작음 (~10K 이상에서 10-20% 이내) 하지만 일관됨.

#### Relative residual at B=1

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

custom/fp32 ≈ cuDSS/fp32 (동일 precision). fp16 / tf32는 1-2 order 뒤짐 — iterative refinement loop 안에서 적합, single-shot drop-in 아님.

### 3.2 Multi batch — factor + solve per system (ms)

각 (case, B)는 **factor+solve per-system** wall, median of 10. cuDSS는 `CUDSS_CONFIG_UBATCH_SIZE=B`, batch-major value buffer (cuPF production 방식).

#### B = 4

| case | cuDSS ubatch / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.0748 | **0.0387** | 0.0388 | 0.0386 |
| case_ACTIVSg2000 | 0.200 | **0.126** | 0.146 | 0.145 |
| case3012wp | 0.163 | **0.106** | 0.123 | 0.117 |
| case6468rte | 0.251 | **0.159** | 0.180 | 0.177 |
| case8387pegase | 0.344 | **0.187** | 0.211 | 0.200 |
| case9241pegase | 0.374 | **0.197** | 0.221 | 0.224 |
| case_ACTIVSg10k | 0.387 | **0.197** | 0.227 | 0.233 |
| case13659pegase | 0.465 | **0.218** | 0.243 | 0.261 |
| case_ACTIVSg25k | 0.905 | 0.428 | 0.451 | **0.420** |
| case_ACTIVSg70k | 2.377 | 1.204 | 1.082 | **1.022** |
| case_SyntheticUSA | 2.681 | 1.348 | **1.221** | 1.286 |

#### B = 32

| case | cuDSS ubatch / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.0448 | **0.0077** | 0.0076 | 0.0077 |
| case_ACTIVSg2000 | 0.125 | **0.0265** | 0.0285 | 0.0282 |
| case3012wp | 0.104 | **0.0248** | 0.0251 | 0.0243 |
| case6468rte | 0.185 | **0.0430** | 0.0457 | 0.0463 |
| case8387pegase | 0.229 | 0.0644 | 0.0580 | **0.0586** |
| case9241pegase | 0.253 | **0.0617** | 0.0682 | 0.0653 |
| case_ACTIVSg10k | 0.294 | **0.0656** | 0.0699 | 0.0656 |
| case13659pegase | 0.329 | **0.0865** | 0.0823 | 0.0843 |
| case_ACTIVSg25k | 0.731 | 0.178 | 0.183 | **0.172** |
| case_ACTIVSg70k | 2.093 | 0.622 | 0.616 | **0.572** |
| case_SyntheticUSA | 2.395 | 0.670 | 0.688 | **0.675** |

#### B = 64

| case | cuDSS ubatch / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.0447 | **0.0059** | 0.0058 | 0.0058 |
| case_ACTIVSg2000 | 0.119 | **0.0191** | 0.0205 | 0.0197 |
| case3012wp | 0.099 | 0.0178 | 0.0194 | 0.0188 |
| case6468rte | 0.179 | **0.0342** | 0.0366 | 0.0367 |
| case8387pegase | 0.220 | **0.0480** | 0.0503 | 0.0525 |
| case9241pegase | 0.243 | 0.0558 | **0.0543** | 0.0565 |
| case_ACTIVSg10k | 0.288 | 0.0563 | 0.0618 | **0.0572** |
| case13659pegase | 0.319 | 0.0732 | 0.0753 | **0.0726** |
| case_ACTIVSg25k | 0.718 | 0.173 | 0.178 | **0.159** |
| case_ACTIVSg70k | 2.071 | **0.564** | 0.579 | 0.580 |
| case_SyntheticUSA | 2.373 | 0.663 | 0.664 | **0.656** |

#### B = 256

| case | cuDSS ubatch / fp32 | custom / fp32 | custom / fp16 | custom / tf32 |
|---|---:|---:|---:|---:|
| case1197 | 0.0436 | **0.0043** | 0.0043 | 0.0043 |
| case_ACTIVSg2000 | 0.114 | 0.0144 | 0.0153 | **0.0144** |
| case3012wp | 0.096 | 0.0139 | **0.0138** | 0.0138 |
| case6468rte | 0.175 | 0.0301 | 0.0313 | **0.0295** |
| case8387pegase | 0.213 | **0.0423** | 0.0426 | 0.0424 |
| case9241pegase | 0.237 | 0.0483 | 0.0604 | **0.0473** |
| case_ACTIVSg10k | 0.283 | **0.0512** | 0.0543 | 0.0517 |
| case13659pegase | 0.311 | 0.0705 | 0.0766 | **0.0658** |
| case_ACTIVSg25k | 0.707 | 0.160 | 0.167 | **0.155** |
| case_ACTIVSg70k | 2.056 | **0.548** | 0.576 | 0.571 |
| case_SyntheticUSA | 2.357 | 0.650 | 0.652 | **0.644** |

### 3.3 custom/tf32 speedup over cuDSS ubatch+mt-auto / fp32 (공정)

(factor + solve per system; `>1.0×` = custom faster)

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

공정한 B=256 tf32 speedup 범위 = **3.60× (USA) ~ 10.16× (case1197)**. Setup A의 5.78~44.7×는 cuDSS batching이 없어서 나온 over-estimate임을 다시 강조한다.

---

## 4. 해석

### 4.1 B = 1 single-batch
- **factor + solve**는 어느 쪽도 clean win 아님: custom/fp32가 factorize 9/11, cuDSS가 solve 11/11. net f+s는 작은 case에서 roughly even, case8387부터 custom/fp32로 기움.
- **analyze**는 작은/중간 case (≤25K)에서 custom 강세 (case1197~case6468rte 5-10× 빠름); 25K 부근에서 close, 두 최대 case에서 cuDSS로 flip. cuDSS setup은 reordering + symbolic factorization이 tiny matrix에서 더 무거움.
- 작은/중간 grid의 single Newton iteration에서 custom/fp32는 cuDSS/fp32와 동일 residual을 더 짧은 total wall로 달성 (주로 analyze gap 덕분).

### 4.2 Multi-batch (B ≥ 4)
- cuDSS도 `UBATCH_SIZE`의 이득을 받음 (B=256에서 per-system이 B=1의 ~25%로). 그러나 custom의 uniform-batch dispatch가 SM utilization을 더 aggressive하게 saturate → B=256에서 measured cases 전반 3-10× per-system win.
- 최대 case (70K, USA)는 ~3.5-3.7×로 saturate — 그 size에서 cuDSS의 parallel triangular solve + level-set factor가 이미 hardware throughput limit 근처, custom의 wall은 spine (per-batch sequential)에 dominate됨.
- 최소 case (case1197)가 최고 비율 (10.16×) — cuDSS가 ubatch로 줄지 않는 per-call setup을 여전히 지불.

### 4.3 Precision dimension
- custom/fp16, custom/tf32는 trailing GEMM의 accuracy를 occupancy headroom과 trade. win은 trailing GEMM이 dominant한 두 최대 case에서 가장 visible: case_SyntheticUSA B=4에서 fp16 path가 fastest (1.221 ms/sys, 그 B의 fp32/tf32 대비 ~10% 빠름).
- bus count ≲ 10K에서는 세 precision이 per-system time에서 noise 이내 — GEMM이 너무 작아 WMMA / PTX advantage가 안 드러남.
- fp16 / tf32 single-shot residual은 큰 case에서 1e-2 ~ 1e-1 — iterative refinement loop 안에서만 권장.

### 4.4 cuDSS `--mt-auto` impact
threading layer는 analyze 비용을 눈에 띄게 줄임 (예: case_ACTIVSg2000 analyze 95 ms는 대부분 host-side mt-aware reordering). analyze 외 wall saving은 이 matrix scale에서 작음; mt layer는 주로 host reordering phase를 돕는다.

---

## 5. Reproducing

```bash
# CLI runner 빌드 (cuDSS install 필요, NVTX optional)
cmake -S custom_linear_solver -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DCLS_BUILD_CUDSS_SCRIPT=ON [-DCLS_ENABLE_NVTX=ON]
cmake --build build-bench -j

# Setup A: raw B=1 single-system cuDSS (custom은 5 B sizes × 3 precisions)
./build-bench/cudss_run <case-dir> --precision fp32 --repeat 10
./build-bench/custom_linear_solver_run <case-dir> --batch <B> --batch-only \
  --precision {fp32|fp16|tf32} --repeat 10

# Setup B: cuDSS ubatch + mt-auto (cuPF pattern)
./build-bench/cudss_run <case-dir> --precision fp32 --mt-auto --batch <B> --repeat 10
./build-bench/custom_linear_solver_run <case-dir> \
  --precision {fp32|fp16|tf32} --batch <B> --batch-only --repeat 10

# Setup A multi-stream nsys profile
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --output=case8387_b1_fp32_multistream_on \
  ./build-bench/custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
  --batch 1 --batch-only --precision fp32 --repeat 10
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --output=case8387_b1_fp32_multistream_off \
  ./build-bench/custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
  --batch 1 --batch-only --precision fp32 --repeat 10 --no-multistream
```

Setup B의 public dataset에 없는 4 cases (`case9241pegase`, `case_ACTIVSg10k`, `case13659pegase`, `case_ACTIVSg70k`)는 NR Jacobian을 regenerate:

```bash
python3 -m gpu-powerflow.python.prepare.convert_m_to_mat \
  --input-root /datasets/power_system/matpower \
  --output-root <writable_mat_dir> \
  --cases case9241pegase case_ACTIVSg10k case13659pegase case_ACTIVSg70k

python3 prepare_datasets/python/prepare_nr_linear_system.py \
  --mat-root <writable_mat_dir> \
  --output-root <writable_nr_dir> \
  --cases case9241pegase case_ACTIVSg10k case13659pegase case_ACTIVSg70k
```

## 6. 관련 문서

| 문서 | 내용 |
|---|---|
| [`01-final-report.md`](01-final-report.md) | canonical master report |
| [`02-comprehensive-sweep.md`](02-comprehensive-sweep.md) | 내부 FP64/FP32/TC full sweep |
| [`04-factorize-progress.md`](04-factorize-progress.md) | factorize 최적화 progress |
| [`../optimal-configuration.md`](../optimal-configuration.md) | optimal dispatch 구성 결정 |
