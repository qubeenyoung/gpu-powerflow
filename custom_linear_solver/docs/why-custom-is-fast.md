# `custom_linear_solver`가 전력망 Jacobian에서 빠른 이유 — 설계 분해

본 문서는 `custom_linear_solver`가 전력망 Newton-Raphson Jacobian 위에서 빠른 **메커니즘**을 그 자체로 분해한다. 비교 분석(vs STRUMPACK, vs cuDSS)은 별도 문서들에 있다(§9 참고).

핵심 명제는 다음 한 줄로 요약된다:

> *전력망 Jacobian의 dense LU work 자체가 너무 작아 GPU compute/bandwidth를 saturate 못 하는 latency-bound 영역에 머문다. 이 영역에서 빠른 솔버는 GPU compute를 더 짜내는 솔버가 아니라, **GPU 호출의 host-side overhead를 모두 제거한 솔버**다. custom_linear_solver는 그 호출 경로를 8단의 도메인 가정 위 설계로 함께 압축한다.*

---

## 1. 출발 — 전력망 Jacobian 위의 *물리적* 측정 사실

`docs/strumpack-vs-custom-multifrontal-8387.md` §2.2 와 ncu 결과 종합:

| 사실 | 측정값 (case8387pegase, N=14908) | 함의 |
|---|---|---|
| max front size | **96** | dense LU 영역 (fsz≥256) 의 38% 도 못 됨 |
| 96.2% of fronts have fsz ≤ 16 | 7,136 / 7,421 panels | 거의 모든 work가 "tiny front" 영역 |
| etree depth | 31 levels | 깊은 dependency chain |
| front arena (FP64) | ~4 MB | 24 GB GPU memory의 0.02% |
| 본 솔버의 factor 본체 (`mf_factor_extend_level`) ncu | SM 4.4%, DRAM 1.8%, FP64 25.3% | **latency-bound** — GPU 어느 자원도 saturate 못 함 |
| 본 솔버의 solve fwd/bwd ncu | SM 1.6~6.6%, DRAM 1.6~2.1% | **latency-bound** |

다른 power-grid 케이스(ACTIVSg25k 등)도 정성적으로 같은 분포 (`docs/related-work-and-contribution.md` §2). 즉 *이 매트릭스 클래스의 본질*이다.

→ **GPU compute를 더 짜내는 것은 leverage 아님.** factor 본체가 SM 4.4% 인데, *"SM 100% 까지 끌어올린다"* 가 best case 인데도 그 자체로 22× 만 빠르게 만든다. 실제로는 latency-bound 에서 SM% 끌어올리기 자체가 매우 어렵다 (per-call work가 너무 작음). leverage는 다른 곳에 있다.

---

## 2. 그러면 leverage는 어디 — *"호출 경로의 host-side overhead 전체"*

같은 case8387pegase 에서 STRUMPACK의 NR iter 2 factor wall=14.32 ms 의 분해 (`docs/strumpack-vs-custom-multifrontal-8387.md` §5.1):

| 구성 | 시간 | 메커니즘 |
|---|---:|---|
| GPU kernel work | ~5 ms (35%) | 1000+ tiny vbatched kernels |
| `cudaLaunchKernel` overhead (~700 calls × 6 μs) | ~4 ms (28%) | 한 NR iter에 700+ launch |
| 페이지락/디바이스 alloc | ~2 ms (14%) | 매 iter buffer 재할당 |
| H2D memcpy | ~2 ms (14%) | 작은 청크 다수 |
| host-side multifrontal scheduling | ~1 ms (7%) | front 별 dispatch loop |

→ **wall 의 65% 가 GPU 작업 *외* host overhead**. 같은 매트릭스 위 custom 의 factor wall=0.64 ms 중 94% 가 GPU kernel work — 나머지 6% 만 overhead.

이 6%는 *"latency-bound 영역에서 host overhead를 어디까지 줄일 수 있는가"* 의 한계점이다. custom 은 그 한계까지 8단의 설계 결정을 통해 압축했다.

---

## 3. *어떻게* — 8단의 설계 결정과 각각 제거한 cost

각 결정은 *power-grid Jacobian 의 도메인 가정* 위에서만 정당하다. 가정과 함께 표시한다.

### D1. CUDA Graph capture + replay (NR iter 의 launch overhead 제거)

**가정**: sparsity pattern 이 NR iter 동안 변하지 않음 (= NR loop 의 정의).

**결정**: `analyze()` 단계에서 factor 와 solve 의 kernel 시퀀스를 각각 `cudaGraph` 로 capture. NR iter 마다는 `cudaGraphLaunch` 한 번씩 (`cudaGraphInstantiate` 는 2회, `cudaGraphLaunch` 는 NR iter 당 2회 = factor + solve).

**측정 증거** (`docs/nr-profile-3way-analysis.md` §3 + nsys `/tmp/bench/nsys/custom_nr_8387.nsys-rep`):

| 구성 | STRUMPACK | custom |
|---|---:|---:|
| NR iter 당 host→device launch | ~700 (`cudaLaunchKernel`) | **1 (`cudaGraphLaunch`)** |
| launch overhead per iter | ~4 ms | ~0.02 ms |

→ **launch overhead 200× 감소**. NR loop 의 정의 자체가 graph capture 의 정당성을 만든다.

### D2. 3-tier 커널 라우팅 (front 분포에 맞춤)

**가정**: max fsz < 160 — 본 도메인 측정 사실 (§1).

**결정**: front 의 fsz 에 따라 세 가지 GPU kernel 중 하나로 dispatch (`docs/fp32-batched-factor-solve-optimization.md`):
- **fsz ≤ 32**: warp-per-front kernel `mf_factor_small_warp_b` — 1 warp/front, 8 warps/block, per-warp shared memory, `__syncwarp()` 만 사용. block barrier 없음.
- **32 < fsz ≤ 159**: shared-resident mid-front kernel `mf_factor_mid_tc32_b` — 프론트 전체를 dynamic shared memory(레벨별 최대 fsz²에 맞춰 sized) 에 한 번 로드, panel LU/U-solve/trailing/extend-add 가 모두 shared 위, L/U 만 write-back.
- **fsz > 159**: 1024-thread/block big-separator kernel — 큰 separator (9~25 fronts/level) 에 다수 warp packing 으로 sequential dependency 은닉.

case8387pegase 에서:
- fsz ≤ 32 fronts: 7,359 / 7,421 (99.2%) → warp-per-front 경로
- 32 < fsz ≤ 96 fronts: 62 / 7,421 (0.8%) → shared-resident 경로
- fsz > 159: 0 (사용 안 됨)

**측정 증거** (`docs/fp32-batched-factor-solve-optimization.md`):
- ACTIVSg25k 의 dominant bottom level (fsz ≤ 16, 4114 fronts): 2.47 → **1.12 ms (compute-bound 76%)**
- 같은 work 를 vbatched 로 처리하면 SM% 가 ~10% (`docs/strumpack-vs-custom-multifrontal-8387.md` §3.2 의 STRUMPACK ncu 참고: `trsm_template_vbatched<8,64>` SM 19%, `<2,32>` SM 0%)

→ **dense LU sweet spot 밖 영역에서 warp-단위 / shared-resident 디자인이 vbatched 디자인보다 적합**. 가정의 정당성은 측정사실 §1 의 max fsz=96.

### D3. Fused factor + extend-add (level 당 kernel launch 1회로)

**가정**: etree 의 부모 panel 이 자식보다 strictly 높은 level 에 있음 — multifrontal LU 의 정의.

**결정** (`src/factorize/multifrontal.cu`): 한 block 안에서 (a) 프론트의 panel LU, (b) trailing update, (c) 부모 front 로 `atomicAdd` extend-add 를 모두 수행. 부모가 strictly 위 level 이라 race-free.

이전 디자인: factor kernel + extend kernel = level 당 2 launch + 1 inter-kernel sync.

**측정 증거**: case8387pegase 의 31 levels × 1 fused kernel = 31 launches per factor. 만약 fused 안 되면 62 launches + 31 syncs. SyntheticUSA 의 72 levels 에서는 누적 효과 더 크다.

→ **graph node 수 절반, level 당 sync 1회 제거**. 직접 launch overhead 감소 (custom 기준)이자 graph 의 sparsity 감소.

### D4. No partial pivoting (row-swap kernel 자체를 제거)

**가정**: NR pre-scaling 으로 Jacobian이 numerically benign — 전력 조류 NR loop 에서 cuPF (또는 호출자) 가 처리.

**결정**: dense LU 를 *no partial pivoting* 으로 진행. pivot row 가 안전하다고 가정 (singular이면 `Status::FactorizationFailed` 로 fail-fast).

비교: STRUMPACK 은 partial pivoting 유지 → `laswp_vbatch_kernel` 47 launches/iter, **SM 0.1% / DRAM 0.2%** = 순수 overhead (`docs/strumpack-vs-custom-multifrontal-8387.md` §3.2).

**측정 증거**: custom 의 ncu 에 `laswp` 같은 row-swap kernel 자체가 없음 — 카테고리 통째로 제거.

→ STRUMPACK 47 launches × ~6 μs launch overhead = ~280 μs / iter 가 사라짐. 더 중요하게는 *"pivoting decision 의 host-side scheduling"* 도 사라짐. 가정의 cost: NR pre-scaling 책임이 호출자 쪽으로 이전.

### D5. Device-resident solve (CPU fallback 제거)

**가정**: RHS/solution이 device 메모리 에 있음 (cuPF 통합 시점에 자연스러움).

**결정** (`src/solve/multifrontal.cu`): solve forward/backward kernel 이 device 포인터를 그대로 받음. analyze 시점에 solve graph 도 함께 capture.

비교: STRUMPACK 의 *"Solve is performed on CPU"* warning — solve 가 호스트 메모리 + CPU 경로 (`docs/strumpack-power-grid-analysis.md` §관찰 2). 매 NR iter 마다 RHS H2D, 솔루션 D2H. solve 22.9 ms (case_ACTIVSg25k) vs cuDSS 0.7 ms 의 격차.

**측정 증거** (case8387pegase, ncu):
- custom solve fwd: 0.34 ms
- STRUMPACK solve: 11.24 ms
- 격차의 대부분이 CPU fallback + H2D/D2H roundtrip

→ 같은 latency-bound 영역에서도 *"GPU 위에 머무는"* 게 훨씬 빠름.

### D6. analyze 시점 메모리 종결 (NR iter 중 alloc/free 없음)

**가정**: NR iter 의 모든 buffer 크기가 analyze 시점에 결정 가능 (sparsity 고정).

**결정**: 모든 `cudaMalloc` 을 analyze 시점에 한 번. NR iter 중에는 새 alloc 없음, 모두 사전 할당된 arena 재사용.

비교: STRUMPACK 의 NR iter 중 `cudaMallocHost` 344 calls + `cudaFreeHost` 344 calls per run (`docs/strumpack-nsys-nr-profile.md` §5). 페이지락 풀 캐시 없음.

**측정 증거** (case8387pegase nsys):
- custom: `cudaMalloc` 26 calls 전체, 모두 setup + analyze 단계. NR iter 자체에는 alloc 없음.
- STRUMPACK: per-iter alloc/free 누적 ~2 ms.

→ NR loop steady-state 의 *"alloc-free-amortization"* cost 가 0 으로 수렴.

### D7. GPU symmetric adjacency 그래프 빌드 (analyze 단축)

**가정**: METIS 가 host 에서 nested dissection 을 수행해야 함 (METIS 자체가 CPU 라이브러리).

**결정** (`src/reordering/metis_nd.cpp` + `matrix::build_symmetric_graph_device`): METIS 입력 그래프 (`xadj`, `adjncy`) 를 GPU 에서 빌드 — `thrust::sort/unique` 로 directed edge keys → CSR 변환. CPU `build_symmetric_adjacency` 의 single-threaded 단계 제거.

**측정 증거** (`docs/fsa-optimization-report.md`):
- 9241 case analyze: 19 ms → 경감 (graph build 가 18.6 ms → 1 ms 수준)
- 소/중규모 case 에서 **analyze −22~34%**

→ analyze 는 NR loop 의 first iter 에만 영향이지만, 자주 행렬을 새로 만드는 시나리오 (Monte-Carlo, contingency screening) 에서는 매번 비용. 줄이는 게 의미 있음.

### D8. 단일 kernel 값 scatter (NR iter 의 value update 단순화)

**가정**: NR iter 의 sparsity pattern 이 고정 → 매 iter 마다 CSR row_ptr/col_idx 는 그대로, values 만 device 위에서 업데이트.

**결정**: NR iter 의 update_values 단계는 host → device memcpy (or in-place 디바이스 update) + `mf_scatter_csr_values` kernel 1회 launch. 이 kernel 이 CSR values 를 ordered multifrontal arena 의 정확한 위치로 직접 scatter.

비교: STRUMPACK 의 `STRUMPACK_update_csr_matrix_values` 는 host-side scheduling + 다중 D2H/H2D + multifrontal arena 재구성. case8387pegase iter2.update_values 가 1.48 ms (custom의 update_values 가 0.11 ms 와 비교).

**측정 증거** (ncu):
- `mf_scatter_csr_values`: SM 4.1%, DRAM **67.0%**, warp 65.8% — **bandwidth-bound (정상)**, 2 instances (NR iter 당 1)
- 단일 kernel call. host scheduling 없음.

→ value 업데이트가 *"GPU 가 자기 bandwidth 의 67% 를 한 번에 쓰는 단일 launch"* 로 압축.

---

## 4. 누적 효과 — *왜 곱셈으로 누적되는가*

case8387pegase 의 NR iter wall 비교:

| 구성 | STRUMPACK | custom | 차이 배수 | 어떤 D 가 만들었나 |
|---|---:|---:|---:|---|
| GPU kernel work | ~5 ms | ~0.6 ms | 8× | D2 (front 라우팅), D3 (fused), D4 (pivot 제거) |
| launch overhead | ~4 ms | ~0.02 ms | 200× | **D1 (CUDA Graph)** |
| alloc / memcpy | ~4 ms | ~0.02 ms | 200× | D6 (analyze-time alloc), D8 (단일 scatter) |
| host scheduling | ~1 ms | ~0.02 ms | 50× | D1 (graph 내부에 compile), D5 (device-resident) |

**총 차이 26× (= 14.32 / 0.64)** = 단일 개선의 합이 아니라 *"각 구성에서 작은 단축이 곱셈으로 누적"*. 가장 큰 단일 leverage 는 D1 (Graph). 하지만 D1 만으로는 부족 — kernel 자체가 효율적이지 않으면 graph 안에 비효율이 컴파일됨.

이게 *"같은 multifrontal LU 인데 26× 차이"* 의 정확한 메커니즘이다. 알고리즘 자체가 아니라 8개 도메인 가정 위 결정의 합 효과.

---

## 5. *왜 이게 STRUMPACK 으로는 불가능했는가* — 정직성

각 결정 (D1~D8) 은 가정을 깬다 — STRUMPACK 이 의도적으로 회피한 가정들:

| D | custom 의 가정 | STRUMPACK 이 회피하는 이유 |
|---|---|---|
| D1 | sparsity 고정 (NR loop) | 일반 sparse direct API 는 sparsity 변경 허용 |
| D2 | max fsz < 160 | 일반 PDE 에서는 큰 dense fronts 정상 — vbatched 가 맞음 |
| D3 | etree level 단조 | 표준이지만 fused 가 일반 lib 에 잘 안 들어감 |
| D4 | NR pre-scaling | 일반 솔버는 numerical 안정성 위해 partial pivoting 필수 |
| D5 | device-resident I/O | host pointer API 가 더 일반적 |
| D6 | NR iter 만 반복 | 일반 사용은 다양한 시나리오 |
| D7 | GPU 있음 | CPU-only 빌드에서 동작해야 함 |
| D8 | 값만 변함 | 일반 sparse update 는 pattern 도 변할 수 있음 |

STRUMPACK 은 *general-purpose sparse direct* 로서 이 모든 가정을 깨야 하는 입력 (큰 PDE, sparsity 변경, pivoting 필요한 ill-conditioned 행렬, host RHS) 을 처리할 수 있어야 한다. 그 generality 의 cost 가 본 매트릭스에서의 26× 손해다.

→ custom 의 우위는 *"알고리즘 발명"* 이 아니라 *"general lib 이 양보 못 하는 8가지 가정을 동시에 받아들임 으로써 얻는 leverage"* 다.

---

## 6. 한계 — 가정이 깨지면 어떻게 되는가

| 가정 깨짐 | 결과 |
|---|---|
| front_total > 8 GB doubles | analyze 가 `Status::AnalysisFailed` 로 fail-fast (`multifrontal.cu:583`). paper 행렬이 정확히 이 경우 (`docs/strumpack-reproduction-report.md` §3.4) |
| 매우 큰 단일 separator (fsz » 1024) | 1024-thread big-separator kernel 의 점진적 효율 저하 |
| numerically ill-conditioned (pivot 0) | `Status::FactorizationFailed`, NR loop 가 fallback 필요 |
| RHS/solution이 host memory | API 가 device 포인터만 받음 — 호출자가 변환 책임 |
| 같은 pattern 위 batched 가 아닌 단발 (single) | graph capture 의 overhead 가 회수 안 됨 — 한 번만 푸는 시나리오엔 비최적 |
| FP32 정확도 부족한 ill-conditioned grid | TC32 path 가 25k/70k 위에서 발산 보고 (`docs/fp32-batched-factor-solve-optimization.md`) |
| sparsity pattern 변경 | NR loop 의 정의 위반 — re-analyze 필요 |

본 솔버는 *"8가지 가정의 모든 적용 영역 안에서만"* 우세하다. 그 안에서는 leverage 가 곱셈으로 누적. 그 밖에서는 STRUMPACK/cuDSS/KLU 같은 general 솔버에 의존해야 함.

---

## 7. 본 분석의 정직성 게이트

- 측정은 case8387pegase 1 case 에서 NR iter 2 steady-state 단일 측정. case3120sp ~ ACTIVSg70k 의 다른 power-grid case 들 (`docs/fp32-batched-factor-solve-optimization.md`) 은 정성적으로 같은 분포지만, ncu kernel bound 분류 는 case8387pegase 만 수행.
- ncu 측정에서 일부 kernel 의 SM% 가 0.0% 인 게 *"GPU 가 아무것도 안 함"* 의 직접적 의미는 아님 — kernel 이 너무 짧아서 ncu 의 sampling 해상도 밖일 수 있음. 다만 다른 측정 (warp%, FP64%) 도 모두 낮아서 latency-bound 결론은 robust.
- vs cuDSS 의 차이 1.24× (`docs/nr-profile-3way-analysis.md`) 는 본 분석에 직접 들어가지 않음. 본 문서는 *"왜 custom 이 빠른가"* 의 메커니즘 분해.
- 정성적으로 같은 분석을 GPU bus dim 30k+ 케이스 (SyntheticUSA n=156255) 에서 검증할 수는 있으나 본 솔버가 그 영역에서 별도 한계로 fail함 (`docs/strumpack-reproduction-report.md` §3.5). 더 강한 일반화는 별도 검증 필요.

---

## 8. 한 줄 요약

전력망 Jacobian 위에서 **GPU 의 compute / bandwidth 는 어차피 saturate 못 함**. 솔버 빠르기를 결정하는 것은 *"호출 경로의 host-side overhead 를 얼마나 잘라낼 수 있는가"* 이고, 그건 도메인 가정을 얼마나 받아들이는가의 함수다.

`custom_linear_solver` 는 8가지 가정 (NR sparsity 고정, fsz<160, no pivot, device I/O, etc.) 위에서 CUDA Graph replay + front-size routing + fused factor/extend + analyze-time memory closure 의 곱셈 효과로 STRUMPACK 대비 26× 빨라진다. 알고리즘 발명이 아닌 *generality 와 trade한 8단의 도메인 특화 설계*가 본질.

---

## 9. 출처

본 저장소:
- `docs/api-and-build-design.md` — 공개 API, 빌드, 복사한 파일 인벤토리 (D1 graph capture, D5 device API 의 출처)
- `docs/related-work-and-contribution.md` — 외부 솔버 landscape, novelty 자체 평가
- `docs/fp32-batched-factor-solve-optimization.md` — D2 (3-tier kernel routing) 의 측정 데이터
- `docs/analyze-bottleneck-and-optimization.md`, `docs/fsa-optimization-report.md` — D7 (GPU symmetric graph) 의 출처와 −22~34% 효과
- `docs/tensor-core-factorize-design.md` — TC32 negative result + small-front kernel 디자인
- `docs/warm-cache-stack-port-expectations.md` — 단일-케이스 vs cuDSS 측정
- `docs/baseline-vs-strumpack.md` — STRUMPACK GPU 의 9가지 한계 매핑
- `docs/strumpack-reproduction-report.md` — STRUMPACK paper Table 2 매트릭스 재현 결과
- `docs/strumpack-power-grid-analysis.md` — wall-clock vs kernel-only 측정
- `docs/strumpack-nsys-nr-profile.md` — STRUMPACK NR profile (1500 launches/iter)
- `docs/nr-profile-3way-analysis.md` — 3-way NR profile (STRUMPACK/cuDSS/custom)
- `docs/strumpack-vs-custom-multifrontal-8387.md` — case8387pegase ncu kernel bound 분류 + front 분포 + Janna paper 데이터셋 분석

외부:
- Spatula (MICRO 2023) — STRUMPACK 0.004% peak on FullChip
- Claus, Ghysels, Boukaram, Li, IJHPCA 2025 — STRUMPACK GPU + BLR 논문 (1.87× over cuDSS 주장의 출처)
- Karypis–Kumar 1998 (METIS), Liu 1986 (etree), Davis 2006 (CSparse), Duff–Reid 1983 (multifrontal) — 알고리즘 표준
