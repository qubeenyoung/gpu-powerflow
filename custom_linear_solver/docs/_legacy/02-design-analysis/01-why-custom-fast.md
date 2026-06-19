# `custom_linear_solver` 가 전력망 Jacobian 에서 빠른 이유 — 설계 분해 + 가속 메커니즘 순위

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: 전력망 Jacobian 은 latency-bound 라 GPU compute 가 아니라 host-side overhead 제거가 leverage 이고, CUDA Graph + 사전 할당 (~80%) 이 STRUMPACK 대비 26× 격차의 주된 원인이다.

본 문서는 두 가지를 함께 담는다: (1) `custom_linear_solver` 가 전력망 Newton-Raphson Jacobian 위에서 빠른 **메커니즘** 의 D1–D8 설계 분해, (2) 그 중 *실제로 무엇이 가속을 만드는가* 의 **순위 있는 정량 분해**. 비교 분석(vs STRUMPACK, vs cuDSS)은 별도 문서들에 있다(§9 참고).

핵심 명제는 다음 한 줄로 요약된다:

> *전력망 Jacobian 의 dense LU work 자체가 너무 작아 GPU compute/bandwidth 를 saturate 못 하는 latency-bound 영역에 머문다. 이 영역에서 빠른 솔버는 GPU compute 를 더 짜내는 솔버가 아니라, **GPU 호출의 host-side overhead 를 모두 제거한 솔버** 다. custom_linear_solver 는 그 호출 경로를 8단의 도메인 가정 위 설계로 함께 압축한다.*

흔히 잘못 framing 되는 두 가지를 명시적으로 정정한다:

- ❌ *"작은 front 가 많아서 빠르다"* — 부분적으로만 참 (factor 의 ~15%)
- ❌ *"pivoting 안 해서 빠르다"* — 직접 기여는 미미 (~1~2%)
- ✅ *"CUDA Graph + 사전 할당 (analyze-시점 메모리 closure) 가 빠른 주된 이유"* — factor 차이의 ~80%

---

## 1. 출발 — 전력망 Jacobian 위의 *물리적* 측정 사실

`docs/04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md` §2.2 와 ncu 결과 종합:

| 사실 | 측정값 (case8387pegase, N=14908) | 함의 |
|---|---|---|
| max front size | **96** | dense LU 영역 (fsz≥256) 의 38% 도 못 됨 |
| 96.2% of fronts have fsz ≤ 16 | 7,136 / 7,421 panels | 거의 모든 work 가 "small front" 영역 |
| etree depth | 31 levels | 깊은 dependency chain |
| front arena (FP64) | ~4 MB | 24 GB GPU memory 의 0.02% |
| factor 본체 (`mf_factor_extend_level`) ncu | SM 4.4%, DRAM 1.8%, FP64 25.3% | **latency-bound** — GPU 어느 자원도 saturate 못 함 |
| solve fwd/bwd ncu | SM 1.6~6.6%, DRAM 1.6~2.1% | **latency-bound** |

다른 power-grid 케이스(ACTIVSg25k 등)도 정성적으로 같은 분포 (`docs/01-orientation/02-related-work-and-novelty.md` §2). 즉 *이 매트릭스 클래스의 본질* 이다.

→ **GPU compute 를 더 짜내는 것은 leverage 아님.** factor 본체가 SM 4.4% 인데 *"SM 100% 까지 끌어올린다"* 가 best case 인데도 그 자체로 22× 만 빠르게 만든다. 실제로는 latency-bound 에서 SM% 끌어올리기 자체가 매우 어렵다 (per-call work 가 너무 작음). leverage 는 다른 곳에 있다.

---

## 2. leverage 는 어디 — *"호출 경로의 host-side overhead 전체"*

같은 case8387pegase 에서 STRUMPACK 의 NR iter 2 factor wall=14.32 ms 의 분해 (`docs/04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md` §5.1):

| 구성 | 시간 | 메커니즘 |
|---|---:|---|
| GPU kernel work | ~5 ms (35%) | 1000+ small vbatched kernels |
| `cudaLaunchKernel` overhead (~700 calls × 6 μs) | ~4 ms (28%) | 한 NR iter 에 700+ launch |
| 페이지락/디바이스 alloc | ~2 ms (14%) | 매 iter buffer 재할당 |
| H2D memcpy | ~2 ms (14%) | 작은 청크 다수 |
| host-side multifrontal scheduling | ~1 ms (7%) | front 별 dispatch loop |

→ **wall 의 65% 가 GPU 작업 *외* host overhead**. 같은 매트릭스 위 custom 의 factor wall=0.64 ms 중 94% 가 GPU kernel work — 나머지 6% 만 overhead. 이 6% 는 *"latency-bound 영역에서 host overhead 를 어디까지 줄일 수 있는가"* 의 한계점이다.

### 2.1 26× 격차의 구성 (case8387pegase, NR iter 2 steady-state)

`docs/04-benchmarks-profiling/05-...case8387.md` §5 + `docs/04-benchmarks-profiling/04-nsys-three-solvers-nr-loop-profile.md`:

| 구성요소 | STRUMPACK [ms] | custom [ms] | 차이 [ms] | 차이 비율 |
|---|---:|---:|---:|---:|
| **GPU kernel work** | ~5.0 | ~0.6 | 4.4 | 17% |
| **launch overhead** (1500 vs 1 launches/iter × 6 μs) | ~4.0 | ~0.02 | 4.0 | 15% |
| **alloc / memcpy** (per-iter cudaMallocHost 등) | ~4.0 | ~0.02 | 4.0 | 15% |
| **host scheduling** (multifrontal dispatch) | ~1.0 | ~0.02 | 1.0 | 4% |
| 위 항목들 사이의 *동시 실행 중첩* | (음수) | (음수) | ~12.6 | ~49% |
| **wall-clock 합** | **14.32** | **0.64** | **13.68** | **100%** |

(동시 실행 중첩 = `cudaLaunchKernel` 과 `cudaMallocHost` 가 host thread 위에서 sequential 누적되면서 GPU kernel 실행과 부분적으로 겹침. wall-clock 의 절반 정도가 이 *"host 위 직렬 누적되어 fully overlap 되지 못한"* 부분.)

→ *"GPU kernel work"* 차이 (4.4 ms) 만이 *"작은 front 라우팅 + 무피벗으로 row swap 제거"* 의 직접 effect. 나머지 host-side cost 의 95% 는 **(1) CUDA Graph + (2) 사전 할당** 두 가지가 동시에 만든다.

---

## 3. *어떻게* — 8단의 설계 결정과 각각 제거한 cost

각 결정은 *power-grid Jacobian 의 도메인 가정* 위에서만 정당하다. 가정과 함께 표시한다.

### D1. CUDA Graph capture + replay — NR iter 의 launch overhead 제거 (주된 leverage #1)

**가정**: sparsity pattern 이 NR iter 동안 변하지 않음 (= NR loop 의 정의).

**결정**: `analyze()` 단계에서 factor 와 solve 의 kernel 시퀀스를 각각 `cudaGraph` 로 capture. NR iter 마다는 `cudaGraphLaunch` 한 번씩 (`cudaGraphInstantiate` 2회, `cudaGraphLaunch` NR iter 당 2회 = factor + solve).

```cpp
// analyze() 끝 (multifrontal_plan.cu):
cudaStreamBeginCapture(stream);
issue_factor_levels(plan, stream);   // L0..Llast 의 모든 factor kernel 호출
cudaStreamEndCapture(stream, &factor_graph);
cudaGraphInstantiate(&factor_graph_exec, factor_graph);
// solve graph 도 동일하게 capture
```

```cpp
// NR iter 의 factorize() 본체 (multifrontal.cu:864):
mf_scatter_csr_values<<<sb, T, 0, stream>>>(...);  // scatter 1회
cudaGraphLaunch(factor_graph_exec, stream);        // replay 1회 (한 줄)
```

→ 31 levels × (factor + extend-add) = 본래 ~62 kernel launch 가 **`cudaGraphLaunch` 1회 + scatter 1회 = 총 2회 host launch** 로 압축.

**측정 증거** (`/tmp/bench/nsys/custom_nr_8387.nsys-rep` 의 `cuda_api_sum`):

| API call | custom 전체 run | STRUMPACK 전체 run |
|---|---:|---:|
| `cudaLaunchKernel` | 179 (대부분 analyze 단계) | **3,038** |
| `cudaGraphInstantiate` | **2** (factor + solve graph) | 0 |
| `cudaGraphLaunch` | **4** (NR iter × 2 × 2 graphs) | 0 |

| 구성 | STRUMPACK | custom |
|---|---:|---:|
| NR iter 당 host→device launch | ~700~1,500 | **1~2 (`cudaGraphLaunch`)** |
| launch overhead per iter | ~4~9 ms | ~0.02 ms |

→ NR iter 당 host launch 750× 차이. 각 launch overhead ~6 μs → STRUMPACK 1500 × 6 μs = 9 ms, custom 2 × 6 μs ≈ 0. **이게 host-side cost 의 가장 큰 단일 원인.**

**왜 STRUMPACK 은 못 하는가**: CUDA Graph 가 *deterministic kernel 시퀀스* 를 요구한다. STRUMPACK 의 partial pivoting → row swap 패턴이 numeric value 에 의존 → graph 재사용 불가. multifrontal scheduler 의 host-side dispatching 도 capture 의도와 충돌. *general-purpose pivoting + dynamic scheduling* 디자인이 CUDA Graph 와 fundamentally 충돌.

### D2. 3-tier 커널 라우팅 (front 분포에 맞춤)

**가정**: 작은 power-grid 케이스는 거의 모든 front 가 fsz ≤ 32 — 본 도메인 측정 사실 (§1).

**결정**: front 의 fsz 에 따라 세 가지 GPU kernel 중 하나로 dispatch (`src/internal/types.hpp` `ClassifyFrontTier`, `src/factorize/{small,mid,big}.cuh`):
- **fsz ≤ 32**: warp-packed sub-group kernel `FactorSmall` — 1 warp/front, 8 warps/block, per-warp shared memory, `__syncwarp()` 만 사용. block barrier 없음. (좁은 레벨 under-fill 시 FactorMid 로 돌리는 occupancy 게이트.)
- **33 ≤ fsz ≤ 64**: shared-resident whole-front kernel `FactorMid` — 프론트 전체를 dynamic shared memory(레벨별 최대 fsz²에 맞춰 sized) 에 한 번 로드, panel LU/U-solve/trailing/extend-add 가 모두 shared 위, L/U 만 write-back. 1 block/front.
- **fsz > 64**: global-resident multi-block kernel `FactorBig` — 큰 separator 를 여러 블록에 분산(pivot + L/U 패널 + trailing multi-block), L/U 패널 타일만 staging.

case8387pegase 에서: front 의 ~99% 가 fsz ≤ 32 → small 경로, 나머지 소수만 mid/big.

**측정 증거**:
- ACTIVSg25k 의 dominant bottom level (fsz ≤ 16, 4114 fronts): 2.47 → **1.12 ms (compute-bound 76%)**
- 같은 work 를 vbatched 로 처리하면 SM% ~10% (STRUMPACK `trsm_template_vbatched<8,64>` SM 19%, `<2,32>` SM 0%)

→ **dense LU sweet spot 밖 영역에서 warp-단위 / shared-resident 디자인이 vbatched 디자인보다 적합**. 정당성은 측정사실 §1 의 max fsz=96.

### D3. Fused factor + extend-add (level 당 kernel launch 1회로)

**가정**: etree 의 부모 panel 이 자식보다 strictly 높은 level 에 있음 — multifrontal LU 의 정의.

**결정** (`src/factorize/front_ops.cuh` extend-add, `src/factorize/{small,mid,big}.cuh`): 한 block 안에서 (a) 프론트의 panel LU, (b) trailing update, (c) 부모 front 로 `atomicAdd` extend-add 를 모두 수행. 부모가 strictly 위 level 이라 race-free.

이전 디자인: factor kernel + extend kernel = level 당 2 launch + 1 inter-kernel sync.

**측정 증거**: case8387pegase 31 levels × 1 fused kernel = 31 launches per factor. fused 안 되면 62 launches + 31 syncs. SyntheticUSA 의 72 levels 에서는 누적 효과 더 크다.

→ **graph node 수 절반, level 당 sync 1회 제거**.

### D4. No partial pivoting (row-swap kernel 자체를 제거)

**가정**: NR pre-scaling 으로 Jacobian 이 numerically benign — 전력 조류 NR loop 에서 cuPF (또는 호출자) 가 처리. 근거는 `02-no-pivoting-proof.md` (H1~H4).

**결정**: dense LU 를 *no partial pivoting* 으로 진행. pivot row 가 안전하다고 가정 (singular 이면 `Status::FactorizationFailed` 로 fail-fast).

비교: STRUMPACK 은 partial pivoting 유지 → `laswp_vbatch_kernel` 47 launches/iter, **SM 0.1% / DRAM 0.2%** = 순수 overhead.

**측정 증거**: custom 의 ncu 에 `laswp` 같은 row-swap kernel 자체가 없음 — 카테고리 통째로 제거.

→ STRUMPACK 47 launches × ~6 μs = ~280 μs / iter 가 사라짐. 더 중요하게는 *"pivoting decision 의 host-side scheduling"* 도 사라짐. 가정의 cost: NR pre-scaling 책임이 호출자 쪽으로 이전.

### D5. Device-resident solve (CPU fallback 제거)

**가정**: RHS/solution 이 device 메모리에 있음 (cuPF 통합 시점에 자연스러움).

**결정** (`src/solve/multifrontal.cu`): solve forward/backward kernel 이 device 포인터를 그대로 받음. analyze 시점에 solve graph 도 함께 capture.

비교: STRUMPACK 의 *"Solve is performed on CPU"* warning — solve 가 호스트 메모리 + CPU 경로. 매 NR iter 마다 RHS H2D, 솔루션 D2H. solve 22.9 ms (case_ACTIVSg25k) vs cuDSS 0.7 ms 의 격차.

**측정 증거** (case8387pegase, ncu): custom solve fwd 0.34 ms vs STRUMPACK solve 11.24 ms. 격차의 대부분이 CPU fallback + H2D/D2H roundtrip.

### D6. analyze 시점 메모리 종결 (NR iter 중 alloc/free 없음) (주된 leverage #2)

**가정**: NR iter 의 모든 buffer 크기가 analyze 시점에 결정 가능 (sparsity 고정).

**정의 (정확화)**:

> **"NR iter (factor / solve) 의 hot loop 안에서 새 `cudaMalloc` / `cudaFree` / `cudaMallocHost` 가 0건이다. 필요한 모든 GPU/host 버퍼가 `analyze()` 단계에서 단 한 번 할당되고, 이후 NR iter 들은 그 사전 할당된 arena 안에서만 동작한다."**

영어로는 *"allocation-free hot loop"* / *"pre-allocated working memory"* 가 정확. 이전 docs 의 *"메모리 종결" / "memory closure"* 는 모호하므로 **"hot-loop 사전 할당"** / **"alloc-free hot loop"** 로 통일한다.

**결정**: 모든 `cudaMalloc` 을 analyze 시점에 한 번. NR iter 중에는 새 alloc 없음, 모두 사전 할당된 arena 재사용.

```cpp
// NR iter 의 factorize() (multifrontal.cu:828): 새 alloc 0건
cudaMemsetAsync(plan.d_front, 0, plan.front_total * sizeof(double), stream);
mf_scatter_csr_values<<<...>>>(...);   // 사전 할당된 arena 에 scatter
cudaGraphLaunch(plan.graph_exec, ...); // pre-built graph replay
```

**측정 증거** (case8387pegase nsys):

| API call | custom (전체) | STRUMPACK (전체) |
|---|---:|---:|
| `cudaMalloc` | 26 (전부 setup + analyze) | 102 (일부 NR iter 중) |
| `cudaFree` | 26 (전부 destroy) | 144 |
| `cudaMallocHost` | **0** | **344** (NR iter 중 매번 페이지락 새로 할당) |
| `cudaFreeHost` | **0** | **344** |

→ STRUMPACK 은 NR iter 마다 페이지락 메모리 매번 할당/해제 (344 / 2 iters ≈ 170 calls/iter × ~30 μs ≈ ~5 ms/iter overhead). custom 은 0. **host-side cost 의 두 번째로 큰 단일 원인.**

### D7. GPU symmetric adjacency 그래프 빌드 (analyze 단축)

**가정**: METIS 가 host 에서 nested dissection 수행 (METIS 자체가 CPU 라이브러리).

**결정** (`src/reordering/metis_nd.cpp` + `matrix::build_symmetric_graph_device`): METIS 입력 그래프 (`xadj`, `adjncy`) 를 GPU 에서 빌드 — `thrust::sort/unique` 로 directed edge keys → CSR 변환. CPU `build_symmetric_adjacency` 의 single-threaded 단계 제거.

**측정 증거**: 9241 case analyze 19 ms → 경감 (graph build 18.6 ms → 1 ms 수준). 소/중규모 case 에서 **analyze −22~34%**.

→ analyze 는 NR loop 의 first iter 에만 영향이지만, 자주 행렬을 새로 만드는 시나리오 (Monte-Carlo, contingency screening) 에서는 매번 비용.

### D8. 단일 kernel 값 scatter (NR iter 의 value update 단순화)

**가정**: NR iter 의 sparsity pattern 고정 → 매 iter 마다 CSR row_ptr/col_idx 그대로, values 만 device 위에서 업데이트.

**결정**: NR iter 의 update_values 단계는 host → device memcpy (or in-place 디바이스 update) + `mf_scatter_csr_values` kernel 1회 launch. 이 kernel 이 CSR values 를 ordered multifrontal arena 의 정확한 위치로 직접 scatter.

비교: STRUMPACK 의 `STRUMPACK_update_csr_matrix_values` 는 host-side scheduling + 다중 D2H/H2D + multifrontal arena 재구성. case8387pegase iter2.update_values 1.48 ms vs custom 0.11 ms.

**측정 증거** (ncu): `mf_scatter_csr_values` SM 4.1%, DRAM **67.0%**, warp 65.8% — **bandwidth-bound (정상)**, 2 instances (NR iter 당 1). 단일 kernel call, host scheduling 없음.

---

## 4. 순위 — 어떤 D 가 실제로 가속을 만드는가

### 4.1 직접 contribution vs precondition

**작은 front** (D2, D3, D8) 의 직접 기여:
- warp-per-front kernel 이 fsz ≤ 32 영역에서 SM% 4% 가능
- STRUMPACK `gemm_template_vbatched<16,16,48>` 가 같은 영역 SM% 10% 이지만 launch 146 instances (vs custom 31 levels × 1 fused = 31 instances + graph 안에 들어가 0)
- 정량: GPU kernel work 5 → 0.6 ms = **4.4 ms 절약, factor wall 의 ~15%**

**무피벗** (D4) 의 직접 기여:
- STRUMPACK `laswp_vbatch_kernel` 47 instances × SM 0.1% ≈ 0.3 ms / iter, custom 은 이 kernel 자체가 없음
- 정량: ~0.3 ms 절약, **factor wall 의 ~2%**

→ *직접* 기여 합 **factor wall 의 ~17%**.

**Precondition (간접) 기여**:
- **무피벗 이 D1 (CUDA Graph) 을 가능케 함**: pivoting 이 있으면 row swap 패턴이 numeric value 에 의존 → 매 iter kernel 시퀀스가 다름 → graph capture 못 함. 무피벗 = deterministic = graph 가능.
- **작은 front 가 "CUDA Graph leverage 가 큰 무대" 를 만듦**: 작은 front = latency-bound (SM% < 10) = launch overhead 가 dominant. 큰 front (compute-bound) 였다면 kernel 자체가 수 ms 이고 launch 가 μs 수준이라 graph leverage 작음. 큰 front 매트릭스 (Janna 행렬, fsz ≥ 1000) 에서 CUDA Graph 는 1.05× 효과만.

### 4.2 순위 정리

| 메커니즘 | 직접 기여 | precondition 으로의 기여 |
|---|---:|---|
| **CUDA Graph capture + replay** (D1) | **~55%** | — |
| **사전 할당된 working arena** (D6) | **~25%** | — |
| 작은 front 별 kernel routing (D2, D3, D8) | ~15% | latency-bound 무대 설정 → D1 leverage 가능 |
| 무피벗 (D4) | ~2% | deterministic 시퀀스 → D1 가능 |
| device-resident solve (D5) | ~3% | — |

→ **D1 + D6 가 단일 dominant (= factor 차이의 80%)**. D2 + D4 는 두 번째 (직접 ~17%) + D1 의 precondition.

### 4.3 ablation — 각 D 를 하나씩 끄면

- CUDA Graph 만 없애면: NR iter 당 ~700 launches 부활 → ~9 ms launch overhead → factor 0.64 → ~10 ms (**15× 손해**)
- 사전 할당 만 없애면: NR iter 당 170+ `cudaMallocHost` 부활 → ~5 ms 추가 → factor 0.64 → ~5.5 ms (**8× 손해**)
- 작은 front routing 만 없애면: SM% ~10% → ~4% → kernel work 0.6 → ~1.5 ms → factor ~1.5 ms (**2.3× 손해**)
- 무피벗 만 없애면: laswp 부활 + graph capture 불가 → factor 0.64 → ~10 ms (**15× 손해**, 단 이건 *무피벗 자체* 가 아니라 *무피벗이 가능하게 만든 graph* 의 손실)

→ **개별 leverage 단계 (descending)**: D1 = D4_precondition > D6 > D2 > D5 > D4_direct.

---

## 5. 누적 효과 — *왜 곱셈으로 누적되는가*

| 구성 | STRUMPACK | custom | 차이 배수 | 어떤 D 가 만들었나 |
|---|---:|---:|---:|---|
| GPU kernel work | ~5 ms | ~0.6 ms | 8× | D2 (front 라우팅), D3 (fused), D4 (pivot 제거) |
| launch overhead | ~4 ms | ~0.02 ms | 200× | **D1 (CUDA Graph)** |
| alloc / memcpy | ~4 ms | ~0.02 ms | 200× | D6 (analyze-time alloc), D8 (단일 scatter) |
| host scheduling | ~1 ms | ~0.02 ms | 50× | D1 (graph 내부에 compile), D5 (device-resident) |

**총 차이 26× (= 14.32 / 0.64)** = 단일 개선의 합이 아니라 *"각 구성에서 작은 단축이 곱셈으로 누적"*. 가장 큰 단일 leverage 는 D1 (Graph). 하지만 D1 만으로는 부족 — kernel 자체가 효율적이지 않으면 graph 안에 비효율이 컴파일됨. 이게 *"같은 multifrontal LU 인데 26× 차이"* 의 정확한 메커니즘이다.

---

## 6. *왜 이게 STRUMPACK 으로는 불가능했는가* — 정직성

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

→ custom 의 우위는 *"알고리즘 발명"* 이 아니라 *"general lib 이 양보 못 하는 8가지 가정을 동시에 받아들임으로써 얻는 leverage"* 다.

---

## 7. 한계 — 가정이 깨지면 어떻게 되는가

| 가정 깨짐 | 결과 |
|---|---|
| front_total > 8 GB doubles | analyze 가 `Status::AnalysisFailed` 로 fail-fast (`multifrontal.cu:583`). paper 행렬이 정확히 이 경우 |
| 매우 큰 단일 separator (fsz » 1024) | 1024-thread big-separator kernel 의 점진적 효율 저하 |
| numerically ill-conditioned (pivot 0) | `Status::FactorizationFailed`, NR loop 가 fallback 필요 |
| RHS/solution 이 host memory | API 가 device 포인터만 받음 — 호출자가 변환 책임 |
| 같은 pattern 위 batched 가 아닌 단발 (single) | graph capture 의 overhead 가 회수 안 됨 |
| FP32 정확도 부족한 ill-conditioned grid | TC32 path 가 25k/70k 위에서 발산 보고 |
| sparsity pattern 변경 | NR loop 의 정의 위반 — re-analyze 필요 |

본 솔버는 *"8가지 가정의 모든 적용 영역 안에서만"* 우세하다. 그 밖에서는 STRUMPACK/cuDSS/KLU 같은 general 솔버에 의존해야 함.

---

## 8. 정직성 게이트

- 측정은 case8387pegase 1 case 에서 NR iter 2 steady-state 단일 측정. case3120sp ~ ACTIVSg70k 의 다른 power-grid case 들은 정성적으로 같은 분포지만, ncu kernel bound 분류는 case8387pegase 만 수행.
- ncu 에서 일부 kernel 의 SM% 가 0.0% 인 게 *"GPU 가 아무것도 안 함"* 의 직접적 의미는 아님 — kernel 이 너무 짧아 ncu sampling 해상도 밖일 수 있음. 다만 다른 측정 (warp%, FP64%) 도 모두 낮아 latency-bound 결론은 robust.
- vs cuDSS 의 차이 1.24× 는 본 분석에 직접 들어가지 않음. 본 문서는 *"왜 custom 이 빠른가"* 의 메커니즘 분해.
- SyntheticUSA (n=156255) 에서는 본 솔버가 별도 한계로 fail. 더 강한 일반화는 별도 검증 필요.

---

## 9. 출처

본 저장소:
- `../main-report.md` — 전체 서사 맥락
- `docs/01-orientation/01-api-and-build-design.md` — 공개 API, 빌드 (D1 graph capture, D5 device API 의 출처)
- `docs/01-orientation/02-related-work-and-novelty.md` — 외부 솔버 landscape, novelty 자체 평가
- `docs/01-orientation/03-lineage-strumpack-not-the-baseline.md` — STRUMPACK GPU 의 9가지 한계 매핑
- [`02-no-pivoting-proof.md`](02-no-pivoting-proof.md) — 무피벗 가정의 정당성 (H1~H4)
- [`03-multifrontal-vs-strumpack.md`](03-multifrontal-vs-strumpack.md) — 소스 레벨 STRUMPACK 비교
- [`04-gemm-fraction-tc-ceiling.md`](04-gemm-fraction-tc-ceiling.md) — trailing GEMM 비중 + TC ceiling
- tier 임계값 근거(small\|mid=32 / mid\|big=64) — `src/internal/types.hpp` 주석
- [`../03-optimization-notes/03-tensor-core-investigation.md`](../03-optimization-notes/03-tensor-core-investigation.md) — TC32 negative result
- [`../04-benchmarks-profiling/03-gemm-fraction-front-distribution.md`](../04-benchmarks-profiling/03-gemm-fraction-front-distribution.md) — front 분포
- `docs/04-benchmarks-profiling/04-nsys-three-solvers-nr-loop-profile.md` — 3-way NR profile (cudaLaunchKernel / cudaMalloc / cudaGraphLaunch 측정 출처)
- `docs/04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md` — case8387pegase ncu kernel bound 분류 + wall-clock 분해 + front 분포

원시 데이터:
- `/tmp/bench/nsys/custom_nr_8387.nsys-rep`, `/tmp/bench/nsys/strumpack_nr_8387.nsys-rep` — cuda_api_sum, cuda_gpu_kern_sum

외부:
- Spatula (MICRO 2023) — STRUMPACK 0.004% peak on FullChip
- Claus, Ghysels, Boukaram, Li, IJHPCA 2025 — STRUMPACK GPU + BLR (1.87× over cuDSS 주장 출처)
- Karypis–Kumar 1998 (METIS), Liu 1986 (etree), Davis 2006 (CSparse), Duff–Reid 1983 (multifrontal) — 알고리즘 표준
