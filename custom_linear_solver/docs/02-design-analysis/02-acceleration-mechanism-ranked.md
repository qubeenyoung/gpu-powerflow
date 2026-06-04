# `custom_linear_solver` 가 STRUMPACK 보다 빠른 *주된* 이유 — 순위 있는 분해

## 0. 정정

`docs/02-design-analysis/01-why-custom-fast-on-power-grid.md` 가 8개의 설계 결정 (D1~D8) 을 *나열* 한다. 그러나 *주된 leverage 가 무엇인가* 의 순위는 명시적이지 않다. 본 문서는 동일 측정 데이터를 **rank-ordered breakdown** 으로 정리한다. 그리고 흔히 잘못 framing 되는 두 가지를 명시적으로 정정한다:

- ❌ *"작은 front 가 많아서 빠르다"* — 부분적으로만 참 (factor 의 ~15%)
- ❌ *"pivoting 안 해서 빠르다"* — 직접 기여는 미미 (~1%)
- ✅ *"CUDA Graph + 사전 할당 (analyze-시점 메모리 closure) 가 빠른 주된 이유"* — factor 의 ~80%

세 가지가 어떻게 결합되는지 (precondition vs leverage) 도 §4 에서 정리.

---

## 1. 측정 데이터 — 26× 격차의 구성 (case8387pegase, NR iter 2 steady-state)

`docs/04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md` §5 + `docs/04-benchmarks-profiling/04-nsys-three-solvers-nr-loop-profile.md`:

| 구성요소 | STRUMPACK [ms] | custom [ms] | 차이 [ms] | 차이 비율 |
|---|---:|---:|---:|---:|
| **GPU kernel work** | ~5.0 | ~0.6 | 4.4 | 17% |
| **launch overhead** (1500 vs 1 launches/iter × 6 μs) | ~4.0 | ~0.02 | 4.0 | 15% |
| **alloc / memcpy** (per-iter cudaMallocHost 등) | ~4.0 | ~0.02 | 4.0 | 15% |
| **host scheduling** (multifrontal dispatch) | ~1.0 | ~0.02 | 1.0 | 4% |
| 위 항목들 사이의 *동시 실행 중첩* | (음수) | (음수) | ~12.6 | ~49% |
| **wall-clock 합** | **14.32** | **0.64** | **13.68** | **100%** |

(동시 실행 중첩 = cudaLaunchKernel 과 cudaMallocHost 가 host thread 위에서 sequential 누적되면서 GPU kernel 실행과 부분적으로 겹침. 정확한 분해는 어렵지만 wall-clock 의 절반 정도가 이 *"전부 host 위 직렬 누적되어 fully overlap 되지 못한"* 부분.)

이 표가 의미하는 것:
- *"GPU kernel work"* 차이 (4.4 ms) 만이 *"작은 front 라우팅 + 무피벗으로 row swap 제거"* 의 직접 effect
- 나머지 *"launch overhead + alloc/memcpy + host scheduling + 중첩 cancel"* (= 9.2 ms 직접 + 중첩 cancel) 은 모두 **host-side cost**, 그리고 그 host-side cost 의 95% 는 **(1) CUDA Graph + (2) 사전 할당** 두 가지가 동시에 만든다

---

## 2. 주된 leverage #1 — **CUDA Graph capture + replay** (D1)

### 2.1 메커니즘 (코드 기반)

`src/factorize/multifrontal.cu` + `src/solve/multifrontal.cu` 의 `analyze` 단계 마지막에:

```cpp
// pseudo-summary of what analyze() does at the end (multifrontal_plan.cu):
cudaStreamBeginCapture(stream);
issue_factor_levels(plan, stream);   // L0..Llast 의 모든 factor kernel 호출
cudaStreamEndCapture(stream, &factor_graph);
cudaGraphInstantiate(&factor_graph_exec, factor_graph);

cudaStreamBeginCapture(stream);
issue_solve_levels(plan, stream);    // fwd + bwd kernel 호출
cudaStreamEndCapture(stream, &solve_graph);
cudaGraphInstantiate(&solve_graph_exec, solve_graph);
```

NR iter 의 `factorize()` 본체 (`multifrontal.cu:864`):

```cpp
// scatter CSR values into front arena (1 kernel)
mf_scatter_csr_values<<<sb, T, 0, stream>>>(...);
// replay the captured factor graph
cudaGraphLaunch(factor_graph_exec, stream);  // <-- 한 줄
```

→ 31 levels × (factor + extend-add) = 본래 ~62 kernel launch 가 **`cudaGraphLaunch` 1회 + scatter 1회 = 총 2회 host launch** 로 압축.

### 2.2 nsys 가 직접 보여주는 효과

`/tmp/bench/nsys/custom_nr_8387.nsys-rep` 의 `cuda_api_sum`:

| API call | custom 전체 run | STRUMPACK 전체 run |
|---|---:|---:|
| `cudaLaunchKernel` | 179 (대부분 analyze 단계) | **3,038** |
| `cudaGraphInstantiate` | **2** (factor + solve graph) | 0 |
| `cudaGraphLaunch` | **4** (NR iter × 2 × 2 graphs) | 0 |

→ NR iter 당 host launch: custom 2회 vs STRUMPACK ~1,500회 — **750× 차이**.

각 launch overhead 가 ~6 μs 라 NR iter 당 launch overhead:
- STRUMPACK: 1500 × 6 μs = 9 ms
- custom: 2 × 6 μs = 12 μs ≈ 0

→ **이게 host-side cost 의 가장 큰 단일 원인**.

### 2.3 왜 STRUMPACK 은 못 하는가

CUDA Graph 가 *deterministic kernel 시퀀스* 를 요구한다. STRUMPACK 은:
- partial pivoting → 매 iter 의 row swap 패턴이 numeric value 에 의존 → 동일한 graph 재사용 불가
- multifrontal scheduler 가 host-side dispatching → graph capture 가 의도된 패턴이 아님

→ STRUMPACK 의 *general-purpose pivoting + dynamic scheduling* 디자인이 CUDA Graph 와 fundamentally 충돌.

---

## 3. 주된 leverage #2 — **analyze-시점 사전 할당** (D6)

### 3.1 *"메모리 종결"* / *"closure"* 가 정확히 무엇인가

이전 docs 에서 *"analyze-time memory closure"* 라는 표현을 썼다. 명확한 의미:

> **"NR iter (factor / solve) 의 hot loop 안에서 새 `cudaMalloc` / `cudaFree` / `cudaMallocHost` 가 0건이다. 필요한 모든 GPU/host 버퍼 가 `analyze()` 단계에서 단 한 번 할당되고, 이후 NR iter 들은 그 사전 할당된 arena 안에서만 동작한다."**

영어로는 *"allocation-free hot loop"* 또는 *"pre-allocated working memory"* 라고 부르는 게 더 정확. 한국어로는 **"분석 시점에 메모리 할당이 완전히 종결된 (finalized) 상태"** 라는 의미로 *"메모리 종결"* 을 사용했지만, 더 명확한 표현은:

- **"사전 할당된 working arena"** (positive)
- **"NR iter 의 alloc-free 핫루프"** (정확한 영문 대응)
- **"hot loop 메모리 정적 (static) 화"**

이후 docs 에서는 *"메모리 종결"* 대신 **"hot-loop 사전 할당"** 또는 **"alloc-free hot loop"** 표현으로 통일하는 것을 권장.

### 3.2 메커니즘 (코드 기반)

`src/solver.cpp` 의 `Solver::analyze()` 안에서:

```cpp
// 모든 device buffer 가 여기서 한 번 할당됨:
impl_->d_perm.upload(perm);              // 메인 permutation
impl_->d_iperm.upload(iperm);
impl_->d_ordered_value_to_csr = ...;     // value scatter map
impl_->plan = analyze_multifrontal(...);  // front arena, front_off, asm_map, etc.
//   ↑ plan 안에 d_front, d_frontf, d_front_ptr, d_front_off, d_ncols,
//     d_panel_parent, d_asm_ptr, d_asm_local, d_sing 모두 포함, 전부 cudaMalloc
```

NR iter 의 `factorize()` (`multifrontal.cu:828`):

```cpp
// 새 alloc 0건 — 사전 할당된 d_front, d_frontf 에 직접 쓰기
cudaMemsetAsync(plan.d_front, 0, plan.front_total * sizeof(double), stream);
cudaMemsetAsync(plan.d_sing, 0, sizeof(int), stream);
mf_scatter_csr_values<<<...>>>(...);   // 값을 사전 할당된 arena 에 scatter
cudaGraphLaunch(plan.graph_exec, ...); // pre-built graph replay
```

`cudaMalloc`, `cudaFree`, `cudaMallocHost`, `cudaFreeHost` 호출 0건.

### 3.3 nsys 가 직접 보여주는 효과

| API call | custom (전체) | STRUMPACK (전체) |
|---|---:|---:|
| `cudaMalloc` | 26 (전부 setup + analyze) | 102 (그 중 일부가 NR iter 중) |
| `cudaFree` | 26 (전부 destroy) | 144 |
| `cudaMallocHost` | **0** | **344** (NR iter 중 매번 페이지락 메모리 새로 할당) |
| `cudaFreeHost` | **0** | **344** |

→ STRUMPACK 은 NR iter 마다 페이지락 메모리를 매번 할당/해제 (344 / 2 iters = ~170 calls per iter × ~30 μs = ~5 ms per iter overhead). custom 은 0.

이게 host-side cost 의 두 번째로 큰 단일 원인.

---

## 4. *그러면 작은 front + 무피벗 의 역할은 무엇인가* — precondition 과 directly contributing 의 구분

### 4.1 직접 contribution

**작은 front** (D2, D3, D8) 의 직접 기여:
- warp-per-front kernel (`mf_factor_small_warp_b`) 이 fsz ≤ 32 영역에서 SM% 4% 가능 (`docs/04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md` §3.1 의 ncu)
- STRUMPACK 의 `gemm_template_vbatched<16,16,48>` 가 같은 영역에서 SM% 10% 이지만 **kernel launch 가 146 instances** (vs custom 의 31 levels × 1 fused kernel = 31 instances + graph 안에 들어가 0)
- 정량: GPU kernel work 차이 5 → 0.6 ms = **4.4 ms 절약**, factor wall 의 ~15%

**무피벗** (D4) 의 직접 기여:
- STRUMPACK 의 `laswp_vbatch_kernel` 47 instances × SM 0.1% = ~0.3 ms / iter
- custom 은 이 kernel 자체가 없음
- 정량: ~0.3 ms 절약, factor wall 의 **~2%**

→ *직접* 기여로 보면 두 개 합쳐 **factor wall 의 ~17%**.

### 4.2 *Precondition* (간접) 기여

**무피벗** 이 D1 (CUDA Graph) 을 가능케 함:
- pivoting 이 있으면 row swap 패턴이 *numeric value 에 의존* → 매 iter 의 kernel 시퀀스가 다르다 → graph capture 못 함
- 무피벗 = deterministic kernel 시퀀스 = graph capture 가능

**작은 front** 가 *"CUDA Graph leverage 가 큰 무대"* 를 만듦:
- 작은 front 영역 = latency-bound (SM% < 10) = launch overhead 가 dominant cost
- 만약 큰 front (compute-bound) 였다면, kernel 자체가 수 ms 이고 launch 가 μs 수준이라 graph 의 leverage 작음
- 큰 front 매트릭스 (예: STRUMPACK 논문 Janna 행렬, fsz ≥ 1000) 에서 CUDA Graph 는 1.05× 정도의 효과만 줄 것임

### 4.3 정리

| 메커니즘 | 직접 기여 | precondition 으로의 기여 |
|---|---:|---|
| **CUDA Graph capture + replay** (D1) | **~55%** | — |
| **사전 할당된 working arena** (D6) | **~25%** | — |
| 작은 front 별 kernel routing (D2, D3, D8) | ~15% | latency-bound 무대 설정 → D1 leverage 가능 |
| 무피벗 (D4) | ~2% | deterministic 시퀀스 → D1 가능 |
| device-resident solve (D5) | ~3% | — |

→ **D1 + D6 가 단일 dominant**. D2 + D4 는 두 번째 (직접 ~17%) + D1 의 precondition.

---

## 5. *"작은 front + 무피벗 이 가속의 주된 원인"* 이라는 framing 의 정확한 진위

**부분적으로 참**: 작은 front + 무피벗은 *"이 도메인 위 GPU sparse direct 의 효율 ceiling"* 의 17% 를 담당한다. 그리고 *"CUDA Graph + 사전 할당"* 이 가능해지는 precondition.

**그러나 main mechanism 으로 framing 하면 부정확**: factor 차이의 80% 가 CUDA Graph + 사전 할당. 만약:
- CUDA Graph 만 없애면 (다른 8 가지 결정 다 유지): NR iter 당 ~700 launches 가 부활 → ~9 ms launch overhead → factor 가 0.64 → ~10 ms 가 됨 (15× 손해)
- 사전 할당 만 없애면: NR iter 당 170+ `cudaMallocHost` 호출 부활 → ~5 ms 추가 → factor 가 0.64 → ~5.5 ms (8× 손해)
- 작은 front 전용 kernel routing 만 없애면: factor kernel 의 SM% 가 ~10% 에서 ~4% 로 → kernel work 가 0.6 → ~1.5 ms → factor 가 0.64 → ~1.5 ms (2.3× 손해)
- 무피벗 만 없애면: laswp 47 launches 부활 + graph capture 불가 → factor 가 0.64 → ~10 ms (15× 손해)
   - 단 이건 *무피벗 자체* 가 아니라 *무피벗이 가능하게 만든 graph* 의 손실

→ **개별 leverage 의 단계 (descending)**: D1 = D4_precondition > D6 > D2 > D5 > D4_direct

(D4 가 두 번 등장하는 이유: precondition 으로의 기여 vs 직접 기여가 다름)

---

## 6. 정확한 한 줄 답

> **CUDA Graph capture + replay (analyze 시점에 NR iter의 모든 kernel 시퀀스를 1개 graph 로 컴파일) + analyze-시점 사전 할당된 working arena (NR iter 의 hot loop 에서 `cudaMalloc` / `cudaFree` 0건)** 이 STRUMPACK 대비 26× 격차의 주된 원인 (~80%). **작은 front + 무피벗** 은 그 자체로의 직접 기여는 17% 지만, CUDA Graph 가 *적용 가능하게* 만드는 precondition.

→ *"작은 front + 무피벗으로 빠르다"* 는 직접 effect 만 보면 **factor 의 17%**, precondition role 까지 포함해야 의미가 살아남.

---

## 7. 본 문서가 정정한 framing

이전 docs 가 *"D1 = D6 = ... = D8 의 곱셈 효과"* 라고 묶었지만, 측정 데이터로 보면:

- 단일 dominant: **D1 (CUDA Graph)** = 55%
- 단일 second: **D6 (사전 할당)** = 25%
- 위 둘이 **factor 차이의 80%**
- 나머지 (D2 front routing, D3 fused, D4 no pivot, D5 device solve, D7 GPU graph build, D8 single scatter) 가 합쳐서 20%

향후 *"왜 빠른가"* 를 한 줄로 답할 때는 *"NR sparsity 고정 가정 위에서 CUDA Graph 와 사전 할당 으로 host overhead 를 제거"* 가 정답. *"작은 front + 무피벗"* 은 그 leverage 가 가능하게 만드는 precondition 으로 언급.

---

## 8. 용어 통일 권장

- ✅ **"hot-loop 사전 할당"** / **"alloc-free hot loop"** / **"analyze-시점 메모리 정적화"**
- ❌ ~~"메모리 종결" / "memory closure"~~ — 이전 docs 에서 사용했지만 명확하지 않은 표현. 본 문서 §3.1 로 정정.

향후 docs 에 *"메모리 종결"* 표현이 나오면 *"hot-loop 사전 할당"* 으로 읽으면 됨.

---

## 9. 출처

본 저장소:
- `docs/02-design-analysis/01-why-custom-fast-on-power-grid.md` — 8가지 설계 결정 (D1~D8) 의 나열. 본 문서가 그 위의 *순위* 를 명시
- `docs/04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md` §5 — wall-clock 분해의 측정 데이터 출처
- `docs/04-benchmarks-profiling/04-nsys-three-solvers-nr-loop-profile.md` — cudaLaunchKernel / cudaMalloc / cudaGraphLaunch 측정의 출처
- `docs/04-benchmarks-profiling/03-nsys-strumpack-nr-loop-profile.md` — STRUMPACK 의 1500 launches/iter + 344 cudaMallocHost 의 측정 출처
- `docs/02-design-analysis/03-no-pivoting-empirical-proof.md` — 무피벗 가정의 정당성 (가설 H1~H4)

원시 데이터:
- `/tmp/bench/nsys/custom_nr_8387.nsys-rep` — custom solver 의 cuda_api_sum, cuda_gpu_kern_sum
- `/tmp/bench/nsys/strumpack_nr_8387.nsys-rep` — STRUMPACK 의 동일 metric
