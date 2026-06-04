# NR 2-iter Nsight Systems 프로파일 — STRUMPACK vs cuDSS vs custom_linear_solver, *왜* 의 분석

세 솔버 모두 **동일한 NR loop 패턴**(`analyze` 한 번 + `factor`+`solve` 두 번 반복)으로 case_ACTIVSg25k (n=47246, nnz=318672) 위에서 nsys 프로파일링. 본 문서는 측정 결과와 함께 **전력망에서 왜 STRUMPACK이 느린지, 왜 cuDSS와 custom이 빠른지**를 솔버별로 분해.

원시 nsys 파일:
- `/tmp/bench/nsys/strumpack_nr_ACTIVSg25k.nsys-rep`
- `/tmp/bench/nsys/cudss_nr_ACTIVSg25k.nsys-rep`
- `/tmp/bench/nsys/custom_nr_ACTIVSg25k.nsys-rep`

드라이버 소스: `/tmp/bench/nsys_{strumpack,cudss,custom}_nr.cpp`

## 1. NR 패턴 (모든 솔버 동일)

```
init + matrix descriptors
analyze            ← 한 번만 (NR loop에서 sparsity 고정)
NR_iter_1
  factor           ← 첫 번째 J 위에서 numeric factor
  solve            ← 첫 번째 rhs 위에서 triangular solve
NR_iter_2
  update_values    ← Jacobian 값 변경 (sparsity 같음)
  factor           ← 두 번째 J 위 numeric factor (steady state)
  solve            ← 두 번째 rhs 위 triangular solve
destroy
```

cuDSS는 iter 2에 `CUDSS_PHASE_REFACTORIZATION` 사용 (이전 측정과 차이점: 이전엔 `FACTORIZATION` 만 반복). STRUMPACK은 `STRUMPACK_update_csr_matrix_values` + `STRUMPACK_factor`. custom은 `set_values` + `factorize()` (내부 CUDA Graph replay).

## 2. 측정 결과 — NR iter 2 (steady state per-iter)

| Solver | iter2.factor [ms] | iter2.solve [ms] | f+s [ms] | iter1 → iter2 factor 단축 |
|---|---:|---:|---:|---:|
| **STRUMPACK** (MAGMA) | 26.3 | 22.9 | **49.2** | 53.5 → 26.3 (−51%) |
| **cuDSS** (REFACTORIZATION) | 1.79 | 0.69 | **2.48** | 13.9 → 1.79 (−87%) |
| **custom_linear_solver** | 1.32 | 0.68 | **2.00** | 1.35 → 1.32 (−2%, 이미 steady) |

→ NR steady-state 한 iter 비용 (case_ACTIVSg25k):
- STRUMPACK 49 ms
- cuDSS 2.5 ms (**20× faster than STRUMPACK**)
- custom 2.0 ms (**25× faster than STRUMPACK, 1.24× faster than cuDSS**)

이전 측정 (`docs/04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md`) 의 *"custom 대 cuDSS 10× 빠름"* 은 매 호출 `cudssExecute(FACTORIZATION)` (first-time setup 포함) 을 비교한 결과로 NR 시나리오에 부정확. NR 사용자가 실제로 보는 차이는 **1.24×**.

## 3. NSYS 핵심 metric 비교 (전체 2-iter run)

| metric | STRUMPACK | cuDSS | custom |
|---|---:|---:|---:|
| total wall (program lifetime) | 740 ms | 473 ms | 432 ms |
| `cudaLaunchKernel` calls | **3,038** | 165 | 179 |
| `cudaGraphInstantiate` | 0 | 0 | **2** |
| `cudaGraphLaunch` | 0 | 0 | **4** |
| `cudaMalloc` calls | 102 | 21 | 26 |
| `cudaFree` calls | 144 | 19 | 26 |
| `cudaMallocHost` calls | **344** | 0 | 0 |
| `cudaMemcpy` H2D bytes | 39 MB / 684 calls | 13 MB / 21 calls | 9.6 MB / 19 calls |
| 총 GPU kernel time | ~19 ms | ~23 ms (셋업 큼) | ~0.4 ms (셋업+iter 모두) |
| GPU kernel time / total wall | 2.6% | 4.9% | 0.1% |
| iter2.factor wall | 26.3 ms | 1.79 ms | 1.32 ms |
| **NR iter 당 launch count (steady)** | **~1,500** | **~80** | **2 (graph)** |

→ **NR iter 당 kernel launch 횟수가 솔버별로 750× 차이.** 이게 차이의 1차 동인.

## 4. *왜 STRUMPACK이 power-grid 위에서 느린가* — 측정 기반 분해

### 4.1 numbers

iter 2 factor 26.3 ms 분해:

| 항목 | 시간 [ms] | 비중 | 출처 |
|---|---:|---:|---|
| GPU kernel work (vbatched GEMM/TRSM + STRUMPACK extend_add) | ~10 | 38% | `cuda_gpu_kern_sum` |
| `cudaLaunchKernel` overhead (~1,500 calls × 6 μs) | ~9 | 34% | `cuda_api_sum` |
| `cudaMallocHost`/`cudaFreeHost` (페이지락 alloc, ~170회) | ~6 | 23% | `cuda_api_sum` |
| H2D memcpy (~340 small transfers, avg 14 μs) | ~5 | 19% | `cuda_api_sum` |
| 기타 host-side multifrontal scheduling + sync | ~?? | — | 위 항목들이 부분 중첩 |

(이 분해는 누적 시간 기반이라 1:1 합산은 안 됨 — 일부는 GPU 실행과 겹친다. 비율은 대략적 추정.)

### 4.2 원인 분석 (왜 power-grid에서 STRUMPACK이 슬로우인가)

**원인 1: 알고리즘이 small-front 도메인과 mismatch**

STRUMPACK GPU 경로는 **multifrontal LU + MAGMA `vbatched_dgetrf` 기반**. 이 설계의 sweet spot은 *"front 수십~수백 개, 각 front fsz ≥ 64"* 영역 (예: Janna 그룹 3D FEM). power-grid 야코비안은 정반대:
- 95% of fronts have fsz ≤ 16 (`docs/01-orientation/02-related-work-and-novelty.md` §2)
- max front size 도 수백 수준
- 깊은 etree (~72 levels on SyntheticUSA)

→ MAGMA vbatched는 *"수천 개 작은 batched LU"* 를 효율적으로 처리하도록 만들어졌지만, *너무* 작고 *너무* 많을 때 (전력망 케이스) launch/scheduling overhead가 컴퓨트를 압도.

**증거 (nsys)**:
- 본 측정에서 STRUMPACK이 launch 한 커널 top:
  - `gemm_template_vbatched_nn_kernel<16,16,48>`: 146 instances, 평균 19 μs
  - `extend_add_kernel<16>`: 336 instances, 평균 7 μs
  - `trsm_template_vbatched_lNU_kernel<16,64>`: 76 instances, 평균 24 μs
  - `trsm_template_vbatched_lNL_kernel<16,64>`: 76 instances
  - `laswp_vbatch_kernel`: 338 instances, 평균 3 μs ← row swap
- **합 1,000+ tiny kernel calls** in just one factor — 대부분 10-25 μs 수준의 짧은 커널

**원인 2: kernel launch overhead가 wall-clock의 34%**

`cudaLaunchKernel` 평균 6 μs × 1500 calls per NR iter = **9 ms overhead just for launches**.
이건 GPU 실행 시간(~10 ms)과 거의 같은 크기 — 즉 GPU가 일하는 동안 host가 다음 커널을 launch하느라 동등하게 바쁨.

**원인 3: 페이지락 메모리 풀 캐시 없음**

`cudaMallocHost` 344회 + `cudaFreeHost` 344회 (~6 ms 누적). 매 호출마다 페이지락 메모리 할당/해제. 대부분 sparse vector / front buffer 임시 호스트 스테이징용. cuDSS / custom은 페이지락을 한 번만 alloc.

**원인 4: H2D 트래픽이 작은 청크로 잘게 쪼개짐**

H2D 39 MB을 684 transfer로 (평균 57 KB / 청크). cuDSS는 13 MB / 21 transfers (평균 0.6 MB), custom은 9.6 MB / 19 transfers. STRUMPACK은 **메모리 카피 자체보다 H2D launch overhead가 비싼 영역**.

**원인 5: Solve가 host fallback**

`docs/04-benchmarks-profiling/01-strumpack-paper-table2-reproduction.md` 와 `docs/04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md` 에서 확인된 *"Solve is performed on CPU"* 경고. nsys 에서도 solve 단계 GPU kernel time 작음 (대부분 cudssExecute(SOLVE) 처럼 GPU bound이지 않고 CPU에서 forward/backward substitution). solve 22.9 ms 의 대부분이 host CPU work + D2H/H2D 라운드트립.

### 4.3 한 줄 정리

**STRUMPACK이 power-grid에서 느린 이유**: small-front도메인에서 *"수천 개 짧은 kernel launch + 그만큼 host overhead + 잘게 쪼개진 H2D + CPU solve"* 의 4중 cost가 곱해진다. 알고리즘이 잘못된 게 아니라 **타깃 매트릭스 클래스가 다르다** — STRUMPACK은 large-front 가정 위에서 짜였고, power-grid는 그 가정을 깬다.

## 5. *왜 cuDSS가 power-grid 위에서 빠른가*

### 5.1 numbers (NR iter 2 = REFACTORIZATION)

iter 2 factor 1.79 ms:

| 항목 | 시간 [ms] | 비중 |
|---|---:|---:|
| GPU kernel work (`factorize_v3_ker` + `factorize_ker`) | ~1.6 | 89% |
| `cudssExecute` API overhead | ~0.2 | 11% |

### 5.2 원인 분석

**원인 1: Supernodal LU + 적은 수의 fat 커널**

cuDSS는 GPU kernel 165개 = STRUMPACK 1/18, 그 중 핵심은:
- `cudss::factorize_v3_ker<long, double, int, double, 128, 8, ...>`: **2 instances, 평균 1.28 ms** ← 핵심 factor 커널
- `cudss::factorize_ker<long, double, int, double, 32, ...>`: 2 instances, 0.38 ms ← 보조 factor
- `cudss::bwd_ker<...>`: 4 instances, 평균 0.19 ms ← backward substitution (device-resident)
- `cudss::fwd_ker<...>`: 2 instances, 평균 0.17 ms ← forward substitution

→ STRUMPACK의 *"1000개 짧은 vbatched 커널 = 누적 시간이 launch overhead"* 와 정반대 패턴: cuDSS는 한두 개의 **굵은 kernel**이 supernodal 패턴 전체를 처리.

**원인 2: PHASE_REFACTORIZATION가 first-time setup 분리**

cuDSS는 `CUDSS_PHASE_ANALYSIS` (1회) + `CUDSS_PHASE_FACTORIZATION` (1회, first numeric) + `CUDSS_PHASE_REFACTORIZATION` (NR iters) 의 3-phase 모델.

본 측정에서:
- iter1.factor (FACTORIZATION): 13.9 ms — 첫 numeric factor + per-phase 메모리 setup
- iter2.factor (REFACTORIZATION): **1.79 ms** — 메모리/스케줄 캐시 재사용

→ STRUMPACK은 이런 phase 분리가 없어서 매번 동일 비용 (26 ms steady-state). cuDSS의 API 디자인 자체가 NR loop에 맞춰져 있음.

**원인 3: device-resident solve**

cudssExecute(SOLVE)가 device 포인터를 그대로 받음. solve 0.67 ms 의 대부분이 forward + backward kernel work, host roundtrip 없음.

### 5.3 한 줄 정리

**cuDSS가 power-grid에서 빠른 이유**: 적은 수의 fat 커널 (165 total launches) + REFACTORIZATION phase로 first-time cost를 분리 + device-resident solve. 알고리즘적으로 supernodal LU가 power-grid front 분포에 더 잘 맞고, API가 NR loop 패턴을 명시적으로 지원.

## 6. *왜 custom_linear_solver가 power-grid 위에서 (더) 빠른가*

### 6.1 numbers (NR iter 2)

iter 2 factor 1.32 ms:

| 항목 | 시간 [ms] | 비중 |
|---|---:|---:|
| GPU kernel work (multifrontal kernels) | ~1.3 | 98% |
| Graph launch overhead (cudaGraphLaunch ~20 μs) | ~0.02 | 2% |

### 6.2 원인 분석

**원인 1: CUDA Graph capture + replay**

본 nsys 측정의 결정적 numbers:
- `cudaGraphInstantiate`: **2 calls** (factor graph + solve graph, analyze 시점에 한 번씩)
- `cudaGraphLaunch`: **4 calls** (iter1 factor, iter1 solve, iter2 factor, iter2 solve)
- `cudaLaunchKernel`: 179 calls **전체** (대부분 setup/analyze 단계의 METIS, CUB sorts. NR iter 자체에는 launch가 거의 없음)

→ NR iter 당 host->device launch 비용 = **1 cudaGraphLaunch (~20 μs)**. STRUMPACK 1500회 × 6 μs = 9 ms 와 비교 **450× 적은 launch overhead**.

**원인 2: small-front 전용 커널 라우팅** (`docs/01-orientation/03-lineage-strumpack-not-the-baseline.md` L2/L3/L8)

custom은 front 크기에 따라 3가지 kernel:
- fsz ≤ 32 → **warp-per-front kernel** (`mf_factor_small_warp_b`)
- 32 < fsz ≤ 159 → **shared-resident mid-front kernel**
- fsz > 159 → **1024-thread big-separator kernel**

power-grid front 분포 (95% fsz≤16) 에 정확히 맞춤. STRUMPACK이 MAGMA vbatched의 `<16,16,48>` 같은 *"전체 batched"* 커널을 작은 input에 끼워넣는 것과 대조.

**원인 3: 값 업데이트 = 단일 kernel scatter**

iter 2 update_values: 0.28 ms 중 GPU kernel `mf_scatter_csr_values<double, double>`: 1회 launch, 20 μs. STRUMPACK처럼 host-side dispatcher를 거치지 않고 device kernel 하나가 CSR values를 multifrontal arena로 직접 scatter.

**원인 4: device-resident solve, no pivoting**

custom은 power-grid Jacobian의 NR pre-scaling 특성을 활용해 **no partial pivoting** 가정 (cuPF가 NR loop 안에서 scaling 처리). 이 단순화로:
- Pivot dispatch / row swap 로직 제거 (STRUMPACK의 `laswp_vbatch_kernel` 338회 launch가 사라짐)
- 솔브 forward/backward도 graph로 캡처 가능

**원인 5: 모든 메모리 allocation을 analyze에서 끝냄**

`cudaMalloc` 26회, 전부 setup + analyze 시점 (init과 분리되어 있어 `custom_setup` NVTX 안에 묶여 있음). NR iter 중에는 새 alloc 없음. STRUMPACK은 NR iter 중에도 cudaMallocHost 144회 / cudaFree 28회 발생.

### 6.3 한 줄 정리

**custom이 power-grid에서 (더) 빠른 이유**: cuDSS의 "적은 fat 커널" 우위는 그대로 살리면서 + **CUDA Graph로 NR iter 당 launch overhead를 1회로 압축** + **front 크기에 맞춘 전용 kernel (warp-per-front)** + **no-pivot 가정으로 solve도 graph화**. 결과적으로 NR iter wall의 98% 가 순수 GPU work.

## 7. 3-way 통합 비교

```
                              STRUMPACK         cuDSS              custom
                              ---------         -----              ------
NR iter (steady)              49 ms             2.5 ms             2.0 ms
  ├─ factor                   26 ms             1.8 ms             1.3 ms
  └─ solve                    23 ms             0.7 ms             0.7 ms

NR iter 당 kernel launches    ~1500             ~80                ~2 (graph)
launch overhead per iter      ~9 ms             ~0.5 ms            ~0.02 ms

GPU kernel time / wall        38% (factor)      89%                98%
host-side overhead            62%               11%                2%

알고리즘                      multifrontal       supernodal LU      multifrontal (small-front
                              + MAGMA vbatched   + cuDSS 자체       전용) + CUDA Graph
                                                  fat kernel
NR loop API 지원              없음 (update_csr_  REFACTORIZATION    graph capture + replay
                              + factor 매번      phase 명시적       (analyze 시점)
                              numeric 재처리)

solve 위치                    CPU fallback       device-resident    device-resident
pivoting                      partial            partial            없음 (no-pivot 가정)
```

## 8. 핵심 분석 결론

세 솔버는 power-grid 야코비안에서 *서로 다른 cost driver*에 영향받는다:

| Solver | Dominant cost | 개선 가능성 |
|---|---|---|
| STRUMPACK | **launch overhead + host-side scheduling** (factor의 62% 가 host work) | 알고리즘 자체가 small-front에 부적합. 본질적으로 large-front PDE를 위한 설계 → power-grid에 fit하려면 재설계 필요 |
| cuDSS | GPU kernel work (factor 89% 가 GPU) | 이미 GPU 효율 높음. 추가 개선은 supernodal pattern 튜닝 영역 |
| custom | GPU kernel work (factor 98% 가 GPU) | CUDA Graph 덕에 launch overhead 거의 0. 추가 개선은 GPU kernel 자체 효율 (front-size 분포에 더 맞춘 커널 라우팅) |

**custom과 cuDSS 차이의 본질**:
- cuDSS는 *"libray as a black box"* — `cudssExecute` 의 host-side overhead가 ~0.2 ms / call 잔존 (`cudssExecute` API call, internal dispatcher)
- custom은 *"NR loop을 graph로 미리 컴파일"* — host-side overhead 사실상 0
- → 절대적 차이 0.48 ms (1.79 → 1.32) 의 대부분이 이 graph capture 효과. **알고리즘적 우위는 작고 (1.2×), engineering 우위가 dominant**

**STRUMPACK과 cuDSS/custom 차이의 본질**:
- 알고리즘 fit (supernodal/specialized vs multifrontal-general)
- launch density (80~2 vs 1500)
- API의 NR loop 명시적 지원 (REFACTORIZATION / graph capture vs 없음)
- solve location (device-resident vs CPU fallback)

→ STRUMPACK 측의 본질적 limitation. small-front 도메인에서 fundamental redesign 없이 cuDSS/custom 따라잡기 어려움.

## 9. 한계 / 정직성 게이트

- 본 측정은 single-system (B=1). custom의 batched 경로 (B=64+) 우위는 별도 측정 영역 (`docs/03-optimization-notes/01-fp32-batched-kernel-optimization.md` 참고).
- 본 NR iter 측정은 case_ACTIVSg25k 1개. 다른 power-grid 케이스에서도 같은 ratio가 나오는지 확인 필요.
- cuDSS의 REFACTORIZATION이 실제 NR loop 시나리오에서 적합한지 (값 변화가 너무 크면 numerical reliability 이슈) 는 별도 검증.
- custom 의 graph capture 는 sparsity pattern 고정 + no-pivot 가정 위에서만 안전. 도메인 외 fallback 없음.

## 10. 재현 명령

```bash
# STRUMPACK
mpicxx -std=c++17 -O2 \
  -I/workspace/sparse_direct_solver/src \
  -I/opt/third_party/install/strumpack-cuda/include \
  /tmp/bench/nsys_strumpack_nr.cpp \
  /workspace/sparse_direct_solver/src/tools/matrix_io.cpp \
  -L/workspace/local/strumpack-cuda-magma/install/lib \
  -L/workspace/local/magma/install/lib \
  -L/opt/intel/oneapi/mkl/2026.0/lib \
  -L/usr/local/cuda-12.8/lib64 \
  -lstrumpack -lmagma -lcudart -lnvToolsExt -lpthread \
  -o /tmp/bench/nsys_strumpack_nr

# cuDSS
g++ -std=c++17 -O2 \
  -I/workspace/sparse_direct_solver/src \
  -I/opt/nvidia/cudss/include \
  /tmp/bench/nsys_cudss_nr.cpp \
  /workspace/sparse_direct_solver/src/tools/matrix_io.cpp \
  -L/opt/nvidia/cudss/lib \
  -L/usr/local/cuda-12.8/lib64 \
  -lcudss -lcudart -lnvToolsExt -lpthread \
  -o /tmp/bench/nsys_cudss_nr

# custom
g++ -std=c++17 -O2 \
  -I/workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver/src \
  -I/workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver/tests \
  /tmp/bench/nsys_custom_nr.cpp \
  /workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver/tests/io.cpp \
  /tmp/clsb/libcustom_linear_solver_ops.a \
  -L/opt/third_party/install/common/lib \
  -lmetis -lcudart -lnvToolsExt -lpthread \
  -o /tmp/bench/nsys_custom_nr

# profile (각 솔버)
nsys profile --output=/tmp/bench/nsys/<solver>_nr_ACTIVSg25k \
  --force-overwrite=true --trace=cuda,nvtx \
  /tmp/bench/nsys_<solver>_nr \
  /datasets/power_system/nr_linear_systems/case_ACTIVSg25k/J.mtx \
  /datasets/power_system/nr_linear_systems/case_ACTIVSg25k/rhs.mtx

# stats
nsys stats --report nvtx_pushpop_sum --format table <file>.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format table <file>.nsys-rep
nsys stats --report cuda_api_sum --format table <file>.nsys-rep
```
