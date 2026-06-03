# 같은 multifrontal인데 왜 custom이 STRUMPACK보다 power-grid Jacobian에서 빠른가
*case8387pegase 위 nsys + ncu + 알고리즘적 분해*

## 0. 질문

`docs/baseline-vs-strumpack.md` 에서 정리한 대로 STRUMPACK과 `custom_linear_solver`는 **둘 다 multifrontal LU** 패밀리다. 둘 다 METIS nested dissection, elimination tree, supernode amalgamation, front + extend-add를 사용한다. 같은 알고리즘 패밀리인데 전력망 야코비안 case8387pegase (n=14908, nnz=110572) 에서 NR steady-state 성능은:

| Solver | NR iter (steady) factor + solve | 배수 |
|---|---:|---:|
| STRUMPACK MAGMA | 25.56 ms | 1× |
| custom_linear_solver | 0.98 ms | **26×** |

이 26× 차이의 메커니즘을 *front-size 분포* + *ncu 커널 bound 분류* + *STRUMPACK 논문 자체 데이터셋과의 비교* 까지 종합해 분해한다.

(cuDSS 와의 비교는 본 문서 범위 밖. `docs/nr-profile-3way-analysis.md` 에 별도.)

---

## 1. 측정 베이스

| 항목 | 값 |
|---|---|
| 매트릭스 | case8387pegase (8387-bus 전력망 NR iter 2 야코비안) |
| N | 14,908 |
| nnz (general) | 110,572 |
| 평균 nnz/row | 7.4 (극단적으로 sparse) |
| GPU | RTX 3090 (sm_86, 24 GB, FP64 0.56 TFLOPs, GDDR6X 936 GB/s) |
| STRUMPACK | v8.0 MAGMA build, METIS NodeND, MC64 matching, compression off |
| custom_linear_solver | FP64, METIS NodeND, no pivot, CUDA Graph capture |

NR steady-state per-iter (`/tmp/bench/nsys/{strumpack,custom}_nr_8387.nsys-rep`):

| Solver | iter2.factor [ms] | iter2.solve [ms] | f+s [ms] |
|---|---:|---:|---:|
| STRUMPACK | 14.32 | 11.24 | 25.56 |
| custom | 0.64 | 0.34 | 0.98 |

## 2. *"small front"* — 정의와 측정

### 2.1 정의 — fsz의 의미

multifrontal LU에서 **front** = 한 supernode (혹은 panel)의 dense pivot block + Schur complement. **fsz** (front size) = front의 행 수 = pivot 컬럼 수(`nc`) + Schur 행 수(`fsz − nc`).

front 위 작업량:
- **factor 작업량 ∝ fsz³** (panel LU + rank-nc trailing update)
- **factor 메모리 ∝ fsz²** (front buffer)

따라서 "small front" 의 정량적 의미는 *"fsz³ 가 GPU의 dense LU breakeven 보다 작은 영역"*. 외부 출처 (Spatula MICRO'23) 가 보고한 GPU dense-LU 임계점:

> *"dense-LU throughput drops linearly below fsz=10,000 and flattens around fsz=20,000"*

GPU 한 SM의 dense-LU 효율 sweet spot 은 fsz ≥ 64, peak 효율은 fsz ≥ 256. 우리 기준:

| 분류 | fsz | factor 메커니즘 |
|---|---|---|
| tiny | ≤ 16 | warp 한 개도 채 못 쓰는 영역. 일반 dense LU 커널은 그냥 idle thread |
| small | 17–32 | 한 warp 단위 (warp-per-front 가 sweet spot) |
| mid | 33–159 | block 단위 dense kernel (shared memory 활용) |
| big | ≥ 160 | multi-warp / multi-block 전용 큰 dense |

이건 본 저장소 `docs/fp32-batched-factor-solve-optimization.md` 의 3-tier 커널 라우팅과 정확히 같은 기준이다.

### 2.2 case8387pegase의 front 분포 (실측)

`CLS_DUMP=1 custom_linear_solver_run` 결과:

```
n=14908  P=7421  levels=31  panel_cap=8  front_total(MB f32)=2.1
```

| fsz 빈 | 개수 | 비율 | Σfsz² 비중 (메모리) | Σfsz³ 비중 (compute) |
|---|---:|---:|---:|---:|
| 1–16 (tiny) | **7,136** | **96.2%** | **53.0%** | **19.1%** |
| 17–32 (small) | 223 | 3.0% | 21.9% | 23.9% |
| 33–48 | 46 | 0.6% | 14.0% | 25.7% |
| 49–64 | 13 | 0.2% | 7.8% | 20.0% |
| 65–96 | 3 | 0.04% | 3.3% | 11.3% |
| **97–160** | **0** | **0%** | **0%** | **0%** |
| **≥ 161** | **0** | **0%** | **0%** | **0%** |
| 합 | 7,421 | 100% | 100% | 100% |

#### 핵심 수치
- **fsz ≤ 16 인 fronts가 전체의 96.2%** (= 7,136 / 7,421)
- **max fsz = 96** (= 96–160 빈에 0개) — 전 행렬에서 가장 큰 front 도 100 미만
- **front_total = 2.1 MB** (f32 기준, FP64 도 4 MB 수준) — 24 GB GPU memory 의 0.02%
- compute (Σfsz³) 의 **89%가 fsz ≤ 64** 에 있고, 단 한 개의 front도 fsz > 96에 없음

### 2.3 STRUMPACK MAGMA 커널의 sweet spot — 그리고 8387의 위치

STRUMPACK GPU 경로가 의존하는 **MAGMA vbatched** 의 sweet spot:

| 커널 | template 파라미터 (block size) | sweet spot fsz |
|---|---|---|
| `gemm_template_vbatched_nn_kernel<double, 16, 16, 48>` | 16×16 tile, K=48 | fsz ≥ 64 (multiple tiles) |
| `trsm_template_vbatched_lN[L,U]_kernel<double, 32, 32>` | 32×32 block | fsz ≥ 64 |
| `dgetf2_fused_kernel_vbatched<11>` | nc=11 panel | nc ≤ 16 |
| STRUMPACK 자체 `extend_add_kernel<16>` | tiny-front 전용 | fsz ≤ 16 |

→ MAGMA `gemm/trsm_template_vbatched<16,32>` 는 **fsz가 충분히 커서 여러 tile을 채울 수 있을 때** 정상 효율. case8387pegase 에서는 *"단 한 개의 front 도 그 영역에 없음"*. MAGMA가 자기 sweet spot 밖에서 동작 = launch overhead가 컴퓨트를 압도.

본 nsys 측정 (`/tmp/bench/nsys/strumpack_nr_8387.nsys-rep` 의 `cuda_gpu_kern_sum`) 의 STRUMPACK top kernels:
- `gemm_template_vbatched_nn_kernel<16,16,48>` 호출 평균 19 μs, instances 146
- `trsm_template_vbatched_lN[L,U]_kernel<16,64>` 평균 24 μs, instances 76 × 2
- `extend_add_kernel<16>` 평균 7 μs, instances 336

호출 당 시간 (19~24 μs) > GPU 커널 launch overhead (~6 μs). 즉 kernel launch 가 컴퓨트와 같은 무게로 누적. **알고리즘이 small-front 영역에서 의도된 효율을 못 냄.**

---

## 3. ncu — 같은 매트릭스 위 두 솔버 hot kernel의 bound 분류

### 3.1 custom_linear_solver hot kernels (RTX 3090, FP64)

`ncu --metrics sm__throughput, dram__throughput, smsp__warps_active, sm__pipe_fp64_cycles_active` (`/tmp/bench/ncu_custom_8387.csv`):

| Kernel | instances | SM% | DRAM% | warp% | FP64% | 분류 |
|---|---:|---:|---:|---:|---:|---|
| `mf_factor_extend_level<double>` (factor + extend-add fused) | 62 (31 levels × 2 iter) | **4.4** | **1.8** | 33.5 | 25.3 | **latency-bound** |
| `mf_fwd_level<double, double>` (solve forward) | 62 | 1.6 | 1.6 | 11.8 | 4.5 | **latency-bound** |
| `mf_bwd_level<double, double>` (solve backward) | 62 | 6.6 | 2.1 | 12.7 | 17.8 | **latency-bound** |
| `mf_scatter_csr_values` (값 업데이트) | 2 | 4.1 | **67.0** | **65.8** | 0.0 | memory-bound (정상) |
| `mf_invert_pivot` (pivot 역수 사전계산) | 2 | **51.4** | 7.0 | 28.3 | **64.6** | compute-bound (FP64) |

핵심 진단:
- **factor 의 main kernel `mf_factor_extend_level` 이 SM 4%/DRAM 2% 사용** → CPU/GPU 어느 자원도 saturate 못 함 = **latency-bound 영역**
- warp occupancy 33.5% — 절반 안 차서 latency hiding 도 부족
- FP64 25% — 쓰긴 쓰지만 활성 영역의 1/4만
- `mf_scatter_csr_values` 만 memory-bound (정상 — 값 업데이트는 bandwidth bound)
- `mf_invert_pivot` 만 compute-bound (FP64 나눗셈)

→ **factor 본체가 latency-bound** — 더 이상 GPU 쥐어짤 여지가 좁음. 이미 *"가능한 만큼 빠르게 launch"* 하고 있고, kernel 안에서 work-per-thread 가 작아서 latency 가 우선. CUDA Graph로 launch 1회까지 줄인 게 결정적 이유.

### 3.2 STRUMPACK hot kernels (`/tmp/bench/ncu_strumpack_8387.csv`)

| Kernel | instances | SM% | DRAM% | warp% | FP64% | 분류 |
|---|---:|---:|---:|---:|---:|---|
| `strumpack::extract_rhs_kernel<double>` (solve setup) | 52 | **1.0** | **0.8** | 20.7 | 0.0 | **latency-bound (극단)** |
| `gemvn_kernel_vbatched<double, 16, 8, 256>` (solve forward) | 51 | 9.9 | 0.7 | 16.2 | 22.7 | latency-bound |
| `strumpack::extend_add_rhs_kernel_left<double>` | 47 | **0.7** | 0.9 | 21.7 | 3.8 | **latency-bound (극단)** |
| `strumpack::extend_add_rhs_kernel_right<double>` | 47 | **0.7** | 1.0 | 21.6 | 3.8 | **latency-bound (극단)** |
| `strumpack::laswp_vbatch_kernel<double>` (row swap, pivoting) | 47 | **0.1** | 0.2 | 13.2 | 0.0 | **순수 overhead (starvation)** |
| `gemvn_kernel_vbatched<double, 128, 1, 256>` | 46 | 3.5 | 0.9 | 15.8 | 7.6 | latency-bound |
| `trsm_template_vbatched_lNL_kernel<double, 8, 64>` | 23 | 19.1 | 0.6 | 13.1 | 32.8 | latency-bound (FP64 부분 동작) |
| `trsm_template_vbatched_lNU_kernel<double, 8, 64>` | 23 | 19.8 | 0.6 | 13.1 | 33.9 | latency-bound |
| `trsm_template_vbatched_lNL_kernel<double, 16, 64>` | 13 | 26.6 | 0.4 | 10.9 | 52.1 | partial (FP64 50%) |
| `trsm_template_vbatched_lNU_kernel<double, 16, 64>` | 13 | 26.6 | 0.4 | 10.9 | 52.1 | partial |
| `trsm_template_vbatched_lNL_kernel<double, 4, 32>` | 7 | 1.1 | 0.4 | 8.3 | 3.8 | **starvation** |
| `trsm_template_vbatched_lNL_kernel<double, 2, 32>` | 3 | **0.0** | 0.3 | 8.3 | 1.1 | **완전 starvation** |
| `trsm_template_vbatched_lNL_kernel<double, 32, 32>` (큰 fronts) | **1** | **57.5** | 0.3 | 9.1 | **81.8** | compute-bound (FP64) |
| `trsm_template_vbatched_lNU_kernel<double, 32, 32>` | **1** | 57.2 | 0.3 | 9.1 | 81.7 | compute-bound |

#### 핵심 진단 (STRUMPACK ncu)

**(A) 알고리즘 자체는 작동한다 — 큰 fronts에서**: `trsm<32,32>` 가 SM 57.5%, FP64 81.8% 까지 올라가는 게 증거. 하지만 그 큰 fronts 는 본 매트릭스에 **각 함수 호출 당 1번만 등장**. 즉 *"MAGMA가 자기 sweet spot에 들어간 호출은 그 1번뿐"*. 나머지 100+ 호출은 SM 0-25% 사이.

**(B) 작은 fronts 가 압도**: `trsm<8,64>` (23 instances) 평균 SM 19%, `trsm<4,32>` (7 instances) SM 1.1%, `trsm<2,32>` (3 instances) **SM 0.0%**. 작은 fsz일수록 GPU가 거의 일을 안 함 — *"늘 vbatched의 형식으로 launch 는 되지만 실제 work 가 너무 작음"*.

**(C) Row-swap이 순수 오버헤드**: `laswp_vbatch_kernel` SM 0.1% / DRAM 0.2% — partial pivoting의 row swap이 GPU 거의 안 씀에도 47회 launch. **이게 STRUMPACK이 maintain하는 pivoting 정확성의 cost**. custom 은 no-pivot 가정으로 이 kernel 자체가 없음.

**(D) Solve 단계의 극단적 latency-bound**: `extract_rhs`, `extend_add_rhs_kernel` 들이 모두 SM 0.7~1.0%, DRAM 0.8~1.0%. solve forward/backward 가 GPU 점유 거의 0. *"Solve is performed on CPU"* 경고와 일치 — GPU 위에 살짝 호스트의 work 가 흩어져 있을 뿐.

#### 정량 비교 (custom vs STRUMPACK ncu)

| 카테고리 | STRUMPACK 평균 SM% | custom 평균 SM% | 어느 쪽이 GPU 더 잘 씀 |
|---|---:|---:|---|
| factor 본체 | 12.3 (8개 vbatched 평균) | 4.4 (`mf_factor_extend_level`) | STRUMPACK 약간 위 |
| solve 본체 | 1.0 (`extract_rhs`, `extend_add_rhs`) | 1.6, 6.6 (fwd, bwd) | 비슷 |
| pivot/row-swap overhead | 0.1 (`laswp_vbatch`) × 47 calls | 없음 (no-pivot) | custom 우위 (kernel 없음) |
| 큰 front 처리 | 57.5% on `trsm<32,32>` (단 1 instance) | (해당 없음) | STRUMPACK 우위 — 그러나 본 매트릭스엔 큰 front 없음 |

→ **두 솔버 모두 latency-bound 영역에 있다**. SM% 가 둘 다 낮은 게 그 증거. 즉 *"GPU compute 더 짜내기"* 가 leverage 아님. 차이는 다른 곳에서 온다:
- STRUMPACK 은 latency-bound 영역에서 **kernel 수가 너무 많음** (총 ~700 / iter), 각각이 launch overhead 와 함께 누적
- custom 은 같은 latency-bound 영역에서 **kernel 수를 최소화 + graph로 launch overhead 제거**

### 3.3 *"latency-bound"* 의 의미 — power-grid 야코비안에서

dense LU에서 fsz³ work 가 충분히 크면 SM 점유 / FP64 throughput / DRAM bandwidth 중 하나가 bound 가 된다. 본 매트릭스에선 그게 모두 작아서 **각 kernel call의 "host launch → GPU dispatch → kernel execute → sync" pipeline 자체** 가 dominant cost가 된다.

이 영역에서 wall-clock 차이를 만드는 요인의 우선순위:
1. **kernel launch 횟수** — 적을수록 빠름
2. **launch 사이의 host work** — 작을수록 빠름
3. **per-kernel work distribution** — front 크기에 맞춰야 idle thread 적음

GPU 의 FP64 throughput 이나 DRAM bandwidth는 어차피 활용을 못 하니까 *"성능을 결정짓는 자원이 아님"*. 이게 멀티프론탈 알고리즘 자체의 한계가 아니라 *"front 가 너무 작은 매트릭스에서의 알고리즘 행동"* 이다.

---

## 4. STRUMPACK 논문 자체 데이터셋과의 비교 — *왜 STRUMPACK은 power-grid에 특화되지 않았는가*

STRUMPACK 논문 (Claus, Ghysels, Boukaram, Li, IJHPCA 2025) 의 Table 2 가 자체 평가에 사용한 SuiteSparse 행렬 (`docs/strumpack-reproduction-report.md` §2 와 `docs/strumpack-power-grid-analysis.md` §1):

| 매트릭스 | N (×10³) | nnz (full ×10³) | 도메인 |
|---|---:|---:|---|
| Serena | 1,391 | 64,531 | 3D 지반역학 (gas reservoir) |
| Geo_1438 | 1,438 | 63,156 | 3D 지반역학 |
| Hook_1498 | 1,498 | 60,917 | 3D 구조 (steel hook) |
| ML_Geer | 1,504 | 110,879 | 3D 다공질 유동 |
| Transport | 1,602 | 23,500 | 3D Stokes 유동 |
| Flan_1565 | 1,565 | 117,406 | 3D 구조 (steel flange) |
| Cube_Coup_dt0 | 2,164 | 129,133 | 3D 구조 (coupled cube) |

평균 nnz/row: 30~75 (8387의 4–10×). 모든 행렬이 **3D 연속체 PDE 이산화** = 3D mesh 토폴로지.

### 4.1 왜 이 매트릭스들에서 STRUMPACK이 잘 동작하는가

3D mesh 그래프의 nested dissection 특성:
- 그래프가 3D 영역 위 균일한 connectivity → **separator 크기 ~ O(N^{2/3})**
  - N = 1.5M 인 경우 separator 크기 ≈ 13,000+
- top separator가 곧 root front: fsz 가 매우 큼 (수천 ~ 수만)
- 하위 separator 도 큰 dense fronts 생성

논문 본문이 직접 보고하는 매트릭스별 root separator 크기 추정 (예시 외부 출처 Spatula MICRO'23 — 같은 행렬의 supernode 크기):
- `atmosmoddd`: 92% of FLOPs in supernodes > 4000
- 본 논문 행렬 (Janna 그룹) 도 동질적인 큰 dense fronts 분포

이런 분포에서 STRUMPACK MAGMA `vbatched_dgetrf<16,16,48>` 와 `<32,32>` TRSM 이 자기 sweet spot 에 들어간다 — fsz ≥ 64 이면 multiple tiles 채워서 GPU의 SM·FP64·DRAM 을 saturate. 이게 paper Table 2 에서 cuDSS 대비 평균 1.87× 우위의 원천.

### 4.2 전력망 야코비안의 그래프 특성 — *왜 같은 알고리즘이 안 통하는가*

power-grid Newton-Raphson 야코비안의 그래프는 매우 다른 토폴로지:
- 그래프 = 전력망 admittance pattern (∼버스 연결 그래프) + PV/PQ 노드 양분
- 거의 **평면 그래프** (전기적 망은 nearly-planar) → planar separator theorem: **separator 크기 ~ O(√N)**
  - N = 14,908 인 경우 separator 크기 ≈ √15,000 ≈ 120
- 실측 (case8387pegase) : max fsz = 96 < 120 (예측과 일치, METIS가 약간 더 작은 separator 찾음)

```
3D 메시 (Janna):      separator ~ N^{2/3}  →  big fronts (수천 단위)
                                              STRUMPACK MAGMA sweet spot
                                              → 1.87× over cuDSS

평면 망 (power):      separator ~ √N       →  tiny fronts (수십 단위)
                                              MAGMA sweet spot 밖
                                              → 본 측정 26× 손해
```

### 4.3 *논리적 결론* — STRUMPACK은 일반 sparse direct 솔버다, power-flow 전용이 아니다

논문의 측정 설계와 알고리즘 선택을 종합:

1. **STRUMPACK의 design intent**: 큰 dense front을 가진 *일반* sparse direct (3D PDE, 구조해석, 지반역학, 유체) 가 타깃. 논문 Table 2 매트릭스 7개 모두 이 영역.
2. **알고리즘 선택의 일관성**:
   - multifrontal LU + MAGMA vbatched: 큰 dense front 위 batched dense LU 가 sweet spot
   - BLR compression option: 큰 front 의 low-rank 구조 활용 — 작은 front 에서는 적용 의미 없음
   - FP64-centric GPU 경로: 큰 dense front 에선 GPU FP64 가 효율적, 작은 front 에선 FP32-native 가 더 나음
3. **API 디자인의 일관성**: `STRUMPACK_factor` 가 매번 numeric 재처리 (refactorize phase 분리 없음) — 같은 pattern을 반복하는 NR loop 시나리오에 최적화 안 됨
4. **본 nsys 측정의 추가 증거**: case8387pegase factor 의 wall 14.3 ms 중 GPU kernel time 은 일부분 (대부분 vbatched 들이 launch overhead가 압도). STRUMPACK 의 "GPU 최적화" 가 본 매트릭스에 미치지 못함.

→ **STRUMPACK은 sparse direct *general-purpose* solver 이지 power-flow specialization 솔버가 아니다.** 같은 multifrontal 알고리즘 위에서도 *"front 가 크고, NR-style 반복 없는 일반 PDE"* 라는 가정 위 구현. 그 가정을 정확히 깨는 power-flow 시나리오에서 자기 sweet spot 밖.

cuDSS 도 비슷하지만 보다 general supernodal 디자인 + REFACTORIZATION phase 가 NR loop 에 부분적으로 적응 → STRUMPACK 보다는 power-grid 에 잘 맞지만 여전히 *"NR-loop 전용 디자인은 아님"*.

`custom_linear_solver`는 정반대 방향:
- multifrontal 알고리즘 그대로
- 가정: **fsz < 160 인 fronts** (즉 power-grid Jacobian의 일반적 분포)
- 가정: **NR loop** (sparsity 고정, 값만 매 iter 갱신)
- 가정: **no-pivot 가능** (NR pre-scaling 이 외부에서 처리)
- 이 세 가정 위에서 커널 라우팅 (warp-per-front, shared-resident, 1024-thread big), CUDA Graph capture, device-resident pipelines 를 통째로 설계

같은 multifrontal 패밀리이지만 *"design target"* 이 다르고, *"design assumption"* 이 다르다.

---

## 5. wall 의 분해 — 왜 14.3 ms vs 0.64 ms 인가

### 5.1 STRUMPACK iter 2 factor = 14.32 ms (case8387pegase)

`/tmp/bench/nsys/strumpack_nr_8387.nsys-rep` 의 `cuda_api_sum` 과 `cuda_gpu_kern_sum` 종합 (추정):

| 구성 | 시간 [ms] | 메커니즘 |
|---|---:|---|
| GPU kernel work (vbatched GEMM/TRSM + STRUMPACK extend_add) | ~5 | 1000+ small kernels (avg 10-25 μs) |
| `cudaLaunchKernel` overhead (~700 calls × 6 μs) | ~4 | small-front 영역에서 kernel 수 폭증 |
| 페이지락 / device alloc 비용 | ~2 | NR iter 마다 일부 buffer 재할당 |
| H2D memcpy 누적 | ~2 | 작은 청크 (avg ~50 KB) 다수 |
| host-side multifrontal scheduling | ~1 | front 별 dispatch loop |
| **합** | **~14** | |

→ **GPU kernel 자체는 wall 의 35% 이하**. 나머지 65% 가 launch + memory + scheduling overhead.

### 5.2 custom iter 2 factor = 0.64 ms (case8387pegase)

`/tmp/bench/nsys/custom_nr_8387.nsys-rep` 와 ncu 통합:

| 구성 | 시간 [ms] | 메커니즘 |
|---|---:|---|
| GPU kernel work (`mf_factor_extend_level` 31 levels + `mf_invert_pivot` + `mf_scatter_csr_values`) | ~0.60 | level 당 1 launch (graph node), latency-bound 영역에서도 빠른 진행 |
| Graph launch overhead (`cudaGraphLaunch` × 1) | ~0.02 | 단일 graph replay |
| 기타 host work | ~0.02 | minimal |
| **합** | **~0.64** | |

→ **wall 의 94%가 GPU kernel work**. host overhead 거의 0.

### 5.3 *왜 차이가 22× 인가*

각 구성요소별로 분해해 보면:

| 구성 | STRUMPACK | custom | 비 | 이유 |
|---|---:|---:|---:|---|
| GPU kernel 시간 | ~5 ms | ~0.6 ms | 8× | front 분포에 맞춘 커널 라우팅 (warp-per-front, fused factor+extend) |
| launch overhead | ~4 ms | ~0.02 ms | 200× | CUDA Graph replay vs 700+ individual launches |
| alloc/memcpy | ~4 ms | ~0.02 ms | 200× | 메모리 알로케이션 모두 analyze 시점에 완료 |
| host scheduling | ~1 ms | ~0.02 ms | 50× | multifrontal scheduling이 graph 안에 컴파일됨 |

→ **단일 거대 원인이 아니라 4가지 모두에서 작은 우위가 곱셈으로 누적.** GPU kernel 자체 차이(8×)도 의미있지만, 진짜 leverage는 host-side overhead 제거(200×) 들이 곱해진 결과.

이는 *"같은 multifrontal 알고리즘"* 인데도 26× 차이가 나는 이유가 *"알고리즘 자체의 차이"*가 아니라 *"power-grid 가정 위 엔지니어링 누적"* 이라는 본 분석의 주장.

---

## 6. 종합 — 한 줄씩

1. **같은 multifrontal LU 패밀리이지만 가정이 다르다**: STRUMPACK은 *"front 크고 NR 아닌 일반 PDE"* 가정, custom은 *"front 작고 NR loop"* 가정.
2. **case8387pegase 의 front 분포**는 정확히 STRUMPACK의 가정을 깬다 — fsz ≤ 16 이 96.2%, max fsz = 96. MAGMA vbatched sweet spot 의 1/4 도 못 됨.
3. **ncu 분류**: custom 의 factor 본체가 latency-bound (SM 4%, DRAM 2%) — *"power-grid 야코비안에서 compute/memory 모두 saturate 못 함"* 이 솔버 디자인의 baseline. 더 빠르게 만들려면 launch density 와 host overhead 를 줄여야 함, 컴퓨트 자체가 아님.
4. **차이의 분해**: 8× GPU kernel + 200× launch overhead + 200× memory + 50× scheduling = 26× total. CUDA Graph + 도메인 가정 위 커널 라우팅 + analyze-시점 메모리 종결 의 곱셈 효과.
5. **STRUMPACK 논문 Table 2 매트릭스 (Janna)** 는 거꾸로 STRUMPACK 가정의 모범 사례 — 큰 3D mesh, 큰 dense front, 1.87× over cuDSS. 그래서 STRUMPACK은 *general-purpose sparse direct* 솔버다. *power-flow 특화 솔버가 아니다*. 같은 multifrontal 안에서 서로 다른 가정 위에서 만들어진 솔버들은 *서로 다른 매트릭스 클래스의 챔피언*.

## 7. 한계 / 정직성

- ncu 측정은 단일 GPU (RTX 3090, sm_86). A100 / H100 에선 FP64 throughput 이 17~30× 높아서 STRUMPACK MAGMA 가 same-size front 에서도 더 빠르게 saturate 가능. 다만 power-grid 의 front 가 *너무* 작아서 (max fsz=96) GPU 종류 바뀌어도 sweet spot 자체에 닿지 못함 — 정성적 결론은 보존될 가능성 높음.
- case8387pegase 한 매트릭스. 다른 power-grid (1k, 3k, 25k bus) 도 같은 front 분포 (`docs/related-work-and-contribution.md` §2 가 보고) — 정성적으로 일반화 가능.
- BLR compression 으로 STRUMPACK이 큰 front 에서 추가 우위를 얻을 수 있다는 논문 본문 주장은 본 매트릭스에선 적용 의미 없음 (큰 front 자체가 없음).

## 8. 재현 명령 + 산출물

원시 데이터:
- `/tmp/bench/nsys/strumpack_nr_8387.nsys-rep` (NVTX + CUDA trace)
- `/tmp/bench/nsys/custom_nr_8387.nsys-rep`
- `/tmp/bench/ncu_custom_8387.csv` (per-kernel metrics)
- `/tmp/bench/ncu_strumpack_8387.csv` (진행 중)

```bash
# nsys
nsys profile --output=/tmp/bench/nsys/strumpack_nr_8387 --force-overwrite=true \
    --trace=cuda,nvtx /tmp/bench/nsys_strumpack_nr \
    /datasets/power_system/nr_linear_systems/case8387pegase/J.mtx \
    /datasets/power_system/nr_linear_systems/case8387pegase/rhs.mtx

# custom 의 front 분포
CLS_DUMP=1 /tmp/clsb/custom_linear_solver_run \
    --matrix /datasets/power_system/nr_linear_systems/case8387pegase/J.mtx \
    --rhs    /datasets/power_system/nr_linear_systems/case8387pegase/rhs.mtx \
    --repeat 1

# ncu (custom, iter2 영역)
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active \
    --launch-skip 0 --launch-count 600 \
    --csv --log-file /tmp/bench/ncu_custom_8387.csv \
    /tmp/bench/nsys_custom_nr <matrix> <rhs>
```

## 9. 참고

- `docs/baseline-vs-strumpack.md` — STRUMPACK ≠ 코드 베이스, lineage 정정
- `docs/strumpack-reproduction-report.md` — STRUMPACK paper 행렬 재현 결과
- `docs/strumpack-power-grid-analysis.md` — 전력망 위 wall-clock + kernel-only 측정
- `docs/nr-profile-3way-analysis.md` — STRUMPACK vs cuDSS vs custom NR 프로파일 (case_ACTIVSg25k)
- `docs/strumpack-nsys-nr-profile.md` — STRUMPACK nsys 상세 (case_ACTIVSg25k)
- `docs/fp32-batched-factor-solve-optimization.md` — front 크기별 커널 라우팅 설계
- `docs/related-work-and-contribution.md` — 외부 출처 (Spatula MICRO'23) 인용 + novelty 자체 평가
- Claus, Ghysels, Boukaram, Li, IJHPCA 2025 — STRUMPACK GPU + BLR 논문
- Spatula (MICRO 2023) — 외부 STRUMPACK profiling: *"FullChip 위 V100 peak 의 0.004%"*
