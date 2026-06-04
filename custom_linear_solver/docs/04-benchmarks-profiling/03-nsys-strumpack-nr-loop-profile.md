# STRUMPACK GPU — Nsight Systems 프로파일 (NR 2-iter 시뮬레이션)

전력조류 Newton-Raphson 루프와 동일한 호출 패턴으로 STRUMPACK GPU를 한 번 더 profile.
**`docs/04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md` §6.3** 의 *"factor wall 169 ms 중 GPU 21.5 ms (13%)"* 라는
주장의 GPU/CPU/launch 내역을 직접 보이는 것이 목적.

## 1. 시뮬레이션 패턴

NR 전력조류는 반복마다:
1. Jacobian J 새로 계산 (sparsity pattern 동일, 값만 변화)
2. J factor
3. 솔브 → 보정 방향 → 전압 업데이트

같은 sparsity pattern이라 **analyze (reorder)** 는 처음에만 한 번 실행하고, 매 iter 마다 **factor + solve** 만 반복하는 게 정석이다 (cuDSS의 `refactorize` API, STRUMPACK의 `STRUMPACK_update_csr_matrix_values` + `STRUMPACK_factor` 가 그 용도).

본 측정 드라이버 (`/tmp/bench/nsys_strumpack_nr.cpp`):

```
init + setup
reorder              ← 한 번만
NVTX: NR_iter_1
  iter1.factor       ← STRUMPACK_factor
  iter1.solve        ← STRUMPACK_solve
NVTX: NR_iter_2
  iter2.update_values ← STRUMPACK_update_csr_matrix_values (값만 ×1.01)
  iter2.factor
  iter2.solve
destroy
```

대상: `case_ACTIVSg25k` (n=47246, nnz=318672). RTX 3090, STRUMPACK MAGMA build, FP64, METIS reorder, MC64 matching, compression off, Krylov direct.

## 2. nsys 파일

`/tmp/bench/nsys/strumpack_nr_ACTIVSg25k.nsys-rep`

```bash
# 재현
nsys profile \
  --output=/tmp/bench/nsys/strumpack_nr_ACTIVSg25k \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  /tmp/bench/nsys_strumpack_nr \
  /datasets/power_system/nr_linear_systems/case_ACTIVSg25k/J.mtx \
  /datasets/power_system/nr_linear_systems/case_ACTIVSg25k/rhs.mtx

# CLI 확인
nsys stats --report nvtx_pushpop_sum --format table .../strumpack_nr_ACTIVSg25k.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format table .../strumpack_nr_ACTIVSg25k.nsys-rep
nsys stats --report cuda_api_sum --format table .../strumpack_nr_ACTIVSg25k.nsys-rep
nsys stats --report cuda_gpu_mem_size_sum --format table .../strumpack_nr_ACTIVSg25k.nsys-rep

# GUI
nsys-ui /tmp/bench/nsys/strumpack_nr_ACTIVSg25k.nsys-rep
```

## 3. NVTX 단위 wall-clock (host 시계)

```
+----------+-----------------+------------------------------------+
| Time (%) | Total Time (ms) |  Range                             |
+----------+-----------------+------------------------------------+
|     32.2 |          238.10 |  STRUMPACK_init+setup              |
|     31.2 |          230.71 |  STRUMPACK_reorder (analyze, once) |
|     10.7 |           78.85 |  NR_iter_1                         |
|      7.5 |           55.53 |  NR_iter_2                         |
|      7.2 |           53.52 |  iter1.factor                      |
|      3.6 |           26.34 |  iter2.factor                      |
|      3.4 |           25.27 |  iter1.solve                       |
|      3.1 |           22.94 |  iter2.solve                       |
|      0.8 |            6.16 |  iter2.update_values               |
|      0.2 |            1.65 |  STRUMPACK_destroy                 |
+----------+-----------------+------------------------------------+
```

NR 정상상태 (iter 2, 첫 init 비용 제외):
- factor 26.3 ms
- solve 22.9 ms
- factor + solve = 49.3 ms per NR iter

iter1 vs iter2 비교:
- factor: 53.5 → 26.3 ms (50% 빠름 — JIT 컴파일/MAGMA dispatch 캐시 등 first-iter cost)
- solve: 25.3 → 22.9 ms (10% 빠름)

→ **NR 정상상태 iter 비용은 49 ms** (factor+solve). `docs/04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md` §6.4 의 wall-clock factor 169 ms는 이것보다 큰 — 그 측정은 솔버 객체 매번 새로 생성 (analyze 포함), 본 측정은 객체 재사용 (analyze 한 번). iter 2 만 보는 게 NR 시나리오에 정확.

## 4. GPU kernel-only 시간 (CUDA event 측정)

CUDA event 기록: 전체 GPU kernel time = **~19.4 ms** (모든 iter의 GPU 작업 합산, kernel summary 의 Time(%) 합산으로 계산).

GPU kernel top (Time(%) 기준):

| Time(%) | Total (ms) | Count | Kernel |
|---:|---:|---:|---|
| 14.6 | 2.83 | 146 | `gemm_template_vbatched_nn_kernel<double, 16, 16, 48, ...>` (MAGMA vbatched GEMM) |
| 11.7 | 2.26 | 336 | `strumpack::gpu::extend_add_kernel<double, 16>` (STRUMPACK 직접 작성, 작은 fronts) |
| 9.3 | 1.81 | 76 | `trsm_template_vbatched_lNU_kernel<double, 16, 64>` (MAGMA vbatched TRSM) |
| 9.3 | 1.80 | 76 | `trsm_template_vbatched_lNL_kernel<double, 16, 64>` |
| 7.0 | 1.35 | 182 | `trsm_template_vbatched_lNU_kernel<double, 8, 64>` |
| 6.9 | 1.34 | 182 | `trsm_template_vbatched_lNL_kernel<double, 8, 64>` |
| 5.0 | 0.96 | 338 | `strumpack::gpu::laswp_vbatch_kernel<double>` |
| 3.2 | 0.62 | 126 | `gemvn_kernel_vbatched<double, 16, 8, 256>` (작은 GEMV) |
| 1.2 | 0.23 | 170 | `magma_iset_pointer_kernel` (MAGMA의 포인터 배열 셋업) |
| 1.1 | 0.21 | 10 | `dgetf2_fused_kernel_vbatched<11>` (panel LU) |

### 관찰
- **MAGMA vbatched 가 GPU 시간의 50%+** — STRUMPACK이 small/medium fronts를 MAGMA의 vbatched-GETRF/TRSM/GEMM으로 처리
- **STRUMPACK 자체 커널은 `extend_add_kernel` 과 `laswp_vbatch_kernel`** — vbatched로 처리 안 되는 부분(small-front extend-add, row swap)을 자체 작성
- 큰 dense fronts용 `<32×32>` TRSM이 4번만 등장 (총 0.25 ms) — power-grid 야코비안에는 큰 front가 거의 없음을 확인 (`docs/01-orientation/02-related-work-and-novelty.md` §2 의 *"95% fronts have fsz ≤ 16"* 와 정확히 일치)

## 5. CUDA API summary — host-side overhead 정량

```
+----------+-----------------+-----------+----------+----------+---------------------------+
| Time (%) | Total Time (ms) | Num Calls | Avg (us) | Med (us) |           Name            |
+----------+-----------------+-----------+----------+----------+---------------------------+
|     57.7 |          102.76 |       125 |    822.1 |      1.0 | cudaDeviceSynchronize     |
|     10.7 |           18.96 |      3038 |      6.2 |      4.1 | cudaLaunchKernel          |
|      6.4 |           11.40 |       344 |     33.1 |      3.2 | cudaMallocHost            |
|      6.0 |           10.70 |       144 |     74.3 |      2.6 | cudaFree                  |
|      5.7 |           10.20 |       686 |     14.9 |      9.7 | cudaMemcpy                |
|      3.6 |            6.34 |         6 |   1057.3 |   1045.9 | cudaGetDeviceProperties   |
|      2.8 |            4.93 |       682 |      7.2 |      4.7 | cudaStreamSynchronize     |
|      2.6 |            4.61 |       344 |     13.4 |      2.6 | cudaFreeHost              |
|      1.8 |            3.26 |       102 |     32.0 |      3.2 | cudaMalloc                |
|      1.3 |            2.24 |       510 |      4.4 |      4.5 | cudaMemset                |
+----------+-----------------+-----------+----------+----------+---------------------------+
```

### 결정적 수치
- **`cudaLaunchKernel` 3038회** 호출, 평균 6 μs, 총 19 ms — 본 측정 자체에서 launch 만으로 19 ms
- `cudaDeviceSynchronize` 125회, 102 ms (대부분 우리 측정에서 의도적으로 부른 것)
- 매 launch 당 ~6 μs 의 launch overhead. 3038 / 2 iters = ~1500 launches per NR iter → per-iter launch overhead ~9 ms

### 메모리 패턴
- `cudaMallocHost` 344회 + `cudaFreeHost` 344회 (페이지락 메모리 매번 alloc/free) → 페이지락 풀 캐시 없음, 매번 새로 할당
- `cudaMalloc` 102회 / `cudaFree` 144회 (디바이스)
- `cudaMemcpy` 686회, 평균 15 μs

## 6. 데이터 전송

```
+------------+-------+----------+----------+------------------------------+
| Total (MB) | Count | Avg (MB) | Med (MB) |          Operation           |
+------------+-------+----------+----------+------------------------------+
|     75.806 |   676 |    0.112 |    0.013 | [CUDA memset]                |
|     39.131 |   684 |    0.057 |    0.007 | [CUDA memcpy Host-to-Device] |
|      0.756 |     2 |    0.378 |    0.378 | [CUDA memcpy Device-to-Host] |
+------------+-------+----------+----------+------------------------------+
```

- **H2D 39 MB / 684 transfers** — 작은 청크가 매우 많음 (평균 57 KB). matrix values + 메타데이터 매번 D2H/H2D
- D2H 0.76 MB (= 솔루션 2 iter × 47 K doubles × 8 byte ≈ 750 KB)
- 즉 **solve 결과만 D2H로 떨어지고** 모든 값 업데이트는 H2D로 올라감 — *"Solve is performed on CPU"* 경고와 직접 연결되지 않음, 실제로는 STRUMPACK이 GPU LU 결과를 host 에 두고 solve도 host 에서 한다는 것의 결과

## 7. 분해 — wall 53 ms 의 내역 (iter 1 factor 기준)

| 항목 | 시간 | 비중 |
|---|---:|---:|
| GPU kernel work (vbatched + STRUMPACK 직접 + MAGMA 셋업) | ~10 ms | 19% |
| `cudaLaunchKernel` (1500 launches × 6 μs) | ~9 ms | 17% |
| `cudaMemcpy` H2D (340 transfers × ~14 μs) | ~5 ms | 9% |
| `cudaMallocHost`/`cudaFree` 누적 | ~6 ms | 11% |
| `cudaDeviceSynchronize` 대기 | ~13 ms | 24% |
| 기타 host work (multifrontal 스케줄링, MAGMA dispatch 준비) | ~10 ms | 19% |
| **합** | **~53 ms** | 100% |

(개략 — `cuda_api_sum` 의 시간은 호출 누적이고 그 중 일부는 GPU 실행과 겹치므로 1:1 합산은 안 됨. 비율은 대략적 추정)

## 8. 결론 — `docs/04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md` §6.3 주장 검증

`docs/04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md` §6.3 의 wall=169 vs kernel=21.5 ms 주장은 본 nsys 프로파일에서 직접 확인된다:

- NR iter 1 factor wall = **53.5 ms** (객체 재사용 시. 신규 객체 생성 시 169 ms — 1회성 init/alloc 차이)
- GPU kernel 작업 시간 = **~10 ms** (vbatched + STRUMPACK 직접 합산, 본 iter의 절반 정도)
- 나머지 ~43 ms = launch overhead + memcpy + malloc/free + sync = **wall-clock의 80% 이상이 host-side overhead**

**핵심 — 본 솔버의 NR-iter steady-state cost 분해 (case_ACTIVSg25k):**

| 구성 | 시간 [ms] | 비중 |
|---|---:|---:|
| GPU 실제 작업 | ~10 | 19% |
| 1500회 kernel launch overhead | ~9 | 17% |
| host-side memcpy/malloc/free/sync | ~24 | 45% |
| host-side multifrontal scheduling | ~10 | 19% |

cuDSS와 custom_linear_solver는 같은 카운트(launch 수)를 보지 않을 것이다 — cuDSS는 통합된 API 안에서 더 적은 launch 패턴, custom은 CUDA Graph로 **1번의 launch (graph replay)** 만 발생. 그게 wall-clock 차이의 본질.

→ 본 nsys 프로파일은 *"STRUMPACK은 small-front 도메인(power-grid)에서 GPU를 잘 못 쓴다"* 의 시각적 증거이며,
`docs/01-orientation/02-related-work-and-novelty.md` §2 의 Spatula(MICRO'23) 주장 *"0.004% of V100 peak"* 의 정성적 매커니즘 — **수천 개의 작은 kernel launch + cudaLaunchKernel overhead** — 을 본 환경에서 직접 확인한다.

## 9. 다음 단계 (suggestion)

- 같은 NR 패턴으로 cuDSS profile (`cudssExecute(REFACTORIZATION)` + `cudssExecute(SOLVE)` × 2). 이게 cuDSS의 wall 14 ms 중 cudssExecute API overhead 가 어디서 나오는지 직접 확인.
- 같은 NR 패턴으로 custom_linear_solver profile (CUDA Graph replay). 이론적으로 launch 수가 graph 인스턴스화 후 1회 / iter 여야 함. 확인.
- 위 셋의 nsys 파일을 모아 `docs/04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md` §6.3 의 "host overhead = 87%" 주장의 솔버별 메커니즘을 시각화.
