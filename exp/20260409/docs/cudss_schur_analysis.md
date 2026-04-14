# cuDSS Schur Complement 분석

> **실험 위치**: `exp/20260409/`  
> **측정 날짜**: 2026-04-09  
> **GPU**: NVIDIA (cuDSS 설치 환경)  
> **CPU**: AMD EPYC 7313 16-Core Processor  
> **행렬 정밀도**: FP32 · **Warmup**: 1 · **Repeats**: 3 (평균 보고)

---

## 1. 배경

Power-flow Newton step은 매 반복에서 sparse 선형계 $J \mathbf{x} = \mathbf{b}$ 를 풀어야 한다.

$$
J = \begin{bmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{bmatrix}
\qquad
\mathbf{x} = \begin{bmatrix} \Delta\theta_{\text{pvpq}} \\ \Delta|V|_{\text{pq}} \end{bmatrix}
$$

| 블록 | 변수 | 크기 |
|------|------|------|
| $J_{11}$ | $\Delta\theta$ (PV + PQ 버스) | $n_\text{pvpq} \times n_\text{pvpq}$ |
| $J_{22}$ | $\Delta|V|$ (PQ 버스만) | $n_\text{pq} \times n_\text{pq}$ |

Schur complement 방식은 $J_{22}$ 블록(Schur 블록)을 외부 dense solver(cuSOLVER)로 처리하고,  
$J_{11}$ 부분만 cuDSS sparse factorization으로 처리한다는 아이디어다.

**실험 목적**: cuDSS full sparse solve와 비교했을 때 Schur complement 방식이 실제로 빠른지 검증한다.

---

## 2. 두 경로 비교

### 2-1. cuDSS Full Solve

```
ANALYSIS  →  FACTORIZATION  →  SOLVE
```

| 단계 | 내용 |
|------|------|
| ANALYSIS | 심볼릭 분석 (reordering + sparsity pattern) |
| FACTORIZATION | $J$ 전체 LU 인수분해 |
| SOLVE | $J \mathbf{x} = \mathbf{b}$ 완전 풀이 |

### 2-2. cuDSS Schur Complement (J22 블록)

```
ANALYSIS  →  FACTORIZATION  →  DataGet(SCHUR)  →  FWD  →  memcpy  →  getrf  →  getrs  →  BWD
```

| 단계 | 내용 |
|------|------|
| ANALYSIS | 심볼릭 분석 + **Schur reordering** (추가 비용 발생) |
| FACTORIZATION | $J_{11}$ 까지만 partial LU (Schur boundary에서 중단) |
| DataGet(SCHUR) | dense Schur matrix $S$ 추출 (cuDSS 내부 → GPU 버퍼) |
| FWD | `SOLVE_FWD_PERM \| SOLVE_FWD` — $b$ 읽어 $x$ 에 씀 |
| memcpy | $x \to b$ 복사 (GENERAL 행렬 전용, SOLVE_DIAG 대체) |
| getrf | cuSOLVER dense LU 인수분해 ($S$ in-place) |
| getrs | cuSOLVER dense 삼각 풀이 → $b[n_\text{pvpq}:] = \mathbf{y}_2$ |
| BWD | `SOLVE_BWD \| SOLVE_BWD_PERM` — $b$ 읽어 $x$ 에 최종 해 |

---

## 3. 성능 비교

### 3-1. 전체 시간 포함 (analysis + factorization + solve, ms)

| case | n_pq | full total | schur total | 배율 |
|------|-----:|-----------:|------------:|-----:|
| case118_ieee | 64 | **9.13** | 9.51 | 1.04× |
| case793_goc | 704 | **12.43** | 21.15 | 1.70× |
| case1354_pegase | 1,094 | **15.19** | 29.24 | 1.92× |
| case2746wop_k | 2,396 | **22.06** | 59.26 | 2.69× |
| case4601_goc | 4,468 | **29.98** | 136.01 | 4.54× |
| case8387_pegase | 6,522 | **44.49** | 241.19 | 5.42× |
| case9241_pegase | 7,796 | **48.36** | 313.46 | 6.48× |

### 3-2. Analysis 제외 비교 (factorization + solve only, ms)

> Analysis는 Newton 반복마다 수행하지 않는다. 실질적인 반복 비용은 analysis를 제외한 시간이다.

| case | n_pq | full (no-analysis) | schur (no-analysis) | 배율 |
|------|-----:|-------------------:|--------------------:|-----:|
| case118_ieee | 64 | **0.15** | 0.33 | 2.2× |
| case793_goc | 704 | **0.35** | 2.95 | 8.5× |
| case1354_pegase | 1,094 | **0.45** | 5.66 | 12.7× |
| case2746wop_k | 2,396 | **0.92** | 17.68 | 19.3× |
| case4601_goc | 4,468 | **0.81** | 51.48 | 63.7× |
| case8387_pegase | 6,522 | **0.94** | 98.75 | 105.1× |
| case9241_pegase | 7,796 | **0.99** | 136.40 | 137.6× |

**Analysis를 제외하면 격차가 훨씬 커진다.** n_pq=7796에서 full은 0.99ms, schur는 136ms로  
**137배 차이**가 난다. Schur complement는 반복 비용 관점에서도 전혀 경쟁력이 없다.

### 3-3. Full solve 단계별 시간 (ms)

| case | n_pq | analysis | factorization | solve | **no-analysis** | **total** |
|------|-----:|---------:|--------------:|------:|----------------:|----------:|
| case118_ieee | 64 | 8.98 | 0.08 | 0.07 | **0.15** | 9.13 |
| case793_goc | 704 | 12.08 | 0.25 | 0.09 | **0.35** | 12.43 |
| case1354_pegase | 1,094 | 14.75 | 0.31 | 0.13 | **0.45** | 15.19 |
| case2746wop_k | 2,396 | 21.14 | 0.75 | 0.16 | **0.92** | 22.06 |
| case4601_goc | 4,468 | 29.18 | 0.58 | 0.22 | **0.81** | 29.98 |
| case8387_pegase | 6,522 | 43.55 | 0.70 | 0.25 | **0.94** | 44.49 |
| case9241_pegase | 7,796 | 47.37 | 0.74 | 0.25 | **0.99** | 48.36 |

Full solve에서 analysis가 전체의 **95~98%** 를 차지한다.  
Factorization + Solve 합산은 모든 케이스에서 1ms 미만.

### 3-4. Schur complement 단계별 시간 (ms)

| case | n_pq | analysis | factor | extract | fwd | diag | getrf | getrs | bwd | **no-analysis** | **total** |
|------|-----:|---------:|-------:|--------:|----:|-----:|------:|------:|----:|----------------:|----------:|
| case118_ieee | 64 | 9.18 | 0.09 | 0.02 | 0.04 | 0.01 | 0.09 | 0.04 | 0.05 | **0.33** | 9.51 |
| case793_goc | 704 | 18.20 | 0.54 | 0.46 | 0.07 | 0.01 | 1.53 | 0.20 | 0.13 | **2.95** | 21.15 |
| case1354_pegase | 1,094 | 23.58 | 0.87 | 1.39 | 0.10 | 0.01 | 2.78 | 0.32 | 0.19 | **5.66** | 29.24 |
| case2746wop_k | 2,396 | 41.58 | 3.05 | 6.08 | 0.11 | 0.01 | 7.32 | 0.79 | 0.32 | **17.68** | 59.26 |
| case4601_goc | 4,468 | 84.53 | 9.97 | 20.77 | 0.16 | 0.01 | 18.57 | 1.44 | 0.55 | **51.48** | 136.01 |
| case8387_pegase | 6,522 | 142.44 | 14.01 | 44.68 | 0.20 | 0.01 | 37.02 | 2.22 | 0.62 | **98.75** | 241.19 |
| case9241_pegase | 7,796 | 177.07 | 16.55 | 62.91 | 0.23 | 0.01 | 53.28 | 2.80 | 0.61 | **136.40** | 313.46 |

---

## 4. Schur 세부 단계 분석

### 4-1. 단계별 복잡도와 병목

| 단계 | 복잡도 | 지배 변수 | case9241 (ms) | 비중 |
|------|--------|-----------|-------------:|-----:|
| analysis | $O(n \cdot \text{fill})$ | $n$ 전체, Schur reordering 추가 | 177.1 | 56.5% |
| factor | $O(n_\text{pvpq}^{1.5\text{–}2})$ | $n_\text{pvpq}$, sparse | 16.6 | 5.3% |
| **extract** | $O(n_\text{pq}^2)$ | $n_\text{pq}^2$, dense 행렬 복사 | **62.9** | 20.1% |
| **getrf** | $O(n_\text{pq}^3)$ | $n_\text{pq}^3$, dense LU | **53.3** | 17.0% |
| getrs | $O(n_\text{pq}^2)$ | $n_\text{pq}^2$, dense 삼각 풀이 | 2.8 | 0.9% |
| fwd + bwd | $O(\text{nnz})$ | sparse triangular | 0.8 | 0.3% |
| diag (memcpy) | $O(n)$ | 전체 벡터 복사 | 0.01 | ~0% |

```
case9241_pegase (total 313ms)

analysis  ████████████████████████████████████████████  177.1ms  56.5%
extract   ████████████████████                           62.9ms  20.1%
getrf     █████████████████                              53.3ms  17.0%
factor    █████                                          16.6ms   5.3%
getrs     █                                               2.8ms   0.9%
fwd+bwd   ▏                                               0.8ms   0.3%
diag      ▏                                               0.0ms   ~0%
```

### 4-2. n_pq 스케일링 실측

| case | n_pq | extract (ms) | getrf (ms) | extract/getrf 비 |
|------|-----:|-------------:|-----------:|:-----------------|
| case118_ieee | 64 | 0.02 | 0.09 | 1 : 4.5 |
| case793_goc | 704 | 0.46 | 1.53 | 1 : 3.3 |
| case2746wop_k | 2,396 | 6.08 | 7.32 | 1 : 1.2 |
| case4601_goc | 4,468 | 20.77 | 18.57 | **1.1 : 1** |
| case9241_pegase | 7,796 | 62.91 | 53.28 | **1.2 : 1** |

- $n_\text{pq}$가 작을 때는 getrf($O(n^3)$)가 extract($O(n^2)$)보다 빠름 — GPU 병렬성 덕분
- $n_\text{pq} > 4000$ 에서는 extract가 getrf를 추월 — dense 행렬 데이터 이동 비용이 지배

### 4-3. Analysis: Full vs Schur 비교

| case | n_pq | full analysis | schur analysis | 배율 |
|------|-----:|-------------:|---------------:|-----:|
| case118_ieee | 64 | 8.98ms | 9.18ms | 1.02× |
| case793_goc | 704 | 12.08ms | 18.20ms | 1.51× |
| case1354_pegase | 1,094 | 14.75ms | 23.58ms | 1.60× |
| case2746wop_k | 2,396 | 21.14ms | 41.58ms | 1.97× |
| case4601_goc | 4,468 | 29.18ms | 84.53ms | 2.90× |
| case8387_pegase | 6,522 | 43.55ms | 142.44ms | 3.27× |
| case9241_pegase | 7,796 | 47.37ms | 177.07ms | 3.74× |

Schur ANALYSIS는 $n_\text{pq}$ 증가에 따라 full보다 최대 3.7배 느려진다.  
Schur reordering(Schur 변수를 인수분해 마지막으로 배치)이 추가 symbolic 연산을 요구하기 때문이다.

### 4-4. Schur 접근법이 경쟁력을 가질 조건

본 실험의 power-flow 케이스에서 Schur가 불리한 근본 이유:

| 문제 | 설명 |
|------|------|
| $n_\text{pq} / \text{dim}$ 비율이 크다 | 35–46% — "작은 Schur 블록"이 아님 |
| Schur matrix가 dense | $n_\text{pq}^2$ 크기의 dense 행렬 전송·풀이 비용 |
| Full sparse factorization이 이미 매우 빠르다 | factorization + solve ≤ 1ms (모든 케이스) |

Schur가 유효해지는 시나리오:

| 조건 | 설명 |
|------|------|
| $n_\text{pq} / \text{dim} \lesssim 5\%$ | Schur 블록이 충분히 작을 때 |
| Schur matrix가 sparse | Dense cuSOLVER 대신 sparse solver 적용 가능 |
| ANALYSIS를 한 번 수행 후 REFACTORIZATION 반복 | Analysis 비용 상각 |

---

## 5. 구현

### 5-1. 전체 API 흐름

```cpp
// ── Constructor ──────────────────────────────────────────────────
// Schur 인덱스 등록 (반드시 ANALYSIS 전에 설정)
int compute_schur = 1;
cudssConfigSet(config_, CUDSS_CONFIG_SCHUR_MODE, &compute_schur, sizeof(int));
cudssDataSet(handle_, data_, CUDSS_DATA_USER_SCHUR_INDICES,
             h_schur_indices_.data(), system_dim_ * sizeof(int32_t));

// ── analyze() ────────────────────────────────────────────────────
cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, data_, A, x, b);

// Schur 블록 크기 확인 및 버퍼 준비
int64_t schur_shape[3];
cudssDataGet(handle_, data_, CUDSS_DATA_SCHUR_SHAPE, &schur_shape, ...);
// schur_shape[0] == n_pq 검증

cudaMalloc(&d_schur_mat_, schur_dim * schur_dim * sizeof(float));
cudssMatrixCreateDn(&schur_mat_obj_, schur_dim, schur_dim, schur_dim,
                    d_schur_mat_, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR);

// cuSOLVER 워크스페이스 (schur_dim 확정 후 한 번만 조회)
cusolverDnCreateParams(&cusolver_params_);
cusolverDnXgetrf_bufferSize(cusolver_handle_, cusolver_params_,
                            schur_dim, schur_dim, CUDA_R_32F, d_schur_mat_, schur_dim,
                            CUDA_R_32F, &work_device_bytes, &work_host_bytes);
cudaMalloc(&d_work_, work_device_bytes);
h_work_ = malloc(work_host_bytes);
cudaMalloc(&d_ipiv_, schur_dim * sizeof(int64_t));

// ── factorize_and_solve() ────────────────────────────────────────
// Phase 1: Partial factorization (J11만, Schur boundary에서 중단)
cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_, data_, A, x, b);

// Phase 2: Schur matrix 추출 (FACTORIZATION 이후에만 유효)
cudssDataGet(handle_, data_, CUDSS_DATA_SCHUR_MATRIX, &schur_mat_obj_, ...);

// Phase 3: Forward solve  (b 읽어 x에 씀)
cudssExecute(handle_, CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD,
             config_, data_, A, x, b);

// Phase 4a: x → b 복사 (GENERAL 행렬 전용, SOLVE_DIAG 대체)
cudaMemcpy(d_rhs_, d_x_, system_dim * sizeof(float), cudaMemcpyDeviceToDevice);

// Phase 4b: Dense LU 인수분해 (Schur matrix in-place)
cusolverDnXgetrf(cusolver_handle_, cusolver_params_,
                 schur_dim, schur_dim, CUDA_R_32F, d_schur_mat_, schur_dim,
                 d_ipiv_, CUDA_R_32F, d_work_, work_device_bytes,
                 h_work_, work_host_bytes, d_info_);

// Phase 4c: Dense 삼각 풀이  b[n_pvpq:] = y2
cusolverDnXgetrs(cusolver_handle_, cusolver_params_,
                 CUBLAS_OP_N, schur_dim, 1,
                 CUDA_R_32F, d_schur_mat_, schur_dim, d_ipiv_,
                 CUDA_R_32F, d_rhs_ + (system_dim - schur_dim), schur_dim, d_info_);

// Phase 5: Backward solve  (b 읽어 x에 최종 해)
cudssExecute(handle_, CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM,
             config_, data_, A, x, b);
```

### 5-2. Schur 인덱스 벡터

변수 순서: $[\underbrace{\Delta\theta_\text{pv},\,\Delta\theta_\text{pq}}_{n_\text{pvpq}},\,\underbrace{\Delta|V|_\text{pq}}_{n_\text{pq}}]$

```cpp
// 마지막 n_pq개를 Schur 블록으로 지정
std::vector<int32_t> h_schur_indices_(system_dim_, 0);
for (int32_t i = system_dim_ - schur_dim_; i < system_dim_; ++i)
    h_schur_indices_[i] = 1;
// → [0, 0, …, 0,  1, 1, …, 1]
//    ← n_pvpq →   ← n_pq →
```

### 5-3. GENERAL 행렬 Buffer 규칙 — 핵심 주의사항

공식 cuDSS sample은 **대칭(Symmetric/LDL^T)** 행렬 기준이다.  
Power-flow Jacobian은 비대칭(GENERAL/LU)이므로 버퍼 동작이 다르다.

#### 대칭(LDL^T) — 공식 sample 기준

```
FWD_PERM | FWD  (A, x, b)   reads b → writes x    x[n-k:] = Schur RHS
SOLVE_DIAG      (A, b, x)   reads x → writes b    b = D⁻¹·x  ← 인자 순서 반전!
external solve               on b[n-k:]            b[n-k:] = y₂
BWD | BWD_PERM  (A, x, b)   reads b → writes x    최종 해
```

SOLVE_DIAG 호출 시 solution/rhs 인자 순서가 **반전**됨:
```cpp
// 일반 규약: (handle, phase, config, data, A, x_output, b_input)
cudssExecute(..., CUDSS_PHASE_SOLVE_FWD_PERM | FWD, ..., A, x, b);  // x=output
cudssExecute(..., CUDSS_PHASE_SOLVE_DIAG,            ..., A, b, x);  // 반전! b=output, x=input
cudssExecute(..., CUDSS_PHASE_SOLVE_BWD | BWD_PERM,  ..., A, x, b);  // x=output
```

#### GENERAL(LU) — 본 구현

```
FWD_PERM | FWD  (A, x, b)   reads b → writes x    x[n-k:] = Schur RHS  (b 불변)
cudaMemcpy                   x → b                 D=I이므로 복사로 대체
external solve               on b[n-k:]            b[n-k:] = y₂
BWD | BWD_PERM  (A, x, b)   reads b → writes x    최종 해
```

GENERAL LU에서 D 행렬은 U 대각에 흡수되므로 SOLVE_DIAG 호출이 불필요하다.  
대신 `cudaMemcpy(b ← x)` 로 동일한 버퍼 이동을 수행한다.  
**이 복사를 생략하면 BWD가 수정되지 않은 원본 $b$ 를 읽어 잘못된 해를 생성한다.**

#### 진단으로 확인한 FWD 버퍼 동작 (case118_ieee)

FWD 실행 직후 `d_rhs_`와 `d_x_` 양쪽을 다운로드해 비교:

```
rhs_j11_max   = 3.53   ← 원본 b[0:n_pvpq] 그대로 (FWD가 b를 수정하지 않음)
rhs_schur_max = 1.67   ← 원본 b[n_pvpq:] 그대로
x_j11_max     = 7.21   ← FWD 출력 y₁ (b와 다름 → FWD가 x에 씀)
x_schur_max   = 2.34   ← Schur RHS b₂' (b₂=1.67과 다름 → FWD가 계산해 x에 씀)
```

### 5-4. CMake 설정

```cmake
find_package(CUDAToolkit REQUIRED)

target_link_libraries(cudss_benchmark PRIVATE
    cupf_core
    CUDA::cusolver    # cuSOLVER dense solver (Xgetrf, Xgetrs)
)
```

---

## 6. 정확도 검증

| case | full residual | schur residual |
|------|-------------:|---------------:|
| case118_ieee | 1.28e-05 | 2.18e-05 |
| case793_goc | 4.52e-04 | 4.21e-04 |
| case1354_pegase | 5.01e-04 | 3.27e-04 |
| case2746wop_k | 1.24e-03 | 1.12e-03 |
| case4601_goc | 8.71e-04 | 5.49e-04 |
| case8387_pegase | 5.49e-02 | 3.56e-02 |
| case9241_pegase | 5.31e-03 | 4.24e-03 |

두 경로 모두 동일한 해를 생성한다. Schur 경로의 잔차가 약간 다른 것은 FP32에서 연산 단계가 더 많기 때문이다.

---

## 7. 결론

### Analysis 포함/제외 비교 요약

| 비교 기준 | full win 배율 범위 |
|-----------|------------------:|
| Analysis 포함 (전체 시간) | 1.0×–6.5× |
| **Analysis 제외 (반복 비용)** | **2.2×–138×** |

Analysis를 제외한 반복 비용에서는 격차가 훨씬 크다.  
n_pq=7796(case9241)에서 full은 **0.99ms**, schur는 **136ms** — **138배 차이**.

### cuDSS Full Solve를 사용해야 하는 이유

| 원인 | 내용 |
|------|------|
| Schur ANALYSIS overhead | Schur reordering으로 full보다 1.5–3.7배 느림 |
| Dense 행렬 추출 비용 | $O(n_\text{pq}^2)$ — case9241에서 62.9ms |
| Dense LU 비용 | $O(n_\text{pq}^3)$ — case9241에서 53.3ms |
| Full sparse solve가 이미 빠름 | Factorization + Solve ≤ 1ms (모든 케이스) |

### Schur가 유효한 시나리오

- $n_\text{pq} / \text{dim} \lesssim 5\%$ 수준의 문제 (power-flow에서는 35–46%)
- Schur complement가 sparse한 문제 (dense cuSOLVER 대신 sparse solver 적용 가능)
- ANALYSIS를 한 번만 수행 후 REFACTORIZATION을 반복하는 경우 (Analysis 비용 상각)
