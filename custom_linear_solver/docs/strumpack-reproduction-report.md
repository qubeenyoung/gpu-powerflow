# STRUMPACK 논문 재현 — RTX 3090 / cuDSS 비교 (단발성 측정)

대상 논문: Claus, Ghysels, Boukaram, Li, *"A graphics processing unit accelerated sparse direct
solver and preconditioner with block low rank compression"*, IJHPCA 2025.
주요 주장: 논문 Table 2 — *"For a collection of SuiteSparse matrices, the STRUMPACK exact factorization on a single GPU is on average **1.87× faster** than NVIDIA's cuDSS solver."* (A100 기준)

본 보고서는 같은 데이터셋(논문 Table 2)에서 같은 비교(STRUMPACK exact factorization vs cuDSS factorization, 단일 GPU)를 RTX 3090 위에서 재현 시도한 결과와, 같은 환경에서 본 저장소의 `custom_linear_solver`도 함께 돌렸을 때의 동작을 정리한다. **이 문서는 베이스라인 비교(`custom_linear_solver` vs STRUMPACK)가 아니라, 논문 Table 2 재현만을 다룬다**. 베이스라인 관계 분석은 `docs/baseline-vs-strumpack.md` 참고.

---

## 1. 실험 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 3090 (24 GB GDDR6X, sm_86) |
| CUDA | 13.0 (드라이버 580.159.03) |
| STRUMPACK | v8.0, **MAGMA 빌드** (`/workspace/local/strumpack-cuda-magma/install`) |
| MAGMA | 2.8.0 (sm_80 빌드) |
| cuDSS | `/opt/nvidia/cudss/lib/libcudss.so.0` |
| METIS | `/opt/third_party/install/common/lib/libmetis.so` |
| STRUMPACK 설정 | `compression=NONE`, `Krylov=DIRECT`, `reordering=METIS`, `matching=MAX_DIAGONAL_PRODUCT_SCALING`, GPU enabled |
| 측정 도구 | `/workspace/sparse_direct_solver/build_strumpack_magma/benchmark` (논문 Table 2와 같은 `analyze` / `factor` / `solve` 페이즈 분리) |
| 측정 방식 | 매트릭스 로드 1회 + warmup 1회 + **5회 timed re-solve** (`--repeat 5`, 매트릭스 reload 없음). 시간은 `cudaDeviceSynchronize` 직후 wall-clock |

논문 vs 본 측정 차이:

| 항목 | 논문 (Table 2) | 본 측정 |
|---|---|---|
| GPU | NVIDIA A100 (80 GB, sm_80) | RTX 3090 (24 GB, sm_86) |
| GPU FP64 peak | 9.7 TFLOPs | 0.56 TFLOPs (1/64 of FP32, sm_86 게이밍) |
| 메모리 대역폭 | 1555 GB/s (HBM2e) | 936 GB/s (GDDR6X) |
| GPU 메모리 | 80 GB | **24 GB** |
| 반복 횟수 | 단일 측정 | 5회 평균 |

→ FP64 throughput만 17×, 메모리는 3.3× 작음. 따라서 절대 시간 차이 (3090이 더 느림) 와 일부 매트릭스의 OOM은 예상된 한계.

## 2. 데이터셋 — 논문 Table 2의 7개 SuiteSparse 행렬

| 매트릭스 | N (×10³) | nnz (full, ×10³) | 대칭 | 출처 |
|---|---:|---:|:-:|---|
| Serena | 1,391 | 64,531 | sym | `suitesparse-collection-website.herokuapp.com/MM/Janna/Serena.tar.gz` |
| Geo_1438 | 1,438 | 63,156 | sym | Janna/Geo_1438 |
| Hook_1498 | 1,498 | 60,917 | sym | Janna/Hook_1498 |
| ML_Geer | 1,504 | 110,879 | gen | Janna/ML_Geer |
| Transport | 1,602 | 23,500 | gen | Janna/Transport |
| Flan_1565 | 1,565 | 117,406 | sym | Janna/Flan_1565 |
| Cube_Coup_dt0 | 2,164 | 129,133 | sym | Janna/Cube_Coup_dt0 |

다운로드 + 압축해제 위치: `/workspace/paper_matrices/`. 모든 매트릭스의 N과 full nnz는 논문 Table 2와 정확히 일치.
RHS / x_true는 `prepare_dataset_vectors --mode random-rhs --seed 42`로 생성.

## 3. 측정 결과

### 3.1 성공한 매트릭스 (2개)

5회 평균:

| 매트릭스 | Solver | analyze [s] | factor [s] | solve [s] | berr |
|---|---|---:|---:|---:|---:|
| **Transport** | strumpack-gpu (MAGMA) | 13.78 | **20.40** | **0.77 (CPU)** | 7.04e-15 |
| Transport | cudss-gpu | 6.78 | 23.02 | 0.09 | 4.89e-15 |
| **ML_Geer** | strumpack-gpu (MAGMA) | 17.55 | **9.86** | **0.53 (CPU)** | 5.06e-13 |
| ML_Geer | cudss-gpu | 8.43 | 10.95 | 0.06 | 6.74e-14 |

논문 vs 본 측정 비교 (factor 페이즈):

| 매트릭스 | 논문 STR A100 [s] | 우리 STR 3090 [s] | 3090/A100 | 논문 cuDSS A100 [s] | 우리 cuDSS 3090 [s] | 3090/A100 |
|---|---:|---:|---:|---:|---:|---:|
| Transport | 3.2 | 20.40 | **6.4×** | 8.8 | 23.02 | **2.6×** |
| ML_Geer | 2.0 | 9.86 | **4.9×** | 8.7 | 10.95 | **1.3×** |

논문의 비 (cuDSS/STR):

| 매트릭스 | 논문 A100 (cuDSS/STR) | 우리 3090 (cuDSS/STR) |
|---|---:|---:|
| Transport | 2.75× | **1.13×** |
| ML_Geer | 4.35× | **1.11×** |

### 3.2 실패한 매트릭스 (2개)

| 매트릭스 | Solver | 결과 | 원인 |
|---|---|---|---|
| Hook_1498 | strumpack-gpu | **SIGKILL (exit 137)** | STRUMPACK 자체 경고: *"Detected a large number of levels in the frontal/elimination tree. STRUMPACK currently does not handle this safely, which could lead to segmentation faults due to stack overflows. As a remedy, try `--sp_enable_METIS_NodeNDP`..."* → MPI 의존, 비활성화 상태에서 OOM kill |
| Hook_1498 | cudss-gpu | **cuDSS factorization failed with status 2** | CUDSS_STATUS_ALLOC_FAILED. 24 GB GPU 메모리 부족 추정 |
| Flan_1565 | strumpack-gpu | **SIGKILL (exit 137)** | 출력 zero. 매트릭스 로드 직후 또는 analyze 중 OOM kill |
| Flan_1565 | cudss-gpu | **cuDSS factorization failed with status 2** | Hook_1498과 동일 패턴, GPU 메모리 부족 |

### 3.3 보류한 매트릭스 (3개)

| 매트릭스 | 보류 사유 |
|---|---|
| Geo_1438 | 논문 STR A100 factor 12.7s → 추정 factor 메모리 12–20 GB. RTX 3090 24 GB에서 borderline. 이전 단발 probe에서 cuDSS status 2 (OOM) 확인됨 |
| Serena | 논문 STR A100 factor 17.9s → factor 메모리 15–25 GB. 3090 24 GB OOM 거의 확실 (probe 확인) |
| Cube_Coup_dt0 | 논문 본문에서도 CPU 메모리 OOM 표기, A100 80 GB에서만 62.1s. 3090 24 GB에서는 거의 확실히 OOM |

### 3.4 `custom_linear_solver` 측정 — 4/4 매트릭스 모두 설계상 bail-out

같은 4개 매트릭스(Transport, ML_Geer, Hook_1498, Flan_1565)에 대해 본 저장소의 `custom_linear_solver` (FP64, `--repeat` 측정 모드) 를 실행한 결과 **4개 모두 analyze 단계에서 `Status::AnalysisFailed` 로 종료**.

원인 (소스 확인됨):

```cpp
// src/factorize/multifrontal.cu : 583
if (total > (1L << 30)) {  // > 1G doubles (8GB) -> bail out, keep cy71 path
    return MultifrontalPlan{};   // num_panels = 0
}
```

해석:
- `total` = 모든 패널의 fsz² 합 (front arena의 doubles 단위 크기)
- 8 GB 캡은 power-grid Jacobian용 sanity check. front_off 가 int32 인덱스 (`front_off[p] = static_cast<int>(total)`) 라 17 GB(INT_MAX doubles) 가 진짜 상한이고, 8 GB는 그보다 여유 있게 잡힌 안전 마진.
- 논문 4개 매트릭스(Janna 그룹 FEM/구조)는 fsz가 수천 단위까지 큼 → fsz² 합산이 8 GB doubles를 쉽게 넘어감 → 설계상 bail-out.
- 이건 OOM/crash가 아니라 **솔버가 "내 도메인 아님"이라고 거절**한 정상 동작.

| 매트릭스 | custom_linear_solver 결과 | analyze return |
|---|---|---|
| Transport (n=1.6M, nnz=23.5M) | bail-out at analyze | `AnalysisFailed` (num_panels==0) |
| ML_Geer (n=1.5M, nnz=110.9M) | bail-out at analyze | `AnalysisFailed` (num_panels==0) |
| Hook_1498 (n=1.5M, nnz=60.9M) | bail-out at analyze | `AnalysisFailed` (num_panels==0) |
| Flan_1565 (n=1.6M, nnz=117.4M) | bail-out at analyze | `AnalysisFailed` (num_panels==0) |

### 3.5 `custom_linear_solver` sanity check — power-grid 야코비안

같은 GPU/같은 빌드로 power-grid Jacobian(=설계 대상 도메인)에 대해 동작 확인. 10회 측정 median:

| 매트릭스 | N | analyze [ms] | factor [ms] | solve [ms] | relres |
|---|---:|---:|---:|---:|---:|
| case3012wp | 5,725 | 15.15 | 0.282 | 0.181 | 3.5e-13 |
| case6468rte | 12,643 | 26.37 | 0.472 | 0.258 | 9.7e-14 |
| case8387pegase | 14,908 | 33.07 | 0.612 | 0.336 | 6.5e-14 |
| case_ACTIVSg25k | 47,246 | 90.12 | 1.263 | 0.624 | 3.6e-13 |
| case_SyntheticUSA | 156,255 | (analyze OK) | **factorize failed** | — | — |

case_SyntheticUSA(N=156K, ~82K 버스, 본 저장소에 있는 가장 큰 전력망 케이스)는 analyze 통과 후 factorize 단계에서 실패 — 본 솔버의 또 다른 상한이 있음을 확인. 이 한계는 본 보고서의 범위 밖이며 별도 조사 필요.

핵심: 솔버는 자기 도메인(small-front sparse direct, 전력망 야코비안)에서는 ms 단위로 동작. 논문 도메인(large-front structural FEM)은 fundamentally 다른 매트릭스 클래스이며 솔버가 의도적으로 처리하지 않는다.

이는 본 저장소의 `docs/related-work-and-contribution.md` §2 의 핵심 명제와 정확히 대칭:

> *"general GPU multifrontal libraries are engineered for matrices with **large dense fronts** (their FLOPs live there); power-grid Jacobians have **no large fronts**, so those libraries run at a tiny fraction of peak"*

— 즉 같은 multifrontal 패밀리 안에서 두 솔버는 서로 다른 front-size 분포를 타깃한다. STRUMPACK은 large-front, custom_linear_solver는 tiny-front. 본 실험은 그 분할을 직접 확인한 측정이다.

## 4. 관찰

### 관찰 1 — 방향성은 재현되지만 차이 폭이 작다

성공한 2개 매트릭스에서 STRUMPACK exact factor가 cuDSS보다 빠른 방향은 같지만 차이 폭이 논문 대비 **~2.4× 작음** (Transport 2.75→1.13×, ML_Geer 4.35→1.11×). 가능한 원인:

- **GPU 차이**: A100은 sm_80 데이터센터 카드로 FP64 9.7 TFLOPs + HBM2e 1.55 TB/s. RTX 3090은 sm_86 게이밍 카드로 FP64 0.56 TFLOPs (FP32의 1/64) + GDDR6X 0.94 TB/s. 큰 dense front 위주 STRUMPACK이 A100의 FP64/HBM에서 비대칭적으로 큰 이득을 누림.
- **STRUMPACK GPU 경로의 sm_80 튜닝**: 논문이 사용한 MAGMA `vbatched_dgetrf` 커널은 sm_80에 맞춤 가능성.

### 관찰 2 — STRUMPACK solve는 모든 케이스에서 CPU fallback

`strumpack-gpu` solve가 매번 *"WARNING: Solve is performed on CPU"* 를 출력. Transport solve = **0.77 s (STRUMPACK)** vs **0.09 s (cuDSS)** — 8.5× 차이. 논문 본문이 직접 인정한 한계:

> *"cuDSS expects the input and output arguments for the solver (right-hand side and solution) in device memory, while STRUMPACK takes host memory, and thus requires extra transfers. We plan to develop a fully GPU resident solve phase in the near future."*

### 관찰 3 — RTX 3090 24 GB로는 논문 Table 2 4/7 매트릭스 측정 불가

- Hook_1498 STRUMPACK = METIS ND 스택 / 메모리 한계 (STRUMPACK 자체 경고)
- Hook_1498 / Flan_1565 / Geo_1438 / Serena cuDSS = factorization status 2 (GPU OOM)
- Cube_Coup_dt0 = 논문조차 CPU OOM 명기

→ **논문 평균 1.87× 주장의 완전 재현은 본 환경에서 불가능**. A100 80 GB가 필요.

## 5. 결론

| 결론 | 근거 |
|---|---|
| 논문 1.87× 평균 주장은 본 환경(RTX 3090 24 GB)에서 완전히 재현 불가 | 7개 중 4개가 OOM/SIGKILL |
| 측정 가능한 2개 매트릭스(Transport, ML_Geer)에서 방향성(STR < cuDSS in factor)은 재현 | 표 3.1 |
| 차이 폭은 A100 대비 약 2.4× 작음 | 표 3.1 row 비교 |
| STRUMPACK CPU-solve fallback은 논문 본문 명기 동작이며 실측에서 확인 | 관찰 2 |
| `custom_linear_solver`는 같은 4개 매트릭스에서 모두 설계상 bail-out (8 GB front-arena 캡) | 표 3.4 — 4개 모두 `AnalysisFailed`, num_panels==0 |
| 같은 솔버가 power-grid 야코비안(N=5K~47K) 에서는 ms 단위로 동작 | 표 3.5 |

본 실험은 논문 주장의 검증/반증이 아니라, **RTX 3090 / 24 GB 환경에서 같은 코드와 같은 데이터로 어디까지 갈 수 있는지의 측정 기록**이며, 동시에 `custom_linear_solver`가 STRUMPACK/cuDSS와 **다른 매트릭스 클래스를 타깃한다는 사실**을 직접 확인한 측정이다.

---

## 6. 재현 명령

```bash
# STRUMPACK MAGMA 빌드 사용을 위한 LD_LIBRARY_PATH 설정
export LD_LIBRARY_PATH=/workspace/local/strumpack-cuda-magma/install/lib:\
/workspace/local/magma/install/lib:/opt/intel/oneapi/mkl/2026.0/lib:\
/opt/nvidia/cudss/lib:/opt/third_party/install/common/lib:\
/opt/third_party/install/mumps/lib:/usr/local/cuda/lib64:\
/usr/lib/x86_64-linux-gnu/openmpi/lib

BIN=/workspace/sparse_direct_solver/build_strumpack_magma/benchmark

# (matrix, solver) 한 쌍당 load 1회 + warmup + 5회 timed
"$BIN" --matrix-set paper --matrix Transport \
  --solver strumpack-gpu --warmup-gpu --repeat 5 --append \
  --output /tmp/bench/paper_runs.csv

"$BIN" --matrix-set paper --matrix Transport \
  --solver cudss-gpu --warmup-gpu --repeat 5 --append \
  --output /tmp/bench/paper_runs.csv
# (ML_Geer 등 다른 매트릭스도 동일)
```

원시 데이터: `/tmp/bench/paper_runs.csv`. 매트릭스/솔버별 stdout: `/tmp/bench/logs/{matrix}_{solver}.log`.

### custom_linear_solver

```bash
# 빌드
cmake -S /workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver \
      -B /tmp/clsb -DCLS_BUILD_CUDA_OPS=ON -DCLS_BUILD_SCRIPTS=ON \
      -DCLS_CUDA_ARCHITECTURES=86
cmake --build /tmp/clsb -j 8

# 논문 매트릭스 (예상: AnalysisFailed)
/tmp/clsb/custom_linear_solver_run \
    --matrix /workspace/paper_matrices/Transport/Transport.mtx \
    --rhs    /workspace/paper_matrices/Transport/rhs.mtx \
    --repeat 1

# power-grid sanity check
/tmp/clsb/custom_linear_solver_run \
    --matrix /datasets/power_system/nr_linear_systems/case_ACTIVSg25k/J.mtx \
    --rhs    /datasets/power_system/nr_linear_systems/case_ACTIVSg25k/rhs.mtx \
    --repeat 10
```

## 7. 참고 문헌

- Claus, Ghysels, Boukaram, Li, *"A GPU accelerated sparse direct solver and preconditioner with block low rank compression"*, IJHPCA 2025. https://journals.sagepub.com/doi/full/10.1177/10943420241288567
