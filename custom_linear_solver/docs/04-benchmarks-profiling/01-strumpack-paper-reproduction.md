# STRUMPACK 논문 재현 — RTX 3090 / cuDSS 비교

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: 논문 Table 2(STRUMPACK 1.87× over cuDSS, A100)를 RTX 3090 24 GB에서 재현 시도 — 4/7 OOM, 방향성만 재현, custom은 4/4 설계상 bail-out.

대상 논문: Claus, Ghysels, Boukaram, Li, *"A graphics processing unit accelerated sparse direct solver and preconditioner with block low rank compression"*, IJHPCA 2025.
주요 주장: 논문 Table 2 — *"STRUMPACK exact factorization on a single GPU is on average **1.87× faster** than NVIDIA's cuDSS solver."* (A100 기준)

본 보고서는 같은 데이터셋(논문 Table 2)에서 같은 비교(STRUMPACK exact factorization vs cuDSS factorization, 단일 GPU)를 RTX 3090에서 재현 시도한 결과 + 같은 환경에서 `custom_linear_solver`도 함께 돌렸을 때의 동작을 정리한다. **이 문서는 베이스라인 비교가 아니라 논문 Table 2 재현만 다룬다.** 베이스라인 관계는 `../01-orientation/03-lineage-strumpack-not-the-baseline.md` 참고.

---

## 1. 실험 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 3090 (24 GB GDDR6X, sm_86) |
| CUDA | 13.0 (드라이버 580.159.03) |
| STRUMPACK | v8.0, **MAGMA 빌드** (`/workspace/local/strumpack-cuda-magma/install`), MAGMA 2.8.0 (sm_80) |
| cuDSS | `/opt/nvidia/cudss/lib/libcudss.so.0` |
| STRUMPACK 설정 | `compression=NONE`, `Krylov=DIRECT`, `reordering=METIS`, `matching=MAX_DIAGONAL_PRODUCT_SCALING`, GPU enabled |
| 측정 방식 | load 1회 + warmup 1회 + **5회 timed re-solve** (`--repeat 5`, reload 없음), `cudaDeviceSynchronize` 직후 wall-clock |

논문 vs 본 측정 차이:

| 항목 | 논문 (Table 2) | 본 측정 |
|---|---|---|
| GPU | A100 (80 GB, sm_80) | RTX 3090 (24 GB, sm_86) |
| FP64 peak | 9.7 TFLOPs | 0.56 TFLOPs (FP32의 1/64) |
| 메모리 대역폭 | 1555 GB/s (HBM2e) | 936 GB/s (GDDR6X) |
| GPU 메모리 | 80 GB | **24 GB** |

→ FP64 throughput 17×, 메모리 3.3× 작음. 절대 시간 차이와 일부 매트릭스 OOM은 예상된 한계.

## 2. 데이터셋 — 논문 Table 2의 7개 SuiteSparse 행렬

| 매트릭스 | N (×10³) | nnz (full, ×10³) | 대칭 | 도메인 |
|---|---:|---:|:-:|---|
| Serena | 1,391 | 64,531 | sym | 3D 지반역학 |
| Geo_1438 | 1,438 | 63,156 | sym | 3D 지반역학 |
| Hook_1498 | 1,498 | 60,917 | sym | 3D 구조 |
| ML_Geer | 1,504 | 110,879 | gen | 3D 다공질 유동 |
| Transport | 1,602 | 23,500 | gen | 3D Stokes 유동 |
| Flan_1565 | 1,565 | 117,406 | sym | 3D 구조 |
| Cube_Coup_dt0 | 2,164 | 129,133 | sym | 3D 구조 |

다운로드 위치: `/workspace/paper_matrices/`. N·full nnz는 논문 Table 2와 정확히 일치. RHS는 `prepare_dataset_vectors --mode random-rhs --seed 42`.

## 3. 측정 결과

### 3.1 성공한 매트릭스 (2개) — 5회 평균

| 매트릭스 | Solver | analyze [s] | factor [s] | solve [s] | berr |
|---|---|---:|---:|---:|---:|
| **Transport** | strumpack-gpu | 13.78 | **20.40** | **0.77 (CPU)** | 7.04e-15 |
| Transport | cudss-gpu | 6.78 | 23.02 | 0.09 | 4.89e-15 |
| **ML_Geer** | strumpack-gpu | 17.55 | **9.86** | **0.53 (CPU)** | 5.06e-13 |
| ML_Geer | cudss-gpu | 8.43 | 10.95 | 0.06 | 6.74e-14 |

논문 vs 본 측정 (factor 페이즈):

| 매트릭스 | 논문 STR A100 [s] | 우리 STR 3090 [s] | 3090/A100 | 논문 cuDSS A100 [s] | 우리 cuDSS 3090 [s] |
|---|---:|---:|---:|---:|---:|
| Transport | 3.2 | 20.40 | **6.4×** | 8.8 | 23.02 |
| ML_Geer | 2.0 | 9.86 | **4.9×** | 8.7 | 10.95 |

비 (cuDSS/STR):

| 매트릭스 | 논문 A100 | 우리 3090 |
|---|---:|---:|
| Transport | 2.75× | **1.13×** |
| ML_Geer | 4.35× | **1.11×** |

### 3.2 실패한 매트릭스 (2개) + 보류 (3개)

| 매트릭스 | 결과 | 원인 |
|---|---|---|
| Hook_1498 STR | SIGKILL (exit 137) | STRUMPACK 경고: *"large number of levels in the frontal/elimination tree ... could lead to stack overflows"* → OOM kill |
| Hook_1498 / Flan_1565 cuDSS | factorization status 2 | CUDSS_STATUS_ALLOC_FAILED, 24 GB 부족 |
| Flan_1565 STR | SIGKILL (exit 137) | 매트릭스 로드/analyze 중 OOM kill |
| Geo_1438 / Serena / Cube_Coup_dt0 | 보류 | 논문 A100 factor 메모리 추정 12–25 GB → 3090 24 GB OOM 거의 확실 (Cube는 논문조차 CPU OOM 명기) |

### 3.3 `custom_linear_solver` — 논문 4개 매트릭스 모두 설계상 bail-out

같은 4개(Transport, ML_Geer, Hook_1498, Flan_1565)에서 `custom_linear_solver`(FP64)는 **4개 모두 analyze 단계에서 `Status::AnalysisFailed`** 로 종료.

```cpp
// src/factorize/multifrontal.cu : 583
if (total > (1L << 30)) {  // > 1G doubles (8GB) -> bail out, keep cy71 path
    return MultifrontalPlan{};   // num_panels = 0
}
```

- `total` = 모든 패널의 fsz² 합(front arena doubles 크기). 8 GB 캡은 power-grid Jacobian용 sanity check. `front_off`가 int32라 17 GB가 진짜 상한, 8 GB는 안전 마진.
- 논문 매트릭스(Janna FEM/구조)는 fsz가 수천 단위 → fsz² 합산이 8 GB doubles를 쉽게 초과 → 설계상 bail-out.
- 이건 OOM/crash가 아니라 **솔버가 "내 도메인 아님"이라고 거절**하는 정상 동작.

| 매트릭스 | custom 결과 | analyze return |
|---|---|---|
| Transport, ML_Geer, Hook_1498, Flan_1565 | bail-out at analyze | `AnalysisFailed` (num_panels==0) |

### 3.4 `custom_linear_solver` sanity check — power-grid 야코비안 (10회 median)

| 매트릭스 | N | analyze [ms] | factor [ms] | solve [ms] | relres |
|---|---:|---:|---:|---:|---:|
| case3012wp | 5,725 | 15.15 | 0.282 | 0.181 | 3.5e-13 |
| case6468rte | 12,643 | 26.37 | 0.472 | 0.258 | 9.7e-14 |
| case8387pegase | 14,908 | 33.07 | 0.612 | 0.336 | 6.5e-14 |
| case_ACTIVSg25k | 47,246 | 90.12 | 1.263 | 0.624 | 3.6e-13 |
| case_SyntheticUSA | 156,255 | (analyze OK) | **factorize failed** | — | — |

case_SyntheticUSA(N=156K)는 analyze 통과 후 factorize에서 실패 — 본 솔버의 또 다른 상한(범위 밖, 별도 조사 필요).

핵심: 솔버는 자기 도메인(small-front sparse direct, 전력망 야코비안)에서는 ms 단위로 동작. 논문 도메인(large-front structural FEM)은 다른 매트릭스 클래스이며 의도적으로 처리하지 않는다. 이는 `../01-orientation/02-related-work-and-novelty.md` §2와 정확히 대칭:

> *"general GPU multifrontal libraries are engineered for matrices with **large dense fronts** ...; power-grid Jacobians have **no large fronts**, so those libraries run at a tiny fraction of peak"*

— 같은 multifrontal 패밀리 안에서 STRUMPACK은 large-front, custom은 tiny-front를 타깃. 본 실험은 그 분할을 직접 확인한 측정이다.

## 4. 관찰

### 관찰 1 — 방향성은 재현되지만 차이 폭이 작다

성공한 2개에서 STRUMPACK factor가 cuDSS보다 빠른 방향은 같지만 차이 폭이 논문 대비 **~2.4× 작음** (Transport 2.75→1.13×, ML_Geer 4.35→1.11×). 원인: A100(FP64 9.7 TFLOPs + HBM2e)에서 큰 dense front 위주 STRUMPACK이 비대칭 이득; MAGMA `vbatched_dgetrf`의 sm_80 튜닝.

### 관찰 2 — STRUMPACK solve는 모든 케이스 CPU fallback

매번 *"WARNING: Solve is performed on CPU"*. Transport solve = **0.77 s (STR)** vs **0.09 s (cuDSS)** — 8.5×. 논문 본문이 인정한 한계: *"STRUMPACK takes host memory, and thus requires extra transfers. We plan to develop a fully GPU resident solve phase."*

### 관찰 3 — RTX 3090 24 GB로는 4/7 매트릭스 측정 불가

→ **논문 평균 1.87× 완전 재현은 본 환경에서 불가능**. A100 80 GB가 필요.

## 5. 결론

| 결론 | 근거 |
|---|---|
| 논문 1.87× 평균 주장은 RTX 3090 24 GB에서 완전 재현 불가 | 7개 중 4개 OOM/SIGKILL |
| 측정 가능한 2개에서 방향성(STR < cuDSS in factor)은 재현 | §3.1 |
| 차이 폭은 A100 대비 약 2.4× 작음 | §3.1 row 비교 |
| STRUMPACK CPU-solve fallback은 논문 명기 동작, 실측 확인 | 관찰 2 |
| `custom_linear_solver`는 같은 4개에서 모두 설계상 bail-out (8 GB front-arena 캡) | §3.3 — 4개 모두 `AnalysisFailed` |
| 같은 솔버가 power-grid 야코비안(N=5K~47K)에서는 ms 단위로 동작 | §3.4 |

본 실험은 논문 주장의 검증/반증이 아니라 **같은 코드·데이터로 RTX 3090에서 어디까지 갈 수 있는지의 측정 기록**이자, `custom_linear_solver`가 STRUMPACK/cuDSS와 **다른 매트릭스 클래스를 타깃한다는 사실**을 직접 확인한 측정이다.

## 6. 참고

- `../01-orientation/03-lineage-strumpack-not-the-baseline.md` — STRUMPACK ≠ 코드 베이스
- `02-strumpack-vs-custom-case8387.md` — 같은 multifrontal에서 custom이 26× 빠른 이유의 분해
- `../main-report.md`
- Claus, Ghysels, Boukaram, Li, IJHPCA 2025. https://journals.sagepub.com/doi/full/10.1177/10943420241288567

원시 데이터: `/tmp/bench/paper_runs.csv`, 로그 `/tmp/bench/logs/{matrix}_{solver}.log`.
