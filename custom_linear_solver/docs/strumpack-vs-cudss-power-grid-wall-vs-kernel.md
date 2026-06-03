# STRUMPACK vs cuDSS vs custom_linear_solver — paper 재현 → 전력망 이동 → 최적화 분석 논리 구조

본 문서는 다음 순서를 한 흐름으로 정리한다.

1. STRUMPACK 논문이 자체 평가에 쓴 SuiteSparse 행렬과 재현 시 우리가 쓴 행렬 (§1, §2)
2. 그 행렬 위에서 측정한 STRUMPACK + cuDSS 결과 (§3)
3. **`custom_linear_solver`가 같은 행렬을 풀지 못하는 이유 분석** (§4)
4. 그래서 동일 환경에서 **전력망 야코비안**으로 이동, 같은 데이터셋 위 STRUMPACK / cuDSS / custom 세 솔버 측정 (§5, §6)
5. **`custom_linear_solver`가 STRUMPACK 대비 전력망 데이터에 어떻게 최적화했는가** — 분석 논리 구조 (§7)
6. 종합 결론 (§8)

세부 측정 데이터는 `docs/strumpack-paper-table2-reproduction.md` (paper 행렬 단독) 와 `docs/lineage-strumpack-not-the-baseline.md` (베이스라인 lineage) 참고. 본 문서는 그 둘을 한 줄기로 잇는 분석 logic 위주.

---

## 1. STRUMPACK 논문이 보고한 SuiteSparse 행렬

대상 논문: Claus, Ghysels, Boukaram, Li, *"A graphics processing unit accelerated sparse direct
solver and preconditioner with block low rank compression"*, IJHPCA 2025.
핵심 주장 (논문 §4 Table 2):

> *"For a collection of SuiteSparse matrices, the STRUMPACK exact factorization on a single GPU is on average **1.87× faster** than NVIDIA's cuDSS solver."* (A100)

논문 Table 2의 행렬 7개 — 전부 SuiteSparse Janna 그룹 (Massimiliano Ferronato의 구조해석/지반/유체 코드 출력):

| 매트릭스 | N (×10³) | nnz (full, ×10³) | 도메인 | 패턴 |
|---|---:|---:|---|---|
| Serena | 1,391 | 64,531 | sym | structural (gas reservoir geomechanics) |
| Geo_1438 | 1,438 | 63,156 | sym | structural (geomechanics 3D) |
| Hook_1498 | 1,498 | 60,917 | sym | structural (steel hook) |
| ML_Geer | 1,504 | 110,879 | gen | flow (3D porous media) |
| Transport | 1,602 | 23,500 | gen | flow (3D Stokes/transport) |
| Flan_1565 | 1,565 | 117,406 | sym | structural (steel flange) |
| Cube_Coup_dt0 | 2,164 | 129,133 | sym | structural (coupled cube) |

공통 성격:
- N ≈ 1.4–2.2 M, nnz ≈ 23–130 M
- 모두 3D 연속체 PDE 이산화 (FEM/FVM)
- 그래프가 3D mesh 토폴로지 → nested dissection separator가 크고 **dense front 큼**
- 결과적으로 multifrontal LU의 FLOPs 대부분이 큰 dense GEMM에 집중 → STRUMPACK이 자기 최적점에서 동작

STRUMPACK 본문이 사용한 비교 환경: NVIDIA A100 (80 GB, sm_80), MAGMA-enabled 빌드, exact factorization 모드 (compression off), 단발 측정.

## 2. 본 환경에서 재현에 사용한 행렬

전부 동일 7개를 SuiteSparse 공식 사이트에서 다운로드:

```
suitesparse-collection-website.herokuapp.com/MM/Janna/{name}.tar.gz
```

다운로드 위치: `/workspace/paper_matrices/{name}/{name}.mtx`. 본 측정 환경:

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 3090 (24 GB GDDR6X, sm_86) — 논문 A100 대비 FP64 1/17, 메모리 1/3.3 |
| STRUMPACK | v8.0, **MAGMA 빌드** `/workspace/local/strumpack-cuda-magma/install`, `compression=NONE`, METIS reorder, MC64 matching |
| cuDSS | `/opt/nvidia/cudss/lib/libcudss.so.0` |
| custom_linear_solver | 본 저장소 `/workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver/`, FP64 mode |
| 측정 도구 | `/workspace/sparse_direct_solver/build_strumpack_magma/benchmark` (STR + cuDSS), `/tmp/clsb/custom_linear_solver_run` (custom). 둘 다 행렬 1회 로드 + warmup + N회 timed re-solve |

행렬별 측정 가능 여부:

| 매트릭스 | STRUMPACK MAGMA | cuDSS | custom_linear_solver | 비고 |
|---|---|---|---|---|
| Transport | ✓ 5회 | ✓ 5회 | **bail-out at analyze** | 측정 가능 / 진단 가능 |
| ML_Geer | ✓ 5회 | ✓ 5회 | **bail-out at analyze** | 측정 가능 / 진단 가능 |
| Hook_1498 | ❌ SIGKILL (METIS 스택) | ❌ status 2 (OOM) | bail-out at analyze | 3090 24 GB 부족 |
| Flan_1565 | ❌ SIGKILL | ❌ status 2 (OOM) | bail-out at analyze | 3090 24 GB 부족 |
| Geo_1438, Serena, Cube_Coup_dt0 | 시도 안 함 (probe에서 OOM 확인) | — | — | 3090에서 측정 불가 보류 |

→ 실제 측정 성공한 paper 행렬은 **Transport, ML_Geer 2개**.

## 3. STRUMPACK + cuDSS 측정 결과 (paper 행렬)

5회 평균 (`docs/strumpack-paper-table2-reproduction.md` §3.1):

| 매트릭스 | Solver | analyze [s] | factor [s] | solve [s] | berr |
|---|---|---:|---:|---:|---:|
| **Transport** | strumpack-gpu (MAGMA) | 13.78 | **20.40** | **0.77 (CPU)** | 7.04e-15 |
| Transport | cudss-gpu | 6.78 | 23.02 | 0.09 | 4.89e-15 |
| **ML_Geer** | strumpack-gpu (MAGMA) | 17.55 | **9.86** | **0.53 (CPU)** | 5.06e-13 |
| ML_Geer | cudss-gpu | 8.43 | 10.95 | 0.06 | 6.74e-14 |

핵심:
- STRUMPACK exact factor < cuDSS factor (방향성은 논문과 일치)
- 차이 폭은 논문 A100 대비 ~2.4× 작음 (Transport 2.75× → 1.13×, ML_Geer 4.35× → 1.11×)
- STRUMPACK solve는 모든 케이스에서 *"WARNING: Solve is performed on CPU"* — 논문 본문이 직접 인정한 한계 *"STRUMPACK takes host memory, ... extra transfers ... plan to develop a fully GPU resident solve phase"*

## 4. `custom_linear_solver`는 paper 행렬을 풀지 못한다 — 이유 분석

같은 4개 행렬에 `custom_linear_solver --repeat 1` 실행 결과 **모두 `Status::AnalysisFailed` 로 즉시 종료**.

### 4.1 1차 원인 — analyze 단계의 front-arena 캡

`src/factorize/multifrontal.cu:577–585`:

```cpp
std::vector<int> front_off(P + 1, 0);
long total = 0;
for (int p = 0; p < P; ++p) {
    const long fsz = mf.front_ptr[p + 1] - mf.front_ptr[p];
    front_off[p] = static_cast<int>(total);
    total += fsz * fsz;
    if (total > (1L << 30)) {  // > 1G doubles (8GB) -> bail out, keep cy71 path
        return MultifrontalPlan{};       // num_panels = 0
    }
}
```

- `total` = 모든 패널의 fsz² 합 (front arena의 doubles 단위 크기, 단위 = 8 바이트)
- **8 GB doubles 한도** 넘으면 빈 plan을 리턴 → `solver.cpp:236`이 `Status::AnalysisFailed` 로 보고
- `front_off[p] = static_cast<int>(total)` — int32 인덱스. 진짜 한계는 INT_MAX = 2.1G doubles = 17 GB. 8 GB 캡은 그보다 여유 잡은 sanity check.
- 의도: power-grid 야코비안에서는 front_total이 수 MB ~ 수백 MB 단위라 8 GB는 절대 넘지 않음. paper 행렬은 fsz가 큰 fronts를 가지므로 8 GB를 쉽게 넘는다.

### 4.2 왜 paper 행렬은 fsz² 합이 큰가 — 그래프 구조

행렬의 fsz (front size) 는 nested dissection separator 크기에 결정된다.

| 매트릭스 클래스 | 그래프 패턴 | ND separator 크기 | front 크기 분포 |
|---|---|---|---|
| Janna 그룹 (paper) | 3D 연속체 mesh, 거의 균질한 connectivity | 큰 separator (3D mesh의 *"2D 단면"*, O(N^{2/3})) | 큰 dense front 다수 |
| 전력망 야코비안 (target) | 거의 평면, sparse, hub-and-spoke 토폴로지 | 매우 작은 separator (그래프가 거의 트리에 가까움) | tiny front 압도적, max front도 수백 수준 |

이 차이는 본 저장소 `docs/related-work-and-novelty.md` §2 가 외부 출처 (Spatula, MICRO'23) 와 함께 다음으로 정량화:

> *"95% of fronts are fsz ≤ 16 (only 5% of the flops); on FullChip (회로) STRUMPACK runs at 0.3 GFLOP/s = 0.004% of V100 peak; on atmosmoddd (덴스 PDE) at 26% of peak"*

### 4.3 따라서 — 8 GB 캡은 OOM이 아니라 *"내 도메인 아님"* 거절

- 8 GB 캡은 **솔버가 의도적으로 둔 sanity check**. 동작 환경 가정(전력망 야코비안: front_total ≪ 8 GB)을 깨는 입력에 대해 깨끗한 실패를 반환.
- crash/SIGKILL이 아니라 `Status::AnalysisFailed` 라는 **API 차원의 정상 실패**.
- 캡을 1L → 3L 등으로 늘려도 front_off int32 인덱스, front-arena GPU 메모리(24 GB 한계), tiny-front 전용 커널들(warp-per-front 등 fsz ≤ 32 가정)이 모두 power-grid 가정에 맞춰 짜여 있어 의미 있는 측정이 불가능. 측정의 honesty 보호.

→ paper 행렬 4개 결과는 모두 *"AnalysisFailed (num_panels==0)"*. 이는 솔버 결함이 아니라 **타깃 매트릭스 클래스 표시**.

## 5. 그래서 전력망 야코비안으로 이동

paper 행렬 위 3-way 비교는 불공정 (custom은 거절, STRUMPACK은 자기 최적점, cuDSS는 OOM). 대신 **`custom_linear_solver`의 타깃 도메인** = 전력망 야코비안 — 같은 매트릭스 위에서 세 솔버를 모두 돌릴 수 있는 환경.

### 5.1 데이터셋

`/datasets/power_system/nr_linear_systems/{case}/J.mtx` — MATPOWER + ACTIVSg 전력망 케이스의 Newton-Raphson iteration 2에서 덤프한 야코비안. 동일 RHS (mismatch vector) + 동일 x_true:

| case | n_bus | N (Jacobian) | nnz |
|---|---:|---:|---:|
| case3012wp | 3,012 | 5,725 | 36,263 |
| case6468rte | 6,468 | 12,643 | 87,845 |
| case8387pegase | 8,387 | 14,908 | 110,572 |
| case_ACTIVSg25k | 25,000 | 47,246 | 318,672 |
| case_SyntheticUSA | 82,000 | 156,255 | 1,052,085 |

### 5.2 실행 환경 + 측정 방식

같은 RTX 3090, 같은 빌드/링크. STRUMPACK은 MAGMA 빌드, cuDSS는 v0.x, custom은 FP64 모드. 행렬 1회 로드 + warmup + **10회 timed**, median/mean 보고.

**측정 방식 두 가지** (이전 측정과 다른 점):

| 측정 | 무엇 | 어떻게 |
|---|---|---|
| **W: wall-clock (per-call latency)** | 솔버 함수 진입~반환의 host 시계 | `std::chrono::steady_clock` 으로 `cudssExecute` / `STRUMPACK_factor` / `solver.factorize()` 호출 wrap |
| **K: GPU kernel-only** | 스트림 위 GPU 작업만 | `cudaEventRecord(start)` → 솔버 호출 → `cudaEventRecord(stop)` → `cudaEventSynchronize` → `cudaEventElapsedTime`. API/CPU side overhead 제외 |

| Solver | wall (W) | kernel (K) | 차이의 의미 |
|---|---|---|---|
| cuDSS | `cudssExecute` 호출 + 내부 스케줄링 + 커널 + 동기화 | 커널만 | cuDSS API call/dispatch overhead |
| STRUMPACK | `STRUMPACK_factor` 호출 (heavy CPU multifrontal 작업 포함) + 커널 + 동기화 | GPU 스트림 위 작업만 | STRUMPACK의 host-side multifrontal 스케줄링/MAGMA dispatch overhead |
| custom | `solver.factorize()` = CUDA graph replay (host side 거의 0) + 커널 + 동기화 | 커널만 | 거의 0 (graph replay 덕분) |

따라서 custom의 W ≈ K (97%, `docs/fp32-batched-kernel-optimization.md`), STRUMPACK/cuDSS는 W ≫ K. 두 measurement 모두 의미 있음 — W는 사용자 NR loop가 보는 per-call latency, K는 순수 GPU 효율.

코드 위치: `/workspace/sparse_direct_solver/src/third_party_solvers/cudss_solver.cpp`, `strumpack_solver.cpp` 의 CUDA event 변경 — `cudaEventRecord` / `cudaEventSynchronize` / `cudaEventElapsedTime` 사용.

## 6. 측정 결과 — STRUMPACK MAGMA vs cuDSS vs custom_linear_solver (power-grid)

원시 데이터:
- `/tmp/bench/power_kernel.csv` — STRUMPACK + cuDSS 새 measurement (GPU-event)
- `/tmp/bench/runs.csv`, `/tmp/bench/strumpack_magma.csv` — 이전 wall-clock measurement
- custom_linear_solver: 10회 median, `CLS_KERNEL_TIME=1` 로 wall + kernel 둘 다

### 6.1 Wall-clock (per-call latency) — NR loop이 보는 실제 latency

| 매트릭스 | N | Solver | analyze [ms] | factor [ms] | solve [ms] |
|---|---:|---|---:|---:|---:|
| case3012wp | 5,725 | strumpack-gpu (MAGMA) | 20.15 | 146.21 | 6.12 |
|  |  | cudss-gpu | 29.19 | 13.06 | 3.25 |
|  |  | custom_linear_solver | 14.84 | **0.28** | **0.18** |
| case6468rte | 12,643 | strumpack-gpu (MAGMA) | 28.84 | 137.14 | 6.50 |
|  |  | cudss-gpu | 34.79 | 13.26 | 3.35 |
|  |  | custom_linear_solver | 25.58 | **0.44** | **0.25** |
| case8387pegase | 14,908 | strumpack-gpu (MAGMA) | 43.02 | 126.32 | 6.50 |
|  |  | cudss-gpu | 42.53 | 13.50 | 3.38 |
|  |  | custom_linear_solver | 32.71 | **0.55** | **0.30** |
| case_ACTIVSg25k | 47,246 | strumpack-gpu (MAGMA) | 122.16 | 169.17 | 9.27 |
|  |  | cudss-gpu | 74.60 | 14.06 | 3.68 |
|  |  | custom_linear_solver | 88.60 | **1.30** | **0.64** |
| case_SyntheticUSA | 156,255 | strumpack-gpu (MAGMA) | 420.81 | 178.14 | 16.71 |
|  |  | cudss-gpu | 198.71 | 9.48 | 3.61 |
|  |  | custom_linear_solver | (OK) | factorize failed | — |

### 6.2 GPU kernel-only (순수 GPU 작업 시간) — 새 measurement

`cudaEventRecord` 로 GPU stream 위 작업만 측정. cuDSS/STRUMPACK은 호스트 API/CPU overhead 제외, custom은 graph replay라 W ≈ K.

| 매트릭스 | Solver | analyze K [ms] | factor K [ms] | solve K [ms] |
|---|---|---:|---:|---:|
| case3012wp | strumpack-gpu | 10.76 | 15.20 | 7.29 |
|  | cudss-gpu | 18.69 | 0.479 | 0.224 |
|  | custom_linear_solver | 14.84 | **0.276** | **0.174** |
| case6468rte | strumpack-gpu | 27.01 | 12.12 | 8.61 |
|  | cudss-gpu | 33.87 | 0.624 | 0.287 |
|  | custom_linear_solver | 25.58 | **0.438** | **0.246** |
| case8387pegase | strumpack-gpu | 50.80 | 25.40 | 11.43 |
|  | cudss-gpu | 28.41 | 1.102 | 0.363 |
|  | custom_linear_solver | 32.71 | **0.544** | **0.297** |
| case_ACTIVSg25k | strumpack-gpu | 127.62 | 21.54 | 51.14 \* |
|  | cudss-gpu | 69.75 | **1.86** | 0.673 |
|  | custom_linear_solver | 88.60 | **1.29** | **0.634** |
| case_SyntheticUSA | strumpack-gpu | 465.95 | 87.59 | 81.22 \* |
|  | cudss-gpu | 198.71 | 5.108 | 1.441 |
|  | custom_linear_solver | — | factorize failed | — |

\* STRUMPACK solve의 K 값은 신뢰도 낮음 — STRUMPACK이 solve를 *"CPU fallback"* 으로 처리하면서 GPU stream 위에 H2D/D2H가 비동기로 흩어져 있어 default-stream CUDA event 측정이 wall-clock보다 큰 값을 보고. STRUMPACK solve의 의미 있는 비교는 wall-clock (§6.1) 으로 봐야 함.

### 6.3 W − K 차이 (host/API overhead 정량)

case_ACTIVSg25k 기준 (per call):

| Solver | factor W [ms] | factor K [ms] | host overhead | overhead 비중 |
|---|---:|---:|---:|---:|
| strumpack-gpu | 169.17 | 21.54 | 147.63 | **87.3%** |
| cudss-gpu | 14.06 | 1.86 | 12.20 | **86.8%** |
| custom_linear_solver | 1.30 | 1.29 | 0.01 | **0.8%** |

해석:
- STRUMPACK W 169 ms 중 GPU 작업은 22 ms (13%), 나머지 147 ms는 host-side multifrontal 스케줄링 / MAGMA dispatch / 동기화. 알고리즘적으로 *"GPU가 충분히 안 쓰임"*.
- cuDSS W 14 ms 중 GPU 작업은 1.9 ms (13%), 나머지 12 ms는 `cudssExecute` API call overhead. 이것도 매 호출 비용.
- custom W 1.3 ms 중 GPU 작업이 1.29 ms (99%). CUDA Graph replay 덕분에 host side 거의 0.

### 6.4 정량 핵심 (case_ACTIVSg25k 기준)

**(A) Wall-clock per-call latency (NR loop이 보는 시간)**

| 항목 | STRUMPACK | cuDSS | custom | custom vs STR | custom vs cuDSS |
|---|---:|---:|---:|---:|---:|
| factor [ms] | 169.17 | 14.06 | **1.30** | **−99.2% (130×)** | **−90.7% (10.8×)** |
| solve [ms] | 9.27 | 3.68 | **0.64** | **−93.1%** | **−82.6%** |
| f+s [ms] | 178.44 | 17.74 | **1.94** | **−98.9%** | **−89.1%** |

**(B) GPU kernel-only (순수 알고리즘/커널 효율)**

| 항목 | STRUMPACK | cuDSS | custom | custom vs STR | custom vs cuDSS |
|---|---:|---:|---:|---:|---:|
| factor K [ms] | 21.54 | 1.86 | **1.29** | **−94.0% (16.7×)** | **−30.6% (1.44×)** |
| solve K [ms] | 51.14 \* | 0.673 | **0.634** | (신뢰도 낮음) | **−5.8% (1.06×)** |
| f+s K [ms] | 72.68 \* | 2.53 | **1.92** | — | **−24.1%** |

### 6.5 두 measurement가 말하는 다른 이야기

| 관점 | 결론 | 해석 |
|---|---|---|
| Wall-clock (per-call latency) | custom이 STR 대비 ~130×, cuDSS 대비 ~10× 빠름 | NR iter loop의 사용자 시점. host/API overhead가 솔버 간 차이의 86%를 결정 |
| GPU kernel-only | custom이 STR 대비 ~17×, cuDSS 대비 ~1.4× 빠름 | 순수 GPU 알고리즘 효율. STRUMPACK은 GPU 활용 자체가 낮음 (factor의 87%가 host work). cuDSS는 GPU 효율 자체는 custom과 비슷 |
| 의미 | **둘 다 진짜 차이** | wall-clock 차이는 *"per-call overhead 엔지니어링 (CUDA Graph 등)"*, kernel 차이는 *"front-size에 맞춘 커널 라우팅"*. STRUMPACK 대비 우위는 두 layer 모두에서, cuDSS 대비 우위는 주로 wall-clock layer에서. |

→ 본 솔버가 cuDSS를 wall-clock으로 10× 이긴다는 주장은 옳지만, 그 이유는 *"GPU 커널이 7× 효율적"* 이 아니라 *"GPU 커널이 1.4× 효율적 + 호출 overhead가 1500× 작음 (CUDA Graph replay)"*. 정직한 분해.

## 7. custom_linear_solver가 STRUMPACK 대비 전력망 데이터에 어떻게 최적화했는가 — 분석 논리 구조

### 7.1 출발 명제 — *"같은 알고리즘 패밀리, 다른 타깃 도메인"*

- STRUMPACK과 `custom_linear_solver` 둘 다 **multifrontal LU** 패밀리 (Duff–Reid 1983)
- 둘 다 **METIS nested dissection** + **elimination tree** + **supernode amalgamation** + **front + extend-add** 사용 (textbook 표준)
- `custom_linear_solver`는 STRUMPACK 코드 베이스가 아니다 (`docs/lineage-strumpack-not-the-baseline.md` §1)
- **차이는 알고리즘 발명이 아니라 front-size 분포 가정과 그 위 커널 엔지니어링**

### 7.2 분석 논리 구조 — 5단 (가설-진단-개입-측정-경계)

#### Step A. 가설: front-size 분포가 dominant cost driver

전력망 야코비안의 front 분포 ≠ Janna FEM의 front 분포. 같은 알고리즘이라도 front 분포에 따라 hot path가 다른 곳에 있다.

- 본 저장소 측정 (`docs/related-work-and-novelty.md` §2, `docs/fp32-batched-kernel-optimization.md`):
  - ACTIVSg25k: 95% of fronts have **fsz ≤ 16**, 5% of FLOPs in those.
  - 70k case factor의 모든 레벨에서 compute < 60% AND DRAM < 60% → **latency/occupancy-bound** (compute-bound도 bandwidth-bound도 아님).
  - max fsz도 수백 수준.
- 외부 출처 (Spatula MICRO'23): STRUMPACK이 FullChip (회로) 위에서 V100 peak의 0.004% = 0.3 GFLOP/s. *"dense-LU throughput drops linearly below 10,000 and flattens around 20,000"*

#### Step B. 진단: STRUMPACK의 9가지 비효율을 power-grid 위에 매핑

| L# | STRUMPACK 한계 | power-grid 위에서의 결과 | 출처 |
|---|---|---|---|
| L1 | Solve 호스트 메모리 + CPU fallback | 매 NR iter마다 device→host→device 라운드트립 | Claus 2025 본문 |
| L2 | tiny front (fsz<32) MAGMA vbatched + `<32×32` naive 커널 | fronts 95%가 이 영역 → 거의 모든 work가 inefficient path | Ghysels & Synk 2022, MICRO'23 |
| L3 | mid front 다중 패스로 매번 global front 재읽기/재쓰기 | DRAM-bound levels에서 큰 트래픽 | Ghysels & Synk 2022 |
| L4 | level당 factor → extend-add 별도 kernel launch | 깊은 etree (SyntheticUSA ~72 levels) 에서 launch overhead 누적 | Ghysels & Synk 2022 |
| L5 | 단일 시스템 API (batched-systems 없음) | NR iter 시나리오에서 매 iter 새로 시작 | Claus 2025 |
| L6 | FP64 중심 (GPU 경로 튜닝이 FP64에 맞춰져 있음) | RTX 3090 FP64 = FP32의 1/64. 자명한 손실 | Claus 2025 |
| L7 | 큰 대칭 행렬 METIS NodeND 스택 한계 | 전력망은 N≤200K로 별 영향 없음 (paper 행렬에서 작용함) | 본 측정 Hook_1498 SIGKILL |
| L8 | 일률 블록 크기 (큰 separator에 맞는 별도 커널 없음) | 9~25개뿐인 큰 separator levels의 occupancy 낮음 | 함수 시그니처 |
| L9 | CUDA Graph capture 미사용 | NR iter 시나리오에서 매번 host launch 누적 | Claus 2025 |

→ L1, L2, L4, L5, L6, L8, L9 이 power-grid 야코비안 위에서 **직접 hot path**.

#### Step C. 개입: 도메인 가정 위에서 각 한계에 대응하는 커널/구조

`custom_linear_solver`가 한 일 (전부 power-grid front 분포 가정 위):

| L# | 개입 | 코드 위치 | 측정 효과 |
|---|---|---|---|
| **L1** | End-to-end device-resident API: set_data/set_rhs/set_solution 디바이스 포인터, solve도 device 솔브 그래프 replay | `src/solve/multifrontal.cu`, `docs/api-and-build-design.md` | 본 측정 case_ACTIVSg25k solve: 0.66 ms (custom) vs 9.27 ms STRUMPACK (CPU 경고) |
| **L2** | **Warp-per-front 전용 커널** (`mf_factor_small_warp_b`): fsz≤32 레벨은 1 warp/front, per-warp shared, `__syncwarp()` 만 | `src/batched/factor_small.cuh` | 본 저장소 `docs/fp32-batched-kernel-optimization.md`: dominant bottom level 2.47→1.12 ms, compute-bound 76% |
| **L3** | **Shared-resident mid-front 커널** (`mf_factor_mid_tc32_b`): 32<fsz≤159 레벨은 프론트 전체를 dynamic shared로 staging, L/U만 write-back | `src/tc/factor_tc.cuh` | DRAM 트래픽 감소, mid-level 단축 |
| **L4** | **Fused factor + extend-add 커널** (`mf_factor_extend_level`): 한 블록 안에서 factor 후 곧바로 parent로 atomic extend-add | `src/factorize/multifrontal.cu` | level당 1개 kernel launch (was 2). 깊은 etree에서 누적 효과 |
| **L5** | **Batched 경로** (`src/batched/*.cuh`): 한 analyze에 B개의 시스템 factor/solve. 1 symbolic + B numeric | `src/batched/multifrontal_batched.cu` | NR 시나리오와 직격 (B개의 NR Jacobian이 같은 pattern) |
| **L6** | **FP32-native batched 경로** (`BatchPrecision::FP32`): 프론트 자체가 FP32, no FP64 master | `src/tc/factor_tc.cuh` | 자체 FP64 baseline 대비 factor+solve −42…−46% |
| **L8** | **1024-thread 큰-separator 커널**: max fsz>159 레벨 — 9~25 fronts에 다수 warp packing해 sequential 의존성 은닉 | `src/tc/factor_tc.cuh` | 70k factor 0.87 → 0.77 ms |
| **L9** | **factor/solve CUDA Graph capture + replay**: analyze 시점 한 번 캡처, 매 NR iter에서 replay → launch overhead 제거 | `docs/api-and-build-design.md` | "kernel time = factor/solve wall-clock의 ~97%" |

#### Step D. 측정: power-grid 위 3-way 비교 결과 (§6) 가 곧 개입의 합산 효과

case_ACTIVSg25k 의 wall-clock factor 차이 (170 ms → 1.3 ms = **130×**) 는 두 layer가 합쳐서 만들어진다 (§6.4 의 W/K 분해):

| Layer | STRUMPACK factor | cuDSS factor | custom factor | 어느 개입이 만든 차이인가 |
|---|---:|---:|---:|---|
| **GPU kernel-only (K)** | 21.5 ms | 1.86 ms | **1.29 ms** | front-size 적합 커널 라우팅 (L2 warp-per-front, L3 shared-resident mid, L8 1024-thread big) |
| **Host overhead (W − K)** | 147 ms (87%) | 12.2 ms (87%) | 0.01 ms (1%) | CUDA Graph capture + replay (L9), device-resident pointers (L1) |

해석:
- STRUMPACK 대비 **factor 130× 차이의 약 88% (147 / 168 ms) 는 host overhead 차이**, 12% 만 GPU 커널 효율 차이. 즉 L9 (CUDA Graph) + L1 (device API) 가 결정적, L2/L3/L8 (커널 라우팅) 은 부수적.
- cuDSS 대비 **factor 11× 차이의 약 95% (12.2 / 13.4 ms) 는 host overhead 차이**, GPU 커널 자체는 1.44× 만 차이. cuDSS는 GPU 위에서 이미 잘 짜여 있고, 차이는 거의 모두 *"매 호출 overhead"* 에서 발생.
- solve 차이도 유사: STRUMPACK은 *L1의 CPU fallback* 자체가 큰 cost, cuDSS는 device-resident지만 매 호출 API overhead.

NR loop의 사용자 시점에서는 **wall-clock latency가 의미 있는 metric** (매 iter 마다 호출하니까). 그러므로:

- L9 (CUDA Graph replay) + L1 (device API) 가 wall-clock 차이를 만든다 = engineering / system layer 우위
- L2 + L3 + L8 (front-size 적합 커널) 이 GPU 커널 효율을 만든다 = algorithmic / kernel layer 우위

둘 다 진짜 발전이지만 *"무엇 덕분에 빠른가"* 의 분해는 다르다 — STRUMPACK 대비는 system + kernel 둘 다, cuDSS 대비는 거의 전부 system.

추가 layer (B=64+ batch 시나리오):
- L5 (batched) + L6 (FP32-native) = batched factor+solve 추가 −25~28% (`docs/fp32-batched-kernel-optimization.md`)

#### Step E. 경계: 어디서 이 최적화가 깨지는가

가정 위에 세운 최적화이므로 가정이 깨지면 무력하다.

| 가정 깨짐 | 결과 |
|---|---|
| front_total > 8 GB doubles | analyze가 `AnalysisFailed`로 깨끗하게 거절 (§4. paper 행렬이 정확히 이 경우) |
| 매우 큰 단일 separator (fsz » 1024) | 1024-thread big-front kernel이 점점 효율 떨어짐. 이론적으로 vbatched-GEMM 같은 STRUMPACK 경로가 더 나을 수 있음 |
| 매트릭스가 mismatch pattern (numerically singular at first try) | 무피벗 LU 가정 깨짐. STRUMPACK은 MC64 matching + partial pivoting으로 대응. custom_linear_solver는 NR pre-scaling을 NR 외부에 가정 |
| FP32 정확도 부족한 NR convergence (대형 ill-conditioned grid) | FP32-native 경로 잠재적 발산 (이미 본 저장소가 TC32에서 25k/70k 발산 보고) |

즉 본 솔버는 *"수만 개의 작은 front, 고정된 pattern, FP32로 충분한 정확도"* 라는 **특정 도메인 가정** 위에서만 우세하다.

### 7.3 결과적으로 "발전"의 의미

- 알고리즘 발명이 아니라 **front 분포 가정 위 커널 엔지니어링**의 누적
- 8 개의 도메인-특화 결정 (L1, L2, L3, L4, L5, L6, L8, L9 의 개입) 이 합쳐서 cuDSS 대비 −90%, STRUMPACK 대비 −99% factor 단축
- 동시에 paper 매트릭스에 대해서는 분석 거절 (도메인 분리의 정직한 표명)

## 8. 결론

| 항목 | 결론 |
|---|---|
| 논문 1.87× 평균 주장 재현 | RTX 3090 24 GB에서 7개 중 2개만 측정 가능, 그 2개에서 방향성 (STR < cuDSS in factor) 만 확인. 차이 폭은 A100 대비 ~2.4× 작음 |
| `custom_linear_solver`의 paper 행렬 처리 | 4/4 행렬 모두 analyze 단계 `Status::AnalysisFailed` — 8 GB front-arena 캡 (`multifrontal.cu:583`) 이 도메인 분리 표시. crash 아닌 정상 거절 |
| 도메인 분리 의미 | STRUMPACK = large-front 타깃 (Janna FEM), `custom_linear_solver` = tiny-front 타깃 (power-grid Jacobian). 같은 multifrontal 패밀리이지만 hot path가 다른 곳 |
| 전력망 위 3-way 비교 (wall-clock per-call latency) | case_ACTIVSg25k 기준: factor STR 169 ms → cuDSS 14 ms → **custom 1.3 ms**. solve STR 9.3 ms → cuDSS 3.7 ms → **custom 0.7 ms** |
| 전력망 위 3-way 비교 (GPU kernel-only, 새로 측정) | factor K: STR 21.5 ms → cuDSS 1.86 ms → **custom 1.29 ms**. cuDSS-custom GPU 효율 차이는 1.44×, STRUMPACK은 17× (cuDSS는 GPU 위에서 이미 잘 동작) |
| 최적화의 본질 | wall-clock 차이의 ~88%는 **host overhead** (CUDA Graph replay [L9] + device-resident API [L1]) — system layer. GPU 커널 효율 차이는 ~12% — kernel layer (warp-per-front L2, shared-resident mid L3, 1024-thread big L8). 두 layer 모두 실재하지만 *"무엇 덕분에 빠른가"* 의 분해는 NR loop 시나리오에서 system 우위가 결정적 |
| 한계 | (a) 가정이 깨진 매트릭스(large front, mid-failure ACTIVSg70k 이상) 에는 무력 (b) head-to-head vs 외부 batched 솔버 (Wang/Fraunhofer, Zhou) 미실행 (`docs/related-work-and-novelty.md` §5) |

본 분석은 *"솔버가 더 빠르다"* 가 아니라 *"솔버가 다른 매트릭스 클래스를 타깃으로 다른 가정 위에서 최적화된 결과, 그 도메인 안에서 STRUMPACK/cuDSS 대비 측정상 우위"* 를 보여주는 logic chain이다.

---

## 9. 출처

본 저장소:
- `docs/strumpack-paper-table2-reproduction.md` — paper Table 2 재현 측정 단독
- `docs/lineage-strumpack-not-the-baseline.md` — lineage, STRUMPACK ≠ 코드 베이스 증거
- `docs/related-work-and-novelty.md` — 외부 솔버 landscape + 인용 + novelty 자체 평가
- `docs/fp32-batched-kernel-optimization.md` — FP32 batched 측정
- `docs/analyze-phase-optimization.md`, `docs/factor-solve-analyze-optimization.md` — 단계별 최적화
- `docs/tensor-core-factor-design.md` — TC32 negative result
- `docs/mysolver-warm-cache-port-plan.md` — single-case path vs cuDSS

외부:
- Claus, Ghysels, Boukaram, Li, *"A GPU accelerated sparse direct solver and preconditioner with block low rank compression"*, IJHPCA 2025
- Ghysels & Synk, *"High-performance sparse multifrontal solvers on GPUs"*, Parallel Computing 2022
- Boukaram et al. (SuperLU_DIST batched), IJHPCA 2024
- *Spatula* (외부 STRUMPACK profiling), MICRO 2023
- Liu 1986 (etree), Karypis-Kumar 1998 (METIS), Davis 2006 (CSparse), Duff-Reid 1983 (multifrontal)

원시 측정 파일:
- `/tmp/bench/paper_runs.csv` — paper 행렬 (5회 timed)
- `/tmp/bench/strumpack_magma.csv` — power-grid STRUMPACK MAGMA (10회)
- `/tmp/bench/runs.csv` — power-grid STRUMPACK no-MAGMA + cuDSS (10회)
- power-grid custom_linear_solver 측정 (10회 median): 본 문서 §6
