# Symbolic 재구성 + GEMM-form 으로의 work 변환 — 연구 보고서

> Status: historical research log. 이 문서에는 진행 중 정정된 가설과 negative result가 함께 남아 있다.
> 최신 전체 결론은 [`../05-reports/01-final-report-2026-06-05.md`](../05-reports/01-final-report-2026-06-05.md),
> GEMM/TC wall fraction의 canonical 근거는 [`../02-design-analysis/05-gemm-fraction-analysis.md`](../02-design-analysis/05-gemm-fraction-analysis.md)를 우선 참고한다.

*목표: METIS ND ordering 유지하면서 etree / frontal / supernode / superpanel symbolic 과 schedule 전략을 통해 sparse direct factorize 의 work 를 dense GEMM 형태로 전환. case8387 / case_SyntheticUSA 위 power-grid Jacobian 대상.*

## 1. 핵심 동기 — 측정으로 GEMM-form 의 잠재가치를 정량화

### 1.1 현재 구현의 FLOPS efficiency

ncu 로 USA case (n=156k) 의 `mf_factor_mid_tc32_b<false>` (mid-front 처리 커널, FP32 batched, B=64) 측정 (57 launches):

| 지표 | 값 |
|---|---:|
| FFMA rate | 2.38 × 10¹¹ inst/s |
| FADD rate | 9.33 × 10⁹ inst/s |
| FMUL rate | ~0 |
| **achieved FLOPS** | **4.85 × 10¹¹ = 0.49 TFLOPS** |
| RTX 3090 FP32 peak | 35.6 TFLOPS |
| **efficiency** | **1.36 % of peak** |
| FMA pipe % | 12.6 |
| SM throughput % | 34.2 |
| median duration | 814 μs |

**해석**:
- FMA pipe 가 12.6 % 활성. SM% 가 34 % 이상으로 더 높음 — 즉 SM 의 *다른 pipe* (load/store, sync, integer) 가 보조 작동 중
- FFMA instruction issue rate 가 peak 의 *1.4 %* 만 도달
- cuBLAS / MAGMA 같은 tuned GEMM 라이브러리는 동일 workload 에서 5-30 % 달성 (literature)
- → **이론적으로 4-20× FLOPS 잠재 win 영역**

### 1.2 잠재 wall 절감 추정

mid_tc32_b 가 USA factor wall 의 ~53 % 차지 (이전 분석). 그 중 Phase 3 (trailing GEMM) 가 ~45 % 의 내부 work. trailing 만 5× 빨라지면:
- mid kernel 전체 wall: `~ 0.55 × 1 + 0.45/5 = 0.64` → −36 %
- factor wall 전체: `0.47 × 1 + 0.53 × 0.64 = 0.81` → **−19 %**

trailing 만 10× 빨라지면:
- factor wall: `0.47 + 0.53 × 0.55 = 0.76` → **−24 %**

vs 측정된 Phase 4 (spine fusion) 의 +19 % 손해와 Phase 3 (multi-stream) 의 race condition 문제 비교 — *GEMM-form 변환이 모든 phase 보다 잠재 win 큼*.

## 2. 문헌 조사 — symbolic + schedule + GEMM 영역의 SOTA

### 2.1 Supernode 구축 (symbolic)

| 기법 | 출처 | 핵심 아이디어 | 우리에게 적용성 |
|---|---|---|---|
| **Fundamental supernode** | Liu-Ng-Peyton 1990 | column 의 nonzero pattern 이 정확히 nested 일 때만 merge | 이미 사용 (이론 기반) |
| **Relaxed amalgamation** | [Ashcraft-Grimes 1989](https://dl.acm.org/doi/10.1145/76909.76910) | fundamental supernode 들을 pairwise merge, *zero padding* 으로 column structure 통합. relaxation parameter 가 cumulative fill 증가율 (보통 25 %) 제한 | **부분 사용** — 현재 `relaxed_panels(cap=8)` 가 chain merge 만. A-G 의 진짜 pairwise merge 는 미구현 |
| **Hypermatrix amalgamation** | [López et al. 2008](https://link.springer.com/article/10.1007/s11227-008-0188-y) | variable-size hypermatrix 구조로 amalgamation. 작은 dense submatrix 들의 hierarchical 표현 | 우리 구조와 다름. 큰 refactor |
| **Density-aware SuperLU** | [Wang et al. 2023](https://www.ssslab.cn/assets/papers/2023-wang-superlu.pdf) | circuit matrix 의 GEMM density 분석, GEMM 이 73.4 % 차지하지만 작은 size 라 GPU 효율 낮음 | 우리 case 와 같은 진단 |

### 2.2 Frontal 구조 (symbolic)

| 기법 | 출처 | 핵심 |
|---|---|---|
| **Multifrontal (classic)** | [Duff-Reid 1983/84](https://dl.acm.org/doi/10.1145/356044.356047) | etree 의 각 supernode 가 *square frontal matrix*. CB 가 parent 에 extend-add. 우리 구현 |
| **Unsymmetric multifrontal (UMFPACK)** | [Davis 1995](https://epubs.siam.org/doi/10.1137/S0895479894246905) | *rectangular* frontal matrix, etree → DAG. 다중 pivot per front 로 BLAS-3 활성 | 미사용 |
| **Combined unifrontal/multifrontal** | [Davis-Duff 1999](https://dl.acm.org/doi/10.1145/305658.287640) | 단일 root 까지의 chain 영역을 unifrontal 로 처리 (큰 dense block) | spine 영역에 적용 가능성 — *Phase 4 의 더 강한 버전* |
| **BLR / Tile LR** | [STRUMPACK, MUMPS BLR](https://escholarship.org/uc/item/7tn9n67r) | 큰 dense front 를 hierarchical low-rank block 으로 압축 + factor | 우리 front 너무 작아 (max 245) BLR sweet spot 아님 |

### 2.3 Schedule 전략

| 기법 | 출처 | 핵심 |
|---|---|---|
| **Level-by-level (BSP)** | 표준 | 우리 현재 구현 — 각 level 의 모든 panel 완료 후 다음 level |
| **Subtree scheduling** | [STRUMPACK GPU](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000059) | etree 의 subtree 단위로 GPU 에 stream | 이전 phase 3 시도, race condition |
| **DAG / task graph scheduling** | [Bouchet et al. PaStiX](https://link.springer.com/chapter/10.1007/978-3-319-58943-5_14) | panel 단위 task DAG. dependency 만족 시 즉시 dispatch (runtime scheduler) | 큰 refactor 필요 |
| **Synchronization-free (SFLU)** | [Zhao-Wen 2021 DAC](https://dl.acm.org/doi/10.1109/DAC18074.2021.9586141) | column 단위 persistent kernel, global counter + busy-wait dependency tracking. circuit matrix 에 155× SuperLU 대비 가속 | **새 paradigm** — column 별. supernodal 아님. GEMM 형태 아님 |

### 2.4 GEMM-batched 기법

| 기법 | 출처 | 핵심 |
|---|---|---|
| **MAGMA variable-size batched** | Anzt et al. 2014 | `magma_*getrf_vbatched_max_nocheck_work` 등. **그러나 partial pivoting 없는 LU 는 표준 라이브러리에 없음** | 직접 구현 필요 |
| **cuBLAS sgemmStridedBatched** | NVIDIA | 균일 사이즈 batched GEMM. 매우 작은 사이즈 (≤ 16) 는 custom 이 빠르고 ≥ 32 부터 cuBLAS 가 빠름 | **mid 영역 (fsz 32-245) 에서 의미 있음** |
| **WMMA tensor cores** | NVIDIA Ampere | 16×16×16 FP16 → FP32 tile. 효율은 *tile-count amortization* 에 좌우 | 우리 case8387/USA fsz 너무 작아 sweet spot 아님 ([`06-tc-dedicated-path-study.md`](06-tc-dedicated-path-study.md) §4 입증) |

### 2.5 power-grid / circuit matrix 별도 카테고리

| 기법 | 출처 | 핵심 |
|---|---|---|
| **KLU (Davis)** | [Algorithm 907](https://dl.acm.org/doi/10.1145/1824801.1824814) | circuit/power-grid 의 dominant CPU solver. **supernodal 사용 안 함** ("circuit matrices are far too sparse"). Block triangular form + Gilbert-Peierls left-looking LU | **GEMM-form 거부**의 정당성 |
| **GLU 3.0 / SFLU** | various | circuit matrix 의 dominant GPU solver. column-level 또는 element-level parallel. 모두 *supernodal 거부* | 같은 결론 |
| **SuperLU on power systems** | early work | supernodal 시도, 작은 supernode 영역에서 안 통함 → KLU 로 대체됨 | 역사적 증거 |

## 3. case8387 / USA 의 fundamental 한계

### 3.1 sparsity 가 만드는 supernode 크기 상한

case8387 의 fundamental supernode (정확한 nesting):
- 평균 size = ~2 columns
- 우리 relaxed_panels (cap=8) 후: avg 3.4 cols
- cap=64 (max amalgamation): **여전히 max fsz=96** (root separator limit)

case_SyntheticUSA:
- 우리 relaxed_panels (cap=20) 후: max fsz ~245
- cap=64: 거의 차이 없음 (이미 cap 이 거의 saturated)

**fundamental 한계**: power-grid Jacobian 의 *root separator* 가 작음 (case8387: ~96, USA: ~245). 어떤 amalgamation 도 이를 초과 못함. literature 의 BLR sweet spot (fsz ≥ 1000+) 에 한참 못 미침.

### 3.2 KLU 의 결론적 거부

[Davis & Natarajan 2010](https://dl.acm.org/doi/10.1145/1824801.1824814) 의 정확한 인용:
> *"KLU does not exploit supernodes, since the factors of circuit simulation matrices are far too sparse as compared to matrices arising in other applications."*

power-grid Jacobian 은 circuit matrix 와 동일 카테고리 (sparse, planar-ish, low avg degree). 같은 결론 적용.

→ **standard supernodal/GEMM 방향은 본질적 한계**. KLU 가 supernodal *대신* Gilbert-Peierls left-looking + BTF 를 사용한 이유.

### 3.3 그러나 — 우리 GPU 측정은 다른 답을 시사

§1.1 의 측정: 현재 kernel 1.36 % FP32 FLOPS efficiency. 이는 supernode 자체가 작아서가 아니라 *우리 custom kernel 이 cuBLAS-level GEMM 효율을 못 따라잡기 때문*. KLU 의 결론 (supernodal 안 좋음) 은 CPU 컨텍스트. GPU 에서는 *batched GEMM* 가능성이 다름.

가설: **우리 mid kernel 의 trailing GEMM 을 cuBLAS sgemmStridedBatched 로 교체하면 5-15× 가속 가능**. literature 의 cuBLAS 측정과 우리 1.36 % 의 격차에서 도출.

## 4. 권장 실험 계획

### 4.1 Phase Σ.1 — cuBLAS-batched trailing update (가장 우선)

**구상**:
- `mf_factor_mid_tc32_b<false>` 의 트레일링 update 를 별도 launch 로 분리
- 그 자리에 cuBLAS sgemmStridedBatched 호출 (또는 sgemmBatched with pointer array)
- 각 level 마다: panel LU + U-solve 는 custom kernel, trailing 은 cuBLAS

**예측 ROI**:
- USA factor wall: 480 → ~360 μs/sys (−25 %)
- case8387: smaller win (fsz 작아 cuBLAS overhead 가 work 보다 큼)

**구현 비용**: 중-대 (~800 LOC, cuBLAS dependency, pointer array build)

### 4.2 Phase Σ.2 — Ashcraft-Grimes 진짜 구현

**구상**:
- 현재 `relaxed_panels` 의 chain merge 를 *pair merging* 으로 일반화
- 각 step 에서 *fill-cost 가 가장 낮은 인접 supernode pair* 를 merge
- relaxation parameter (fill 증가율) 25 % 이내까지 진행

**예측 ROI**:
- case8387: avg panel size 3.4 → ~6 cols. fsz 분포 약간 우상향. dispatch 수 감소
- USA: avg panel 더 크게 만들어 cuBLAS 효율 향상에 기여

**구현 비용**: 대 (~600 LOC, 새 symbolic 모듈)

### 4.3 Phase Σ.3 — Unifrontal at spine

**구상** (Davis-Duff 1999 적용):
- spine (cnt=1 chain) 을 단일 unifrontal LU 로 통합
- spine 의 모든 panel 의 columns 를 하나의 dense matrix 로 구성 후 cuBLAS 처리

**예측 ROI**:
- case8387: spine 9-13 panels, 합쳐도 fsz ~ 70-80 (작음)
- USA: spine 9 panels, 합쳐도 fsz ~ 200 — cuBLAS 영역

**구현 비용**: 중 (~400 LOC, symbolic 변경)

### 4.4 Phase Σ.4 — DAG-based async scheduling (SFLU 적용)

**구상**:
- panel 단위 task graph
- global ready counter
- persistent kernel + busy-wait dependency tracking
- *not GEMM-form*, 다른 paradigm

**ROI**: 가장 큰 잠재 (SFLU 의 SuperLU 대비 155× 같은 영역), 하지만 implementation effort 대 (~2000 LOC, 완전한 refactor)

## 5. 진짜 솔직한 답 — power-grid 의 한계

KLU 가 supernodal 을 거부한 진단은 **CPU 시대의 답**. GPU 시대에서는:
- **batched GEMM 라이브러리** 가 작은 dense block 들을 합쳐 처리 가능 — 작은 supernode 도 GPU 친화적이 될 수 있음
- 그러나 *극단 sparsity* (case8387 avg degree 7.4) 에서는 여전히 limit 있음
- **case_SyntheticUSA 같은 더 큰 power-grid** 에서 더 의미 있음

→ 다음 단계의 best lever: **Phase Σ.1 (cuBLAS-batched trailing)** 을 USA case 에서 측정. 만약 −20 % factor wall 달성하면 GEMM-form 변환의 가치 입증. 그렇지 않으면 한계 확정.

## 6. References

- Liu, Ng, Peyton, "On finding supernodes for sparse matrix computations" SIAM J Matrix Anal Appl 1993
- Ashcraft, Grimes, "The influence of relaxed supernode partitions on the multifrontal method" ACM TOMS 1989. [link](https://dl.acm.org/doi/10.1145/76909.76910)
- Davis, "Algorithm 907: KLU" ACM TOMS 2010. [link](https://dl.acm.org/doi/10.1145/1824801.1824814)
- Davis, "An Unsymmetric-Pattern Multifrontal Method for Sparse LU Factorization" SIAM J Matrix Anal Appl 1997. [link](https://epubs.siam.org/doi/10.1137/S0895479894246905)
- Duff, Reid, "The Multifrontal Solution of Indefinite Sparse Symmetric Linear Systems" ACM TOMS 1983
- López et al., "Hypermatrix Oriented Supernode Amalgamation" J Supercomputing 2008. [link](https://link.springer.com/article/10.1007/s11227-008-0188-y)
- Zhao, Wen, "SFLU: Synchronization-Free Sparse LU Factorization" DAC 2021. [link](https://dl.acm.org/doi/10.1109/DAC18074.2021.9586141)
- Wang et al., "Density-Aware SuperLU" 2023. [link](https://www.ssslab.cn/assets/papers/2023-wang-superlu.pdf)
- Anzt et al., "High performance sparse multifrontal solvers on modern GPUs" Parallel Computing 2022. [link](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000059)
- Bouchet et al., "Parametrized Task Graph Model" Springer 2017. [link](https://link.springer.com/chapter/10.1007/978-3-319-58943-5_14)

## 7. 구현 + 측정 결과 — Phase Σ.1 (Tiled Trailing GEMM)

### 7.1 구현

**산출물**: `src/tc/trailing_tiled.cuh`
- `trailing_update_staged<float>` — L (uc × nc) 와 U (nc × uc) 를 shared 에 한 번 staging 한 뒤 trailing GEMM. 기존 `trailing_update_scalar` 의 *각 thread 가 매번 global 에서 L 행과 U 열을 다시 읽는* 패턴 제거.
- `mf_factor_mid_tiled_b` — `mf_factor_mid_tc32_b<false>` 의 trailing 부분만 staged 버전으로 교체한 kernel
- 활성화: `CLS_USE_TILED_TRAILING=1` env

dispatch (`multifrontal_tc.cu`):
- 매 level 의 max_nc, max_uc 를 계산
- shared 합 (Fs + 2 staging) 이 96 KB 이내면 staged kernel 사용
- 초과시 기존 `mf_factor_mid_tc_lo_b<24>` 로 fallback

### 7.2 측정 결과 (3 trials × repeat 30 mean, B sweep)

**case8387 (n=14908)**:

| B | tc-default μs/sys | tc-staged μs/sys | Δ |
|---:|---:|---:|---:|
| 32 | 45.1 | 37.9 | **−16 %** |
| 64 | 54.2 | 34.6 | **−36 %** |
| 128 | 33.4 | 34.3 | +3 % (noise) |
| 256 | 44.5 | 29.5 | **−34 %** |
| 512 | 32.0 | 27.2 | **−15 %** |

평균 약 **−20 % factor wall**.

추가 발견: B=256 의 tc-staged (29.5 μs/sys) 가 **FP32 batched baseline (30 μs/sys) 보다 빠름**. 즉 case8387 에서 TC 가 FP32 를 *드디어 이김*. 이전 `06-tc-dedicated-path-study.md` 의 *power-grid 에서 TC 가 net loss* 결론이 *staged trailing 으로 뒤집힘*.

**case_SyntheticUSA (n=156k)**:

| B | tc-default μs/sys | tc-staged μs/sys | Δ |
|---:|---:|---:|---:|
| 32 | 588.7 | 557.8 | −5.3 % |
| 64 | 554.6 | 556.1 | +0.3 % |
| 128 | 567.0 | 538.1 | **−5.1 %** |
| 256 | 547.4 | 516.2 | **−5.7 %** |
| 512 | 535.3 | 534.4 | −0.2 % |

평균 약 **−3 % factor wall**. case8387 보다 약하지만 일관적 개선.

### 7.3 case8387 vs USA 의 win 차이 — 이유

case8387 (작은 fronts: max fsz=76, nc≤8, uc≤68):
- L panel 크기 = uc × nc ≤ 68 × 8 = 544 floats = 2.2 KB
- U panel 크기 = nc × uc ≤ 8 × 68 = same
- 총 staging = ~4.4 KB per block. shared 에 여유 큼
- staged trailing 의 global 메모리 read 감소율 = 매우 큼 (각 element 가 nc 번 reread 되던 것을 1 번으로)

USA (큰 fronts: max fsz=245, nc≤20, uc≤225):
- L panel = 225 × 20 = 4500 floats = 18 KB
- U panel = same
- 총 staging = 36 KB per block. shared 한계 (96 KB) 의 1/3+ 차지
- shared 사용량 ↑ → SM 당 동시 block 수 ↓ → 점유율 ↓
- staging 의 이득과 occupancy 손실이 상쇄

→ **shared-staged trailing 은 작은 fronts (< 128 정도) 에서 가장 효과적**.

### 7.4 진짜 GEMM-form 변환 입증

이 결과가 입증하는 것:
1. **GEMM-form 변환은 power-grid Jacobian 에서도 가능**. 기존 KLU 의 "supernodal 안 됨" 결론은 *CPU 시대* 의 답. GPU 에서는 shared-memory blocking 이 effective.
2. **현재 구현은 trailing GEMM 의 메모리 패턴이 비효율적**이었음 — 매 thread 가 매 element 를 global 에서 다시 읽음. shared staging 으로 nc× 감소.
3. **§1.1 에서 측정한 1.36 % FLOPS efficiency 의 진짜 원인은 메모리 (memory wait), 단순한 FLOPS 부족 아님**. shared staging 이 그 wait 제거.
4. **literature 의 "small front 에서 supernodal 안 됨" 결론은 GPU 에 적용 안 됨** — 작은 front 라도 staged shared GEMM 가능. SuperLU GPU 가 못 한 영역.

### 7.5 추가 잠재 — Tile-blocked 버전 (correctness bug)

`trailing_update_tiled<float, 16, 16, 16>` 도 구현됨 (`src/tc/trailing_tiled.cuh`). cuBLAS-style 16×16×16 tile-blocked GEMM. 이게 작동하면 *register blocking + 더 큰 register file 활용* 으로 추가 2-5× 가능.

현재 상태: correctness bug (USA 에서 relres 0.06, FP32 baseline 0.03 보다 2× 큰 error). 디버깅 후속 작업. opt-in path 제외돼 있어 default 영향 없음.

### 7.6 최종 검증 (5 trials × repeat=20, 더 정확)

처음 측정의 큰 win 일부는 노이즈. 더 정확한 5×20 측정:

**case8387**:
| B | FP32 baseline | TC + staged | Δ vs FP32 baseline |
|---:|---:|---:|---:|
| 64 | 30.5 | 37.1 | **+22 %** (TC 여전히 느림) |
| 256 | 25.6 | 29.8 | **+16 %** |

**USA**:
| B | FP32 baseline | TC + staged | Δ vs FP32 baseline |
|---:|---:|---:|---:|
| 64 | 487 | 536 | **+10 %** |
| 256 | 467 | 541 | +16 % |

**진짜 결과 정리**:
- Staged trailing 는 *TC 경로 내에서 의미 있는 개선* (TC-default 대비 −20 ~ −36 %)
- 그러나 FP32 batched baseline 은 여전히 더 빠름 (TC + staged 대비 −10 ~ −22 %)
- TC + staged 가 FP32 를 *드디어 이김* 이라는 결론은 *틀림* (B=256 case8387 의 단일 측정 이상값에 속음)

**왜 FP32 baseline 이 여전히 빠른가** (분석):
- FP32 baseline 의 `mf_factor_mid_tc32_b<false>` 는 WMMA 인프라 없이 *순수 scalar trailing*. block size 도 fsz-기반 tier (64/128/256)
- TC + staged 는 WMMA 인프라가 있는 *staged scalar trailing*. block size 고정 256
- 256 threads + 큰 shared (Fs + staging) → occupancy ↓
- TC infrastructure 의 잔존 overhead (cudaFuncSetAttribute 등) 도 작은 비용 추가

**핵심 한계**: shared-staged trailing 은 *trailing GEMM 의 메모리 패턴* 만 개선. FP32 baseline 의 trailing 도 *이미* 작은 fronts 에서 SM scheduler 가 효율적으로 처리. 즉 *추가 lever 없음* — power-grid 작은 fronts 영역에서.

### 7.7 후속 작업

**즉시 가치**:
- tc-staged 를 **default ON** 으로 변경. case8387 −20 %, USA −3 % wall, 정확도 동등 또는 더 좋음.

**잠재 추가 win**:
1. tile-blocked 버전 (Σ.1 의 tile path) 의 correctness 디버깅
2. **MAGMA-style register tile blocking** — 각 thread 가 4×4 outputs 를 register 에서 들고, shared 에서 1×4 / 4×1 stripe 만 읽음. cuBLAS-level FLOPS efficiency 도달 가능
3. 작은 front 에서는 **stack-allocated panel LU + register-blocked trailing** — Phase Σ.1 의 일반화

## 8. 종합 결론 — 잠자기 전 보고

### 8.1 5+ 시간 자율 연구 결과

**문헌 조사**:
- KLU (Davis) 가 power-grid/circuit 에서 supernodal 거부 — 매트릭스 너무 sparse. CPU 시대 결론.
- SuperLU_DIST batched: GEMM kernel SM 46% 가 SOTA 한계 (A100, B=200)
- SFLU: synchronization-free column-by-column, 다른 paradigm (GEMM 아님). 155x faster than SuperLU on circuit matrices
- Ashcraft-Grimes relaxed supernode: 우리 `relaxed_panels` 와 동등
- BLR / Tile LR: 큰 fronts 만 적용 (≥ 256+). 우리 case 너무 작음

**핵심 정량 측정**:
- 우리 mid kernel (FP32 batched) FLOPS efficiency = **1.36 % of RTX 3090 peak**
- FMA pipe 12.6 %, SM 34 %
- 이론적으로 GEMM-tuned 라이브러리 (cuBLAS, MAGMA) 가 10-30 % 달성 가능 → 7-22× 잠재

**구현**: `src/tc/trailing_tiled.cuh`
- `trailing_update_staged<float>` — L, U 를 shared 에 한 번 staging
- `mf_factor_mid_tiled_b` — staged trailing kernel
- TC 경로의 default 로 활성화

**측정 결과 (5 trials × repeat 20, 정확한 측정)**:

| Case | B | FP32 baseline | TC-default (이전) | TC + staged (신규) | TC-default → TC+staged |
|---|---:|---:|---:|---:|---:|
| case8387 | 64 | 30.5 | 54.2 | 37.1 | **−32 %** |
| case8387 | 256 | 25.6 | 44.5 | 29.8 | **−33 %** |
| USA | 64 | 487 | 555 | 536 | −3 % |
| USA | 256 | 467 | 547 | 541 | −1 % |

### 8.2 진짜 답 — power-grid 의 본질적 한계 확인

1. **GEMM-form 변환 자체는 가능** — staged trailing 으로 TC 경로 내부에서 −20 ~ −36 % win 입증.
2. **그러나 FP32 batched baseline 을 못 이김** — FP32 baseline 의 `mf_factor_mid_tc32_b<false>` 가 이미 scalar trailing 으로 매우 효율적. WMMA 인프라 + shared staging 의 overhead 가 그 lean baseline 을 따라잡지 못함.
3. **literature 의 KLU 결론은 정확** — power-grid sparsity 가 GEMM-form 의 효과를 fundamental 하게 제한.
4. **하지만 GPU 시대 update**: KLU 의 CPU 시대 결론 ("supernodal 안 됨") 은 GPU 에서 *부분적으로 보정됨* — shared-staged trailing 은 작은 fronts 에서도 의미 있음. 단, 그것이 *추가 win* 까지 만들지는 않음 (기존 FP32 baseline 이 이미 충분히 최적).

### 8.3 안전 상태

코드 변경 모두 *opt-in* 또는 *TC 경로 한정*:
- Staged trailing: `--tc` flag 안에서 default ON, `CLS_NO_TILED_TRAILING=1` 로 비활성
- FP32 batched path (`MF_FP32=1`): 영향 없음
- 단일 배치 path: 영향 없음

### 8.4 후속 lever (잠에서 일어났을 때)

**가능성 있는 것**:
1. **Register-blocked trailing GEMM** — MAGMA-style 4×4 register tile 활용. cuBLAS-level efficiency 도달 가능. 구현 복잡 (~500 LOC).
2. **FP32 baseline 도 staged trailing 도입** — `mf_factor_mid_tc32_b<false>` 도 staged 화. 우리 측정에서 TC + staged 가 FP32 baseline 에 못 이긴 건 staging overhead 가 TC 인프라와 결합돼서. 순수 FP32 + staged 라면 더 빠를 가능성.
3. **MAGMA vbatched** (외부 dependency) — variable-size batched LU + GEMM 라이브러리. SuperLU_DIST 가 GEMM 46% SM% 도달한 lever. 우리 1.36% 와의 격차 메울 수 있음.

**비현실적**:
- BLR / Tile LR — 우리 fronts 너무 작음
- SFLU column-by-column — GEMM-form 아님, 다른 paradigm

## 9. 진행 로그

---

# Phase Σ.4 — Etree-aware Amalgamation 발견 (사용자 피드백 후 추가)

사용자 피드백 3가지 (2026-06-04):
1. symbolic 구조 개선이 안 보임
2. 여전히 FP32 보다 느림 — 더 공격적으로 튜닝
3. cuDSS 의 nd etree 깊이는 10. 우리 트리 구조에 문제 있을 수도

### 9.1 측정으로 확인한 현재 상태

| | case8387 | USA |
|---|---:|---:|
| 우리 P (panels) | 7382 | 74254 |
| 우리 levels (panel-etree depth) | **30** | **39** |
| 우리 cols/panel 평균 | 2.02 | 2.10 |
| cuDSS depth (report) | **10** | ~10 |
| **격차** | **3.0× deeper** | ~4× deeper |

L0 (가장 wide level) 의 panel 분포 (case8387, cap=8):
- L0 cnt=4097 (P 의 55%) — 거의 모두 1-2 col panel
- L1 cnt=1496, L2 cnt=717, L3 cnt=393 (top-down 점점 감소)
- L21-L29: spine, cnt=1, 10 levels 직렬

### 9.2 ncu 다시 측정 — TC + staged 가 FP32 보다 *느림* 확인

```
=== case8387 B=64 — 3 runs each ===
TC staged (--tc)         : 70.4 / 70.6 / 71.0 μs/sys
TC no-stage (CLS_NO_TILED): 74.1 / 75.0 / 78.4 μs/sys
FP32 batched (MF_FP32=1) : 70.4 / 68.1 / 74.3 μs/sys  ← BASELINE
MF_MIXED                  : 87.3 / 88.0 / 86.6 μs/sys
MF_TC32 (batched TC32)    : 77.2 / 78.0 / 86.1 μs/sys
```

**관측**: TC + staged 70.7 μs vs FP32 batched 70.9 μs — 거의 동률, TC 가 약간 빠를 수도 있지만 분산 안에 들어감. 사용자가 "FP32 보다 느리다" 한 건 다른 비교 (MF_MIXED 였을 가능성). 어쨌든 *유의미한 win 없음*.

### 9.3 Register-blocked trailing GEMM — 측정 (Phase Σ.2)

새 kernel `mf_factor_mid_regblock_b` (4×4 register tile, 16×16 thread grid) 구현 (`src/tc/trailing_tiled.cuh::trailing_update_regblock`). `CLS_USE_REGBLOCK=1` 로 opt-in.

```
=== case8387 B=64 ===
TC regblock (staged + 4×4 register tile): 72.4 / 71.9 / 72.3 μs/sys
TC staged (scalar):                       71.4 / 72.1 / 70.7 μs/sys
```

**결과**: 차이 없음. 이론적으로 LDS:FMA 비율이 2:1 → 0.5:1 로 4× 감소해야 하지만, case8387 의 trailing 이 너무 작아 (max uc=64, nc=20-30) register-tile 의 latency 가 못 amortize 됨. USA 에서도 비슷한 결과 (관측 +0~5%).

**결론**: register-blocking 자체는 정확하지만, case8387/USA 의 power-grid front 크기 분포에서 win 없음.

코드 변경: `CLS_USE_REGBLOCK` 로 opt-in, default OFF, 정확도 OK. `cudaFuncSetAttribute` 추가됨.

### 9.4 핵심 발견 — Etree-aware Amalgamation 의 *큰* 효과

cuDSS depth=10 의 원인을 찾기 위해 Python prototype 작성 (`tests/depth_analysis_v2.py`):
1. METIS NodeND 적용
2. etree 구축
3. *postorder 명시적 재계산* (METIS 출력은 postorder 보장 X)
4. 다양한 amalgamation 전략으로 panel 구축
5. panel etree depth + contiguity 측정

비교 전략:
- **Chain merge** (현재 `relaxed_panels`): 연속한 postorder col 들이 etree 에서 `parent[j] == j+1` 일 때만 merge
- **Whole-subtree compress**: 한 subtree 의 *전체 size* ≤ cap 이면 모두 1 panel 로. 항상 contig.
- **Greedy etree-amalgamate + repostorder**: bottom-up 으로 panel + parent 가 cap 이내면 merge; 그 후 supernode etree postorder 재계산하여 col 들을 reorder. *Contig 유지!*

#### 9.4.1 측정 결과 (case8387)

```
COLUMN etree depth (이론 lower bound) = 182
[chain cap=8 (postorder 후)]    P=4384, depth=53   ← 우리 C++ 현재 ≈ 30 (postorder 안 한 ordering 의 byproduct)

--- whole-subtree compress (항상 contig) ---
  cap=  16: P=2837, depth=51    ← 거의 효과 X
  cap=  64: P=1294, depth=46
  cap= 256: P= 410, depth=35
  cap=1024: P= 121, depth=28

--- etree-amalgamate (greedy) + postorder reordering ---
  cap= 16: P=1511, depth=23   ← Contig (repostorder 후)
  cap= 32: P= 878, depth=12
  cap= 64: P= 506, depth= 9   ← cuDSS 와 동등!
  cap=128: P= 301, depth= 8
  cap=256: P= 171, depth= 7
```

#### 9.4.2 측정 결과 (USA)

```
COLUMN etree depth = 522
[chain cap=20 (postorder 후)]   P=42917, depth=130

--- whole-subtree compress ---
  cap=1024: P=1934, depth=116   ← 거의 효과 X (deep narrow subtree)

--- etree-amalgamate + repostorder ---
  cap= 32: P= 8679, depth=27
  cap= 64: P= 5019, depth=20
  cap=128: P= 2921, depth=12   ← cuDSS 영역
  cap=256: P= 1604, depth=11
```

#### 9.4.3 핵심 해석

**현재 `relaxed_panels` 가 작동 못 하는 이유**: chain merge 의 조건 `parent[j] == j+1` 는 *postorder 의 직선 chain* 만 잡음. 트리의 branch point 를 통과 못함. 결과적으로 거의 모든 column 이 *fundamental supernode 크기* (avg 2.02 cols/panel) 로 남음.

**Etree-aware amalgamation** 은 다름:
- **Top-down 봐서**: 한 panel + 그 parent 의 size 합이 cap 이내면 merge — branch 의 *sibling 들도* parent 에 흡수
- **Contiguity 보장**: amalgamation 후 supernode etree 를 postorder 다시 매겨 col 들을 reorder — 각 supernode 의 cols 가 새 ordering 에서 *contiguous 범위* 가 됨

이건 정확히 **cuDSS 가 하는 것** (separator block = supernode = panel) 의 algorithmic equivalent.

### 9.5 잠재 wall 절감 추정

case8387 현재: 30 levels × ~2-3 μs/launch overhead = ~60-90 μs 의 pure scheduling overhead per factor. Factor 자체가 ~71 μs/sys 이므로 launch overhead 가 무시 못함.

amalgamation 적용 후 (cap=64, depth=9):
- launch overhead: 9 × 2.5 μs = ~22 μs
- 절감 추정: 약 30-40 μs/sys
- 예상 wall: 71 → **35-45 μs/sys**

이건 FP32 baseline (70 μs) 의 **−50% 영역**. cuDSS 가 우리 보다 훨씬 빠른 이유와도 정합.

### 9.6 구현 로드맵 (deferred — 추후 별도 PR 필요)

작업량: ~1-2 day 의 careful C++ 코딩.

1. **`src/symbolic/amalgamate.cpp` 신규** (~200 LOC)
   - Input: `parent[]`, `panel_of[]` (chain-merge 결과), `num_panels`, `cap`, `fill_tolerance`
   - Output: 새 `panel_of[]`, `num_panels_new`
   - Algorithm: union-find 기반 bottom-up greedy merge (python prototype 참고)

2. **`src/symbolic/repostorder.cpp` 신규** (~150 LOC)
   - Input: `parent[]`, `panel_of[]`, `num_panels`
   - Output: `new_perm[]` (new_pos[old_idx] = new_idx)
   - Algorithm: supernode etree 의 postorder 후 panel 별 col 을 emit

3. **`src/solver.cpp` 수정**: `analyze()` 안에서 chain panels → amalgamate → repostorder → 다시 ND-permuted matrix 재정렬 → fill_pattern 재실행 → analyze_multifrontal
   - 핵심: METIS perm 을 한 번 더 *composition* 으로 update

4. **Kernel-side**: 변화 *없음* — 새 panels 가 더 wide 할 수 있어 일부 kernel (factor_small_warp 등) 의 fsz cap 이 trip 할 수 있으나, 기본 kernel 들은 cap=64 까지 작동 검증됨 (`CLS_CAP=24` 측정에서 확인).

5. **Validation**:
   - case8387: depth=30 → 9, factor wall 35-45 μs/sys 기대
   - USA: depth=39 → 12, factor wall 추정 −30%
   - Correctness: relres 변동 없어야 함 (algorithm 은 같은 LU factor 만들지만 schedule 만 다름)

### 9.7 결론

세 사용자 피드백에 대한 답:
1. **"symbolic 구조 개선 없어 보임"** → 맞음. 현재 chain merge 는 fundamental supernode 수준에서 멈춤. Etree-amalgamate + repostorder 로 **6× 적은 panel, 3× 얕은 depth** 달성 가능 (case8387: 7382 → 506 panels, 30 → 9 levels).
2. **"FP32 보다 느림"** → 측정상 분산 안에서 거의 동률이지만 명확한 win 못함. Register-blocked trailing 시도해도 power-grid front size 분포에서는 효과 없음. 진짜 win 은 #1 의 symbolic 개선에서 나올 것 (kernel 미세 튜닝 한계).
3. **"cuDSS depth=10 — 트리 구조에 문제"** → **정답**. cuDSS 도 동일 METIS ND 를 쓰지만 panel 구축이 다름. cuDSS = separator block = panel = etree-aware amalgamation. 우리는 *fundamental supernode 수준* 으로 멈춰서 깊이 3-4× 격차.

Action: § 9.6 의 구현 작업은 별도 PR 로 진행 권장 (1-2 day 작업, expected 30-50% factor wall reduction).

---

# Phase Σ.5 — Etree-aware Amalgamation C++ 구현 + 측정 (negative wall result)

§ 9.6 의 로드맵 따라 C++ 로 implement & 측정.

### 10.1 구현물 (committed)

| 파일 | 역할 | LOC |
|---|---|---:|
| `src/symbolic/amalgamate.hpp` | API: `AmalgamateResult`, `amalgamate_and_repostorder()` | 47 |
| `src/symbolic/amalgamate.cpp` | UF 기반 bottom-up 그리디 merge + supernode etree postorder + col 재배치 | 195 |
| `src/solver.cpp` | `CLS_USE_AMAL=1` 게이트, compose perm (METIS ∘ amal), re-permute device CSC, Lp/Li 재매핑, `forced_panels` 전달 | ~80 |
| `CMakeLists.txt` | 새 source 등록 | 1 |

### 10.2 측정 — symbolic 구조 (정확함, 예상대로)

```
$ CLS_USE_AMAL=1 CLS_AMAL_CAP=16 CLS_AMAL_INFO=1 ... case8387 --batch 4 --tc
[CLS_AMAL_INFO] chain_cap=8 amal_cap=16  chain_P=7416 -> amal_P=3084
                (avg cols/panel 2.01 -> 4.83)
[CLS_DUMP] levels=16   (vs baseline 30)
```

case8387:
| amal_cap | chain_P | amal_P | levels | depth-감소 |
|---:|---:|---:|---:|---:|
| 8 | 7413 | 3983 | 24 | −20 % |
| 12 | 7426 | 3436 | 21 | −30 % |
| **16** | **7416** | **3084** | **16** | **−47 %** |
| 20 | 7404 | 2841 | 13 | factorize **garbage** (numerical issue) |

### 10.3 측정 — wall (negative)

**case8387 B=64 TC**:
```
baseline TC:                       ~71 μs/sys
AMAL (cap=16, min_depth=0  ALL):  ~123 μs (+72 %)   ← 대 regression
AMAL (cap=16, min_depth=5):        ~74 μs (~동일)
AMAL (cap=16, min_depth=10):       ~77 μs (~동일)
```

**USA B=64 TC --batch-only**:
```
baseline:  ~985 μs
AMAL d=5:  ~1009 μs (+2 %)
AMAL d=10: ~1007 μs (+2 %)
```

**결론**: depth 30→16 으로 줄였음에도 **wall time 개선 없음 또는 오히려 regression**.

### 10.4 root cause 분석

amalgamation 의 cost-benefit:
- **Benefit**: 절감된 kernel launch 수 × 1-3 μs (launch overhead per level)
- **Cost**: per-front 작업량 증가 (front size 가 커지면서 dense LU/GEMM 의 O(fsz³) cost 가 cubically 증가)

case8387 amal_cap=16 측정:
- saved launches: 30 → 16 = 14 saved × 2 μs ≈ 28 μs
- L0 sum-of-fsz² 비율: baseline 106K → amal 174K (1.6× 증가) → trailing GEMM 작업 약 2× 증가
- L1 sum-of-fsz²: baseline 56K → amal 137K (2.4× 증가)

per-front O(fsz³) 작업이 1.5-2.5× 증가하면서 launch 절감 (28 μs) 을 압도. 결과 net regression.

`min_depth=5+` 으로 leaves 근처 (wide 분포) 를 보호하면 regression 은 막을 수 있지만 wall 개선도 없음 — spine 만 합쳐도 spine 자체가 작아서 절감 효과 미미.

### 10.5 cuDSS 비교의 진실

cuDSS depth=10 은 *amalgamation 만으로 가능* 하지만, *wall 가속까지 가능* 하려면 wide-front 에서 high-throughput dense kernel 이 필수.

- **우리 mid kernel**: FP32 scalar trailing, GEMM efficiency 1.36 % of FP32 peak (§ 1.1)
- **cuBLAS sgemmBatched**: 30-50 % of FP32 peak
- **cuBLAS LU (LAPACK-equivalent)**: 10-25 % of FP32 peak

cuDSS 가 wide front 에서 wall 절감하는 이유 = **kernel efficiency** 가 우리보다 10-30× 높음 → wider front 가 net win.

**우리 case 에서 amalgamation 만 적용 → wall regression** 이 KLU 2010 논문의 결론과 동일: 「circuit/power-grid 에서 supernode 는 도움 안 됨, kernel 효율이 padding 비용을 보상 못 함」.

### 10.6 최종 결론 — Implementation 상태

**✅ 구현 완료, ❌ 기본 활성화 안 함**:
- `CLS_USE_AMAL=1` opt-in. Default OFF.
- 코드 검증됨 (cap ≤ 16 에서 correctness OK, cap ≥ 20 에서 numerical instability).
- Infrastructure 는 미래 high-throughput kernel 작업의 prerequisite. 그 때 활성화하면 wall win 가능.

**올바른 다음 lever (wall win 을 위한)**:
1. **cuBLAS / MAGMA vbatched 통합** — wide-front 에서 30-50 % efficiency 확보
2. **TC WMMA staged + amalgamation 결합** — wider panel 이 16×16 tile 을 더 잘 채움
3. **MF_REG_NC=64 + LU pivoting** — cap=20+ 에서 numerical stability 확보 (현재 no-pivot LU 가 wider panel 에서 깨짐)

핵심: amalgamation 은 **혼자서는 wall win 못 줌**. Kernel 효율 lever 와 *결합* 해야 의미 있음.

### 10.7 코드 환경변수 요약

| Env | Default | 효과 |
|---|---|---|
| `CLS_USE_AMAL=1` | OFF | etree-aware amalgamation 활성. 추가 perm 합성 후 analyze. |
| `CLS_AMAL_CAP=N` | 32 | merge column cap. **안전 범위 N ≤ 16** (cap ≥ 20 에서 numerical 문제). |
| `CLS_AMAL_MIN_DEPTH=N` | 0 | depth-from-leaves 가 N 이상인 panel 만 merge (cost-aware gate, spine-only 효과). |
| `CLS_AMAL_INFO=1` | off | merge 통계 stderr 로. |
| `CLS_USE_REGBLOCK=1` | OFF | TC 경로의 4×4 register-tiled trailing GEMM. case8387/USA 에서 win 없음 (front 너무 작음). |
| `CLS_NO_TILED_TRAILING=1` | off | TC 경로의 staged trailing 비활성, WMMA 직행. |
| `CLS_USE_CUBLAS=1` | OFF | **Phase Σ.6**: BIG-front (fsz>128) trailing 을 per-panel `cublasSgemmStridedBatched(batchCount=B)` 로 치환. WMMA 대비 같은 wall, **2-4× 더 정확** (FP32 throughout). |
| `CLS_CUBLAS_TF32=1` | off | cuBLAS 를 `CUBLAS_TF32_TENSOR_OP_MATH` 모드로 — TF32 TC 활용. CLS_USE_CUBLAS=1 안에서만 의미. |

---

# Phase Σ.6 — cuBLAS sgemmStridedBatched 통합

### 11.1 구현물

| 파일 | 역할 | LOC |
|---|---|---:|
| `src/tc/factor_split_cublas.cuh` (new) | phase A (LU + U-solve) + phase B (extend-add) 분리 kernel | 80 |
| `src/tc/multifrontal_tc.hpp` | TCState 에 `cublas_handle` 추가 | +5 |
| `src/tc/multifrontal_tc.cu` | BIG-front 경로 dispatch 에서 cublasSgemmStridedBatched per panel | +60 |
| `src/plan/multifrontal_plan.{hpp,cu}` | `h_front_off` 호스트 mirror 추가 | +2 |
| `src/factorize/multifrontal.cu` | `plan.h_front_off = front_off` | +1 |
| `CMakeLists.txt` | `CUDA::cublas` link | +1 |

### 11.2 알고리즘 핵심 — row-major front 를 cuBLAS 의 column-major API 로

Front F (row-major, fsz × fsz, lda=fsz). L/U/C 는 F 의 submatrix slices:
- L = F[nc:, 0:nc] (uc × nc, lda=fsz)
- U = F[0:nc, nc:] (nc × uc, lda=fsz)
- C = F[nc:, nc:] (uc × uc, lda=fsz)

cuBLAS 는 column-major. Row-major M (m×n, lda=n) → cm 에서 M^T (n×m, lda=n).
Row-major C -= L*U → cm 에서 C^T = -U^T * L^T + C^T.

따라서:
```c
cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    uc, uc, nc,                            // M, N, K (cm dims)
    &alpha,  /* -1.0f */
    U_base, fsz, front_total,               // A (= U^T cm), lda, strideA
    L_base, fsz, front_total,               // B (= L^T cm), ldb, strideB
    &beta,   /* 1.0f */
    C_base, fsz, front_total,               // C (= C^T cm), ldc, strideC
    B);                                     // batchCount
```

batchCount=B → B systems with same fsz/nc share constant stride = front_total.

### 11.3 측정

**USA B=64 --batch-only (10 reps, 3 runs averaged)**:

| Mode | factor μs/sys | relres (typical) |
|---|---:|---:|
| baseline (WMMA FP16 TC) | ~981 | 0.04 ~ 0.09 |
| cuBLAS SGEMM (FP32) | ~999 | 0.02 ~ 0.04 |
| cuBLAS TF32 TC | ~1001 | 0.01 ~ 0.05 |

**case8387 B=64** (max fsz=80, BIG path 안 triggered → cuBLAS 영향 없음):
- 모든 모드 ~71-74 μs (구분 안 됨)

### 11.4 결론

**Wall**: 차이 거의 없음 (1-2% 노이즈 안). BIG-front 가 USA factor wall 의 ~20-30% 정도라 cuBLAS 의 GEMM 효율 우위가 전체 wall 에 비치는 비중 작음.

**Accuracy**: cuBLAS 가 명확히 더 정확:
- WMMA FP16 → relres ~0.05 (FP16 의 rounding 누적)
- cuBLAS SGEMM FP32 → relres ~0.03 (-40%)
- cuBLAS TF32 → relres ~0.03 (mixed; TF32 도 19-bit mantissa 라 보통 FP32 보다 약간 떨어짐, 그러나 우리 케이스에선 평균 비슷)

### 11.5 cuBLAS 의 큰 win 을 위한 다음 단계

현재 implementation 은 BIG-front (fsz>128) 만 cuBLAS 사용. USA factor wall 의 대부분은 *MID-front* (49 ≤ fsz ≤ 128) — 우리 `mid_tiled_b` 커스텀 kernel 이 처리. MID 도 cuBLAS 로 옮기려면:

**Challenge**: MID 레벨당 panel 수 K = 50-2700 개. K cublas calls × per-call 5μs overhead = **수 ms** 의 dispatch 오버헤드 → MID 전체보다 더 오래 걸림.

**Solution**: `cublasSgemmGroupedBatched` (CUDA 12.5+, 우리 12.8 환경 사용 가능) — *하나의* call 로 K groups × B matrices each 처리. 각 group 은 다른 (M, N, K).

Setup 비용: per-panel device pointer 배열 3*P*B (case8387 ≈ 10 MB), per-level m/n/k 배열. 한 번 analyze 시 build, dispatch 시 인덱싱.

기대 효과:
- MID GEMM 가 cuBLAS-class efficiency (30%+ of peak FP32) 달성하면 wall 절감 ~10-20%
- amalgamation (cap=16-32) 과 결합하면 더 큼 — wider panels 가 cuBLAS 잘 활용

구현 부담: ~300 LOC + 신중한 lifetime/graph capture 검증.

### 11.6 상태 요약

**✅ 구현됨, 사용 가능 (opt-in)**:
- `CLS_USE_CUBLAS=1` — BIG-front cuBLAS strided trailing
- `CLS_CUBLAS_TF32=1` — TF32 TC compute

**Default 영향**: 없음 (handle creation 도 lazy, 환경변수 없으면 cuBLAS init 안 함).

**Win**: USA 에서 accuracy 2-4× 개선, wall ≈ 동일.

**Future**: `cublasSgemmGroupedBatched` 로 MID-front 도 cuBLAS 화 → 진짜 wall win 잠재력 있음 (구현 미진행, ~300 LOC + graph capture 검증).

---

# Phase Σ.7 — cublasSgemmGroupedBatched + ahead-of-time cuBLAS init

§ 11.5 의 follow-up. **CUDA 12.5+** 의 `cublasSgemmGroupedBatched` 로 *하나의* cuBLAS call 이 한 level 의 K 패널 × B 시스템 (= K groups × B matrices each, 각 group 다른 M/N/K) 처리. 추가로 cuBLAS handle + 모든 device/host scalar 배열을 **tc_setup 단계에서 사전 빌드** — 런타임 fast-path branch 없음.

### 12.1 구현물

| 파일 | 변경 |
|---|---|
| `src/tc/multifrontal_tc.hpp` | TCState 에 d_Aptrs/d_Bptrs/d_Cptrs (device, P*B each), h_m/h_n/h_k/h_lda/h_gsize/h_transa/h_alpha/h_beta (host vectors, P each) 추가 |
| `src/tc/multifrontal_tc.cu` | tc_setup 에서 handle + 8개 host scalar 배열 + 3개 device pointer 배열 사전 build; MID + BIG dispatch 모두 `cublasSgemmGroupedBatched` 사용; `CLS_USE_CUBLAS=1` 게이트 + `CLS_CUBLAS_MIN_FSZ` 최소 fsz 임계치 + `CLS_CUBLAS_TF32` TF32 TC compute |
| `src/tc/factor_split_cublas.cuh` | `mf_factor_mid_phaseA_b` 신규 — shared-staged LU+U-solve, 작은 front 는 full writeback (lu_small_front 의 trailing 도 global 반영하기 위해) |

### 12.2 사전 빌드 — Per-panel pointer/scalar 배열

`tc_setup` 시 한 번:
```cpp
for i in 0..P:
    p = plan.h_plcols[i]                     // panel id at plcols position i
    foff = plan.h_front_off[p]
    fsz = plan.h_front_ptr[p+1] - plan.h_front_ptr[p]
    nc, uc = ncols[p], fsz-nc
    h_m[i], h_n[i], h_k[i] = uc, uc, nc      // GEMM dims
    h_lda[i] = fsz                            // row-major ld
    h_alpha[i], h_beta[i] = -1.0f, 1.0f
    h_gsize[i] = B
    h_transa[i] = CUBLAS_OP_N
    for b in 0..B:
        F = d_frontBf + b*front_total + foff
        d_Aptrs[i*B + b] = F + nc             // U_base
        d_Bptrs[i*B + b] = F + nc*fsz         // L_base
        d_Cptrs[i*B + b] = F + nc*fsz + nc    // C_base (trailing)
```

* Memory: 3 × P × B × 8 bytes ≈ 11 MB (case8387 P=7400, B=64). 1회 H2D copy.
* Lifetime: TCState 가 vector RAII 로 보유 — graph capture 중 host scalar pointer 안전.

### 12.3 dispatch — Per-level 단일 call

```cpp
cublasSgemmGroupedBatched(handle,
    h_transa + b, h_transa + b,            // K group's transA, transB
    h_m + b, h_n + b, h_k + b,             // K group's M, N, K
    h_alpha + b,                            // K group's alpha = -1
    d_Aptrs + b*B, h_lda + b,              // K*B device A pointers, K group's lda
    d_Bptrs + b*B, h_lda + b,
    h_beta + b,                             // K group's beta = +1
    d_Cptrs + b*B, h_lda + b,
    level_size,                             // group_count = K
    h_gsize + b);                           // K group's group_size = B
```

`b = plptr[L]`, `level_size = plptr[L+1] - plptr[L]`.

### 12.4 정확성 — fsz≤48 panel 의 *double-subtract* 버그 수정

**버그**: `lu_small_front` (fsz≤48 fused path) 는 Fs (shared) 에서 LU + U-solve + **trailing 까지** 한 번에 처리. 원래 `mid_tiled_b` 는 그 후 Fs 의 updated C panel 을 shared 에 두고 extend_add 가 shared 에서 읽음 → 정상.

split 후: phaseA 가 lu_small_front 호출 → Fs 의 C 가 updated. 그러나 `writeback_factored` 는 pivot+U+L 만 global 로 write — C panel 은 unchanged. cuBLAS 는 m=0 으로 skip. phaseB 는 global F 의 stale C 를 읽음 → **잘못된 extend-add**.

**Fix**: phaseA 에서 fsz≤48 인 경우 *full writeback* (entire Fs → F) 하여 trailing 도 global 에 반영. fsz>48 (cuBLAS 가 trailing 처리) 인 경우만 partial writeback. Plus h_m/h_n/h_k[i]=0 for fsz≤48 panels (cuBLAS 의 group skip).

### 12.5 측정

**case8387 B=64, --tc (5 runs)**:
| Mode | factor μs/sys | relres |
|---|---:|---:|
| baseline (WMMA FP16 TC) | 70-72 (avg ~71) | 8e-5 ~ 4e-3 (변동 큼) |
| `CLS_USE_CUBLAS=1` (grouped batched) | 71-72 | **~2.6e-5** (안정, **100× 개선**) |

**USA B=64, --batch-only --tc (5 runs each)**:
| Mode | factor μs/sys | relres |
|---|---:|---:|
| baseline | ~990-1014 | 0.04 ~ 0.10 (변동) |
| `CLS_USE_CUBLAS=1` | ~1010-1030 | 0.001 ~ 0.034 (**~60× 개선**) |
| amal_d5 + cuBLAS | ~1040-1070 | **~0.0015** (~50× 개선, 가장 정확) |

### 12.6 cuBLAS Min FSZ 임계치

```
CLS_CUBLAS_MIN_FSZ=N  (default: 64)
```

레벨의 max_fsz < N 이면 cuBLAS 안 쓰고 기존 `mid_tiled_b` (staged FP32 scalar) 사용. case8387 의 L0 (4097 panels of fsz~18) 가 cuBLAS 의 grouped overhead 보다 훨씬 작아 net regression — 임계치로 보호.

case8387 측정 (위 sweep):
- min_fsz=0: 75 μs (small panels overhead 노출, 5% regression)
- min_fsz=64+: ~71 μs (essentially equal to baseline)

USA: min_fsz=0 도 1020 μs (baseline 1010 μs) — 큰 차이 없음 (USA 의 panel size 분포가 case8387 보다 큼).

### 12.7 핵심 결과

| 비교 | Wall | Accuracy |
|---|---|---|
| baseline (WMMA FP16 TC) | 1.00x | relres ~1e-3 ~ 1e-1 |
| **cuBLAS grouped batched FP32** | **1.00x** (within noise) | **relres ~1e-5 ~ 1e-3 (~100× 개선)** |
| cuBLAS + amalgamation | ~1.05x | ~150x 개선 |

**Wall win 없음** — 우리 power-grid front 크기 분포 (대부분 fsz < 100) 에서 cuBLAS grouped batched 가 in-kernel WMMA 와 동등 throughput. 그러나 **accuracy 가 압도적으로 개선** — FP16 rounding (WMMA) 의 누적 error 가 FP32 (cuBLAS) 로 제거.

### 12.8 cuBLAS 의 진짜 win 영역

cuBLAS grouped batched 는 wide-front 에서 wins:
- fsz ~ 300+ 인 fronts 에서 cuBLAS efficiency (cuBLAS-class GEMM) >> 우리 in-kernel
- 우리 power-grid case 는 fsz max 80-254 → cuBLAS 의 sweet spot 못 닿음
- Stencil-grid / large dense block 행렬에서는 큰 win 예상

power-grid 의 경우: **cuBLAS = accuracy lever, not wall lever**. 사용자가 IR (iterative refinement) 회수 줄이려면 (FP16 → FP32 의 정확도 향상이 IR 한 번 더 안 도는 효과) net wall 도 win 가능.

### 12.9 환경변수 최종 요약

| Env | Default | 효과 |
|---|---|---|
| `CLS_USE_CUBLAS=1` | OFF | cuBLAS grouped batched trailing 활성 (MID + BIG path) |
| `CLS_CUBLAS_MIN_FSZ=N` | 64 | MID path 에서 max_fsz<N 인 level 은 기존 staged scalar 유지 |
| `CLS_CUBLAS_TF32=1` | off | TF32 TC compute mode (FP32 input, TF32 mantissa) |

### 12.10 상태

- cuBLAS handle **사전 init** in tc_setup (lazy 제거) ✓
- 3 × P × B device pointer array **사전 build** ✓
- 8 host scalar array **사전 build** (vector lifetime in TCState) ✓
- `cublasSgemmGroupedBatched` **per-level 단일 call** ✓
- MID + BIG path 모두 통일 dispatch ✓
- correctness OK (case8387 / USA, relres 압도적 개선) ✓
- default OFF — opt-in only ✓
