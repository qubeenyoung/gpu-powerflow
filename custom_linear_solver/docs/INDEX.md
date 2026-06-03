# `custom_linear_solver` 문서 색인

15 docs, 그룹별 권장 읽기 순서 포함. 각 항목은 *"무엇에 답하는 문서인가"* + *"누가 읽으면 좋은가"*.

---

## A. 설계 / API (먼저 읽기)

### [api-and-build-design.md](api-and-build-design.md)
*공개 API + 빌드 구성 + 복사한 파일 인벤토리*
cuDSS-like phase API (`set_data` / `analyze` / `factorize` / `solve`) 의 형태, 빌드 옵션 (`CLS_BUILD_CUDA_OPS`, `CLS_INTERNAL_GRAPH`), 어떤 소스가 어떤 디렉토리로 복사됐는지. **솔버를 처음 쓰는 사람**.

### [related-work-and-novelty.md](related-work-and-novelty.md)
*외부 GPU sparse direct 솔버 landscape + 본 작업의 정직한 novelty 평가*
STRUMPACK / SuperLU_DIST / GLU 3.0 / cuDSS / Wang-Fraunhofer / Zhou batched / Spatula 의 외부 출처 기반 비교 (22 sources, 25 verified claims). 본 솔버가 *"기존에 없는 것"* 과 *"기존에 있는 것"* 의 분해. **포지셔닝 / 논문 작성** 단계에서 읽기.

---

## B. 솔버 내부 최적화 엔지니어링 (어떻게 빠른가)

### [analyze-phase-optimization.md](analyze-phase-optimization.md)
*`Solver::analyze()` 단계의 분해와 최적화 기록*
analyze 가 어떤 단계로 구성되는지 (CSR↔CSC, METIS ND, etree, fill pattern, multifrontal plan, CUDA graph capture), 각 단계의 시간 + 줄인 방법. **analyze 가 NR 처음 한 번만 돌지만 시간 길어서 신경 쓰일 때**.

### [factor-solve-analyze-optimization.md](factor-solve-analyze-optimization.md)
*analyze + factor + solve 모두 30% 단축 목표의 종합 최적화 리포트*
case3120sp ~ case_ACTIVSg70k 위 baseline vs optimized 측정, A1 (GPU symmetric graph), A2 (parallel root induce), F1 (adaptive mixed precision) 등의 적용 + 효과. **최적화 cycle 의 기록**.

### [fp32-batched-kernel-optimization.md](fp32-batched-kernel-optimization.md)
*B=64–128 batched 모드에서 FP32-native factor+solve −25~28% 의 측정*
warp-per-front tiny-front 커널 (`mf_factor_small_warp_b`), shared-resident mid-front 커널, 1024-thread big-separator 커널의 설계. **front size 별 routing 의 측정 근거**.

### [tensor-core-factor-design.md](tensor-core-factor-design.md)
*FP16/Tensor core를 batched factor에 적용 시도 + 음의 결과*
TC32 경로 (FP32-native + FP16 trailing WMMA) 설계, K=nc≤20 의 thin contraction 영역에서 TC 가 FP32 FMA 못 이김 + 25k/70k에서 발산하는 negative result. ***"왜 TC 를 끄는가"* 의 reasoning**.

### [mysolver-warm-cache-port-plan.md](mysolver-warm-cache-port-plan.md)
*`perf/warm-cache-stack` (mysolver) 기법을 본 솔버 + cuPF Mixed 경로에 적용한 예상*
cuPF Mixed B=1 에서 cuDSS 못 이기는 원인이 *"batched 경로 라우팅 문제"* 인 측정, 단일-케이스 경로의 warm-cache-stack 도입 계획. **cuPF Mixed 통합 시점에 참고**.

---

## C. STRUMPACK 비교 (lineage / 재현 / 측정 / 분석)

### [lineage-strumpack-not-the-baseline.md](lineage-strumpack-not-the-baseline.md)
*"`custom_linear_solver`가 STRUMPACK의 fork인가?" — 아니라는 증명 + 9가지 한계 매핑*
git history (subtree merge 추적) + `mysolver/solver.hpp` 의 *"M0 milestone delegates to KLU + API mimics cuDSS"* 인용. STRUMPACK 의 알려진 한계 L1~L9 (CPU solve, tiny front 비효율, 일률 블록 등) 와 본 솔버의 대응 매핑. **lineage 질문 / STRUMPACK 과의 관계** 정리.

### [strumpack-paper-table2-reproduction.md](strumpack-paper-table2-reproduction.md)
*STRUMPACK 논문 (IJHPCA 2025) Table 2 행렬 7개의 RTX 3090 재현 시도*
SuiteSparse Janna 그룹 (Serena, Geo_1438, Hook_1498, ML_Geer, Transport, Flan_1565, Cube_Coup_dt0). 2/7 측정 성공, 4/7 OOM/SIGKILL, 3/7 보류. 논문의 1.87× 평균 주장 검증의 한계. **재현 가능성 / paper 환경 와의 차이** 정리.

### [strumpack-vs-cudss-power-grid-wall-vs-kernel.md](strumpack-vs-cudss-power-grid-wall-vs-kernel.md)
*전력망 5 case 위 STRUMPACK MAGMA vs cuDSS — wall-clock 과 GPU kernel-only 분리 측정*
CUDA event 로 kernel-only timing 추가 (host overhead 분리). case_ACTIVSg25k 에서 wall (170 ms / 14 ms) vs kernel (22 ms / 1.9 ms) — host overhead 가 87% 라는 분해. **"왜 wall-clock 10× 차이"* 의 근거**.

### [nsys-strumpack-nr-loop-profile.md](nsys-strumpack-nr-loop-profile.md)
*STRUMPACK GPU 의 NR 2-iter 시뮬레이션을 Nsight Systems 로 profile*
case_ACTIVSg25k 위 STRUMPACK 한 솔버만 — 1500 launches/iter, MAGMA vbatched 의 16/8/32 tile 분포, `cudaMallocHost` 344회 등. STRUMPACK 의 NR iter 비용 내역. **STRUMPACK 의 GPU 활용 패턴**.

### [nsys-three-solvers-nr-loop-profile.md](nsys-three-solvers-nr-loop-profile.md)
*세 솔버 (STRUMPACK + cuDSS + custom) NR 2-iter 모두 profile + 솔버별 *"왜 빠르고 왜 느린가"* 분해*
같은 case_ACTIVSg25k 위 세 솔버 비교. cuDSS REFACTORIZATION (1.79 ms) vs FACTORIZATION (13.9 ms) 의 차이가 *"이전 측정이 cuDSS에게 불공정"* 했음을 정정. **3-way 비교의 시각화**.

### [strumpack-vs-custom-multifrontal-case8387.md](strumpack-vs-custom-multifrontal-case8387.md)
*"같은 multifrontal 인데 왜 custom 이 빠른가" — case8387pegase 위 ncu + 알고리즘 deep dive*
front-size 분포 (96.2% fsz ≤ 16, max fsz = 96), ncu 커널 bound 분류 (둘 다 latency-bound), STRUMPACK 논문 Janna 행렬과의 비교 (3D mesh 의 큰 separator vs 평면망의 작은 separator). **STRUMPACK 이 "general sparse direct" 이지 "power-flow 전용" 이 아닌 이유**.

### [why-custom-fast-on-power-grid.md](why-custom-fast-on-power-grid.md)
*비교 분석 없이 `custom_linear_solver` 자체의 8가지 설계 결정 (D1~D8) 만으로 *"왜 빠른가"* 를 분해*
D1 CUDA Graph capture, D2 3-tier kernel routing, D3 fused factor+extend, D4 no pivot, D5 device-resident solve, D6 analyze-time memory closure, D7 GPU symmetric graph build, D8 단일 kernel value scatter. 각각의 가정 + 측정 증거 + 제거한 cost. **솔버 자체의 설계 분해**.

---

## D. 수치 분석 / 증명

### [no-pivoting-empirical-proof.md](no-pivoting-empirical-proof.md)
*"왜 pivoting 없이 풀리는가" — 가설 H1~H4 + 4 power-grid case 실측 증명*
H1 (structural symmetry), H2 (diag nonzero + weak DD), H3 (Wilkinson growth factor < 1), H4 (backward error ≈ ε_m). 이론적 유도 (Ybus → graph 대칭, R/X ≪ 1, normal V) + SciPy SuperLU 로 partial-pivot vs no-pivot 실측. 깨지는 경계 (voltage collapse, BTF block, R/X ≫ 1 등) 명시. **"무피벗 가정의 정당성"의 증명**.

---

## E. 권장 읽기 순서

처음 읽는 사람:
1. `api-and-build-design.md` (이게 뭐 하는 솔버인지)
2. `why-custom-fast-on-power-grid.md` (어떻게 빠른가)
3. `no-pivoting-empirical-proof.md` (왜 안전한가)
4. `related-work-and-novelty.md` (어디 위치한 작업인가)

성능 측정 / 비교에 관심:
1. `nsys-three-solvers-nr-loop-profile.md`
2. `strumpack-vs-custom-multifrontal-case8387.md`
3. `strumpack-vs-cudss-power-grid-wall-vs-kernel.md`

STRUMPACK 의 관계 정리:
1. `lineage-strumpack-not-the-baseline.md` (코드 lineage)
2. `strumpack-paper-table2-reproduction.md` (논문 재현 시도)
3. `nsys-strumpack-nr-loop-profile.md` (STRUMPACK 단독 profile)

내부 최적화 디테일:
1. `fp32-batched-kernel-optimization.md`
2. `factor-solve-analyze-optimization.md`
3. `analyze-phase-optimization.md`
4. `tensor-core-factor-design.md` (TC 음의 결과)
5. `mysolver-warm-cache-port-plan.md`
