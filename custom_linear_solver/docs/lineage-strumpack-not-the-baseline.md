# `custom_linear_solver`의 베이스라인 — STRUMPACK이 아닌 이유, 그리고 발전 사항 매핑

질문 두 가지:

1. `custom_linear_solver`의 베이스라인이 STRUMPACK인가? **`external/lin_solver` 자체가 STRUMPACK 베이스인가?**
2. 그렇지 않다면, STRUMPACK과 비교했을 때 무엇이 다르고 무엇이 발전한 것인가?

분석 출처: 본 디렉터리 `src/` + git history (`gpu-powerflow` 저장소 + 같은 워크스페이스의
`sparse_direct_solver` 저장소) + 같은 워크스페이스의 STRUMPACK 소스
(`/opt/third_party/src/strumpack`).

논문 Table 2 재현 측정 결과는 별도 문서(`docs/strumpack-paper-table2-reproduction.md`) 참고.

---

## 1. lineage — `custom_linear_solver` ← `external/lin_solver` ← (자체 코드, KLU 위임 시작 → cuDSS-API 미메틱)

### 1.1 git 추적

`gpu-powerflow` 저장소(`/workspace/sparse_direct_solver/gpu-powerflow`)의 git history:

```
0531888  Initial commit: GPU-accelerated Newton-Raphson Power Flow solver
990ccec  Squashed 'external/lin_solver/' content from commit 7782774
ef5bd4a  Merge commit '990ccec...' as 'external/lin_solver'
734e725  chore: 정리 docker matlab solver setup
            "Add custom_linear_solver as the cuPF-integrated version of external/lin_solver."
4345ea6  analyze: GPU symmetric adjacency graph build (replace CPU adj_build + CSC download)
1569374  factor: adaptive FP64-master/FP32-LU mixed precision (small/medium); ...
9563768  tensor cores: etree-respecting amalgamation + dynamic-shared multi-k WMMA ...
...
```

즉:

```
sparse_direct_solver (현재 워크스페이스, src/mysolver/ 포함)
        |
        | (git subtree squash, 2026-05-28)
        v
gpu-powerflow/external/lin_solver/
        |
        | (cuPF 통합용 추출, 커밋 734e725)
        v
gpu-powerflow/custom_linear_solver/
```

### 1.2 `external/lin_solver` 안의 실제 소스 인벤토리

서브트리 임포트 시점(commit 7782774)의 `src/` 구조:

```
src/
├── mysolver/                          ← 자체 GPU 멀티프론탈 솔버
│   ├── solver.cpp / solver.hpp        ← cuDSS 흉내 API (analyze/factorize/solve)
│   ├── symbolic/
│   │   ├── elimination_tree.cpp       ← 헤더에 "Liu 1986" 인용
│   │   ├── multifrontal.cpp
│   │   ├── supernode.cpp
│   │   └── schedule.cpp
│   ├── reordering/
│   │   ├── metis_nd.cpp               ← METIS ND
│   │   ├── gpu_nd.cu                  ← GPU symmetric graph + ND 가속
│   │   └── mc64.cpp                   ← (실험)
│   ├── factorize/
│   │   ├── dense_lu.cpp
│   │   ├── sparse_lu.cpp
│   │   ├── supernodal.cpp
│   │   └── own_pipeline.cpp
│   └── gpu/
│       ├── gpu_mf.cu (1345 lines)     ← 멀티프론탈 GPU 커널
│       ├── gpu_factor.cu
│       ├── gpu_solve.cu
│       └── gpu_spmv.cu
└── third_party_solvers/               ← 벤치마크용 외부 솔버 래퍼
    ├── cudss_solver.cpp
    ├── strumpack_solver.cpp           ← STRUMPACK은 여기, 단일 래퍼
    ├── mumps_solver.cpp
    ├── pastix_solver.cpp
    ├── superlu_solver.cpp
    ├── suitesparse_solvers.cpp (klu, umfpack)
    ├── glu_solver.cpp
    ├── pangulu_solver.cpp
    └── pardiso_solver.cpp
```

### 1.3 결정적 증거 — `mysolver/solver.hpp`의 자체 설명

```cpp
// mysolver public API.
//
// The external call pattern is intentionally 1:1 with the cuDSS phase model
// (cf. cuDSS Getting Started "Workflow": analyze -> factorize -> solve).
// In a Newton-Raphson loop the Jacobian sparsity is fixed, so analyze() runs
// once and only factorize()/solve() repeat as values change.
//
// M0 milestone: the internal implementation delegates numeric work to
// SuiteSparse KLU (a faithful wrapper with the cuDSS I/O contract).  Later
// milestones replace each phase with native GPU implementations:
//   AnalyzeResult -> perm/iperm + elimination tree + supernode + schedule
//   FactorState   -> L/U numeric values + block metadata
```

핵심:

- **API 모방 대상: cuDSS** (STRUMPACK 아님)
- **초기 numeric backend: KLU** (STRUMPACK 아님)
- 이후 마일스톤에서 phase 별로 GPU 자체 구현으로 단계적 교체

### 1.4 STRUMPACK 참조 grep 결과

```bash
# mysolver 소스 안의 STRUMPACK / Ghysels / Claus / Boukaram 참조 개수
src/mysolver/gpu/gpu_mf.cu              : 0
src/mysolver/symbolic/elimination_tree.cpp : 0
src/mysolver/symbolic/multifrontal.cpp  : 0
src/mysolver/reordering/metis_nd.cpp    : 0
src/mysolver/solver.cpp                 : 0
src/mysolver/gpu/gpu_factor.cu          : 0
```

`custom_linear_solver/src/`에서도 동일하게 0개.

### 1.5 STRUMPACK의 실제 역할 — third-party 벤치마크 래퍼

`lin_solver`/`sparse_direct_solver` 양쪽에서 STRUMPACK은 **`src/third_party_solvers/strumpack_solver.cpp`** 한 파일(120 라인)에서만 등장. 이 파일은 STRUMPACK C API를 호출하는 얇은 어댑터로, 다른 third-party 솔버(cuDSS, MUMPS, PaStiX, SuperLU 등)와 동일한 위상이다. mysolver와 직접 호출 관계 없음.

---

## 2. 결론 (베이스라인 질문에 대한 답)

| 가설 | 검증 | 결론 |
|---|---|---|
| `custom_linear_solver`는 STRUMPACK 포크인가? | grep 0, 헤더에 cuDSS 모방 명기 | **아님** |
| `external/lin_solver`가 STRUMPACK 베이스인가? | `mysolver/`는 KLU 위임으로 시작 → 자체 GPU 멀티프론탈로 단계적 교체. STRUMPACK은 third-party 폴더의 한 래퍼 | **아님** |
| 둘 다 멀티프론탈 LU 패밀리인 것은 사실인가? | 그렇다 | 알고리즘 패밀리 공유 ≠ 코드 베이스 |

같은 알고리즘 패밀리지만 코드 lineage는 독립이다. 멀티프론탈 LU는 Duff–Reid 1983 이래의 표준 기법으로 STRUMPACK 독점이 아니다.

| 공통 알고리즘 요소 | 표준 출처 (양쪽 다 사용) |
|---|---|
| Fill-reducing reorder | Karypis–Kumar 1998 (METIS ND) |
| Elimination tree | Liu 1986 (path-compressed) |
| Symbolic factorization | Davis 2006 (CSparse) |
| Supernode amalgamation | Duff–Reid, Ashcraft 1990s |
| Multifrontal extend-add | Duff–Reid 1983 |

`elimination_tree.cpp:14`에 명시적으로 *"Liu 1986"* 인용 (custom_linear_solver와 mysolver 모두).

---

## 3. 그래도 STRUMPACK 대비 무엇이 다른가 — 발전 사항 매핑

### 3.1 STRUMPACK GPU 경로의 알려진 한계 (출처: STRUMPACK 논문 / 외부 profiling / 본 측정)

| L# | 한계 | 출처 |
|---|---|---|
| **L1** | Solve가 호스트 메모리 + CPU fallback | Claus et al. IJHPCA 2025 본문 ("STRUMPACK takes host memory ... extra transfers. We plan to develop a fully GPU resident solve phase in the near future.") |
| **L2** | Tiny-front (fsz<32) 처리 비효율 | Ghysels & Synk 2022, Claus et al. 2025: MAGMA vbatched + 자체 `<32×32` 커널, 벤더 batched는 *"not sufficiently optimized"* for tiny fronts |
| **L3** | 회로/전력망 류 그래프에서 V100 peak의 **0.004%** 사용 | Spatula, MICRO 2023 (외부 STRUMPACK profiling) |
| **L4** | 레벨당 다중 커널 launch (factor → extend 분리) | Ghysels & Synk 2022 |
| **L5** | 단일 시스템 API — B개의 시스템(같은 symbolic, 다른 값) batched 지원 없음 | Claus et al. 2025 (SuperLU_DIST가 batched 있으나 FP64-only — Boukaram et al. 2024) |
| **L6** | FP64 중심 (template로 FP32 가능하나 GPU 경로 튜닝은 FP64에 맞음) | Claus et al. 2025 |
| **L7** | 큰 대칭 행렬에서 METIS ND 스택 오버플로우. NodeNDP 권장하나 MPI 의존 | STRUMPACK 자체 stderr 경고 — 본 측정 Hook_1498 SIGKILL에서 직접 관찰 (`docs/strumpack-paper-table2-reproduction.md` §3.2) |
| **L8** | 일률적 블록 크기 — 큰 separator front (fsz>159, occupancy 낮음)에 맞는 별도 커널 없음 | 함수 시그니처 검사 |
| **L9** | CUDA Graph capture 미사용 (또는 제한적) | Claus et al. 2025 |

### 3.2 `custom_linear_solver`의 대응 매핑

| L# | STRUMPACK 한계 | custom_linear_solver의 대응 | 위치 / 측정 효과 |
|---|---|---|---|
| **L1** | CPU solve fallback | **End-to-end device-resident API** (`set_data`/`set_rhs`/`set_solution` device 포인터, `solve()`는 device 솔브 그래프 replay) | `docs/api-and-build-design.md` "Public API Shape" + `src/solve/multifrontal.cu`. 본 측정 Transport에서 STRUMPACK solve = 0.77 s (CPU 경고) vs cuDSS = 0.09 s — 우리는 cuDSS 식 device-resident |
| **L2** | Tiny-front 비효율 | **Warp-per-front 전용 커널** (`mf_factor_small_warp_b`): `fsz≤32` 레벨은 1 warp/front, 8 warps/block, 프론트를 per-warp shared로 coalesced 스테이지, `__syncwarp()`만 | `src/batched/factor_small.cuh`. `docs/fp32-batched-kernel-optimization.md`: dominant bottom level 2.47→1.12 ms, **compute-bound 76%** 달성 |
| **L3** | 일반 그래프 peak 0.004% | (L2/L8과 함께) front-size 분포 자체에 맞춘 커널 라우팅 — 우리도 latency-bound이지만 그 안에서 가장 빠른 lever 사용 | `docs/related-work-and-novelty.md` §2: 95% of fronts are fsz≤16 (only 5% of flops), 75% compute / 33% DRAM (3 bands) |
| **L4** | 레벨 직렬화 | **Fused factor+extend-add 커널** (`mf_factor_extend_level`): 한 블록이 프론트 factor 후 곧바로 부모 프론트로 extend-add (`atomicAdd`). 부모는 strictly 높은 레벨이라 race-free | `src/factorize/multifrontal.cu`. *"Halves graph nodes, removes one inter-kernel sync per level"* — SyntheticUSA의 ~72 레벨 깊은 etree에서 누적 효과 |
| **L5** | 단일 시스템 API | **Batched 경로** (`src/batched/*.cuh`): 한 번의 analyze로 B개의 시스템 factor/solve. 1 symbolic + B numeric | `src/batched/multifrontal_batched.cu`. NR 전력조류(모든 NR iter가 같은 pattern, 값만 바뀜)에 직격 |
| **L6** | FP64 중심 | **FP32-native batched 경로** (`BatchPrecision::FP32`): 프론트 자체가 FP32, no FP64 master | 자체 FP64 baseline 대비 **factor+solve −42…−46%** (`docs/fp32-batched-kernel-optimization.md`) |
| **L7** | METIS NodeND 스택 한계 | **GPU symmetric graph build** (`matrix::build_symmetric_graph_device`) + `metis_nd_from_graph`: CPU `build_symmetric_adjacency` 직렬 단계 제거. 전력망 야코비안은 N≤200K로 stack 한계 자체 발생 안 함 | `src/reordering/metis_nd.cpp`. `docs/factor-solve-analyze-optimization.md`: 9241 case analyze 19→ms→경감, **analyze −22…−34%** (소/중규모) |
| **L8** | 일률 블록 크기 | **3-tier 블록 크기**: tiny(fsz≤32) warp-per-front, mid(32<fsz≤159) shared-resident, big(fsz>159) **1024-thread** | `src/tc/factor_tc.cuh` + 본 doc `docs/fp32-batched-kernel-optimization.md` §3. 70k factor 0.87→0.77 ms |
| **L9** | CUDA Graph 미사용 | **Factor/solve CUDA Graph capture & replay**: analyze 시점에 한 번 캡처, 매 NR iter에서 replay | `docs/api-and-build-design.md` analyze 단계. *"kernel time = factor/solve wall-clock의 ~97%"* — launch overhead < 3% |

### 3.3 측정으로 확인된 차이 (이번 측정 + 기존 보고서)

본 STRUMPACK 재현 측정 (`docs/strumpack-paper-table2-reproduction.md` §3.1):

| 매트릭스 | STRUMPACK MAGMA (RTX 3090) | cuDSS (RTX 3090) | 관찰 |
|---|---|---|---|
| Transport | factor 20.4s / solve **0.77s (CPU)** | factor 23.0s / solve 0.09s | STRUMPACK solve CPU fallback 명시적 — L1 그대로 |
| ML_Geer | factor 9.9s / solve **0.53s (CPU)** | factor 11.0s / solve 0.06s | 동일 |
| Hook_1498 | **SIGKILL** | **OOM** | STRUMPACK 자체 경고 — L7 그대로 |

기존 보고서 — 전력망 야코비안 (custom_linear_solver의 타깃):

| case | n | orig FP32 batched f+s [ms/sys] | new FP32-opt [ms/sys] | 개선 |
|------|--:|---:|---:|---:|
| case3120sp | 5,991 | 0.040 | 0.034 | −16% |
| case6470rte | 12,485 | 0.083 | 0.067 | −19% |
| case9241pegase | 17,036 | 0.122 | 0.102 | −16% |
| ACTIVSg25k | 47,246 | 0.391 | 0.292 | **−25%** |
| ACTIVSg70k | 134,104 | 1.240 | 0.895 | **−28%** |

(출처: `docs/fp32-batched-kernel-optimization.md`)

cuDSS와의 단일 시스템(B=1) 비교 (cuPF 통합 경로, `docs/mysolver-warm-cache-port-plan.md`):

| | precision | cuDSS | custom |
|---|---|---|---|
| 9k | FP64 | 0.99 ms | **0.90 ms** ✓ |
| 25k | FP64 | 1.45 ms | **1.10 ms** ✓ |

---

## 4. 솔직한 한계 (출처: `docs/related-work-and-novelty.md` §5)

발전 사항이 측정상 보인다고 해서 "더 빠르다"고 단정할 수는 없다. 미실행 항목:

1. **헤드투헤드 미실행**: 본 측정은 STRUMPACK MAGMA vs cuDSS만 다룬다. custom_linear_solver vs cuDSS 직접 비교는 단일-케이스 FP64에서만 확인됐고, FP32 batched 모드 head-to-head는 미정리.
2. **선행 연구의 정밀도 미확인**: Wang/Fraunhofer batched power-flow 솔버가 FP32인지 FP64인지 미공개. FP32-native novelty 주장의 기반이 약함.
3. **IR-vs-FP32 accuracy ablation 부재**: FP32 factor가 NR 외부 루프의 수렴 횟수에 영향 주는가?
4. **일반화 미확인**: RTX 3090 / MATPOWER+ACTIVSg 케이스에 한정.

---

## 5. 한 줄 결론

- **베이스라인 관계**: STRUMPACK은 알고리즘 패밀리 동료(멀티프론탈 LU)이자 third-party 벤치마크 비교 대상일 뿐, `external/lin_solver`도 `custom_linear_solver`도 STRUMPACK 코드 베이스가 아니다. API는 cuDSS 모방, 초기 numeric backend는 KLU 위임에서 시작해 자체 GPU 멀티프론탈로 단계 교체된 lineage.
- **발전 매핑**: STRUMPACK GPU의 알려진 9가지 한계(L1~L9) — 호스트 솔브, tiny-front 비효율, 멀티 커널 launch, FP64 중심, 단일 시스템 API, METIS 스택, 일률 블록 크기, CUDA Graph 미사용 — 에 대한 각각의 대응 커널/구조를 power-grid 야코비안 타깃에 맞춰 구현.

---

## 6. 출처

본 디렉터리:
- `docs/api-and-build-design.md` — 공개 API + 빌드 + 복사 인벤토리
- `docs/related-work-and-novelty.md` — 외부 솔버 landscape + 인용 + novelty 자체 평가
- `docs/fp32-batched-kernel-optimization.md` — FP32 batched 발전 + 측정
- `docs/analyze-phase-optimization.md` — analyze 분해
- `docs/factor-solve-analyze-optimization.md` — analyze/factor/solve 통합 최적화
- `docs/tensor-core-factor-design.md` — TC32 + 텐서코어 음의 결과
- `docs/mysolver-warm-cache-port-plan.md` — 단일-케이스 vs cuDSS 측정
- `docs/strumpack-paper-table2-reproduction.md` — 본 문서와 짝, 논문 Table 2 재현 시도

외부:
- Claus, Ghysels, Boukaram, Li, *"A GPU accelerated sparse direct solver and preconditioner with block low rank compression"*, IJHPCA 2025
- Ghysels & Synk, *"High-performance sparse multifrontal solvers on GPUs"*, Parallel Computing 2022
- Boukaram et al. (SuperLU_DIST batched), IJHPCA 2024
- *Spatula* (외부 STRUMPACK profiling), MICRO 2023
- Liu 1986 (etree), Karypis–Kumar 1998 (METIS), Davis 2006 (CSparse), Duff–Reid 1983 (multifrontal) — 알고리즘 표준 출처
