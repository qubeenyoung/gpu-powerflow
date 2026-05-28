# cuDSS 재현 프로젝트 — 자동화 실행 플랜

> **대상**: Discord에 연결된 Claude Code 인스턴스
> **전제**: `sparse-direct-solver:3090` 컨테이너가 이미 실행 중이고 (`--gpus all` 활성),
> 셸이 `/workspace/sparse_direct_solver`에 들어와 있다.
> **목표**: cuDSS의 `Analysis → Factorize → Triangular Solve` 파이프라인을 자체 구현해서
> SuiteSparse + MATPOWER NR 벤치마크에서 정확도(`berr`, `absolute_error`)와
> 단계별 실행시간을 cuDSS / KLU 대비 동등 또는 합리적 범위로 끌어올린다.
> **운영 방식**: Claude Code가 `벤치마크 → 결과 분석 → 코드 수정 → 재빌드 → 재벤치마크`
> 사이클을 자율적으로 반복하고, 주기마다 Discord에 보고한다.

---

## 0. 참고 자료 (이 플랜과 함께 항상 열어둘 것)

| 종류 | URL | 어디에 쓰나 |
|---|---|---|
| cuDSS 공식 문서 | <https://docs.nvidia.com/cuda/cudss/index.html> | API / Workflow / Advanced Features. 재현 대상의 정의 근거 |
| ├ Getting Started | <https://docs.nvidia.com/cuda/cudss/getting_started.html> | **cuDSS Workflow** 그림 — `analyze → factorize → solve` 단계 정의의 표준 |
| ├ Functions | <https://docs.nvidia.com/cuda/cudss/functions.html> | `cudssExecute()`의 phase enum, config 옵션. mysolver 공개 API 설계 시 1:1 매핑 |
| ├ Advanced Features | <https://docs.nvidia.com/cuda/cudss/advanced_features.html> | Hybrid memory, MGMN, refactorization, iterative refinement — 후순위 기능 출처 |
| └ Tips & Tricks | <https://docs.nvidia.com/cuda/cudss/tips_and_tricks.html> | 환경변수, reordering 알고리즘 선택 등 휴리스틱 출처 |
| CUDALibrarySamples 루트 | <https://github.com/NVIDIA/CUDALibrarySamples> | cuDSS 호출 예제 모음 |
| ├ `cuDSS/simple` | <https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuDSS/simple> | **가장 먼저 읽을 것.** `cudssMatrixCreateCsr` → `cudssExecute(ANALYSIS/FACTORIZATION/SOLVE)` 최소 흐름 |
| ├ `cuDSS/simple_get_set` | <https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuDSS/simple_get_set> | config 옵션 (reorder algo, pivot tol, hybrid memory) 설정·조회 방법 |
| ├ `cuDSS/simple_mgmn_mode` | <https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuDSS/simple_mgmn_mode> | 다중 GPU. 본 프로젝트 1차 범위 밖. M5 이후 고려 |
| └ `cuSOLVERSp2cuDSS` | <https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSOLVERSp2cuDSS> | 마트릭스 마켓 입력 처리 / 단계별 타이머. **벤치마크 비교 측정 패턴**으로 활용 |

원칙:
- **mysolver의 외부 API는 cuDSS의 phase 분리(ANALYSIS / FACTORIZATION / SOLVE)와 1:1 일치시킨다.**
  내부 구현은 자유지만, 외부 호출 패턴은 위 `cuDSS/simple`을 그대로 따라간다.
- 모든 사이클의 변경 사항은 위 문서/예제의 어느 부분과 대응되는지 commit 메시지에 짧게 적는다
  (예: `cycle 12: reorder algo enum (cf. cudss/functions.html#cudssConfigParam_t)`).

### 0.1 선택적 참고 — GLU3.0 (회로 LU 가속 사례)

cuDSS는 closed-source라 내부 알고리즘을 직접 읽을 수 없다. 비교 검토용으로 다음을 선택적으로 참고:

| 종류 | URL | 어디에 쓰나 |
|---|---|---|
| 논문 | <https://arxiv.org/pdf/1908.00204> | "GLU3.0: Fast GPU-based Parallel Sparse LU Factorization for Circuit Simulation". 회로 매트릭스(SuiteSparse `memplus`/`rajat*`/`onetone2`)에 특화된 GPU LU 설계. **symbolic 단계 / level scheduling / supernode 처리** 의사결정 시 비교용으로 읽기 |
| 코드 | <https://github.com/sheldonucr/GLU_public> | 컨테이너 내 이미 빌드돼 있음 (`/opt/third_party/install/glu3/bin/glu3-lu_cmd`). 라이브러리가 아닌 **CLI 바이너리**라 mysolver와 같은 in-process 비교는 불가. 비교가 필요하면 subprocess wrapper로 별도 측정 |

활용 가이드:
- **필수 참고가 아님**. cuDSS docs / CUDALibrarySamples를 우선 본 뒤, 막히는 부분에 한해 GLU3.0 논문을 참조
- 본 프로젝트의 1차 회귀 비교군은 `cudss-gpu` / `klu` / `umfpack`. GLU3.0은 회로 매트릭스 한정 보조 비교군
- M3 (Numeric Factorize) 또는 M5 (최적화) 사이클에서 회로 매트릭스 성능이 정체되면 GLU3.0의 설계 선택을 자료로 추가 — 단, **GLU3.0 코드를 mysolver에 직접 포팅하지 말 것** (라이센스/구조가 달라 혼선)
- commit 메시지에 GLU3.0 참고 시 prefix `(cf. GLU3.0 §X.Y)` 형태로 명시

---

## 1. 작업 컨벤션 / 경로 (현재 컨테이너 기준)

| 항목 | 경로 |
|---|---|
| 작업 루트 | `/workspace/sparse_direct_solver` |
| 호스트 마운트 (선택) | `/workspace/host_sparse_direct_solver` |
| 데이터셋 루트 | `/datasets` |
| SuiteSparse 매트릭스 | `/datasets/benchmark_matrices/matrices/<name>/<name>.mtx` |
| MATPOWER NR 시스템 | `/datasets/power_system/nr_linear_systems/<case>/{J,rhs,x_true}.mtx` |
| 벤치마크 결과 CSV | `report/benchmark/*.csv` |
| 자동화 상태 | `report/autorun/STATE.json` |
| 자동화 로그 | `report/autorun/log/` |
| Nsight 결과 | `report/nsys/`, `report/ncu/` |
| cuDSS 라이브러리 | `/opt/nvidia/cudss` (`CUDSS_DIR`) |

### 1.1 새로 만들 소스 트리

기존 `src/benchmark/third_party_solvers.cpp` 옆에 본 프로젝트 솔버를 추가한다.
이름은 `mysolver` (또는 팀 합의 명칭)로 고정하고, 벤치마크 러너의 `--solver` 옵션에 등록한다.

```
src/
├── benchmark/
│   ├── third_party_solvers.cpp        # 기존
│   └── solver_registry.cpp            # 신규: mysolver 등록
├── mysolver/
│   ├── reordering/
│   │   ├── metis_nd.hpp / .cpp        # METIS ND 래퍼
│   │   └── graph_prep.cu              # CSR ⇄ adjacency, D2H/H2D
│   ├── symbolic/
│   │   ├── elimination_tree.cu        # etree 구성
│   │   ├── supernode.cu               # supernode/superpanel 식별
│   │   ├── dependency.cu              # count_dep_fwd_bwd 대응
│   │   └── schedule.cu                # level-set / topo sort
│   ├── factorize/
│   │   ├── factorize_dense_block.cu   # supernode 단위 dense LU (cuBLAS GEMM/TRSM)
│   │   └── factorize_driver.cu        # cuDSS factorize_ker 대응
│   ├── solve/
│   │   ├── perm.cu                    # perm_ker 대응
│   │   ├── forward.cu                 # fwd_ker
│   │   └── backward.cu                # bwd_ker
│   ├── common/
│   │   ├── csr.hpp                    # CSR 컨테이너 (host + device)
│   │   ├── device_buffer.hpp          # 할당/스트림 도우미
│   │   └── error.hpp                  # cudaCheck, cuBLAS check 등
│   ├── solver.hpp                     # public API
│   └── solver.cpp
├── tools/                              # 기존 profiling 인프라 활용
└── autorun/
    ├── run_cycle.py                    # 자동화 루프의 단일 사이클
    ├── analyze_csv.py                  # CSV 파싱 + 회귀 검사
    ├── discord_report.py               # Discord webhook 전송
    └── state.py                        # STATE.json 읽기/쓰기
```

### 1.2 본 솔버 Public API (고정 — cuDSS 페이즈와 1:1)

```cpp
namespace mysolver {

struct AnalyzeResult {  // ANALYSIS 산출물 (재사용 가능): perm, etree, supernode, schedule
};
struct FactorState {    // FACTORIZATION 산출물: L, U 수치 + 블록 메타
};

AnalyzeResult analyze(const CsrMatrixView& A);                   // 1회
void factorize(const CsrMatrixView& A,
               const AnalyzeResult& a,
               FactorState* out);                                // NR 반복마다
void solve(const FactorState& f,
           const AnalyzeResult& a,
           DeviceVectorView rhs,
           DeviceVectorView x_out);                              // NR 반복마다

}  // namespace mysolver
```

뉴턴-랩슨 반복에서 자코비안 희소 구조가 변하지 않으므로 `analyze`는 1회만 부르고,
값이 바뀐 `A`로 `factorize`만 다시 호출하는 워크플로우를 강제한다 (cuDSS Getting Started의
Workflow 그림과 동일).

### 1.3 입출력 / 메트릭 (기존 문서 그대로 사용)

- 입력 단위: `A * x = rhs`, 동봉되는 `x_true.mtx`로 절대 오차 측정
- `compute_error` API로 `berr`, `absolute_error` 동시 측정
- 벤치마크 CSV 컬럼:
  `matrix,group,solver,rows,cols,nnz,density,analysis_ms,factor_ms,solve_ms,berr,absolute_error,success,message`
- 프로파일링: `PROFILE_SCOPE`, `PROFILE_CUDA_SCOPE`, NVTX 범위
- Nsight Systems / Nsight Compute로 단계별 / 커널별 시간 검증

---

## 2. 마일스톤 (M0 ~ M5)

각 마일스톤은 **반드시** 다음 4가지를 만족해야 다음으로 진입한다.

1. **기능 종료조건 (Functional)**: 정확도 기준 (`berr`, `absolute_error`)
2. **회귀 종료조건 (No regression)**: 이전 마일스톤 케이스에서 메트릭 악화 없음
3. **성능 종료조건 (Performance)**: cuDSS 또는 KLU 대비 상대 시간 기준
4. **문서/로그 종료조건**: 해당 마일스톤의 변경 요약이
   `report/autorun/log/MILESTONE-Mx.md`로 commit 되어 있음

종료조건을 모두 만족하면 STATE.json의 `current_milestone`을 다음으로 올린다.

---

### M0. 기준선 (Baseline & Harness)

**목표**: 솔버 본체는 비워두고, 자동화 루프와 벤치마크 인프라가 끝까지 잘 돌아가는지 검증.

**구현 항목**
- `mysolver::analyze/factorize/solve` 더미 구현: 내부적으로 **KLU 호출**로 답을 만들고
  반환만 한다 (cuDSS와 같은 입출력 계약을 갖춘 wrapper)
- `solver_registry.cpp`에서 `--solver mysolver` 옵션 추가
- 동일 매트릭스셋에서 `klu`, `umfpack`, `cudss-gpu`, `mysolver` 비교 측정
- `src/autorun/run_cycle.py` 동작: 빌드 → 벤치마크 → CSV 파싱 → Discord 1회 리포트 → STATE 업데이트
- **참고**: `CUDALibrarySamples/cuSOLVERSp2cuDSS`의 `--timer` 옵션 구현을 보고 단계별 타이머 패턴 차용

**벤치마크 매트릭스셋 (smoke)**
- SuiteSparse: `Hamm/memplus`, `Rajat/rajat27`, `Wang/wang3`
- MATPOWER: `case30`, `case118`, `case1197`

**Exit criteria**
- `mysolver`가 위 매트릭스 전부에서 `success=true`
- 절대오차/`berr`가 KLU와 동일 (당연: 내부적으로 KLU 호출 중)
- 자동화 사이클이 1회 완주 (`STATE.json`의 `cycle_count >= 1`)
- Discord에 baseline 리포트 1건 도착

---

### M1. Reordering: METIS ND

**목표**: cuDSS Analysis의 첫 단계인 METIS ND 기반 fill-in 감소 permutation을 자체 구현.

**참고**: cuDSS docs의 reordering algorithm 선택 옵션
(`cudss/tips_and_tricks.html`, `cudssConfigParam_t`의 `CUDSS_CONFIG_REORDERING_ALG`)을 보고
어떤 ND 변형을 default로 쓰는지 비교 — 본 프로젝트는 표준 METIS ND부터 시작.

**구현 항목**
- `reordering/graph_prep.cu`:
  - CSR → adjacency (symmetric pattern `A + Aᵀ`)
  - 디바이스에서 그래프 데이터 구성 (cuDSS의 `copy_csr_columns_ker`,
    `trans_nnz_per_row_ker`, `xadj_ker`, `adjncy_ker` 흐름 모사)
  - D2H 복사
- `reordering/metis_nd.cpp`: host에서 `METIS_NodeND` 호출, perm/iperm 산출
- H2D 복사로 디바이스에 perm/iperm 적재
- `mysolver::analyze` 안에서 위 흐름이 NVTX 범위 `analyze.reorder`로 묶이도록
- M0의 KLU 폴백은 유지하되, **perm/iperm 만 mysolver가 생성하고 이후 단계는 KLU에 위임**해서
  end-to-end 정확도가 깨지지 않도록 한다 (점진적 교체 원칙)

**검증**
- `berr <= 1e-10`, `absolute_error <= 1e-8` (KLU 폴백 덕분에 자동 충족)
- 같은 매트릭스에 대해 mysolver가 만든 perm vs. SuiteSparse `KLU_analyze` perm의 fill-in 차이
  (`nnz(L+U)`) 기록: 5% 이내 권장. 5% 초과 시 회귀로 간주하지 않되 리포트에 반드시 표기
- Nsight Systems 트레이스에 `analyze.reorder` 범위가 잡히고, host와 device 시간 비율이
  PPT의 추정값(전체 45ms 중 GPU 3.5ms) 패턴과 유사한지 확인

**Exit criteria**
- 정확도: M0 수준 유지 (KLU 폴백)
- 기능: smoke 셋 전부에서 mysolver 자체 perm 사용
- 성능: 본 단계는 host bound이므로 절대시간 목표는 없음. **회귀 없음**만 요구
- 추가 셋: SuiteSparse 전체 5종(`memplus, rajat27, wang3, onetone2, rajat15`) +
  MATPOWER `case30..case8387pegase` 통과

---

### M2. Symbolic Factorize: Elimination Tree + Supernode + Dependency

**목표**: cuDSS의 symbolic 단계가 만들어내는 자료구조를 자체 생성.

**참고**: cuDSS docs의 `cudssExecute` ANALYSIS phase 설명 + `CUDALibrarySamples/cuDSS/simple_get_set`
에서 `cudssDataGet`으로 꺼낼 수 있는 메타데이터 항목 확인 — symbolic 산출물을 어디까지 외부에
노출해야 하는지 결정의 근거로 사용.

**구현 항목**
- `symbolic/elimination_tree.cu`:
  - perm된 A에 대해 column elimination tree 구성
  - cuDSS는 binary tree 형태 (PPT 슬라이드 7: `Grid=1023 = depth 10 binary tree`)이므로,
    필요한 경우 etree를 **balanced binary tree로 변환**하는 후처리 포함
- `symbolic/supernode.cu`:
  - 연속된 동일 sparsity-pattern 컬럼을 supernode로 묶기
  - supernode → superpanel 그룹핑 (PPT의 `supernode_map_*`, `define_superpanel_ker` 모사)
- `symbolic/dependency.cu`:
  - factorize/solve가 공유할 dependency 그래프 구성
  - PPT 슬라이드 7의 `count_dep_fwd_bwd_ker`가 만드는 정보에 해당
  - 출력: forward/backward 두 순서의 step 리스트
- `symbolic/schedule.cu`:
  - elimination tree level별로 supernode를 묶어 GPU launch 단위로 변환
  - 같은 level의 supernode는 독립적으로 실행 가능

**검증**
- mysolver etree와 SuiteSparse `cs_etree` (CXSparse) 결과 비교: 부모-자식 관계 일치율 100%
- supernode 개수 / 평균 크기 / 최대 supernode 크기를 CSV 보조 컬럼으로 기록
- Nsight: `analyze.symbolic` 범위 안에서 `dependency_map_ker`에 해당하는 작업이
  etree depth 횟수만큼 호출되는지 (PPT 슬라이드 7과 동일 패턴)
- 최종 출력 perm/etree/supernode로 **KLU 대신 SuiteSparse UMFPACK** 또는 자체 reference factorize/solve
  를 돌려도 정확도 유지

**Exit criteria**
- 정확도: M0/M1 수준 유지
- 기능: smoke 셋 + `case_ACTIVSg2000`, `case3012wp` 통과
- 성능: numeric 단계가 여전히 폴백이므로 시간 목표 없음
- 산출물: `report/benchmark/symbolic_stats.csv` (supernode 통계) 생성

---

### M3. Numeric Factorize

**목표**: M2의 symbolic 산출물을 사용해 GPU에서 LU 수치 분해.

**참고**: cuDSS docs의 `cudssConfigParam_t` 중 pivoting 관련 옵션
(`CUDSS_CONFIG_PIVOT_TYPE`, `CUDSS_CONFIG_PIVOT_THRESHOLD`) — 본 프로젝트는 PPT 슬라이드 5의
"supernode 내부 local pivoting"을 기본으로 채택하고, 옵션화는 M5에서.

**구현 항목**
- `factorize/factorize_dense_block.cu`:
  - supernode 단위 dense block LU
  - 내부에서 cuBLAS GEMM/TRSM 호출 (Tensor Core 적용은 후속 과제)
- `factorize/factorize_driver.cu`:
  - level-by-level 실행: 같은 level의 supernode는 한 번의 그리드 런치로 묶기
  - cuDSS의 `factorize_ker` (작은 supernode들) + `factorize_v3_ker` (큰 supernode들) 역할 분리에 대응하는
    **두 경로 (small-path / large-path)** 구현
  - small-path: warp/single-block thread coarsening
  - large-path: cuBLAS 호출
- **Pivoting**: cuDSS와 동일하게 **supernode 내부에서만 partial pivoting** (local pivoting).
  Pivot 정보는 `FactorState`에 보관
- `independent_ker` 대응: 의존성이 없는 supernode 묶음을 한 번에 처리하는 빠른 경로

**검증**
- 정확도: `berr <= 1e-10`, `absolute_error <= 1e-8`
- 회귀: M2까지의 모든 케이스에서 메트릭 악화 없음
- **반복 안정성**: 동일 매트릭스를 100회 factorize 했을 때 `berr` 분산 < 5%
- Nsight Compute로 `factorize_*` 커널의 occupancy / memory throughput 측정해서 로그에 시계열 추적

**Exit criteria**
- 정확도: 위 기준 충족
- 기능: 전체 SuiteSparse 5종 + `case30..case8387pegase` 통과
- 성능 (1차):
  - SuiteSparse smoke (≤ 40K rows): `mysolver factor_ms ≤ KLU factor_ms × 2.0`
  - MATPOWER `case6468rte`: `mysolver factor_ms ≤ cudss-gpu factor_ms × 3.0`
  - 두 기준이 모두 깨지면 다음 마일스톤 진입 금지

---

### M4. Triangular Solve

**목표**: 분해 결과를 사용한 GPU 전진/후진 대입.

**참고**: cuDSS docs의 SOLVE phase + refactorization 절. Solve 단계가 ANALYSIS의 schedule을
재사용한다는 점 확인 (본 프로젝트도 동일).

**구현 항목**
- `solve/perm.cu`: 시작/끝의 `perm_ker` (RHS와 해 벡터에 perm/iperm 적용)
- `solve/forward.cu`:
  - L에 대한 sparse triangular solve
  - level-set 스케줄 사용 (M2의 dependency 결과 재사용)
  - cuDSS의 `fwd_ker`가 grid 5480, 458로 두 번 등장 → small/large 경로 분리
- `solve/backward.cu`:
  - U에 대한 후진 대입, 동일하게 small/large 경로
  - `upd_marker_bwd_ker` 대응: backward 진행 중 의존성 마킹 보조

**검증**
- 정확도: `berr <= 1e-10`, `absolute_error <= 1e-8`
- M3까지 회귀 없음
- 반복 RHS 시나리오 (동일 A, 100개의 다른 b): solve 시간이 b 개수에 거의 선형 비례
- Nsight: `solve.<matrix>.mysolver` 범위에서 `perm → fwd → fwd → bwd → bwd → perm` 순서가
  PPT 슬라이드 9 호출 구조와 일치

**Exit criteria**
- 정확도: 충족
- 기능: 전체 벤치마크셋(SuiteSparse 5 + MATPOWER 9)에서 `success=true`
- 성능:
  - MATPOWER `case6468rte`: `solve_ms ≤ cudss-gpu solve_ms × 5.0`
  - SuiteSparse smoke: `solve_ms ≤ KLU solve_ms × 2.0`

---

### M5. 통합 / 최적화 / 안정화

**목표**: cuDSS 대비 비율을 끌어내리고, 대형 케이스까지 안정화.

**참고**: cuDSS Advanced Features 문서의 hybrid memory, deterministic mode, refactorization,
iterative refinement 절 — 이 단계에서 도입 후보로 검토.

**구현 항목 (우선순위)**
1. Reordering의 GPU 전처리 비율 증대 (현재 host bound)
2. Factorize large-path에서 batched cuBLAS 사용
3. Solve에서 RHS multiple 지원 (NRHS > 1)
4. 메모리 풀링 (반복 factorize 시 alloc/free 비용 제거)
5. (옵션) Tensor Core 활용 — 슬라이드 3의 차년도 연구 주제와 연결되는 부분으로,
   FP16/TF32 적용 가능한 GEMM 위치 식별 및 정확도 보존 확인

**Exit criteria**
- 정확도: 유지
- 성능 종합:
  - MATPOWER `case6468rte`: 전체 wall time ≤ cudss-gpu × 2.0
  - SuiteSparse `Rajat/rajat15` (37K): 전체 wall time ≤ cudss-gpu × 3.0
- 대형 케이스: `case_ACTIVSg25k`, `case_SyntheticUSA`에서 OOM/Crash 없이 정상 종료

---

## 3. 자동화 루프 (Claude Code의 동작 명세)

### 3.1 STATE.json 스키마

```json
{
  "current_milestone": "M2",
  "milestone_status": "in_progress",
  "cycle_count": 37,
  "last_cycle_at": "2026-05-26T01:23:45Z",
  "last_report_at": "2026-05-26T01:00:00Z",
  "active_matrix_set": "smoke",
  "best_metrics_per_case": {
    "case6468rte": {"factor_ms": 12.3, "solve_ms": 1.1, "berr": 7.4e-12}
  },
  "regression_alerts": [],
  "todo": [
    "implement supernode_map kernel",
    "verify etree against cs_etree on rajat15"
  ],
  "blocked_reason": null
}
```

### 3.2 단일 사이클 (`src/autorun/run_cycle.py`)

매 사이클은 다음 단계를 **이 순서대로** 수행한다.

1. **읽기 단계**
   - `STATE.json` 로드
   - 현재 마일스톤의 Exit criteria 로드
2. **계획 단계** (Claude Code가 LLM으로 수행)
   - 직전 사이클의 CSV / Nsight 로그를 읽고 가장 효과가 큰 작업 1개 선택
   - 변경 범위는 **단일 파일 또는 단일 알고리즘 모듈**로 제한 (큰 리팩토링 금지)
   - 변경 의도 + 기대 효과를 `report/autorun/log/<cycle>.md`에 먼저 기록
3. **편집 단계**
   - 소스 편집 (위 모듈 한정)
   - 빌드: `cmake --build /tmp/profile-build --target benchmark -j`
   - 빌드 실패 시: 즉시 롤백 (`git restore`) → STATE에 `blocked_reason` → Discord 알림
4. **검증 단계**
   - 현재 마일스톤의 `active_matrix_set`으로 벤치마크 실행
   - 결과 CSV를 `analyze_csv.py`로 파싱:
     - 정확도 위반 (`berr > threshold` 또는 `success=false`)
     - 회귀 (`best_metrics_per_case` 대비 `factor_ms` 또는 `solve_ms` +10%↑)
     - 마일스톤 Exit criteria 충족 여부
5. **상태 갱신 단계**
   - 정확도 위반: **무조건 롤백**, `regression_alerts` 기록, Discord 즉시 알림
   - 회귀만 있고 정확도는 OK:
     - 의도된 회귀(예: 폴백 제거 직후)면 변경 유지하고 `regression_alerts`에만 기록
     - 그 외에는 롤백
   - Exit criteria 충족: `current_milestone`을 다음으로 올리고 `milestone_status: ready_for_review`
   - 그 외: `cycle_count += 1` 후 계속 진행
6. **리포트 단계** — **모든 보고는 Discord로**. 보고를 건너뛰는 사이클은 없다.

   사이클 내부에서 다음 시점에 자동 전송한다:

   | 시점 | 트리거 | 보고 종류 |
   |---|---|---|
   | 편집 + 빌드 완료 직후 | 항상 | **change_report** — 무엇을 어떻게 바꿨는지, diff 요약 |
   | 벤치마크 완료 직후 | 항상 | **benchmark_report** — 어떤 셋을 돌렸고 결과 메트릭이 어땠는지 |
   | 마일스톤 전환 시 | 발생 시 | **milestone_report** — exit metrics와 다음 마일스톤 입장 안내 |
   | 정확도 위반 / 빌드 실패 / 롤백 | 발생 시 | **alert_report** — 즉시 전송, 다른 보고와 합치지 않음 |
   | heartbeat | 마지막 보고 후 **2시간** 경과 시 (위 보고로 갱신되지 않은 경우) | **heartbeat** — 현재 진행 상황 한 줄 |

   같은 사이클 안에서 위 보고들은 **분리된 메시지**로 전송한다 (묶지 않음).
   Webhook 실패 시 `report/autorun/log/discord_outbox/`에 적재 후 다음 사이클에서 재시도.

### 3.3 종료 조건 (Stop conditions)

다음 중 하나라도 만족하면 루프 중단하고 사람에게 알림:

- 동일한 변경 시도가 3사이클 연속 실패 (롤백)
- `blocked_reason`이 24시간 이상 해소되지 않음
- 모든 마일스톤 종료조건 충족 (M5 완료)
- 디스크 사용량 95% 초과 / GPU 메모리 OOM 3회 연속

### 3.4 안전장치

- 빌드는 항상 별도 트리(`/tmp/profile-build`)에서. 호스트 마운트의 build 디렉토리를 망가뜨리지 않음
- 데이터셋 (`/datasets`) 읽기 전용 취급. 절대 수정 금지
- Git 관련 안전장치는 §3.5에 별도 정의

### 3.5 Git 워크플로우 / 브랜치 정책

작업 공간은 이미 `git init`이 끝난 상태라고 가정한다. 자동화 루프는 다음 규칙을 **위반 없이** 따른다.

#### Master 불변 원칙 (Optimal-on-master)

- `master`는 **항상 "현재까지의 최선 구현(optimal)"** 만 보관한다
- master 머지 직전 조건 (모두 만족해야 함):
  - 빌드 성공
  - smoke 셋 accuracy 통과 (`berr ≤ 1e-10`, `absolute_error ≤ 1e-8`)
  - 직전 master 대비 **회귀 없음** (factor_ms / solve_ms 둘 다 +10% 이내)
  - 단위 테스트 모두 통과
- **자동화 루프는 master에 직접 commit하지 않는다.** master는 머지로만 갱신
- 마일스톤 완료 시 master에 `git tag m<N>` (예: `m0`, `m1`, …, `m5`) 부여

#### 브랜치 구조

| 브랜치 | 용도 | Claude Code가 push? |
|---|---|---|
| `master` | optimal 보존 | ❌ 머지로만 갱신 |
| `feat/mysolver-autorun` | 자동화 루프의 상시 작업 브랜치 (사이클 단위 commit) | ✅ |
| `experiment/<short-name>` | 크게 다른 시도 — master/feat을 오염시키지 않고 격리 | ✅ |
| `fix/<issue>` | 짧은 정확도/빌드 수정. 통과하면 즉시 머지 후 삭제 | ✅ |

#### 언제 새 branch를 파나 (분기 기준)

다음 중 **하나라도** 해당하면 사이클을 시작하기 전에 `experiment/<name>` 브랜치를 따로 판다:

- 변경 예상 LOC > 300
- 기존 자료구조(예: supernode 표현, dependency 스케줄 포맷)를 갈아치우는 변경
- pivoting 전략처럼 정확도 위반 위험이 큰 알고리즘 변경
- "이게 통할지 모르겠지만 일단 해보자" 류의 탐색적 시도
- 동일 변경 의도가 3사이클 연속 revert된 직후 — feat 브랜치를 깨끗하게 두고 experiment로 옮긴다

experiment 브랜치는 안정화되면 (smoke + 해당 마일스톤 셋 통과 + 회귀 없음) `feat/mysolver-autorun`으로 머지, 그 다음 표준 흐름으로 master로 올라간다.

#### Commit 메시지 컨벤션

**모든** commit은 다음 포맷을 따른다 (autorun이 자동 생성하는 commit 포함):

```
<prefix>(cycle <N>): <one-line intent under 72 chars>

<2~5줄 본문>
- changed: 어떤 파일/함수/커널이 바뀌었나
- why:     왜 바꿨나 (직전 사이클 결과의 어떤 신호 때문에)
- impact:  어떤 메트릭이 어떻게 움직일 것으로 기대했나
- result:  실제로 어떻게 움직였나 (벤치 결과 한 줄 요약)

refs:
  - cudss/<docs anchor>     # 가능하면
  - (cf. GLU3.0 §X.Y)        # 회로 매트릭스 관련 변경일 때
  - PPT slide N              # 슬라이드 관찰값 근거일 때
```

| prefix | 사용처 |
|---|---|
| `feat` | 새 기능/모듈 추가 |
| `perf` | 성능 개선 (정확도 무변동) |
| `fix` | 정확도/빌드/크래시 수정 |
| `refactor` | 동작 동일, 구조만 변경 |
| `test` | 단위/통합 테스트 추가 |
| `docs` | PLAN.md, log, 주석 |
| `revert` | 자동 롤백 |

예시: `perf(cycle 38): split supernode_map into small/large kernel branches`

#### 머지 규칙

- `feat → master` 머지는 위 "Master 불변 원칙" 조건 충족 시에만 + 항상 `git merge --no-ff`
  (마일스톤/개선 단위가 history에서 식별되도록)
- `experiment → feat` 머지는 experiment 브랜치가 3사이클 연속 회귀 없이 안정 + 해당 변경의
  의도된 성능 효과가 측정으로 확인된 후
- 머지 직후 Discord에 `milestone_report` 또는 `change_report`로 알림

#### 롤백 규칙

- 자동 롤백은 **`git revert HEAD`** (history 보존). `git reset --hard`는 금지
- revert commit 메시지: `revert(cycle <N>): <원래 의도> — <롤백 사유>`
- 단위 테스트 실패 또는 빌드 실패 시 **commit 자체를 만들지 않는다**:
  - 변경 파일은 `git stash`로 보관 후 alert_report 첨부
  - 같은 stash가 24시간 안에 활용되지 않으면 drop

#### 사이클당 commit 단위 (요약)

- **사이클당 1 commit이 기본**. 빌드/테스트 실패로 commit이 안 만들어진 사이클도 있을 수 있음
- 사이클이 너무 커지면 사이클을 분할 (여러 사이클 = 여러 commit) — 한 commit에 여러 의도 섞지 말 것
- revert는 commit이므로 별도 사이클 카운트를 잡지 않고 같은 사이클 내에서 처리

---

## 4. 벤치마크 매트릭스 (마일스톤 × 매트릭스셋)

| 매트릭스셋 | M0 | M1 | M2 | M3 | M4 | M5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| smoke (memplus, rajat27, wang3 + case30, 118, 1197) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| suitesparse-full (5종) |  | ✅ | ✅ | ✅ | ✅ | ✅ |
| matpower-mid (case_ACTIVSg2000, case3012wp) |  |  | ✅ | ✅ | ✅ | ✅ |
| matpower-large (case6468rte, case8387pegase) |  |  |  | ✅ | ✅ | ✅ |
| matpower-xl (case_ACTIVSg25k, case_SyntheticUSA) |  |  |  |  |  | ✅ |

`benchmark --matrix-set <name>`이 위 셋을 받을 수 있도록 `solver_registry.cpp`에 매핑 추가.

비교 솔버는 매 사이클 함께 측정:
- `klu` (CPU reference)
- `umfpack` (CPU 비교군)
- `cudss-gpu` (재현 목표)
- `mysolver` (본 솔버)
- `glu3` (회로 매트릭스 한정 보조 비교, **선택**) — `/opt/third_party/install/glu3/bin/glu3-lu_cmd`를
  subprocess로 호출하는 별도 wrapper. SuiteSparse 회로 셋(memplus/rajat*/onetone2)에서만 켜고
  MATPOWER 셋에서는 끈다. wall-time만 측정 가능 (단계별 분해 불가)

---

## 5. 메트릭 / 종료조건 임계값

| 메트릭 | 임계값 | 비고 |
|---|---|---|
| `berr` 정확도 한계 | `1e-10` | 위반 시 즉시 롤백 |
| `absolute_error` 정확도 한계 | `1e-8` | 위반 시 즉시 롤백 |
| factor_ms 회귀 임계 | +10% | best 대비. 의도된 변경은 STATE에 명시 |
| solve_ms 회귀 임계 | +10% | 동일 |
| 사이클당 변경 라인 수 권장 | < 300 LOC | 큰 변경은 여러 사이클로 분할 |
| 동일 변경 연속 실패 한도 | 3회 | 초과 시 stop |
| 동일 케이스 OOM 한도 | 2회 | 초과 시 해당 케이스 제외 후 리포트 |

---

## 6. Discord 리포팅 명세

### 6.1 기본 규칙

- **모든 자동화 산출물은 Discord 채널을 단일 진실 공급원(single source of truth)으로 한다.** Discord에 안 올라간 진행은 없는 것으로 취급
- Webhook 미설정 (`DISCORD_WEBHOOK_URL` 빈 값) 시 자동화 루프는 **시작하지 않는다**. 사람이 명시적으로 `--no-discord` 플래그를 주면 로컬 로그 모드로 전환
- 메시지는 **종류별로 분리** 전송. 한 사이클에서 보통 `change_report` 1건 + `benchmark_report` 1건이 페어로 나온다
- 모든 메시지에 공통 헤더 포함: `[mysolver] cycle N | <milestone>:<status> | <UTC timestamp>`
- 첨부: 직전 사이클 CSV 일부(best/worst 3개), 변경 diff summary, Nsight 결과 경로

### 6.2 메시지 종류와 포맷

#### change_report (편집+빌드 완료 직후 — 항상)

```
[mysolver] cycle 38 | M2:in_progress | 2026-05-26T01:23:45Z
🛠 change_report
  files:
    - src/mysolver/symbolic/supernode.cu  (+47 / -12)
    - src/mysolver/symbolic/dependency.cu (+5  / -3)
  intent: split supernode_map into two kernels (small/large branch)
  build:  OK   (/tmp/profile-build, 18.4s)
  unit:   12/12 passed (tests/mysolver)
  🔗 ref: cudss/functions.html#cudssExecute (ANALYSIS phase)
  next:   run smoke + suitesparse-full benchmark
```

빌드 실패 시 이 메시지가 alert_report로 승격 (아래 참조).

#### benchmark_report (벤치마크 완료 직후 — 항상)

```
[mysolver] cycle 38 | M2:in_progress | 2026-05-26T01:31:02Z
📊 benchmark_report
  set:   smoke + suitesparse-full   (8 matrices, 4 solvers)
  wall:  6m 12s
  accuracy:  ✅ berr ≤ 9.1e-12 (worst rajat27), absolute ≤ 4.3e-9
  vs best:
    🟢 case30   factor_ms 0.71 → 0.68   (-4%)
    🟡 case3012 factor_ms 19.2 → 21.4   (+11%, regression flag)
    ⚪ memplus  unchanged
  vs cudss-gpu (case6468rte): mysolver 38.4ms / cudss 12.1ms (3.17x)
  vs klu        (smoke):       mysolver 1.6x KLU                ✅
  exit-criteria progress: 3/5 met for M2
  📎 report/benchmark/cycle_0038.csv
```

#### milestone_report (마일스톤 전환 시)

```
[mysolver] 🎉 milestone_report  M1 → M2
  cycle 42 | 2026-05-26T03:11:08Z
  exit metrics passed:
    - smoke berr max         : 7.4e-12   (limit 1e-10)   ✅
    - suitesparse-full        : all success              ✅
    - matpower (case30..8387) : all success              ✅
    - fill-in vs KLU perm     : +3.1% on rajat15         ✅ (limit 5%)
  next milestone goals: elimination tree, supernode partition, dependency scheduling
  active_matrix_set: smoke + suitesparse-full + matpower-mid
```

#### alert_report (정확도 위반 / 빌드 실패 / 롤백 / OOM — 즉시)

```
[mysolver] 🚨 alert_report  cycle 39
  type: ACCURACY_VIOLATION
  what: berr = 3.2e-04 on rajat15 (limit 1e-10)
  cause: factorize_v3_ker pivot threshold change (cycle 39)
  action: git revert HEAD; STATE.regression_alerts += 1
  next plan: revert + add unit test for pivot path before retry
```

alert_report는 다른 보고와 묶지 않고 **즉시 단독 전송**. 같은 종류 alert가 1시간 내 5건 초과면 자동화 루프를 일시정지하고 `STATE.blocked_reason`을 채운다.

#### heartbeat (정적 사이클 — 마지막 보고 후 2시간 경과)

```
[mysolver] 💓 heartbeat  cycle 51 | M3:in_progress | 2026-05-26T08:00:00Z
  last change   : 1h 47m ago (cycle 50, factorize small-path tuning)
  last bench    : 1h 47m ago (no regression)
  current task  : profiling case6468rte with ncu
  blocked       : no
```

heartbeat는 정상 상태에서도 "살아있다"는 신호로 전송. 이게 4시간 이상 비면 사람이 들여다봐야 한다.

### 6.3 보고 내용 회수 / 검색

- 모든 송신 메시지는 `report/autorun/log/discord_sent/<cycle>_<type>.json`에도 동일하게 적재 (재현용)
- Discord 검색은 헤더의 `cycle N` 토큰으로 grep 가능하게 모든 메시지에서 동일 포맷 유지

---

## 7. 검증 / 단위 테스트 (마일스톤 평행)

각 모듈에 대해 **마일스톤 진입 전에** 최소 다음 테스트 작성 — 자동화 루프와 별개로 게이트 역할:

| 모듈 | 테스트 |
|---|---|
| reordering | 5x5 toy graph에 대해 알려진 ND perm과 일치, CSR 대칭화 결과가 `A+Aᵀ` 패턴과 일치 |
| etree | `cs_etree` 출력과 부모-자식 관계 비교 (rajat27 포함) |
| supernode | 분할 합이 n과 일치, supernode 내 sparsity pattern 동일성 검증 |
| dependency | level의 위상정렬 — 같은 level 안에서 의존성 없음 |
| factorize | small dense 8x8 randomized 100건에 대해 reference LU와 일치 (`norm < 1e-12`) |
| solve | `L*y=b`를 dense LU로 푼 결과와 mysolver triangular solve 결과의 max-norm 일치 |
| 엔드투엔드 | `CUDALibrarySamples/cuDSS/simple`의 5x5 토이 예제를 mysolver로 풀어 동일 해 확인 |

테스트는 `tests/mysolver/`에 두고, 자동화 루프의 **검증 단계**에서 벤치마크 전에 무조건 실행.

---

## 8. 위험요소 / 알려진 함정

- **METIS 라이센스 / 빌드**: 이미지에 `/opt/third_party/install/common`로 이미 빌드되어 있으므로,
  CMake에서 link만 하면 됨. 새로 빌드 시도하지 말 것
- **cuDSS와 1:1 이름 매칭 강박 금지**: PPT의 커널명은 Nsight 트레이스 관찰명이지 강제된 API가 아님.
  같은 역할의 묶음 구현으로 충분
- **Binary tree 가정**: cuDSS가 etree를 균형 이진트리로 변환하는 것은 관찰 기반 추정.
  본 프로젝트는 자연 etree로 시작하고, M2 후반에 균형화가 필요한지 데이터로 결정
- **MATPOWER NR 시스템의 numeric 특성**: 일부 케이스의 자코비안이 매우 ill-conditioned일 수 있음.
  `berr`가 클 때 mysolver 버그인지 매트릭스 특성인지 구별하려면 항상 cuDSS / KLU의 `berr`와 함께 보고
- **CUDA 이벤트 타이밍 noise**: `cuda_ms`는 워밍업 후 3회 측정 중앙값을 사용.
  `analyze_csv.py`에 중앙값 계산 포함
- **cuDSS preview 버전**: docs에 "API is subject to change"라고 명시되어 있음.
  현재 이미지의 `nvidia-cudss-cu12==0.7.1.6` 동작을 기준으로 하고, 버전 업그레이드는 별도 사이클로 격리

---

## 9. 시작 명령 (Claude Code의 첫 사이클)

컨테이너 내부, `/workspace/sparse_direct_solver`에 진입했다고 가정:

```bash
# 0) 환경 확인
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
test -d /opt/nvidia/cudss && echo "cuDSS ok"
test -f /datasets/power_system/nr_linear_systems/case6468rte/J.mtx && echo "datasets ok"
test -n "$DISCORD_WEBHOOK_URL" && echo "discord ok" || { echo "DISCORD_WEBHOOK_URL not set — refusing to start (see §6.1)"; exit 1; }

# 1) Git 기준점 셋업 (§3.5 참고)
#    master는 항상 optimal을 유지하므로, 자동화 루프는 feat 브랜치에서만 작업한다
git config user.name  "mysolver-autorun"
git config user.email "autorun@mysolver.local"
git add -A
git diff --cached --quiet || git commit -m "chore: initial snapshot before mysolver autorun"
git branch -M master
git tag -f baseline-pre-m0
git checkout -b feat/mysolver-autorun

# 2) 디렉토리 스캐폴딩
mkdir -p src/mysolver/{reordering,symbolic,factorize,solve,common}
mkdir -p src/autorun tests/mysolver report/{benchmark,autorun/log,nsys,ncu}
touch report/autorun/STATE.json

# 3) M0 베이스라인 빌드 (timer + NVTX 켠 상태)
cmake -S . -B /tmp/profile-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TIMER=ON -DENABLE_CUDA_TIMER=ON -DENABLE_NVTX=ON
cmake --build /tmp/profile-build --target benchmark -j

# 4) 베이스라인 벤치마크
/tmp/profile-build/benchmark \
  --matrix-set smoke \
  --solver klu,umfpack,cudss-gpu \
  --output report/benchmark/baseline_reference.csv
# mysolver는 M0 더미 추가 후 같은 셋으로 다시 측정 → report/benchmark/baseline_mysolver.csv

# 5) 자동화 루프 진입
python3 src/autorun/run_cycle.py \
  --plan PLAN.md \
  --state report/autorun/STATE.json
```

이후부터는 `run_cycle.py`가 PLAN.md(이 문서)와 STATE.json만 보고 반복 실행한다.

---

## 10. 산출물 체크리스트 (M5 종료시점)

- [ ] `src/mysolver/` 전체 구현 (KLU 폴백 제거 상태)
- [ ] `tests/mysolver/` 단위 테스트 통과
- [ ] `report/benchmark/`에 마일스톤별 baseline CSV
- [ ] `report/nsys/`에 cuDSS와 mysolver의 비교용 timeline
- [ ] `report/autorun/log/MILESTONE-M{0..5}.md` 마일스톤 요약
- [ ] `report/autorun/STATE.json`의 최종 상태가 `current_milestone: M5, milestone_status: done`
- [ ] Discord 채널에 M5 완료 메시지 도달
- [ ] cuDSS 공식 문서 / CUDALibrarySamples 참고 위치를 commit history에서 추적 가능
- [ ] `master`가 M5 optimal 상태이고 `m0`~`m5` 태그가 전부 부여되어 있음
- [ ] `experiment/*` 브랜치는 정리됨 (머지된 것은 삭제, 보존할 것은 명시적으로 남김)
