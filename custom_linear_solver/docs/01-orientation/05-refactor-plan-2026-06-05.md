# 리팩토링 계획 — 2026-06-05

> 베이스: [`04-batched-precision-and-dispatch-map.md`](04-batched-precision-and-dispatch-map.md) § 0.1 의 사용자 지시 11개.
> 목적: 단일 정밀도 모델 (fp64/fp32/tc), 단일 batched 경로, env var 제거, 죽은 코드/실험 코드 제거, 일관된 이름·구조.

## 1. 현재 vs 목표 디렉터리 구조

### 1.1 현재 (src/, tests/ — refactor 전)

```
src/
├── batched/             # 일반 batched 경로 (B≥1)
│   ├── factor_kernels.cuh      (261 lines) FP64/FP32 + Mixed + TC kernels
│   ├── factor_small.cuh        (104) warp-packed leaf kernel
│   ├── lu_device.cuh           (222) panel LU / U solve / trailing primitives
│   ├── multifrontal_batched.cu (670) host orchestration + dispatcher
│   ├── multifrontal_batched.hpp (93)
│   ├── scatter.cuh             (29) value scatter
│   ├── solve_kernels.cuh       (215) fwd/bwd level kernels
│   └── solve_small.cuh         (196) warp-packed solve
├── factorize/           # B=1 전용 (delete)
│   ├── multifrontal.cu         (1311) 단일 시스템 factor
│   └── multifrontal.hpp        (39)
├── solve/               # B=1 전용 (delete)
│   ├── multifrontal.cu         (322) 단일 시스템 solve
│   └── multifrontal.hpp        (27)
├── tc/                  # TC dedicated + 실험 코드 (대부분 delete)
│   ├── factor_kernels_tc.cuh   (83) old-TC kernels (Mixed master + WMMA)
│   ├── factor_no_trailing.cuh  (222) NT clone 측정용 — delete
│   ├── factor_split_cublas.cuh (252) cuBLAS 실험 — delete
│   ├── factor_tc.cuh           (201) tc32 (FP32 native + WMMA) — *유지, 새 tc 의 본체*
│   ├── multifrontal_tc.cu      (783) TC 전용 entry point — delete (batched 와 통합)
│   ├── multifrontal_tc.hpp     (101) — delete
│   ├── spine_kernel.cuh        (83) spine 실험 — delete
│   └── trailing_tiled.cuh      (441) mid_tiled, regblock, regblock_h16 — 정리 (mid_tiled만 유지)
├── symbolic/
│   ├── amalgamate.cpp          (257) — delete
│   ├── amalgamate.hpp          (60) — delete
│   ├── elimination_tree.{cpp,hpp}    유지
│   ├── multifrontal.{cpp,hpp}        유지
│   └── supernode.{cpp,hpp}            유지
├── plan/                          유지
├── matrix/                        유지
├── reordering/                    유지
└── solver.{hpp,cpp}               유지 (API 단순화 후)

tests/
├── io.{cpp,hpp}                   유지
├── run_custom_solver.cu           유지 (env var 제거, --precision flag 로)
├── run_cudss_solver.cpp           유지
├── tc_trailing_microbench.cu      delete
└── wmma_pack_microbench.cu        delete
```

### 1.2 목표 (refactor 후)

```
src/
├── matrix/                    CSR view 등
├── reordering/                METIS nested dissection
├── symbolic/                  elimination tree, supernode, multifrontal analysis
├── plan/                      MultifrontalPlan
├── factorize/                 numeric factorize (batched, B≥1)
│   ├── factorize.{cu,hpp}     public entry + dispatcher
│   ├── kernels.cuh            정밀도별 factor kernels (fp64, fp32, tc)
│   ├── small_kernels.cuh      warp-packed leaf kernel
│   └── lu_device.cuh          panel LU / U solve / trailing primitives
├── solve/                     numeric solve (batched, B≥1)
│   ├── solve.{cu,hpp}
│   ├── kernels.cuh
│   └── small_kernels.cuh
└── solver.{hpp,cpp}           façade

deprecated/                    # 옛 코드 보존 (참고용)
├── README.md                  무엇이 왜 옮겨졌는지
├── single_system/             옛 B=1 dedicated path
│   ├── factorize_multifrontal.cu/hpp
│   └── solve_multifrontal.cu/hpp
├── tc_dedicated/              옛 TC 전용 경로
│   ├── multifrontal_tc.cu/hpp
│   ├── factor_split_cublas.cuh
│   └── spine_kernel.cuh
├── precision_mixed/           Mixed mode 의 kernel 들
│   ├── factor_extend_mixed_b.cuh
│   └── factor_extend_mixed_tc_b.cuh
├── selinv/                    selinv 사전계산 path
│   └── invert_pivot.cuh
├── amalgamation/              symbolic amalgamation
│   ├── amalgamate.cpp/hpp
│   └── notes.md
├── profiling_no_trailing/     NT clone 측정용
│   └── factor_no_trailing.cuh
└── microbench/
    ├── tc_trailing_microbench.cu
    └── wmma_pack_microbench.cu

tests/                         테스트 / 벤치 진입점만
```

## 2. 정밀도 모델 (instruction #5)

### 2.1 단순화

| 옛 enum | 처리 |
|---|---|
| FP64 | 유지 → `Precision::FP64` |
| FP32 | 유지 → `Precision::FP32` |
| Mixed | **삭제** (FP64 master + FP32 working) |
| TC | **삭제** (Mixed + FP16 WMMA — Mixed 자체가 사라지므로 같이) |
| TC32 | **rename → `Precision::TC`** (FP32 + FP16 WMMA) |

### 2.2 API (instruction #6)

env var 제거, config 로:

```cpp
namespace custom_linear_solver {

enum class Precision { FP64, FP32, TC };

struct SolverConfig {
    bool use_matching = false;
    bool use_parallel_nested_dissection = true;
    bool enable_shift_retry = true;
    double shift_retry_epsilon = 1.0e-8;
    int panel_cap = 8;
    Precision precision = Precision::FP64;
};

// batched_setup 의 prec 인자 제거 — Solver 생성 시점에 config 로 결정됨
Status batched_setup(int batch);
```

runner CLI 도 env 가 아니라 `--precision fp64|fp32|tc` 플래그로.

### 2.3 삭제되는 env var

- `MF_NO_MIXED`, `MF_FP32`, `MF_MIXED`, `MF_TC`, `MF_TC32` — 전부 삭제. runtime 정밀도는 config 만.
- `CLS_USE_SELINV`, `MF_NO_SELINV` — selinv 자체 삭제 (instruction #9).
- `CLS_PROFILE_NO_TRAILING` — 측정 코드 삭제 (instruction #8).
- `CLS_NO_TILED_TRAILING` — mid_tiled 가 default 단일 path 가 되므로 토글 불필요.
- `CLS_USE_AMAL`, `CLS_AMAL_CAP`, `CLS_AMAL_MIN_DEPTH`, `CLS_AMAL_INFO` — amalg 삭제 (instruction #9).
- `CLS_USE_CUBLAS`, `CLS_CUBLAS_TF32`, `CLS_CUBLAS_MIN_FSZ` — TC dedicated path 의 실험 (삭제).
- `CLS_USE_REGBLOCK`, `CLS_USE_REGBLOCK_H16`, `CLS_USE_SPINE`, `CLS_USE_MULTISTREAM`, `CLS_BYPASS_GRAPH`, `CLS_USE_PIVOTING`, `CLS_TC_SETUP_DBG` — TC dedicated 의 (삭제).
- `CLS_NO_MULTISTREAM` — multi-stream subtree dispatch 는 *유지* (성능에 본질적). 토글은 *제거*하고 항상 ON.
- `CLS_USE_SMALL_WARP`, `CLS_NO_SHARED_FACTOR`, `CLS_WARP_DBG` — single-system path 의 토글 (single-system 삭제와 함께).
- `CLS_CAP` — research override. 일단 *유지 검토* (실험에 자주 쓰임).
- `CLS_DUMP`, `CLS_PANEL_DUMP`, `CLS_TREE_INFO`, `CLS_KERNEL_TIME` — debug print, 유지 가능 (instruction 외).

남는 env (디버그 / 빌드 토글만):
- `CLS_DUMP`, `CLS_PANEL_DUMP`, `CLS_TREE_INFO`, `CLS_KERNEL_TIME` — 측정 보조.
- `CLS_CAP` — research panel cap override (별도 결정).

## 3. 커널 rename 표 (instructions #2, #5)

| 옛 이름 | 새 이름 | 비고 |
|---|---|---|
| `mf_factor_extend_level_b<double>` | `factor_big_fp64` | top-level FP64 big-front kernel |
| `mf_factor_extend_level_b<float>` | `factor_big_fp32` | top-level FP32 big-front kernel |
| `mf_factor_extend_tc32_b` | `factor_big_tc` | TC (FP32 + WMMA) big-front kernel |
| `mf_factor_extend_mixed_b` | **삭제** | Mixed |
| `mf_factor_extend_mixed_tc_b` | **삭제** | 옛 TC (Mixed + WMMA) |
| `mf_factor_mid_tc32_b<true>` | `factor_mid_tc` | TC mid kernel (실제 WMMA) |
| `mf_factor_mid_tc32_b<false>` | **삭제** (mid_tiled 로 통합) | "tc32" 이름인데 TC 안 씀 |
| `tc::mf_factor_mid_tiled_b` | `factor_mid_fp32` | FP32 mid kernel (shared L/U staging) |
| `mf_factor_small_warp_b<T>` | `factor_small<T>` | leaf-level warp-packed |
| `mf_invert_pivot_b<T>` | **삭제** | selinv 사전계산 (instruction #9) |
| `mf_factor_*_NT_b` | **삭제** | NT clones 측정용 (instruction #8) |
| `tc::mf_factor_mid_tiled_h16_b` 등 regblock 변형 | **삭제** | 실험 코드 |
| `mf_fwd_level_b<T>` | `solve_fwd<T>` | forward solve, level 단위 |
| `mf_bwd_level_b<T>` | `solve_bwd<T>` | backward solve, level 단위 |
| `mf_fwd_small_warp_b<T>` | `solve_fwd_small<T>` | warp-packed forward |
| `mf_bwd_small_warp_b<T>` | `solve_bwd_small<T>` | warp-packed backward |
| `mf_fwd_level_pp_b<T>` | **삭제 검토** | within-pivot pivoting (실험) |
| `gather_rhs_b<RT, YT>` | `gather_rhs` | |
| `scatter_sol_b<YT, ST>` | `scatter_sol` | |
| `scatter_batched<FT, VT>` | `scatter_values` | factor 의 CSR→front scatter |

## 4. dispatcher 정책 (instruction #7)

### 4.1 새 결정 트리 (모호함 제거)

```
factor_level(b, e, precision)
├─ scan level → max_fsz, max_uc (analyze 단계에서 precompute 검토)
│
├─ tier = front_size_tier(max_fsz)        // SMALL=32, MID=128 (통일), BIG=>128
│
├─ if tier == SMALL:  factor_small<FT>          // FP64/FP32/TC 동일 dispatcher
├─ if tier == MID:
│   ├─ FP64 → factor_big_fp64 (mid 에 별도 kernel 없음, big 와 공유)
│   ├─ FP32 → factor_mid_fp32
│   └─ TC   → factor_mid_tc
└─ if tier == BIG:
    ├─ FP64 → factor_big_fp64
    ├─ FP32 → factor_big_fp32
    └─ TC   → factor_big_tc
```

### 4.2 MID_THRESH 통일 (refactor R2)

옛: 128 (TC32) vs 159 (FP32). 새: **128** 로 통일.
이유: WMMA tile = 16, 그 8배 = 128. FP32 의 159 는 shared 96KB 한도 측정치 — 명확한 정책은 *128 + shared 점유 자동 조정*.

### 4.3 dispatcher 가 분기 외에 하는 일

옛: dispatcher 가 per-level max_fsz, max_uc, level_max_nc 를 host 에서 매번 계산.
새: analyze 단계의 plan 에 *level metadata* 로 미리 채우기 (refactor R6). dispatcher 는 단순 lookup.

## 5. 주석 / 시각화 정책 (instructions #10, #11)

### 5.1 주석 표준 스타일

- 파일 헤더: 책임 영역 1-3 줄.
- public function: doxygen-style 짧은 설명 + 인자.
- kernel: 실행 흐름을 "Phase 1 / Phase 2 / Phase 3" 로 단계 표시.
- 복잡한 흐름: ASCII art (예: dispatcher tree, etree, front layout).
- 실험 흔적 / 측정 로그 / phase Σ.X 같은 historical reference 는 제거. *코드는 현재 상태만 설명.*

### 5.2 변수 이름

- `fsz` / `nc` / `uc` / `cb` 는 *유지* (multifrontal 도메인 용어, 짧고 일관).
- `B` (batch) 는 `batch_size` 또는 `B` 둘 다 OK — local 에선 `B`, 인자/필드 에선 `batch_size`.
- `FT` (front type) → `T` 또는 `Scalar` — 도메인 hint 없는 일반 타입.
- `d_*` (device pointer) prefix 는 유지.

## 6. Phase 별 작업 순서

### Phase 1 (현재) — 인벤토리 + 계획
본 문서.

### Phase 2 — deprecated/ 폴더 + 안전한 이동
*빌드 영향 없는 (다른 곳에서 참조 안 되는) 파일* 먼저 이동:
- `tests/{tc_trailing,wmma_pack}_microbench.cu` → `deprecated/microbench/`
- `src/tc/factor_split_cublas.cuh` → `deprecated/tc_dedicated/`
- `src/tc/spine_kernel.cuh` → `deprecated/tc_dedicated/`
- `src/tc/factor_no_trailing.cuh` → `deprecated/profiling_no_trailing/`

### Phase 3 — 정밀도 모델 단순화
- `SolverConfig::precision` 도입.
- `BatchPrecision` enum 축소 (FP64, FP32, TC) + `Precision` 으로 rename.
- runner: env var 제거 → `--precision` flag.
- `solver.hpp/cpp` 의 `tc_*` API 제거.

### Phase 4 — Mixed / 옛 TC / selinv / amalg 코드 삭제 (deprecated/ 로)
- `mf_factor_extend_mixed_b` → deprecated/precision_mixed/.
- `mf_factor_extend_mixed_tc_b` → deprecated/precision_mixed/.
- `mf_invert_pivot_b` + 관련 dispatcher 라인 → deprecated/selinv/.
- `src/symbolic/amalgamate.{cpp,hpp}` + 호출 라인 → deprecated/amalgamation/.

### Phase 5 — 단일 시스템 path 삭제
- `src/factorize/multifrontal.{cu,hpp}` → deprecated/single_system/.
- `src/solve/multifrontal.{cu,hpp}` → deprecated/single_system/.
- `solver.cpp` 의 single-system API 를 batched(B=1) 로 redirect.
- (옵션) `solver.hpp` 에서 `factorize()` / `solve()` 시그니처 유지하고 내부에서 batched(B=1) 호출.

### Phase 6 — TC dedicated path 삭제 + 통합
- `src/tc/multifrontal_tc.{cu,hpp}` → deprecated/tc_dedicated/.
- `solver.hpp/cpp` 의 `tc_setup/tc_factorize/tc_solve` 삭제.
- `src/tc/factor_tc.cuh` 의 kernel 들 (`mf_factor_extend_tc32_b`, `mf_factor_mid_tc32_b<true>`) 을 새 `src/factorize/` 로 이동.

### Phase 7 — 디렉터리 / 파일 / 커널 rename
- `src/batched/` → `src/factorize/` + `src/solve/`.
- `src/tc/factor_tc.cuh` 의 잔재 → `src/factorize/kernels.cuh`.
- 커널 rename (§ 3 표).

### Phase 8 — dispatcher 정책 정리
- MID_THRESH 통일.
- 죽은 `case TC32:` 제거.
- 분기 트리 단순화.
- (옵션) level metadata 를 plan 에 흡수.

### Phase 9 — 주석 / 시각화 / 변수명 정리
- 헤더 주석 표준화.
- 실험 흔적 제거.
- ASCII 다이어그램 추가.
- 변수 이름 통일.

### Phase 10 — 빌드 / 테스트 검증 + final iteration
- 각 phase 후 빌드 통과 확인 (incremental).
- 회귀 테스트: case8387 / USA 의 factor + solve relres 매 phase 측정.
- final docs/README 업데이트.

## 7. 호환성 / 외부 영향

### 7.1 cuPF 등 외부 사용자

- `Solver::tc_*` API 제거 — cuPF 가 쓰는지 확인 후 migration path 명시.
- `Solver::batched_setup(int B, BatchPrecision)` → `Solver::batched_setup(int B)` (precision 은 config) — breaking.
- env var 제거 — 사용처 (benchmark scripts, cuPF wrapper) 의 migration 필요.

### 7.2 측정 문서 정정

doc 09, 10 의 분석은 *옛 default* 기준이라 새 default 와 다름.
새 측정 표 (FP64/FP32/TC × 케이스 × B) 를 한 번 새로 찍어 doc 11 로 추가.

### 7.3 외부 의존성

- METIS — 유지.
- cuBLAS — TC dedicated 의 실험에서만 썼음. 삭제 시 link 제거 검토 (현재 cublas 이 `target_link_libraries` 에 있음).

## 8. 변경 사항 추적

각 phase 의 commit 단위로 진행. 매 phase 후:
1. 빌드 확인.
2. case8387 / USA 위 sanity check (relres < 1e-2 for FP32, < 1e-10 for FP64).
3. 본 문서의 phase 체크.
