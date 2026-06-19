# 코드 구조 & 컨벤션 — `src/factorize`·`src/solve` 레이아웃과 리팩토링 규칙

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: `src/factorize/`·`src/solve/` 각 4파일(phases/kernels/dispatch/scatter|permute) 대칭 구조와 옛→새 경로 매핑, 그리고 naming/style/SRP/문서화 리팩토링 컨벤션.

이 문서는 두 부분이다.
1. **§1–§7 파일 레이아웃** — `src/factorize/`(4파일) + `src/solve/`(4파일)의 phases/kernels/dispatch/scatter 대칭 패턴, plan/build 분리, 옛→새 경로 매핑.
2. **§8–§13 리팩토링 컨벤션** — `src/` 전반의 naming, 상수 canonical화, SRP 함수 분리, 주석 규칙.

---

# Part A. 파일 레이아웃

**계기**: 11개 파일로 흩어졌던 `src/factorize/` (size 기준 small/mid/big + 기능 기준 primitives/scatter/stage/panel_lu/u_solve/trailing/extend_add/writeback/factorize_front) 가 너무 분산됨. 연산 수준으로 통합해 **4 파일**로 정리. 동일 패턴을 `src/solve/` 에도 적용.

## 1. 최종 파일 구조

```
src/factorize/                          src/solve/
├── phases.cuh   (376줄)                ├── phases.cuh   (110줄)
├── kernels.cuh  (374줄)                ├── kernels.cuh  (175줄)
├── dispatch.cuh (274줄)                ├── dispatch.cuh (137줄)
└── scatter.cuh  ( 28줄)                └── permute.cuh  ( 50줄)
   = 1052줄                                = 472줄
```

총 8 파일, 1524줄. 동일 4-역할 패턴:
- **phases.cuh** : `__device__` 빌딩 블록 (per-phase)
- **kernels.cuh** : `__global__` 진입점 (per-tier × precision 변종)
- **dispatch.cuh** : host측 per-level dispatch (multifrontal.cu 에서 옮겨옴)
- **scatter.cuh / permute.cuh** : I/O setup 전용 `__global__` kernel

## 2. factorize/ 파일별 역할

### `phases.cuh` — device 빌딩 블록

per-front phase 4개 + stage/writeback/orchestrator. 모두 `__device__ __forceinline__`.

| 함수 | Phase | 비고 |
|------|------|------|
| `stage_in_async<T>` | pre | global → shared (cp.async on Ampere+, fallback) |
| `lu_small_front<T>` | 1 | fused Phase 1+3 (fsz ≤ 48) |
| `lu_panel_factor<T>` | 1 | split form for mid/big (row-fused for nc ≤ 12) |
| `lu_small_warp<FT>` | 1 | warp-specialized (32-lane, __syncwarp) — small kernel용 |
| `u_panel_solve<T>` | 2 | U-panel triangular solve |
| `trailing_update_scalar<T>` | 3 | 직접 scalar GEMM, no staging |
| `trailing_update_staged<T>` | 3 | shared-staged scalar GEMM (mid 기본) |
| `trailing_update_wmma_f32` | 3 | FP16 WMMA 16×16×16 trailing |
| `trailing_update_wmma_tf32_f32` | 3 | TF32 WMMA 16×16×8 trailing |
| `extend_add<DstT, SrcT>` | 4 | CB → parent atomicAdd |
| `writeback_factored<MT, WT>` | post | shared → global L/U writeback |
| `factorize_front<T, F>` | orchestrator | Phase 1+2+3 canonical 순서 |

### `kernels.cuh` — kernel 진입점

7개 `__global__` kernel, tier 3개 × 정밀도 변종.

| Tier | Kernel | 구성 |
|------|--------|------|
| small | `factor_mid<T>` | 1 warp = 1 (front, batch). 8 warps/block. per-warp shared. fused LU. |
| mid | `factor_mid<T>` | 1 block = 1 (front, batch). 256 thread. 전체 shared stage-in. staged scalar trailing. |
| mid | `factor_mid_tc` | mid + FP16 WMMA trailing |
| mid | `factor_mid_tf32` | mid + TF32 WMMA trailing |
| big | `factor_big<T>` | 1 block = 1 (front, batch). 1024 thread. global memory direct (no stage). scalar trailing. |
| big | `factor_big_tc` | big + FP16 WMMA trailing (small shared scratch for WMMA fragments) |
| big | `factor_big_tf32` | big + TF32 WMMA trailing |

각 kernel 은 `phases.cuh` 의 빌딩 블록을 조립한 얇은 orchestrator (≈30-60줄 each).

### `dispatch.cuh` — host측 level dispatch

per-level → per-range → per-tier 3층 dispatch. multifrontal.cu 에서 옮겨옴.

| 함수 | 역할 |
|------|------|
| `issue_factor_levels` | 모든 etree level 순회. multi-stream subtree fork/join. |
| `issue_factor_level_range_split` | T-split (opt-in): level 안 mixed tier 를 sub-range로 분할 |
| `issue_factor_level_range` | range 의 max_fsz / precision / shared budget 기준으로 kernel 선택 |

### `scatter.cuh` — 별개

`scatter_values<FT, VT>` kernel. 매 factorize 호출 시 입력 CSR `values` 를 front arena의 적절한 위치에 atomicAdd 로 뿌리는 setup. factorize phase 와 별개.

## 3. solve/ 파일별 역할

### `solve/phases.cuh` — device 빌딩 블록

| 함수 | 방향 | Phase |
|------|------|------|
| `fwd_substitute<T>` | forward | 1 (warp-parallel panel substitution) |
| `fwd_cb_update<T>` | forward | 2 (CB rows atomic update) |
| `bwd_load_rhs_and_x<T>` | backward | 1 (gather from y) |
| `bwd_cb_subtract<T>` | backward | 2 (CB contribution to rhs) |
| `bwd_substitute<T>` | backward | 3 (warp-parallel panel substitution) |

`MF_MAX_NC = 64` 상수도 여기.

### `solve/kernels.cuh` — 4개 `__global__`

| Tier | Forward | Backward | Block |
|------|---------|----------|------|
| small | `solve_fwd_small<T>` | `solve_bwd_small<T>` | 8 warps, 1 warp per (front, batch) |
| regular | `solve_fwd<T>` | `solve_bwd<T>` | 64-256 thread, 1 block per (front, batch) |

solve 는 factor 와 달리 mid/big 으로 더 안 쪼갬 (work per front 가 가벼워 그럴 필요 없음).

### `solve/dispatch.cuh`

`issue_solve_levels` — forward pass (leaves → root) 후 backward pass (root → leaves). 각 pass 안에서 level max_fsz 로 small / regular kernel 선택. multi-stream 안 씀 (solve 는 work 가 작아 stream 동시성 이득 미미).

### `solve/permute.cuh` — I/O 변환 (factorize/scatter 대칭)

| kernel | 시점 | 역할 |
|--------|------|------|
| `gather_rhs<RT, YT>` | solve 진입 | `y[k] = rhs[perm[k]]` (orig → ND order) |
| `scatter_sol<YT, ST>` | solve 종료 | `sol[perm[k]] = y[k]` (ND → orig) |

template 으로 RT/YT/ST 정밀도 독립 (mixed-precision 가능).

## 4. dependency graph

```
multifrontal.cu
   ├── factorize/scatter.cuh        (값 scatter, 입력 setup)
   ├── factorize/dispatch.cuh
   │    └── factorize/kernels.cuh
   │         └── factorize/phases.cuh
   ├── solve/permute.cuh            (rhs/sol perm, I/O)
   ├── solve/dispatch.cuh
   │    └── solve/kernels.cuh
   │         └── solve/phases.cuh
   └── multifrontal.hpp             (Plan + State 정의)
```

linear chain, 양쪽 모두 phases → kernels → dispatch → multifrontal.cu. cycle 없음, 대칭 구조.

## 5. multifrontal.cu 의 잔여 역할

이번 정리 후 multifrontal.cu (270줄, 정리 전 627줄) 에 남은 것:
- `State` (plan 별 runtime state) 의 destructor + setup
- `factorize` / `solve` 진입 함수 (factorize_impl, solve_impl)
- CUDA Graph capture / replay 로직 (`CLS_INTERNAL_GRAPH`)

factor dispatch 는 `factorize/dispatch.cuh` 로, solve dispatch + I/O perm 은 `solve/dispatch.cuh` / `solve/permute.cuh` 로 모두 이동.

### 5a. plan/build (analyze pipeline) 분리

`Solver::analyze()` 가 100+ 줄의 CSR → CSC → METIS → etree → fill_pattern → analyze 파이프라인을 직접 들고 있던 것을 `src/plan/build.{hpp,cpp}` 로 추출.

```
src/plan/build.hpp  ( 52줄) — PlanBuildOptions / PlanBuildResult / build_plan_from_csr 선언
src/plan/build.cpp  (162줄) — par_for, permute_symmetric_pattern, build_plan_from_csr 본체
                              (CLS_DUMP_FRONTS hook 포함)
```

| 책임 | 위치 |
|------|------|
| 8-step analyze pipeline (CSR→CSC→METIS-ND→permute→etree→fill_pattern→multifrontal plan) | `plan/build.cpp` |
| `par_for`, `permute_symmetric_pattern` | `plan/build.cpp` (internal) |
| Solver impl 의 `analyze()` — thin adapter (16줄) | `src/solver.cpp` |

`PlanBuildOptions` 는 SolverConfig 의 일부 (use_parallel_nested_dissection / panel_cap / pure_fp32) 만 받는 좁은 구조체. plan/ 이 solver.hpp 에 의존하지 않도록 분리. `PlanBuildResult` 는 MultifrontalPlan + perm/iperm (host) + d_perm/d_iperm/d_ordered_value_to_csr (device) 를 한 번에 반환. 결과: `solver.cpp` 700+ → 217 줄.

### 5b. utility 통합 — multifrontal.hpp

`is_fp32_front(Precision)`, `round_up_int(int, int)` 두 작은 utility 가 multifrontal.cu / factorize/dispatch.cuh / solve/dispatch.cuh 에 각자 local `static` 으로 (이름만 다르게: `is_fp32_front_solve`, `round_up_int_dispatch`) 중복 정의되어 있던 것을 multifrontal.hpp 에 inline 로 통합. 세 곳 모두 이미 multifrontal.hpp 를 include.

### 5c. deprecated/ 로 옮긴 실험 커널

| 폴더 | 본 커널 | 결정 사유 |
|------|--------|----------|
| `deprecated/mid_warp/` | `factor_mid_warp<T>` | T4.1: barrier 0% 달성했으나 occupancy 추락 → net loss. docs/10 |
| `deprecated/mid_opt/` | `factor_mid_opt<T>` | P1+P2: sync −64% 했으나 wall −1~−4% 그침. docs/14 |

P1 (reciprocal multiply) 만 default kernel `lu_panel_factor` 에 흡수 retained.

## 6. 정리 이전 구조 (참고)

```
src/factorize/  (정리 전, 11파일)
├── mid.cuh / mid.cuh / big.cuh        → kernels.cuh (mid 의 trailing variants 는 phases.cuh)
├── mid_warp.cuh / mid_opt.cuh           → deprecated/
├── primitives.cuh / stage.cuh / panel_lu.cuh / u_solve.cuh /
│   trailing.cuh / extend_add.cuh / writeback.cuh / factorize_front.cuh → phases.cuh (전부 device 빌딩 블록)
└── scatter.cuh                          (그대로)
```

`small/mid/big.cuh` 의 7개 kernel 도 한 파일 (`kernels.cuh`) 로 통합.

## 7. 옛 docs 의 파일 경로 참조 — 옛→새 매핑

**factorize/**:
| 옛 경로 | 새 경로 |
|---------|---------|
| `src/factorize/primitives.cuh` | `src/factorize/phases.cuh` |
| `src/factorize/stage.cuh` | `src/factorize/phases.cuh` |
| `src/factorize/panel_lu.cuh` | `src/factorize/phases.cuh` |
| `src/factorize/u_solve.cuh` | `src/factorize/phases.cuh` |
| `src/factorize/trailing.cuh` | `src/factorize/phases.cuh` |
| `src/factorize/extend_add.cuh` | `src/factorize/phases.cuh` |
| `src/factorize/writeback.cuh` | `src/factorize/phases.cuh` |
| `src/factorize/factorize_front.cuh` | `src/factorize/phases.cuh` |
| `src/factorize/mid.cuh` | `src/factorize/kernels.cuh` |
| `src/factorize/mid.cuh` | `src/factorize/kernels.cuh` |
| `src/factorize/big.cuh` | `src/factorize/kernels.cuh` |
| `src/factorize/mid_warp.cuh` | `deprecated/mid_warp/mid_warp.cuh` |
| `src/factorize/mid_opt.cuh` | `deprecated/mid_opt/mid_opt.cuh` |
| `src/multifrontal.cu:issue_factor_*` | `src/factorize/dispatch.cuh` |

**solve/**:
| 옛 경로 | 새 경로 |
|---------|---------|
| `src/solve/primitives.cuh` (`fwd_*`, `bwd_*`, `MF_MAX_NC`) | `src/solve/phases.cuh` |
| `src/solve/primitives.cuh` (`gather_rhs`, `scatter_sol`) | `src/solve/permute.cuh` |
| `src/solve/mid.cuh` | `src/solve/kernels.cuh` |
| `src/solve/big.cuh` | `src/solve/kernels.cuh` |
| `src/multifrontal.cu:issue_solve_levels` | `src/solve/dispatch.cuh` |

**plan + utility 통합**:
| 옛 경로 | 새 경로 |
|---------|---------|
| `src/solver.cpp:Solver::analyze (8-step pipeline)` | `src/plan/build.cpp:build_plan_from_csr` |
| `src/solver.cpp:par_for, permute_symmetric_pattern` | `src/plan/build.cpp` (internal) |
| `src/multifrontal.cu:is_fp32_front (static)` | `src/multifrontal.hpp:is_fp32_front (inline)` |
| `src/multifrontal.cu:round_up_int (static)` | `src/multifrontal.hpp:round_up_int (inline)` |
| `src/factorize/dispatch.cuh:round_up_int_dispatch (static)` | `src/multifrontal.hpp:round_up_int` (canonical) |
| `src/solve/dispatch.cuh:is_fp32_front_solve (static)` | `src/multifrontal.hpp:is_fp32_front` (canonical) |

---

# Part B. 리팩토링 컨벤션 (2026-06-08)

`gpu-powerflow/custom_linear_solver/src` 정리 규칙이다. 긴 계획을 쓰지 말고, 아래 규칙을 한 모듈씩 적용한다.

기준: [CUTLASS/NVIDIA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html), [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), [Doxygen](https://www.doxygen.nl/manual/docblocks.html). 충돌 시 `기존 public API 호환성 → CUTLASS/NVIDIA CUDA 관례 → Google C++ → Doxygen` 순서로 따른다.

## 8. 실행

- 먼저 `src/` 전체 구조를 읽는다.
- 한 번에 하나만 고친다: 이름, 상수, 함수 분리, 파일 이동, 문서 갱신.
- 순서: `solver → multifrontal → plan → factorize → solve → matrix/symbolic/reordering/profile → tests/docs`.
- 사용자 승인 없이 진행한다.
- 성능 정책, 수치 결과는 보존한다.
- public api 수정 허용.
- 모듈 경계 변경 시 include, CMake, docs를 같이 갱신한다.
- 리팩토링 조건을 만족할 때까지 반복하여 점검한다.

```bash
cmake -S gpu-powerflow/custom_linear_solver -B gpu-powerflow/custom_linear_solver/build-refactor \
  -DCMAKE_BUILD_TYPE=Release -DCLS_BUILD_CUDA_OPS=ON -DCLS_BUILD_SCRIPTS=ON \
  -DCLS_BUILD_CUDSS_SCRIPT=OFF -DCLS_CUDA_ARCHITECTURES=86
cmake --build gpu-powerflow/custom_linear_solver/build-refactor -j
```

## 9. 이름

- 파일명, namespace, 변수, 함수 인자, struct field: `snake_case`.
- Type, struct, class, enum: `PascalCase`.
- namespace-scope constant: `inline constexpr kPascalCase`.
- scoped enum만 사용하고 enumerator는 `kPascalCase`.
- macro는 preprocessor가 필요할 때만 쓰고 `CLS_ALL_CAPS`.
- 기존 함수명은 이 repo 관례에 맞춰 `snake_case`를 유지한다. Google 함수명 `PascalCase`는 이 프로젝트의 local deviation이다.
- 새 `camelCase`와 의미 없는 약어를 만들지 않는다.
- `d_` / `h_` prefix는 device/host pointer 또는 mirror에만 쓴다.

짧은 이름 허용 범위:

- 좁은 loop scope의 `i`, `j`, `k`.
- CUDA 문맥의 `lane`, `warp`, `thread_idx`, `block_idx`.
- precision 약어 `FP64`, `FP32`, `FP16`, `TF32`.
- device hot loop의 매우 좁은 local 약어. host dispatch와 plan code에서는 풀어쓴다.

## 10. 상수와 정책

다음 값은 하나의 canonical header로 모은다. 파일마다 재정의하지 않는다.

```cpp
inline constexpr int kWarpSize = 32;
inline constexpr int kSmallFrontMax = kWarpSize;  // 32  (small|mid 경계)
inline constexpr int kMidFrontMax = 64;           // mid|big = whole-front 점유 교차점
inline constexpr int kSmallTierWarpsPerBlock = 8;
inline constexpr int kMaxPivotColumns = 64;
inline constexpr int kTensorCorePivotColumnCap = 32;
inline constexpr int kFusedMidFrontMax = 48;      // mid kernel 의 fused(P1+2+3) 경로 상한
inline constexpr std::size_t kDynamicSharedMemoryOptInBytes = 99 * 1024;  // sm_86 opt-in shared
inline constexpr int kNumFrontBuckets = 3;        // small/mid/big
inline constexpr int kMaxSubtreeStreams = 8;
inline constexpr int kArenaAlignmentBytes = 256;
// big 커널의 내부 bounded-shared staging 상한(티어 경계 아님): 159(float)/111(double).
constexpr int WholeFrontSharedMax(bool fp64);
```

> tier 경계는 `kMidFrontMax=64`(2026-06-18 통합). 옛 4-tier 의 `MID_THRESH=128`·`MID_SHARED_BUDGET=96KB` 는
> 더 이상 없다 — `159/111` 은 이제 big 커널 내부 staging 상한으로만 남았다(`WholeFrontSharedMax`).

공통 API는 `src/internal/types.hpp` 의 실제 시그니처를 따른다(Google C++ 스타일, `PascalCase`).

```cpp
enum class FrontTier { kSmall, kMid, kBig };

FrontTier ClassifyFrontTier(int front_size, bool fp64);
int FrontBucket(int front_size, bool fp64);
constexpr int RoundUpToMultiple(int value, int alignment);
```

`plan/analyze.cu`, `factorize/dispatch.cuh`, `solve/dispatch.cuh`는 같은 classifier를 사용한다. "must match" 주석은 제거한다.

## 11. 함수 분리 (SRP)

함수 하나는 한 책임만 가진다.

- scan: host metadata를 읽고 stats를 만든다.
- policy: tier, thread count, shared memory size를 결정한다.
- launch: kernel을 호출한다.
- orchestration: level, subtree, pass 순서를 결정한다.

## 12. 주석

- 주석은 이유와 invariant를 설명한다.
- 코드가 이미 말하는 내용을 반복하지 않는다.
- magic number 설명은 canonical constant 이름으로 대체한다.
- historical experiment 이름은 새 코드 주석에 넣지 않는다.
- ASCII diagram은 layout, dependency, launch hierarchy처럼 그림이 필요한 경우만 쓴다.

## 13. 금지

- `git reset --hard`, `git checkout --`, 대량 삭제.
- 사용자 변경 되돌리기.
- 리팩토링 중 새 heuristic 추가.
- 새 build flag/env var 추가.
- include cycle 생성.
- 성능 근거 없는 kernel signature 대개편.
