# Factorize / Solve 소스 레이아웃 (2026-06-06 정리)

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

## 2. 파일별 역할

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
| small | `factor_small<T>` | 1 warp = 1 (front, batch). 8 warps/block. per-warp shared. fused LU. |
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

## 3. solve/ 의 파일별 역할

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

`issue_solve_levels` — forward pass (leaves → root) 후 backward pass (root → leaves). 각 pass 안에서 level max_fsz 로 small / regular kernel 선택. multi-stream 안 씀 (factor 와 달리 solve 는 work 가 작아 stream 동시성 이득 미미).

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

## 5a. plan/build (analyze pipeline) 분리 (2026-06-06)

`Solver::analyze()` 가 100+ 줄의 CSR → CSC → METIS → etree → fill_pattern → analyze 파이프라인을 직접 들고 있던 것을 `src/plan/build.{hpp,cpp}` 로 추출.

```
src/plan/build.hpp  ( 52줄) — PlanBuildOptions / PlanBuildResult / build_plan_from_csr 선언
src/plan/build.cpp  (162줄) — par_for, permute_symmetric_pattern, build_plan_from_csr 본체
                              (CLS_DUMP_FRONTS hook 포함)
```

| 책임 | 위치 |
|------|------|
| 8-step analyze pipeline (CSR→CSC→METIS-ND→permute→etree→fill_pattern→multifrontal plan) | `plan/build.cpp` |
| `par_for` (thread-fanout helper), `permute_symmetric_pattern` | `plan/build.cpp` (internal) |
| Solver impl 의 `analyze()` — thin adapter (16줄) | `src/solver.cpp` |

`PlanBuildOptions` 는 SolverConfig 의 일부 (use_parallel_nested_dissection / panel_cap / pure_fp32) 만 받는 좁은 구조체. plan/ 이 solver.hpp 에 의존하지 않도록 분리. `PlanBuildResult` 는 MultifrontalPlan + perm/iperm (host) + d_perm/d_iperm/d_ordered_value_to_csr (device) 를 한 번에 반환.

결과: `solver.cpp` 700+ → 217 줄로 축소. analyze 외 다른 phase API (set_data / setup / factorize / solve) 는 그대로 다 staying in solver.cpp.

## 5b. utility 통합 — multifrontal.hpp

`is_fp32_front(Precision)`, `round_up_int(int, int)` 두 작은 utility 가 multifrontal.cu / factorize/dispatch.cuh / solve/dispatch.cuh 에 각자 local `static` 으로 (이름만 다르게: `is_fp32_front_solve`, `round_up_int_dispatch`) 중복 정의되어 있던 것을 multifrontal.hpp 에 inline 로 한 곳에 통합. 세 곳 모두 multifrontal.hpp 를 이미 include 하므로 별도 include 추가 없음.

## 5c. deprecated/ 로 옮긴 실험 커널 (2026-06-06)

| 폴더 | 본 커널 | 결정 사유 |
|------|--------|----------|
| `deprecated/mid_warp/` | `factor_mid_warp<T>` | T4.1: barrier 0% 달성했으나 occupancy 추락 → net loss. docs/10 |
| `deprecated/mid_opt/` | `factor_mid_opt<T>` | P1+P2: sync −64% 했으나 wall −1~−4% 그침. docs/14 |

P1 (reciprocal multiply) 만 default kernel `lu_panel_factor` 에 흡수 retained.

## 6. 정리 이전 (참고)

```
src/factorize/  (정리 전, 11파일)
├── small.cuh         → kernels.cuh 안으로
├── mid.cuh           → kernels.cuh + phases.cuh (trailing variants) 로 분리
├── big.cuh           → kernels.cuh 안으로
├── mid_warp.cuh      → deprecated/mid_warp/
├── mid_opt.cuh       → deprecated/mid_opt/
├── primitives.cuh    → phases.cuh 의 일부
├── stage.cuh         → phases.cuh 의 일부
├── panel_lu.cuh      → phases.cuh 의 일부
├── u_solve.cuh       → phases.cuh 의 일부
├── trailing.cuh      → phases.cuh 의 일부
├── extend_add.cuh    → phases.cuh 의 일부
├── writeback.cuh     → phases.cuh 의 일부
├── factorize_front.cuh → phases.cuh 의 일부
└── scatter.cuh       (그대로)
```

`primitives/stage/panel_lu/u_solve/trailing/extend_add/writeback/factorize_front` 는 전부 device 빌딩 블록이라 한 파일 (`phases.cuh`) 로 통합. trailing 의 4개 변종 (scalar, staged, WMMA F16, WMMA TF32) 도 모두 한 곳. `small/mid/big.cuh` 의 7개 kernel 도 한 파일 (`kernels.cuh`) 로 통합.

## 7. 옛 docs 의 파일 경로 참조

본 리팩토링 이전 문서의 경로 매핑:

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
| `src/factorize/small.cuh` | `src/factorize/kernels.cuh` |
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
| `src/solve/small.cuh` | `src/solve/kernels.cuh` |
| `src/solve/big.cuh` | `src/solve/kernels.cuh` |
| `src/multifrontal.cu:issue_solve_levels` | `src/solve/dispatch.cuh` |

**plan + utility 통합 (2026-06-06)**:
| 옛 경로 | 새 경로 |
|---------|---------|
| `src/solver.cpp:Solver::analyze (8-step pipeline)` | `src/plan/build.cpp:build_plan_from_csr` |
| `src/solver.cpp:par_for, permute_symmetric_pattern` | `src/plan/build.cpp` (internal) |
| `src/multifrontal.cu:is_fp32_front (static)` | `src/multifrontal.hpp:is_fp32_front (inline)` |
| `src/multifrontal.cu:round_up_int (static)` | `src/multifrontal.hpp:round_up_int (inline)` |
| `src/factorize/dispatch.cuh:round_up_int_dispatch (static)` | `src/multifrontal.hpp:round_up_int` (canonical 사용) |
| `src/solve/dispatch.cuh:is_fp32_front_solve (static)` | `src/multifrontal.hpp:is_fp32_front` (canonical 사용) |
