# `custom_linear_solver` 문서 색인

루트에는 이 색인만 두고, 세부 문서는 주제별 폴더에 읽는 순서대로 번호를 붙였다.
2026-06-04 초기 정리는 `01`-`04`에 API/설계/최적화/측정을 나눴고, 2026-06-04 오후부터 2026-06-05까지 추가된 실험 로그와 최종 보고서는 기존 흐름에 흡수하거나 `05-reports/`로 분리했다.

## 폴더 구조

- `01-orientation/`: API, 연구 포지셔닝, 코드 lineage.
- `02-design-analysis/`: 왜 빠른지, 어떤 가정이 안전한지, STRUMPACK과 설계가 어떻게 다른지, GEMM/TC wall 근거.
- `03-optimization-notes/`: 구현 최적화 기록, 연구 로그, 음의 결과.
- `04-benchmarks-profiling/`: 논문 재현, wall/kernel 측정, Nsight Systems/Compute 분석, bottleneck sweep.
- `05-reports/`: 날짜별 최종 보고서, comprehensive sweep 원본 표, 시간순 세션 로그.
- `2606012_lab_meeting/`: lab meeting용 case/level front-size 분포 추적 CSV와 요약.

## 중복 정리 기준

- 최신 전체 결론은 `05-reports/01-final-report-2026-06-05.md`를 우선 참고한다.
- 전체 benchmark table의 canonical source는 `05-reports/02-comprehensive-sweep-2026-06-05.md`다.
- GEMM/TC wall fraction과 TC speedup ceiling의 canonical 근거는 `02-design-analysis/05-gemm-fraction-analysis.md`다.
- 시간순 작업 로그는 `05-reports/03-session-summary-2026-06-05.md`에 보존했다.
- `03-optimization-notes/archive/` 는 현재 코드에 반영되지 않은 시도 (research log, 회귀, deprecated 코드) 를 모아둔다. 같은 lever 재시도 시 동일한 실수 회피용. 폴더 자체 README 가 분류 표를 가진다.
- 측정 근거가 독립적인 문서는 삭제하지 않고 보존했다. 대신 색인에서 “대표 문서”, “증거 문서”, “historical log”의 역할을 분리했다.

## 최적 경로 (2026-06-08 정리)

V9h GEMM stack (docs/15) + EXP-B `__launch_bounds__(512, 2)` (docs/17) 가 **`Precision::TF32` 의 기본 동작으로 baked-in** 되었다. 별도 build flag 가 더 이상 필요 없으며, runtime CLI 로 모든 비교가 가능하다.

### Default 빌드

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

런타임 선택:

```bash
# V9h + LB(512, 2) — 권장 (USA B=64 −5.7% vs V0)
build/custom_linear_solver_run <case> --batch 64 --precision tf32

# V0 baseline (비교용)
build/custom_linear_solver_run <case> --batch 64 --precision tf32_wmma
```

### 정밀도 5종 비교 가능

| `--precision` | front | Phase-3 trailing            | 비고                           |
|---------------|-------|-----------------------------|--------------------------------|
| `fp64`        | FP64  | scalar FP64                 | reference 정확도               |
| `fp32`        | FP32  | staged-scalar FP32          | ~2× FP64                       |
| `fp16`        | FP32  | FP16 WMMA m16n16k16         | FP16 input, FP32 accumulate    |
| `tf32_wmma`   | FP32  | TF32 WMMA m16n16k8 + Csc    | V0 baseline (docs/15 §1)       |
| `tf32`        | FP32  | TF32 PTX mma.m16n8k8/k4     | **V9h + LB recommended**       |

### 런타임 knobs

| Knob (SolverConfig / CLI)                | 기본    | 용도                                          |
|-----------------------------------------|---------|-----------------------------------------------|
| `precision` (`--precision`)              | `FP64`  | 위 5종 중 선택                                |
| `panel_cap` (`--panel-cap`)              | `8`     | supernode panel width cap (1..64)             |
| `use_multistream_subtrees` (`--no-multistream`) | `true` | 독립 subtree 를 별도 stream 으로            |
| `analyze_emit_info` (`--analyze-info`)   | `false` | front-size + subtree 분포 stderr 덤프          |
| `analyze_dump_fronts_path` (`--dump-fronts <path>`) | `""` | per-front CSV 출력                       |

CMake 옵션은 `CLS_INTERNAL_GRAPH` (default ON, capturable mode 일 때 OFF) 가 유일하게 빈번한 빌드-타임 토글이다. 정밀도 / 경로 / 디버그 덤프는 더 이상 build flag 가 아니다.

### 폐기된 / 삭제된 R&D 변형 (재시도 시 docs 필독)

- **M2 persistent subtree-walking** — prototype wall +240~820% 회귀, 코드 전체 삭제. docs/19, docs/20 (`.deprecated.md`).
- **V1 PTX_VARIANT / V3 LB_256_4 / V4 BIG_T_512 / V9 always-k4 / SMALL_WARPS_16 / factor_mid_warp / factor_mid_opt** — 모두 회귀 또는 noise. 매크로/커널 제거 (docs/10, docs/14, docs/15, docs/17, docs/18).
- **T-split (CLS_USE_TIER_SPLIT, CLS_TIER_SPLIT_B_MIN)** — case-specific opt-in (USA-target). marginal gain (B≥16 -4~-6%) + case8387 noise/회귀 + B<32 회귀로 gate 필요. dispatch path 복잡도 비례 ROI 낮음 → 코드 제거 (2026-06-07). docs/12 (`.deprecated.md`) 의 측정 데이터만 보존.
- **연구용 env 변수 5종** (`CLS_CAP`, `CLS_DUMP`, `CLS_PANEL_DUMP`, `CLS_TREE_INFO`, `CLS_DUMP_FRONTS`, `CLS_DISABLE_MULTISTREAM`) — SolverConfig / CLI 로 이전 후 env 경로 제거 (2026-06-08).
- **`CLS_NO_CP_ASYNC`, `CLS_NO_ROWFUSED_LU`, `CLS_NO_RECIP_PIV`** — A/B 토글로 노출돼 있던 baked-in 최적화. macros 제거, 코드는 항상 ON (2026-06-08).

## 01. Orientation

1. [API and Build Design](01-orientation/01-api-and-build-design.md)
   cuDSS-like phase API, 빌드 옵션, 복사된 소스 인벤토리.

2. [Related Work and Novelty](01-orientation/02-related-work-and-novelty.md)
   GPU sparse direct solver landscape와 본 작업의 novelty/한계.

3. [Lineage: STRUMPACK Is Not the Baseline](01-orientation/03-lineage-strumpack-not-the-baseline.md)
   코드 lineage 정정, STRUMPACK 한계 L1-L9와 본 솔버의 대응 매핑.

4. [Batched Precision and Dispatch Map](01-orientation/04-batched-precision-and-dispatch-map.md)
   5개 정밀도 모드 × front-size tier × env/CMake lever 의 단일 dispatch 표. 리팩토링 베이스라인.

6. [Factorize / Solve File Layout 2026-06-06](01-orientation/06-factorize-file-layout-2026-06-06.md)
   `src/factorize/` 11파일 → **4파일** + `src/solve/` 3파일 → **4파일** 통합 정리 (양쪽 모두 phases / kernels / dispatch / scatter or permute 대칭 패턴). multifrontal.cu (627줄 → 280줄) 의 factor + solve dispatch 함수가 각 모듈로 이동. 옛 경로 → 새 경로 매핑 표. mid_warp/mid_opt 실험 커널이 `deprecated/` 로 이동.

7. [Claude Refactor Instructions 2026-06-08](01-orientation/07-claude-refactor-instructions-2026-06-08.md)
   CUTLASS/NVIDIA 관례, Google C++ Style Guide, Doxygen 스타일을 이 코드베이스에 맞춘 간결한 리팩토링 규칙. 모듈 순서, 네이밍, CUDA 파일 역할, canonical dispatch policy, 함수 분리, 문서화 기준을 정의.

## 02. Design Analysis

1. [Why Custom Is Fast on Power Grid](02-design-analysis/01-why-custom-fast-on-power-grid.md)
   CUDA Graph, 3-tier kernel routing, no-pivot, device-resident solve 등 D1-D8 설계 분해.

2. [Acceleration Mechanism Ranked](02-design-analysis/02-acceleration-mechanism-ranked.md)
   D1-D8 중 실제 성능 기여를 순위화한 문서.

3. [No-Pivoting Empirical Proof](02-design-analysis/03-no-pivoting-empirical-proof.md)
   power-grid NR Jacobian에서 pivoting 없이 정확한 이유와 실패 경계.

4. [Multifrontal Layout and Level Batching vs STRUMPACK](02-design-analysis/04-multifrontal-layout-and-level-batching-vs-strumpack.md)
   front layout, level batching, extend-add fusion, no-pivot, solve 경로 차이를 소스 레벨로 비교.

5. [GEMM Fraction Analysis](02-design-analysis/05-gemm-fraction-analysis.md)
   trailing GEMM의 이론 FLOP 비중과 실측 wall 비중을 분리해 TC headroom을 정량화한 최신 근거 문서.

6. [Tier Threshold Rationale 2026-06-06](02-design-analysis/06-tier-threshold-rationale-2026-06-06.md)
   SMALL_THRESH=32, MID_THRESH=128 의 design 근거. SMALL은 warp=32 lane HW alignment + sync cost trade-off, MID는 sm_86 99 KB shared budget + 256 thread block 효율. FP64 자동 fall-through, WMMA tile constraint (power-grid 에서 non-binding), sm_90 Hopper 시 MID 상향 여지 분석 포함.

## 03. Optimization Notes

1. [FP32 Batched Kernel Optimization](03-optimization-notes/01-fp32-batched-kernel-optimization.md)
   B=64-128 batched 모드, FP32-native 경로, small/mid/big front routing 측정.

2. [Factor/Solve/Analyze Optimization](03-optimization-notes/02-factor-solve-analyze-optimization.md)
   analyze, factorize, solve를 함께 줄인 종합 최적화 리포트.

3. [Analyze Phase Optimization](03-optimization-notes/03-analyze-phase-optimization.md)
   `Solver::analyze()` 단계별 병목과 세부 최적화 기록.

4. [Tensor-Core Factor Design](03-optimization-notes/04-tensor-core-factor-design.md)
   Tensor Core/TC32 시도와 왜 끄는지에 대한 초기 음의 결과.

5. [Mysolver Warm-Cache Port Plan](03-optimization-notes/05-mysolver-warm-cache-port-plan.md)
   `perf/warm-cache-stack` 기법을 본 솔버/cuPF Mixed 경로에 적용하는 계획.

6. [(Archived) TC Dedicated Path Study](03-optimization-notes/archive/06-tc-dedicated-path-study.md)
   TC 전용 경로 초기 설계와 negative result. 최신 TC 판단은 final report와 GEMM fraction 문서를 우선 참고.

7. [(Archived) Symbolic GEMM Research](03-optimization-notes/archive/07-symbolic-gemm-research.md)
   symbolic 재구성, staged trailing, cuBLAS/pivoting 연구 로그와 정정된 가설.

8. [(Archived) Tree Restructuring Research Plan](03-optimization-notes/archive/08-tree-restructuring-research-plan.md)
   subtree/spine/multistream 연구 로그, 보류된 sibling amalgamation, race condition 기록.

9. [Non-GEMM Sync Bottleneck Plan 2026-06-06](03-optimization-notes/09-non-gemm-sync-bottleneck-plan-2026-06-06.md)
   factor_mid/big의 panel-LU `__syncthreads` 41% 병목 해소 4-phase 계획 (T4.1 warp-per-front mid, T4.2 sub-block, T4.3 cp.async stage-in, T4.4 persistent within-level). 이전 실패한 시도와의 차별성, Go/No-Go 기준 포함.

10. [T4.1+T4.3+T4.2.A Results 2026-06-06](03-optimization-notes/10-t4.1-t4.3-results-2026-06-06.md)
    docs/09 계획의 T4.1 (mid_warp) + T4.3 (cp.async stage-in) + T4.2.A (row-fused panel LU) 구현 후 실측. T4.1: barrier 41%→0% 확인하나 load imbalance로 occupancy 추락 → 기본 OFF env opt-in 유지. T4.3: USA B=1 factor_mid duration −11% (ncu) — 기본 ON sm_80+. T4.2.A: nc≤12 gate로 case8387 잠재 −7%, USA 회귀 차단 — 기본 ON. **메타-결론**: ncu barrier stall 41% → wall 단축 비례하지 않음 (~−3~−5%만). occupancy / per-thread serial work 가 sync 절감을 상쇄. 다음 round는 scatter_values / solve 또는 GEMM 영역 권장. cudaFuncSetAttribute 누락의 silent corruption 발견. **(2026-06-06 추가)** T4.1 `factor_mid_warp` 는 `deprecated/mid_warp/` 로 rollback; T4.3 cp.async + T4.2.A row-fused 는 `src/factorize/phases.cuh` 의 default kernel 에 유지.

11. [(Archived) Small-tier Packing Experiment 2026-06-06](03-optimization-notes/archive/11-small-packed-experiment-2026-06-06.md)
    analyze-time (nc, fsz, parent) 사전 정렬 + multi-front-per-warp packed kernel + nc=2 unrolled fast path. case8387 L=2의 730 fronts가 130개 contiguous run으로 잘 묶이지만, subtree multi-stream fragmentation 으로 launch 수 10-20개 폭증 → B=1 +65~+133% 회귀, B=64 case8387 +7% 회귀. USA B=64만 −5.7% 작은 win. 메타-결론: small tier 잠재 ROI는 packing이 아니라 launch overhead 줄이기 (이미 D1 CUDA Graph로 해결).

12. [(Archived, deprecated) Per-level Tier Split Experiment 2026-06-06](03-optimization-notes/archive/12-tier-split-experiment-2026-06-06.deprecated.md)
    USA-target opt-in (T-split): analyze-time fsz 오름차순 정렬 + dispatch wrapper 로 mixed 레벨을 small/mid/big sub-range 로 쪼개 적합한 kernel 로 routing. USA B≥16 −4~−6%, case8387 noise, B<32 회귀. **2026-06-07 폐기** — case-specific opt-in 비례 ROI 낮음, `CLS_USE_TIER_SPLIT`/`CLS_TIER_SPLIT_B_MIN` env + dispatch wrapper + analyze sort 블록 모두 제거. 측정 데이터는 historical log 로 보존.

13. [Panel LU + U-solve Bottleneck Analysis 2026-06-06](03-optimization-notes/13-panel-lu-u-solve-bottleneck-2026-06-06.md)
    mid / big tier 의 Phase 1 (panel LU) + Phase 2 (U-solve) 의 sync 수 분석 (case8387 mid 17 / USA mid 61 / USA big 61+), thread under-utilization 식별, 5개 개선안 제안. P1 (reciprocal multiply, FDIV→RCP×mul 4x cyc 단축) 실측 USA B=1 **−2%**, default ON. P2 (Phase 1+2 fusion, sync 67% 절감), P3 (parallel U-solve), P4 (bank conflict pad), P5 (warp-spec) deferred. 메타-결론: docs/10 §9 의 "sync 절감 ≠ wall 단축" ceiling 직면, GEMM/dispatch 영역이 더 큰 lever.

14. [(Archived) factor_mid_opt Experiment 2026-06-06](03-optimization-notes/archive/14-factor-mid-opt-experiment-2026-06-06.md)
    docs/13의 P1+P2+P4를 결합한 별도 커널 (`deprecated/mid_opt/mid_opt.cuh`, **2026-06-06 rollback**). P4 (shared padding) 폐기: stage-in의 integer division 오버헤드가 +85% 회귀. P3는 P2 fusion에 흡수. 최종 P1+P2 결과: USA B≥16에서 −1~−4% wall, case8387 noise/회귀. sync USA 61→22 (−64%) vs wall −1~−4% — docs/10 §9 "sync ≠ wall" 메타-결론 정량 재확인. **별도 kernel rollback**, P1 (reciprocal multiply) 만 default kernel `phases.cuh:lu_panel_factor` 에 흡수 유지. 결론: mid/big micro-optimization 한계 도달, GEMM/dispatch lever로 이동.

15. [TF32 trailing GEMM 가속 — 통합 보고서 2026-06-06](03-optimization-notes/15-tf32-ptx-trailing-experiment-2026-06-06.md)
    10개 변형 (V1-V9h) 시도, 7개 실패, **2개 채택** (V0 default + V9h opt-in stack), 나머지는 코드/매크로 삭제 (2026-06-07 정리). **default = V0** (WMMA m16n16k8 TF32 trailing) — 모든 case × B 에서 안전 winner. **opt-in = V9h stack** = `-DCLS_TF32_BIG_PTX=1 -DCLS_TF32_MMA_AREUSE=1 -DCLS_TF32_SKIP_CONVERT=1 -DCLS_TF32_MID_PTX=1 -DCLS_TF32_MID_K4_HYBRID=1 -DCLS_TF32_BIG_LB_512_2=1`: USA B=1 **-10%**, B=64 -2.7%; ACTIVSg25k **-5% B=1 / -4.1% B=64**; case8387 무영향. V9h 의 K4_HYBRID 는 dispatch 단계에서 `use_k4 = (KP_k4 < KP_k8)` heuristic 으로 mid level 마다 m16n8k4 vs m16n8k8 동적 선택 — nc=8 dominant case8387 mid (77%) 는 k8 자동 선택해 always-k4 의 +27% 회귀 회피, nc=12 dominant ACTIVSg25k (59%) 는 k4 자동 선택해 25% K-padding 절약. **삭제된 실패 매크로**: `CLS_TF32_PTX_VARIANT` (V1 m16n8k8 단독, +13% case8387), `CLS_TF32_BIG_T_512` (V4 thread cap), `CLS_TF32_BIG_LB_256_4` (V3 aggressive LB, USA B=1 +41%), `CLS_SMALL_WARPS_16` (docs/18 EXP-D, +4~+10%), `CLS_TF32_MID_K4` (V9 always-k4, case8387 mid +27%). 영구 학습 5가지: (i) m16n8k8 A 매핑이 PTX 문서 외삽과 다름, (ii) inline asm `"+f"(arr[var])` 는 var compile-time + `#pragma unroll` + break, (iii) `wmma::__float_to_tf32` 는 mma `.tf32` 와 redundant, (iv) m16n8k4 layout 단순, (v) **새 `__global__` kernel 의 `cudaFuncSetAttribute(.., MaxDynamicSharedMemorySize, 99 KB)` 등록 필수** (누락 시 graph capture mode 에서 segfault).

19. [(Archived, deprecated) Architecture R&D: Persistent Subtree-walking (M2) — Design + 측정 2026-06-07](03-optimization-notes/archive/19-m2-persistent-subtree-rd-2026-06-07.deprecated.md)
    docs/15-18 micro-opt ceiling 후 시도한 M2 (Rennich-Davis 2014) persistent subtree-walking R&D. Step 1 (etree shape 측정 + budget-fit DFS) 만 완료. **결론**: 잠재 게인 estimate 는 컸으나 (-10~-15% case8387) prototype 단계 (docs/20) 에서 실패. 본 문서는 historical design log.

20. [(Archived, deprecated) M2 R&D — Prototype Implementation Status 2026-06-07](03-optimization-notes/archive/20-m2-prototype-status-2026-06-07.deprecated.md)
    M2 prototype kernel + dispatch 구현. **실측 결과 wall +240-820% B=64 회귀** — 1-block-per-(subtree, batch) sequential 처리로 SM 자원 활용률 추락. parallel-within-subtree redesign 필요하나 비용/위험 대비 ROI 불명확 → **무기한 보류, 코드 전체 삭제 (2026-06-07)**. 본 문서는 historical R&D log; 재시도 시 필독.

21. [Solve Spine + Multi-Stream Subtree 2026-06-07](03-optimization-notes/21-solve-spine-multistream-2026-06-07.md)
    B=1 PF Jacobian 솔브 wall **−5.8% 평균** (best 9241pegase −8.6%). 두 lever 채택: (i) **spine fusion** = cnt=1 chain (root 영역 5–9 panel) 을 단일 persistent kernel 로 융합, per-level launch overhead 제거; (ii) **multi-stream subtree** = factor 의 fork/join 패턴을 솔브에 이식, subtree 별 stream 동시 진행. ncu 진단: solve kernel SM throughput 1–5%, launch+memory latency bound. 시도 후 폐기 6건 (SMALL_THRESH 확장, shared-mem staging, bwd cb-parallel regular, narrow band persistent, cooperative-grid, FP32 모드) 모두 회귀 또는 무효 — 같은 lever 재시도 시 참고. 다음 단계 후보 (시도 안 함): B≥16 batched 측정, cuPF NR-loop wall, selinv + CB batched-GEMM. 4 case 모두 residual baseline 수준 유지.

22. [FP16 PTX → Default 2026-06-08](03-optimization-notes/22-fp16-ptx-default-2026-06-08.md)
    `--precision fp16` 의 trailing GEMM 을 WMMA m16n16k16 → **PTX `mma.m16n8k16` 으로 전환** (tf32 가 V9h PTX 디폴트인 것과 대칭). `Precision::FP16` = PTX (new default), `Precision::FP16_WMMA` = legacy WMMA (`--precision fp16_wmma`). 새 `factor_mid_fp16_ptx` + `factor_big_fp16_ptx` 커널 + `trailing_update_mma_fp16_ptx` device helper (Csc readback 없는 직접 accumulator → `F`). ACTIVSg10k B=64: **fp16 PTX 0.0321 ms/sys vs fp16_wmma 0.0348 ms/sys = −7.7%**. 같은 변경 round 에 `perf/factorize-b1-10pct-codex` 의 B=1 factor 가속 (factor_small_single + factor_panel_chain_staged_single + plain-extend bit) 통합 — ACTIVSg10k B=1 factor −9%, 13659pegase −5%, 70k −6%. residual baseline 수준. 폐기: stash@{1} 의 `Precision::TC` 하이브리드 모드 (사용자 결정).

30. [Big-MB removal exposes tensor-core 2026-06-09](03-optimization-notes/30-big-mb-removal-exposes-tensor-core-2026-06-09.md)
    `CLS_NO_BIG_MB`로 big underfill의 multi-block 경로를 제거하면 **모든 big이 fused 단일 커널**이 되어 tf32/fp16이 모든 big에서 TC trailing을 쓴다 (기본 구성은 underfill에서 tf32도 scalar-MB로 빠져 TC 미사용). 측정(factor_per_sys, factorize만): no-MB 끼리 **TC가 fp32를 −1~−12% 일관되게 이김** (USA B=1 tf32 −11.3%, fp16 −12.5%). **단 caveat**: 진짜 default fp32(MB)와 비교하면 **B=1은 fp32(MB)가 최速**(TC가 +16% 짐), B=64만 TC −3~4% 승. no-MB는 절대적으로 느린 구성(MB가 둘 다 도움). 근본(thin-K memory-bound, doc 28) 불변 — no-MB는 TC를 *보이게* 할 뿐. **랩미팅=no-MB로 TC 가속 제시, 교수보고=MB 포함 전체+caveat.** LDB bank-conflict fix(a12f059) baked-in.

31. [FP16 Tensor Core fused trail+extend 2026-06-09](03-optimization-notes/31-fp16-fused-trail-extend-2026-06-09.md)
    big-front FP16 TC trailing accumulator를 parent extend-add로 직접 drain해 C-block global write/read 왕복을 제거. `case_SyntheticUSA` B=1 factorize에서 FP32 대비 **1.274x**를 serial-ND 고반복으로 확인. `CLS_FUSE_FP16_TRAIL_EXTEND=ON` 기본값, TF32 fuse는 별도 실험 옵션으로 OFF.

32. [Mid-tier FP16 Tensor Core trailing 2026-06-09](03-optimization-notes/32-mid-fp16-tensor-core-2026-06-09.md)
    mid-tier 후보(`uc>=32`, `10<=nc<=32`)에 FP16 PTX TC trailing + parent direct drain을 적용. `case_ACTIVSg25k` default parallel-ND B=1 factorize 평균 **1.195x**(0.799229→0.668790 ms/sys), `CLS_MID_FP16_TC=ON` 기본값. 8387/USA 회귀를 피하기 위해 현재 policy는 `20000<=n<80000` 에서만 mid TC를 활성화.

33. [Large-batch Tensor Core follow-up 2026-06-09](03-optimization-notes/33-large-batch-tensor-core-followup-2026-06-09.md)
    B=64/256에서 fp32 대비 1.2-1.4x를 목표로 FP16 force-all, big-only, TF32 mid TC + big fused drain, cuBLAS grouped TF32 trailing, direct-shared mid TF32, right-looking blocked TF32 mid/big, shared-resident big-low split을 재측정. 후속으로 direct-low mid TF32, 33..48/49..64 split(`CLS_MID_LOW_SPLIT`), column-owned U-solve(`CLS_TF32_COLUMN_USOLVE`), `CLS_RESPECT_PANEL_CAP`, global big-high thread-count(`CLS_BIG_TF32_THREADS_256/384`), shared big-low 512 threads, big-only `CLS_FUSE_TF32_TRAIL_EXTEND`를 비교. 최신 repeat=61 large-case 정책은 default 512-thread global big-high TF32 fallback + shared big-low 512 threads + no TF32 fuse, B64 `cap31`, B256 `cap32`: 25K **1.222/1.275x**, USA **1.241/1.231x**로 목표권. 8387은 fsz<=16 내부가 `(fsz,nc,uc)=(6,2,4),(4,2,2),(5,2,3)`처럼 K=1/2 tiny trailing dominant라 TC tile을 amortize하지 못해 여전히 미달; 9..32를 mid로 올린 tiny-TC, unique scatter, `fsz<=8` small-bucket split도 ratio 기준으로 폐기. TF32 residual은 큰 케이스에서 대체로 `~5e-2`.

34. [GPU Sparse Direct / Tensor Core Literature Survey 2026-06-09](03-optimization-notes/34-gpu-sparse-direct-tc-literature-survey-2026-06-09.md)
    8387 blocker와 large-case TC 성공을 문헌 기준으로 재정렬. Duff/Reid/Liu/Davis-Duff multifrontal, SuperLU/SuperLU_DIST supernodal, KLU/circuit sparse LU, STRUMPACK GPU/BLR, cuDSS, MAGMA/batched GEMM/batched LU, cuBLAS grouped GEMM, PTX/Ampere TF32, mixed-precision refinement, GPU ACOPF/NR power-flow, 2026 sparse-direct review까지 넓게 정리. 결론: 25K/USA large-case 정책은 dense-update 노출 관점에서 정당하지만, 8387은 K=1/2 tiny front dominant라 per-front TC가 구조적으로 맞지 않음. 다음 우선순위는 parent `extend_add` fan-in/profile 기반 재설계, 49..128/big-high grouped GEMM 제한 재검토, TF32 factor + refinement/NR 수렴성 평가.

35. [Parent Update Fan-in Profile 2026-06-09](03-optimization-notes/35-parent-update-fanin-profile-2026-06-09.md)
    `--dump-fronts` CSV에 `parent,asm_len,extend_elems` 분석 컬럼을 추가하고 8387 cap28, 25K cap31/32, USA cap31/32의 parent update 분포를 측정. 8387은 total `uc^2` extend elems 229K 중 `<=16` update가 25%, `uc>32`는 11.7%, fan-in 9+ parent는 21.2%라 고충돌 parent reduce가 지배적이지 않음. 진단 옵션 `CLS_DISABLE_EXTEND_ADD=ON`으로 parent update를 통째로 제거해도 8387 repeat=31 speedup은 cap28 B64/B256 `1.05/1.06x`, cap32 `1.10/0.99x`라 1.2x 미달. 25K/USA는 `>256` update가 56..64%로 large-case parent redesign 후보이나, 8387 TC-ratio gap을 단독으로 닫을 가능성은 낮음.

36. [8387 Tensor Core Upper-bound Check 2026-06-09](03-optimization-notes/36-case8387-tc-upper-bound-2026-06-09.md)
    8387 cap24/28/30 front dump로 TC coverage를 정량화. FLOP proxy상 strong TC 후보(`nc>=8,uc>=16`)는 49..56 fronts가 60..66%를 차지하지만, 전체 naive TF32 tile 효율은 약 13%이고 실제 wall은 thousands tiny small fronts + parent/scatter/front memory에 묶임. 진단 옵션 `CLS_DISABLE_SMALL_FACTOR=ON`을 추가해 small tier를 통째로 제거해도 B256 최고 약 `1.12x`; `CLS_DISABLE_SMALL_FACTOR=ON` + `CLS_DISABLE_EXTEND_ADD=ON` combined upper bound도 B256 cap sweep 16..32 최고 `1.16x`라 목표 미달. 추가로 `--serial-nd` + cap36..64 구조 cap 실험도 FP32부터 residual `1e37..inf`로 invalid. 결론: 8387은 local small/mid/extend knob나 단순 panel cap 확대가 아니라 deeper ordering/panelization 변화, many-front packed TC, 또는 low-fill sparse-LU 별도 branch가 필요.

37. [Many-front Packed Tensor Core Feasibility 2026-06-09](03-optimization-notes/37-many-front-packed-tc-feasibility-2026-06-09.md)
    8387 tiny-front를 여러 개 묶어 `mma.m16n8k8` 한 tile에 넣는 block-diagonal packing 가능성을 수식으로 평가. 안전한 packing은 `uc<=8`에서 `min(floor(16/uc), floor(8/uc))` fronts/tile이 한계이고 cross-front output은 버려야 함. cap24/28/30의 `small<=16` packed tile 효율은 약 **6.5%**뿐이며 useful work share도 21..22%. dominant `(6,2,4)`, `(4,2,2)`, `(5,2,3)`는 TF32 peak를 낙관적으로 잡아도 scalar 대비 이득이 불가능한 수준. 결론: 현재 tiny front를 그대로 packed TC로 구현하지 말고, 먼저 ordering/panelization으로 `uc,nc`를 키워야 함.

38. [8387 Ordering Seed Sweep 2026-06-09](03-optimization-notes/38-case8387-ordering-seed-sweep-2026-06-09.md)
    `--metis-seed` 런타임 knob를 추가하고 8387에서 serial/parallel ND seed `{1,7,17,42,99}` × cap `{20,24,28,30,32}` × B64/B256을 sweep. serial-ND repeat=7 best min(B64,B256)은 seed7 cap32의 **1.038x**뿐. parallel-ND repeat=7은 seed1 cap32 B64 **1.183x**, seed42 cap30 B256 **1.182x**처럼 한 배치만 좋아지는 점은 있었지만 best paired min은 seed42 cap32 **1.093x**. repeat=31 검증에서도 seed7 cap28 B64/B256 **1.078/1.118x**, seed42 cap32 **1.088/1.055x**로 목표 미달. 결론: cheap ordering variance는 8387 TC enabler가 아니며, 남은 후보는 더 깊은 panelization/amalgamation 변경, low-fill sparse-LU branch, 또는 8387 제외한 TC claim.

39. [8387 Sibling Panel Amalgamation 2026-06-09](03-optimization-notes/39-case8387-sibling-panel-amalgamation-2026-06-09.md)
    default-OFF diagnostic `CLS_SIBLING_PANEL_AMALGAMATE`를 추가해 같은 parent를 가진 연속 relaxed panels만 cap 안에서 병합하고, candidate `panel_parent[p] > p` 및 모든 `asm_idx>=0`이면 채택. 8387에서 sibling cap8/cap16 모두 invariant는 통과하고 panels를 약 **9%** 줄이며 padded fill은 **1.025..1.026x**만 증가. 그러나 repeat=31 결과는 cap8 serial B64/B256 **1.037/1.015x**, cap16 serial **1.037/1.015x**, cap16 parallel **1.127/1.102x**로 목표 미달. 결론: safe sibling merge는 feasible하지만 TF32-covered work를 충분히 특화 증폭하지 못하고 FP32 공통 work/fill/scheduling variance도 같이 키움.

40. [Literature Deep Dive Follow-up 2026-06-09](03-optimization-notes/40-literature-deep-dive-followup-2026-06-09.md)
    34번 문헌 survey의 후속 deep dive. cuDSS 최신 docs/blog, SuperLU_DIST batched sparse direct, STRUMPACK/BLR, PanguLU/Caracal/GLU/Basker/KLU, CUTLASS/cuBLAS grouped GEMM, small Tensor Core batched GEMM, Tensor-Core SpMM synergy, mixed precision refinement, power-grid ACOPF/SABLE까지 확장. 결론: large 25K/USA는 현 mid/big TF32 정책 유지, 8387은 `CLS_MID_TF32_LOW_TC` dispatch size gate가 만든 low-mid force-all 진단만 닫은 뒤 per-front TC가 아니라 tree/assembly, grouped mid/big, low-fill branch, NR/refinement metric으로 이동.

41. [8387 Low-mid TF32 Force-all Diagnostic 2026-06-09](03-optimization-notes/41-case8387-low-mid-tf32-force-all-2026-06-09.md)
    default-OFF `CLS_MID_TF32_LOW_TC_FORCE_ALL`을 추가해 8387(`n=14908`)에서도 33..48 low-mid direct-shared TF32 경로를 실제로 dispatch하도록 진단. cap32 seed42 front dump 기준 새 low-TC 후보는 `fsz>16, uc>=16, 4<=nc<=32` 69개, 33..48 band는 18개뿐. repeat=7에서는 seed7 cap28 B64/B256 **1.110/1.128x**가 보였지만 repeat=31에서 **0.996/1.055x**로 붕괴. seed42 cap32 repeat=31도 **1.131/0.985x**. 결론: 기존 size gate는 blind spot이었지만, 켜도 8387 paired B64/B256 목표에는 부족하므로 low-mid per-front TF32는 종료.

42. [Extend-add uc-bucket timing diagnostic 2026-06-09](03-optimization-notes/42-extend-add-uc-bucket-timing-2026-06-09.md)
    default-OFF timing-only `CLS_EXTEND_SKIP_UC_LE/GT`를 추가해 parent extend-add를 `uc<=16`/`uc>16` bucket으로 생략. 8387 cap32 seed42 dump는 extend elems의 `uc<=16` 69.4%, `uc>16` 30.6%로 분리되지만, repeat=31 결과는 `uc<=16` 제거 후 최고 **1.098x**, `uc>16` 제거 후 최고 **1.065x**로 B64/B256 목표 미달. 25K/USA sanity check는 full TF32가 이미 이기며 bucket skip은 FP32/TF32를 같이 줄이는 형태. 결론: parent extend-add는 8387의 숨은 TC enabler가 아니며, 다음은 8387 제외 claim / upstream panelization / 별도 low-fill branch 중 선택.

43. [case13659 TC follow-up 2026-06-09](03-optimization-notes/43-case13659-tc-followup-2026-06-09.md)
    `/tmp/cls_missing_nr/case13659pegase`에서 stable, low-mid TF32 force-all, seed/cap sweep, mid direct-fuse, blocked mid TF32, `CLS_SMALL_FRONT_MAX_16`, cuBLAS grouped mid, FP16 force-all, `--no-multistream`을 확인. stable은 cap31 B64/B256 **1.014/1.132x**, low-mid force-all repeat=31 best는 cap31 **1.188/1.112x**, repeat=7 seed/cap sweep도 valid cap<=32 paired min 최고 약 **1.073x**. mid-fuse/blocked/small16/cuBLAS/FP16/no-multistream 모두 paired target 실패. 결론: 13K도 local per-front TC toggle로는 목표권이 아니며, accepted TC claim은 25K/USA 중심으로 유지.

44. [Tensor Core target pass/fail matrix 2026-06-09](03-optimization-notes/44-tc-target-pass-fail-matrix-2026-06-09.md)
    목표 대비 accepted evidence를 정리. stable large-case TF32 정책은 repeat=61 기준 25K B64/B256 **1.222/1.275x**, USA **1.241/1.231x**. 추가로 70K `/tmp/cls_missing_nr/case_ACTIVSg70k` seed99 cap29 repeat=61에서 B64/B256 **1.257/1.257x**를 확인. 반대로 8387/13K는 low-mid force, seed/cap, extend bucket, small16, cuBLAS, FP16 등 local per-front TC 토글이 모두 paired target 실패. 결론: TC 목표는 25K/70K/USA large cases에서 달성, 8387/13K 포함은 upstream panelization 또는 low-fill branch가 필요.

45. [Expanded Reference Map 2026-06-09](03-optimization-notes/45-expanded-reference-map-2026-06-09.md)
    문헌군을 annotated bibliography 형태로 재정리. Liu/Davis-Duff sparse-direct foundations, cuDSS vendor baseline, STRUMPACK/SuperLU_DIST GPU multifrontal, KLU/Basker/GLU/PanguLU/Caracal low-fill sparse-LU, PTX/cuBLAS/CUTLASS/small batched GEMM, sparse-derived Tensor Core synergy, mixed-precision refinement, power-flow/ACOPF/SABLE을 현재 의사결정과 매핑했다. 결론: large 25K/70K/USA TC claim은 유지, 8387/13K는 per-front tiny TC가 아니라 symbolic shape 변화 또는 low-fill branch 없이는 목표권이 아님.

46. [Panel-chain amalgamation diagnostic 2026-06-09](03-optimization-notes/46-panel-chain-amalgamation-2026-06-09.md)
    default-OFF `CLS_PANEL_CHAIN_AMALGAMATE`를 추가해 parent-child 연속 panel chain만 cap 안에서 병합하고 multifrontal invariant를 검증. 8387/13K cap32 seed42에서 panel 수는 약 `1..2%`만 줄고 padded fill은 `~1.01..1.03x`, final dump는 여전히 `fsz<=16`이 8387 `97.2%`, 13K `97.7%`. repeat=31 검증은 8387 seed7 cap28 B64/B256 **1.091/0.941x**, 13K seed7 cap31 **1.112/1.004x**, seed99 cap31 **1.042/1.110x**로 paired target 실패. 결론: safe chain merge도 8387/13K TC enabler가 아니며, 남은 구조 후보는 nonlocal panelization 또는 low-fill sparse-LU branch.

47. [Sibling + chain amalgamation combo 2026-06-09](03-optimization-notes/47-sibling-chain-amalgamation-combo-2026-06-09.md)
    `CLS_SIBLING_PANEL_AMALGAMATE`와 `CLS_PANEL_CHAIN_AMALGAMATE`를 함께 켠 가장 강한 보수적 symbolic merge 조합을 확인. cap32에서는 8387 panel이 6404, 13K panel이 8776 수준까지 줄지만 여전히 `fsz<=16`이 8387 `96.89%`, 13K `96.35%`. repeat=31 검증은 8387 best 후보도 **0.982/1.007x**, **1.009/1.032x**, 13K best 후보도 **1.028/1.117x**, **0.980/0.976x**로 target 실패. cap40/48은 FP32부터 relres가 수백 이상으로 invalid. 결론: conservative amalgamation 계열은 종료, 다음은 nonlocal symbolic redesign 또는 low-fill sparse-LU branch.

48. [Batch-dimension packed Tensor Core feasibility 2026-06-09](03-optimization-notes/48-batch-dimension-packed-tc-feasibility-2026-06-09.md)
    B=64/256의 배치 차원을 Tensor Core tile 축으로 쓰는 접근을 수식과 현재 front arena layout 기준으로 평가. 표준 GEMM처럼 batch를 dense 축에 넣으면 `L_b * U_{b'}` cross-batch 항이 생겨 factorization이 틀리므로, 안전한 형태는 독립 batch/front를 block-diagonal로 pack하고 off-diagonal 출력을 버리는 방식뿐이다. 이 경우 `mma.m16n8k8` 효율은 docs/37 many-front packing과 동일하게 dominant `(6,2,4)`, `(4,2,2)`, `(5,2,3)`에서 `~3..6%` 수준. 결론: multi-batch는 후보 수와 launch amortization에는 도움되지만 tiny-update TC underfill을 해결하지 못하므로 구현하지 않고, nonlocal panelization / low-fill sparse-LU / large-case scoped TC claim으로 이동.

49. [Nonlocal panelization and parallel-ND threshold diagnostic 2026-06-09](03-optimization-notes/49-nonlocal-panelization-parnd-threshold-2026-06-09.md)
    default-OFF `CLS_VALIDATED_RUN_PANEL_AMALGAMATE`와 parallel-ND recursion/base-threshold CMake knobs를 추가해 8387/13K 구조 lever를 확인. 임의 연속 panel greedy merge는 cap `2/8/16/32` 모두 `multifrontal_symbolic` invariant를 통과하지 못해 fallback됐다. ND threshold extreme(deep `small=1000,large=4000`, shallow `small=8000,large=30000`)도 front 분포를 low-fill regime 밖으로 밀지 못했고, repeat=7 mini-sweep best는 8387 deep seed7 cap28 **1.089/0.998x**, 13K deep seed99 cap28 **1.112/1.188x**로 paired target 실패. 결론: cheap structural knobs도 종료, 남은 8387/13K 포함 경로는 closure 기반 symbolic panelization 또는 low-fill sparse-LU branch.

50. [TC-closure + mid128 + small16 low-fill pass 2026-06-09](03-optimization-notes/50-tc-closure-mid128-small16-lowfill-pass-2026-06-09.md)
    default-OFF `CLS_TC_CLOSURE_PANEL_AMALGAMATE`를 추가해 parent-front containment와 최종 `multifrontal_symbolic` invariant를 통과하고 `fsz>32, uc>=16, 4<=nc<=32`로 Tensor-Core-routable해지는 consecutive panel group만 병합. 여기에 `CLS_MID_TF32_TC_THREADS_128`와 `CLS_SMALL_FRONT_MAX_16`을 결합한 low-fill 정책이 8387/13K에서 처음으로 repeat=61 paired target을 통과했다. serial ND 기준 8387 seed7 cap30 B64/B256 **1.244/1.207x**, 13K seed99 cap32 **1.255/1.241x**. cap>32는 FP32 residual부터 invalid, `CLS_TF32_DIRECT_NTJ8_16`은 회귀. 이 정책은 25K에 universal replacement가 아니므로, 현재 결론은 low-fill(8387/13K) 정책과 large-case(25K/70K/USA) stable mid/big TC 정책을 분리하는 two-policy result.

51. [TF32 Ozaki TC2 accuracy check 2026-06-10](03-optimization-notes/51-tf32-ozaki-tc2-accuracy-2026-06-10.md)
    default-OFF `CLS_TF32_OZAKI_TC2`를 추가해 TF32 MMA 입력을 `x0=cvt.rna.tf32(x)`, `x1=cvt.rna.tf32(x-x0)`로 분해하고 `L0U0+L1U0+L0U1+L1U1`을 Tensor Core로 누적한다. global/mid direct/shared blocked/small warp TF32 경로에 적용. low-fill repeat=31에서 8387 residual **3.97e-2→4.77e-5**, 13K **6.47e-3→1.33e-4**로 FP32 band에 들어왔고, 25K도 `~2e-4` band로 복구. USA는 같은 조건 FP32가 이미 `~2.7e-2`라 Ozaki TF32가 FP32 floor와 같거나 낮다. 비용: extra MMA 3회 + 변환 때문에 8387/13K speed margin은 줄어듦.

52. [TF32 Ozaki speed follow-up 2026-06-10](03-optimization-notes/52-tf32-ozaki-speed-followup-2026-06-10.md)
    default-OFF `CLS_TF32_OZAKI_TC2_FIRST_ORDER`를 추가해 `L1U1` tail-tail 항을 생략했다. 정확도는 FP32 band를 유지하고 full TC2 대비 약 **1%** 빨라진다. low-fill repeat=31: 8387 B64/B256 **1.190/1.151x**, 13K **1.228/1.205x**. large-case repeat=7: 25K **1.240/1.258x**, USA **1.292/1.299x**. `CLS_TF32_OZAKI_STAGE_DIRECT`는 변환을 shared에 stage했지만 shared footprint/occupancy 비용이 커서 8387/13K 모두 회귀해 폐기. 결론: first-order는 채택 후보지만, 8387 B256의 원래 raw-TF32 margin이 얇아 1.2x를 복구하지 못한다.

53. [TF32 오차보정 + 혼합정밀 정제 (정확도 회복 통합) 2026-06-09](03-optimization-notes/53-tf32-error-correction-and-mixed-precision-refinement-2026-06-09.md)
    저정밀 factor 의 정확도를 회복하는 **2-레이어** 연구 통합. **레버 A (채택, opt-in)**: trailing GEMM 내부 **TF32x3→x5 오차보정**(Ootomo–Yokota). FP32 입력 hi/lo 분할(`tf32t` 10-bit 마스크, hi+lo==x 정확) + lo 2단계 분할로 13-bit 잔차 복원(5-항 mma), **모든 곱을 TC 바깥 SIMT RN 으로 누적**(TC 의 RZ 가 보정항 편향 제거) → standalone FP32 등가(~2e-7). in-solver `ACTIVSg70k` tf32 relres **0.062→0.052**, **USA 는 무효**(conditioning floor, doc 27). `CLS_TF32X3` default OFF (`100daa6`). **레버 B**: 바깥 **IR(`--ir`)/GMRES(`--gmres`)** — 저정밀 factor 를 preconditioner 로, FP64 잔차로 전체 오차 흡수, ill-conditioned 에 보편적(`a4fcbe5`/`142d486`). 두 레버 stackable. 부수 발견: trailing 진짜 병목은 **uncoalesced C-drain store 2.5×**(doc 30), thin-K mma 아님. scalar 프로브(`*_SIM`,`FP16_CB_SIM`)·fused trail+extend·`factor_mid_fp16_ptx` 위치/상태 표 포함.

29. [Panel LU / U-solve barrier-bound 2026-06-09](03-optimization-notes/29-panel-lu-usolve-barrier-bound-2026-06-09.md)
    big 커널 phase breakdown: **panel LU + U-solve = 61%**, trailing 19.5%, extend 19.2% (70K/USA 동일). ncu `factor_big_panel`: **barrier stall 12–13 지배**, long_scoreboard 2.0, **L2 hit 92%**, DRAM 1.7%, warps_active 66% → memory 아니라 **순차 pivot 의 1024-thread `__syncthreads` 체인이 임계경로**. lever: ① blocked LU(rank-b, 전폭 update sync nc→nc/b) ② block-size 256/512 축소(barrier 비용↓, 가장 싼 실험) ③ pipeline/warp-spec ④ U-solve sync 융합. TC 가설 연결: panel↓ → factor 빨라지고 trailing 비중↑로 TC 가시성↑ (단 trailing 은 thin-K memory-bound라 TC-vs-fp32 격차는 작게 유지). 다음: block-size A/B 먼저.

28. [Tensor-core ceiling: thin (K=nc) memory-bound trailing 2026-06-09](03-optimization-notes/28-tensor-core-ceiling-thin-gemm-2026-06-09.md)
    과제 목표(TC 가속) 진단. TC vs **fp32**: B=1 **+4~12% 느림**, B=64 −2~−6%(목표 미달). ncu: fused big TC tensor pipe **0.2%**, trailing 분리해도 **0.4%**(batched-TC 전제 반증, long_scoreboard 11=memory). 근본: big trailing = **C[uc×uc]−L[uc×nc]·U[nc×uc]**, 실측 **M=N=uc≈140, K=nc≈20** → mma m16n8k8 기준 **K=2.5 tile뿐**이라 staging 이 mma 압도(memory-bound). amalgamation 으로 front 키워도 uc 만 커지고 **K=nc(panel width)는 안 커짐** → TC 근본적으로 못 살림. batched-TC-trailing 구현·측정 후 revert(dead-end). 다음 lever: panel LU/U-solve(doc 29).

27. [Big-trailing L/U staging + USA fp32 conditioning floor 2026-06-09](03-optimization-notes/27-big-trailing-lu-staging-and-usa-fp32-conditioning-2026-06-09.md)
    (1) `factor_big_staged` (commit 23ec691): scalar big trailing(fp64/fp32)이 L/U 패널을 global 에서 ~uc 배 중복 read 하던 것을 shared 에 1회 staging(`trailing_update_staged`). 산술강도 1/16 FLOP/byte → memory-bound. filled big(B=64): **factor_big −23%, end-to-end factor −11%(USA fp64)/−5.7%(25K)/−5.9%(8387)**, fp32 −1.6%. budget 초과 scalar fallback, `CLS_NO_BIG_STAGED` 토글. (2) **USA fp32 multi-batch 의 큰 relres 는 데이터셋 문제**: cuDSS 레퍼런스도 USA fp32 floor **~1e-3** (fp64 e-12) — Jacobian 이 fp32 로는 ill-conditioned. 우리 relres 변동(1e-3↔0.02–0.06)은 floor(데이터셋) + parallel-ND ordering 변동(사전 존재, staging 무관)의 합. 후속 lever: fp64/mixed, deterministic ordering, iterative refinement(docs/18 M5). **교훈**: flaky 케이스에서 baseline 대조 없이 correctness 판단 금지 — 유효한 −11% 최적화를 revert 직전까지 갔다.

26. [Mid tier 분할 + front-per-block packing 연구 2026-06-09](03-optimization-notes/26-mid-tier-split-and-packing-research-2026-06-09.md)
    mid tier 를 `front_bucket()` 으로 mid-low(≤64)/mid-high bucket 분할 (commit 073ab65, kNumTiers 3→4) — heterogeneous mid level 의 작은 front 가 tighter `fsz_cap` 으로 dispatch, mid-low 점유율 28%→73% (shared 29KB→10KB). **그러나 factor_mid wall time 불변** (83.0→82.5ms). 진단: factor_mid 는 B=64 factor 의 37% 지배 커널이고 **latency-bound** — ncu 점유율 73% 인데 stall long_scoreboard 8.4 (global load 지연) 잔존, SM 22% / DRAM 16%, load 합착(1.83 sec/req)·L2 hit 42% (cold→DRAM). **이론: phase-aligned staging stall** — `stage_in_async` 의 `__pipeline_wait_prior(0)` 가 모든 block 을 동시 위상으로 메모리 대기시켜 occupancy(=naive packing) 가 못 채움. **추후 대상: pipelined front-per-block packing** (stage f+1 ∥ factorize f 로 위상 정렬 파괴) — 판별 예측: pipelined 만 long_scoreboard 감소, naive packing 무효(occupancy 와 동일, 이미 음성). mid-split 의 shared headroom 이 packing 전제. 일반화: staged front 커널(mid+big) 의 Little's-law 위상정렬 잠복 은닉 문제.

23. [Fsz-band-split factor dispatch 2026-06-08](03-optimization-notes/23-fsz-band-split-dispatch-2026-06-08.md)
    factor dispatch 의 launch SHAPE 만 바꾼 추상 정리. *"한 launch 의 일감을 균질화"* 가 본질 — 같은 (subtree, level) 안 panel 을 fsz-band (`≤32 / ≤48 / ≤64 / ≤96 / ≤128 / >128`) 로 stable-sort 후 같은 band 끼리만 모아 sub-launch. 효과 출처 두 갈래: (E1) shared `fsz_cap²` 정합성, (E2) tier 경계 (small/mid/big) 를 가로지르는 panel 의 재라우팅 — mixed level 의 tiny front 가 mid kernel 에서 small kernel 로 풀려나옴 (이게 dominant). 활성 게이트: `n ≥ 40K → B>1`, `16K ≤ n < 40K → B ≥ 128`, 그 외 OFF. **실측 대 예상**: 70K/USA B=64/256 wall **−11~15%** (예상 범위), 25K B=256 wall **−12%**, 13K B=256 direct no-GEMM **−30.1%** (예상 상한, thin margin), B=1 영향 0. doc 9 의 14 reverted experiments (fine band split, packed warp 4/16, `__launch_bounds__`, scatter 512 block, `__ldg`, contiguous `asm_local` fast path 등) 가 이 layer 의 *local optimum* 확정. 5 함수 (`dispatch_fsz_band` 분류, `wants_factor_band_order` 정책, `build_factor_band_order` 데이터, `use_factor_band_order` 가드, dispatcher 의 split-loop 실행) SRP 분담. 다음 30% 는 이 layer 밖 (25K mid-front packing / 70K plan-etree 재설계).

18. [(Archived) Small/Mid Memory-bound 분석 2026-06-07](03-optimization-notes/archive/18-small-mid-memory-bound-2026-06-07.md)
    docs/16 의 small (memory-latency, scoreboard 203%) + mid (sync+memory, scoreboard 257%) 의 memory-bound 출처 세부 진단. ncu DRAM/L1 breakdown: **mid 는 DRAM 580 GB/s = peak 의 69% (read 313 + write 267) — bulk stage-in/writeback 이 주범**, atomic 은 0 sectors/ns (extend_add contention 가정 reject). small 은 DRAM 32% 만 → memory-latency hiding 부족. **EXP-D 실측**: SMALL_WARPS 8 → 16 (docs/16 S1 가설) **wall +4 ~ +10% 회귀**. 원인 — small kernel 의 warp 들이 독립적이라 (1 warp = 1 front-batch) block 키워도 warps_in_flight 안 늘어남, smem 만 2× → S1 가설 reject. 남는 lever (모두 본 라운드 미실행): M1 front packing (5-10% 예상), M2 persistent subtree-walking (Rennich-Davis 2014; 5-10%), M5 정밀도 강하 (Ootomo-Yokota correction; 5-10% trade off). 결론: **본 codebase 의 small/mid memory simple micro-opt 는 거의 다 ceiling 도달**. 진짜 lever 는 architecture-level (weeks of work). docs/15-17 의 GEMM/occupancy micro-opt 와 함께 ceiling 시리즈의 마지막 점 marking.

17. [Big-tier Occupancy via `__launch_bounds__` 2026-06-07](03-optimization-notes/17-big-tier-occupancy-launch-bounds-2026-06-07.md)
    docs/16 의 BG3 (bigT 256/512 + launch_bounds) 부분 실행. **EXP-B 결과**: factor_big_tf32 의 `__launch_bounds__(512, 2)` → barrier stall **1801% → 1340% (-26%)**, register/thread 48→64 (compiler spill 허용), warps_active 거의 동일. wall: USA B=64 **-2.7%**, B=256 **-3.4%**, case8387 B=1 **-6.3%**, regression 없음. LB(256, 4) aggressive 변형은 warps_active 49% 추락 + **USA B=1 +41% catastrophic regression** → 폐기. **V9h (docs/15) 와 결합 시 USA B=64 -5.7%, ACTIVSg25k B=64 -3.1%, B=256 -4.1%** — orthogonal lever (big tier vs mid tier). **EXP-A (warp-spec look-ahead pipeline, docs/16 BG1) 본 라운드 미실행**: prior art 없음 (Rennich-Davis 2016 명시적 unsolved), 본 codebase 의 sync→wall 변환률 ~1/10 일관 (docs/14 + 본 문서) → BG1 의 20-40% 예측이 5-10% 로 조정, 300+ 줄 위험 비례 ROI 낮음. EXP-A 단계적 분해 plan 은 §5.4 — A-step2 (1 warp panel + 31 idle, no-pipeline 측정) 부터 시작 권장.

16. [Large-batch Bottleneck Analysis 2026-06-06](03-optimization-notes/16-large-batch-bottleneck-analysis-2026-06-06.md)
    docs/15 V9h 의 wall 게인이 noise 수준 (2-5%) 에 머문 이유 + 진짜 lever 위치. **B-sweep** (B=1..256) saturation point: small B≈256 (latency-bound), mid B≈64 (sync+memory), big B≈16 (extreme sync). **ncu B=64** stall: small barrier 0% / scoreboard 203% (memory latency), mid barrier 94% / inst/cycle 0.42, **big barrier 1907% / inst/cycle 0.26 / DRAM 23%**. ncu 의 1907% barrier stall = 32 warp 중 평균 19 warp 가 sync 대기 → big tier 의 1 block/SM thread cap 과 phase 1 panel LU 의 sequential 구조 결합 → **big_tf32 의 컴퓨트 자원 90% 가 sync 로 묶임**. **GEMM micro-opt 의 ceiling**: trailing GEMM 자체가 wall 의 30% × 이론 max 가속 2× = ceiling 15%, V9h 실측 5-10% 가 그 절반. 진짜 lever 는 **non-GEMM 의 70%, 특히 phase 1 panel LU 의 sync wait** — 본 문서가 tier 별 제안 제시 (small: warps/block 8→16 (S1), mid: front packing M1, big: **warp-specialized panel LU BG1 — 20-40% wall 잠재, docs/13 의 P5 deferred 재활성화**). USA 같이 big dominant case 에 BG1 우선, ACTIVSg25k 같이 mid dominant 에 M1.

## 04. Benchmarks and Profiling

1. [STRUMPACK Paper Table 2 Reproduction](04-benchmarks-profiling/01-strumpack-paper-table2-reproduction.md)
   STRUMPACK 논문 Table 2 행렬의 RTX 3090 재현 시도.

2. [STRUMPACK vs cuDSS Wall vs Kernel](04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md)
   power-grid case 위 wall-clock과 GPU kernel-only timing 분리.

3. [STRUMPACK NR Loop Nsys Profile](04-benchmarks-profiling/03-nsys-strumpack-nr-loop-profile.md)
   STRUMPACK 단독 NR 2-iter Nsight Systems 프로파일.

4. [Three-Solver NR Loop Nsys Profile](04-benchmarks-profiling/04-nsys-three-solvers-nr-loop-profile.md)
   STRUMPACK, cuDSS, custom을 같은 NR steady-state 시나리오에서 비교.

5. [STRUMPACK vs Custom Multifrontal on case8387](04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md)
   같은 multifrontal 계열인데 power-grid Jacobian에서 custom이 빠른 이유를 ncu/front-size 분포로 분석.

6. [Single-Batch Bottleneck on case8387](04-benchmarks-profiling/06-single-batch-bottleneck-case8387-fp64.md)
   pre-batched single-system path의 dispatch/level bottleneck, cap sweep, FP64/FP32 비교.

7. [Batched Bottleneck FP64 case8387 B=1..256](04-benchmarks-profiling/07-batched-bottleneck-fp64-case8387-b1-b256.md)
   FP64 uniform-batch factor/solve 분리, B별 kernel 분포와 ncu bound 분석.

8. [Batched Throughput FP32 case8387 B=2..1024](04-benchmarks-profiling/08-batched-throughput-fp32-case8387-b2-b1024.md)
   FP32 batched factorize throughput saturation, SM/DRAM sweep, small/mid kernel 병목.

9. [Batched Memory-Bound case8387/USA B=4,64,256 (FP64)](04-benchmarks-profiling/09-batched-membound-case8387-usa-b4-b64-b256.md)
   FP64 강제, case8387 / SyntheticUSA × B=4/64/256 의 dominant 커널 bound 분류 (memory / compute / latency) 와 메모리 바운드 완화 후보 (M1-M6).

10. [Batched Memory-Bound case8387/USA B=4,64,256 (FP32)](04-benchmarks-profiling/10-batched-membound-case8387-usa-b4-b64-b256-fp32.md)
    pure FP32, 같은 (case × B) 위의 bound 재분류. FP64 의 memory wall 이 FP32 에서 어떻게 *invert_pivot 의 FP64 compute* 로 이동하는지 직접 비교.

11. [FP32 Factorize GEMM vs Non-GEMM 2026-06-06](04-benchmarks-profiling/11-fp32-factorize-gemm-vs-nongemm-2026-06-06.md)
    refactored 코드 기준, case8387/USA × B=1/64에서 trailing GEMM과 비-GEMM의 wall 분리 (skip-trailing 차분). 이론 FLOP 71-77% vs 실측 wall 11-41% 격차의 원인 분석과 aggressive TC 활용 4가지 제안 (T1-T4), `--precision tf32` 구현 메모.

12. [Front and GEMM Size Distribution 2026-06-06](04-benchmarks-profiling/12-front-gemm-distribution-2026-06-06.md)
    case8387/USA의 fsz, nc, uc 분포와 trailing GEMM 크기를 small/mid/big tier별 히스토그램으로 분해. nc=2 (small) / nc=8 (8387 mid) / nc=20 (USA mid+big) 의 모드, WMMA padding (FP16 K=16: 37-50% padding vs TF32 K=8: 0-17% padding) 정량화. tier별 factorize 구조 ASCII 시각화 + ncu 병목 + 최적화 여지 ROI 종합. T1 TF32 권장의 핵심 정량 근거.

13. [Multi-stream Tier Impact 2026-06-06](04-benchmarks-profiling/13-multistream-tier-impact-2026-06-06.md)
    subtree multi-stream (`num_subtrees=K=8`) A/B. case8387 B=64 −22%, USA B=64 −14%, USA B=1 −9% wall. case30/118는 single이 −6~−10% 빠름 (overhead 우세). 메커니즘: **Hyper-Q 동시 커널 실행으로 SM의 idle cycle을 다른 stream의 block이 채움**. per-kernel time 자체가 단축 (case8387 mid −40%, small −16~−22%). barrier stall이 큰 경우 가장 효과, memory bandwidth contention이 우세하면 회귀. K=8이 현 워크로드에 적절.

## 05. Reports

1. [Final Report 2026-06-05](05-reports/01-final-report-2026-06-05.md)
   최신 전체 요약, 최적 dispatch, 권장 mode, 코드 변경 요약.

2. [Comprehensive Sweep 2026-06-05](05-reports/02-comprehensive-sweep-2026-06-05.md)
   5 cases × 3 modes × 5 batch sizes full benchmark table.

3. [Session Summary 2026-06-05](05-reports/03-session-summary-2026-06-05.md)
   2026-06-04 ~ 2026-06-05 작업의 시간순 로그와 phase별 판단 기록.

4. [Benchmark vs cuDSS 2026-06-07](05-reports/04-bench-vs-cudss-2026-06-07.md)
   bus-count > 1K MATPOWER NR Jacobians (case1197~case_SyntheticUSA) 위에서 cuDSS fp32 vs custom {fp32, fp16, tf32} × B∈{1, 4, 32, 64, 256} sweep. case8387 B=1 fp32 의 subtree 멀티스트림 ON/OFF nsys 프로파일 동봉. cuDSS 는 B=1 single-system 으로만 측정 (5.8~44.7× 의 speedup 은 cuDSS 가 batching 없는 baseline 이라 과대평가). NVTX timeline 으로 multi-stream overlap 시각 확인.

5. [Benchmark vs cuDSS ubatch+mt-auto 2026-06-07](05-reports/05-bench-vs-cudss-ubatch-2026-06-07.md)
   docs/04 의 후속. cuDSS 를 **`CUDSS_CONFIG_UBATCH_SIZE` (uniform batch) + `cudssSetThreadingLayer` (mt-auto)** 로 설정해 cuPF production 동작과 동일하게 측정. 11 케이스 (case1197~case_SyntheticUSA, n 2.4K~156K) × {cuDSS-ubatch-mtauto-fp32, custom fp32/fp16/tf32} × B∈{1,4,32,64,256}. 단일배치 / 다중배치 섹션 분리, analyze 도 포함. cuDSS 가 실제 batching 으로 가속 받은 후의 win 은 3.6~10.2× (B=256, custom/tf32 기준), 큰 케이스일수록 saturate.

8. [B=1 factorize 10% progress 2026-06-07](05-reports/08-factorize-b1-10pct-codex-progress-2026-06-07.md)
   `perf/factorize-b1-10pct-codex` worktree 의 exact B=1 factorize 변경 기록. `scatter_values_unique`, B=1 full factor graph, `factor_small_single`, FP16 B=1 scalar mid, spine panel-chain fusion, cap18 policy. 3012/13659 strict best-mode 10%+ 달성, 여러 fp16 mode-specific 30% 내외 개선, 회귀/폐기 실험 목록 포함.

9. [Factorize non-GEMM 30% target 2026-06-08](05-reports/09-factorize-non-gemm-30pct-2026-06-08.md)
   10K+ case 에 대해 B=1 / B=64 / B=256 을 분리해 baseline vs 현재 후보를 재측정하고, 문헌 검색과 기존 실험을 엮은 설계/결과/원인 보고서. direct no-trailing 기준 `case13659pegase B=256` 은 setup-time band order 로 `0.047062 → 0.032911` **−30.1%** (repeat-301) 까지 도달했으나 margin 이 작고, 25K+/B=1 전반 목표는 아직 미완. setup-time band-split dispatch 는 regenerated 25K B=256 direct no-GEMM −12.4%, 70K/USA B=64/256 direct no-GEMM 약 −13.9%~−18.4%. 70K B=256 graph-node profile 은 `factor_small` 30 ms / `factor_mid` 20 ms / `factor_big` 14 ms / front memset 9 ms per call 로 **front-memory+small/mid 병목** 확인. extend-add 는 13K/25K/70K B=256 no-trailing 의 12.9%/20.4%/18.7% 예산. contiguous `asm_local` fast path 와 `CLS_EXTEND_PLAIN_SAFE` 전역 plain-safe split 모두 회귀라 폐기. 결론: **parent-update 는 contended atomic 문제 아니라 mostly-uncontended atomic/write traffic 문제**. 14 건 reverted micro-opt 목록 + literature 인용 (Rennich 2014, SuperLU_DIST 2024, Abdelfattah HPEC 2019, Volkov-Demmel 2008, NVIDIA cuBLAS/Ampere whitepaper) 동봉.

## 권장 읽기 순서

처음 읽는 사람:

1. `01-orientation/01-api-and-build-design.md`
2. `02-design-analysis/01-why-custom-fast-on-power-grid.md`
3. `02-design-analysis/03-no-pivoting-empirical-proof.md`
4. `05-reports/01-final-report-2026-06-05.md`

최신 성능 결론:

1. `05-reports/01-final-report-2026-06-05.md`
2. `05-reports/02-comprehensive-sweep-2026-06-05.md`
3. `02-design-analysis/05-gemm-fraction-analysis.md`

성능 측정/비교:

1. `04-benchmarks-profiling/04-nsys-three-solvers-nr-loop-profile.md`
2. `04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md`
3. `02-design-analysis/04-multifrontal-layout-and-level-batching-vs-strumpack.md`
4. `04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md`

병목 분석:

1. `04-benchmarks-profiling/06-single-batch-bottleneck-case8387-fp64.md`
2. `04-benchmarks-profiling/07-batched-bottleneck-fp64-case8387-b1-b256.md`
3. `04-benchmarks-profiling/08-batched-throughput-fp32-case8387-b2-b1024.md`

연구 로그:

1. `03-optimization-notes/archive/06-tc-dedicated-path-study.md`
2. `03-optimization-notes/archive/07-symbolic-gemm-research.md`
3. `03-optimization-notes/archive/08-tree-restructuring-research-plan.md`
4. `05-reports/03-session-summary-2026-06-05.md`

STRUMPACK과의 관계:

1. `01-orientation/03-lineage-strumpack-not-the-baseline.md`
2. `04-benchmarks-profiling/01-strumpack-paper-table2-reproduction.md`
3. `04-benchmarks-profiling/03-nsys-strumpack-nr-loop-profile.md`

내부 최적화:

1. `03-optimization-notes/01-fp32-batched-kernel-optimization.md`
2. `03-optimization-notes/02-factor-solve-analyze-optimization.md`
3. `03-optimization-notes/03-analyze-phase-optimization.md`
4. `03-optimization-notes/04-tensor-core-factor-design.md`
5. `03-optimization-notes/05-mysolver-warm-cache-port-plan.md`
