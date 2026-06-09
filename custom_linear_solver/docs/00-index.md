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
