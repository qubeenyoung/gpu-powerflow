# `custom_linear_solver` 문서 색인

루트에는 이 색인만 두고, 세부 문서는 주제별 폴더에 읽는 순서대로 번호를 붙였다.
2026-06-04 초기 정리는 `01`-`04`에 API/설계/최적화/측정을 나눴고, 2026-06-04 오후부터 2026-06-05까지 추가된 실험 로그와 최종 보고서는 기존 흐름에 흡수하거나 `05-reports/`로 분리했다.

## 폴더 구조

- `01-orientation/`: API, 연구 포지셔닝, 코드 lineage.
- `02-design-analysis/`: 왜 빠른지, 어떤 가정이 안전한지, STRUMPACK과 설계가 어떻게 다른지, GEMM/TC wall 근거.
- `03-optimization-notes/`: 구현 최적화 기록, 연구 로그, 음의 결과.
- `04-benchmarks-profiling/`: 논문 재현, wall/kernel 측정, Nsight Systems/Compute 분석, bottleneck sweep.
- `05-reports/`: 날짜별 최종 보고서, comprehensive sweep 원본 표, 시간순 세션 로그.

## 중복 정리 기준

- 최신 전체 결론은 `05-reports/01-final-report-2026-06-05.md`를 우선 참고한다.
- 전체 benchmark table의 canonical source는 `05-reports/02-comprehensive-sweep-2026-06-05.md`다.
- GEMM/TC wall fraction과 TC speedup ceiling의 canonical 근거는 `02-design-analysis/05-gemm-fraction-analysis.md`다.
- 시간순 작업 로그는 `05-reports/03-session-summary-2026-06-05.md`에 보존했다.
- `03-optimization-notes/06`-`08`은 historical research log다. 진행 중 정정된 가설, negative result, race condition 기록을 보존하되 최신 결론의 기준점은 아니다.
- 측정 근거가 독립적인 문서는 삭제하지 않고 보존했다. 대신 색인에서 “대표 문서”, “증거 문서”, “historical log”의 역할을 분리했다.

## 01. Orientation

1. [API and Build Design](01-orientation/01-api-and-build-design.md)
   cuDSS-like phase API, 빌드 옵션, 복사된 소스 인벤토리.

2. [Related Work and Novelty](01-orientation/02-related-work-and-novelty.md)
   GPU sparse direct solver landscape와 본 작업의 novelty/한계.

3. [Lineage: STRUMPACK Is Not the Baseline](01-orientation/03-lineage-strumpack-not-the-baseline.md)
   코드 lineage 정정, STRUMPACK 한계 L1-L9와 본 솔버의 대응 매핑.

4. [Batched Precision and Dispatch Map](01-orientation/04-batched-precision-and-dispatch-map.md)
   5개 정밀도 모드 × front-size tier × env/CMake lever 의 단일 dispatch 표. 리팩토링 베이스라인.

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

6. [TC Dedicated Path Study](03-optimization-notes/06-tc-dedicated-path-study.md)
   TC 전용 경로 초기 설계와 negative result. 최신 TC 판단은 final report와 GEMM fraction 문서를 우선 참고.

7. [Symbolic GEMM Research](03-optimization-notes/07-symbolic-gemm-research.md)
   symbolic 재구성, staged trailing, cuBLAS/pivoting 연구 로그와 정정된 가설.

8. [Tree Restructuring Research Plan](03-optimization-notes/08-tree-restructuring-research-plan.md)
   subtree/spine/multistream 연구 로그, 보류된 sibling amalgamation, race condition 기록.

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

## 05. Reports

1. [Final Report 2026-06-05](05-reports/01-final-report-2026-06-05.md)
   최신 전체 요약, 최적 dispatch, 권장 mode, 코드 변경 요약.

2. [Comprehensive Sweep 2026-06-05](05-reports/02-comprehensive-sweep-2026-06-05.md)
   5 cases × 3 modes × 5 batch sizes full benchmark table.

3. [Session Summary 2026-06-05](05-reports/03-session-summary-2026-06-05.md)
   2026-06-04 ~ 2026-06-05 작업의 시간순 로그와 phase별 판단 기록.

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

1. `03-optimization-notes/06-tc-dedicated-path-study.md`
2. `03-optimization-notes/07-symbolic-gemm-research.md`
3. `03-optimization-notes/08-tree-restructuring-research-plan.md`
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
