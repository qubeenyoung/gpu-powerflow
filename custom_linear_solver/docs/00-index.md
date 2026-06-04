# `custom_linear_solver` 문서 색인

루트에는 이 색인만 두고, 세부 문서는 주제별 폴더에 읽는 순서대로 번호를 붙였다.

## 폴더 구조

- `01-orientation/`: API, 연구 포지셔닝, 코드 lineage.
- `02-design-analysis/`: 왜 빠른지, 어떤 가정이 안전한지, STRUMPACK과 설계가 어떻게 다른지.
- `03-optimization-notes/`: 구현 최적화 기록과 음의 결과.
- `04-benchmarks-profiling/`: 논문 재현, wall/kernel 측정, Nsight Systems/Compute 분석.

## 중복 정리 기준

- `02-design-analysis/01-why-custom-fast-on-power-grid.md`: 설계 결정 D1-D8을 나열하는 대표 설계 문서.
- `02-design-analysis/02-acceleration-mechanism-ranked.md`: 같은 D1-D8 중 실제 leverage 순위를 정리하는 요약 문서.
- `02-design-analysis/04-multifrontal-layout-and-level-batching-vs-strumpack.md`: graph 효과가 아니라 layout/kernel granularity 차이를 소스 레벨에서 설명하는 문서.
- `04-benchmarks-profiling/04-nsys-three-solvers-nr-loop-profile.md`: STRUMPACK/cuDSS/custom의 NR steady-state 비교는 이 문서를 우선 참고.
- `04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md`: paper 재현에서 power-grid로 넘어가는 bridge 성격의 historical 측정 문서.

측정 근거가 독립적인 문서는 삭제하지 않고 보존했다. 대신 색인에서 “대표 문서”와 “증거 문서”의 역할을 분리했다.

## 01. Orientation

1. [API and Build Design](01-orientation/01-api-and-build-design.md)  
   cuDSS-like phase API, 빌드 옵션, 복사된 소스 인벤토리.

2. [Related Work and Novelty](01-orientation/02-related-work-and-novelty.md)  
   GPU sparse direct solver landscape와 본 작업의 novelty/한계.

3. [Lineage: STRUMPACK Is Not the Baseline](01-orientation/03-lineage-strumpack-not-the-baseline.md)  
   코드 lineage 정정, STRUMPACK 한계 L1-L9와 본 솔버의 대응 매핑.

## 02. Design Analysis

1. [Why Custom Is Fast on Power Grid](02-design-analysis/01-why-custom-fast-on-power-grid.md)  
   CUDA Graph, 3-tier kernel routing, no-pivot, device-resident solve 등 D1-D8 설계 분해.

2. [Acceleration Mechanism Ranked](02-design-analysis/02-acceleration-mechanism-ranked.md)  
   D1-D8 중 실제 성능 기여를 순위화한 문서.

3. [No-Pivoting Empirical Proof](02-design-analysis/03-no-pivoting-empirical-proof.md)  
   power-grid NR Jacobian에서 pivoting 없이 정확한 이유와 실패 경계.

4. [Multifrontal Layout and Level Batching vs STRUMPACK](02-design-analysis/04-multifrontal-layout-and-level-batching-vs-strumpack.md)  
   front layout, level batching, extend-add fusion, no-pivot, solve 경로 차이를 소스 레벨로 비교.

## 03. Optimization Notes

1. [FP32 Batched Kernel Optimization](03-optimization-notes/01-fp32-batched-kernel-optimization.md)  
   B=64-128 batched 모드, FP32-native 경로, small/mid/big front routing 측정.

2. [Factor/Solve/Analyze Optimization](03-optimization-notes/02-factor-solve-analyze-optimization.md)  
   analyze, factorize, solve를 함께 줄인 종합 최적화 리포트.

3. [Analyze Phase Optimization](03-optimization-notes/03-analyze-phase-optimization.md)  
   `Solver::analyze()` 단계별 병목과 세부 최적화 기록.

4. [Tensor-Core Factor Design](03-optimization-notes/04-tensor-core-factor-design.md)  
   Tensor Core/TC32 시도와 왜 끄는지에 대한 음의 결과.

5. [Mysolver Warm-Cache Port Plan](03-optimization-notes/05-mysolver-warm-cache-port-plan.md)  
   `perf/warm-cache-stack` 기법을 본 솔버/cuPF Mixed 경로에 적용하는 계획.

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

## 권장 읽기 순서

처음 읽는 사람:

1. `01-orientation/01-api-and-build-design.md`
2. `02-design-analysis/01-why-custom-fast-on-power-grid.md`
3. `02-design-analysis/03-no-pivoting-empirical-proof.md`
4. `01-orientation/02-related-work-and-novelty.md`

성능 측정/비교:

1. `04-benchmarks-profiling/04-nsys-three-solvers-nr-loop-profile.md`
2. `04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md`
3. `02-design-analysis/04-multifrontal-layout-and-level-batching-vs-strumpack.md`
4. `04-benchmarks-profiling/02-strumpack-vs-cudss-power-grid-wall-vs-kernel.md`

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
