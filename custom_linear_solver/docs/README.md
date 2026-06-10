# `custom_linear_solver` 문서 색인

> **상태**: canonical   **갱신**: 2026-06-10
> **한 줄**: tiny-front 전력망 Jacobian 용 배치 GPU multifrontal 솔버 문서의 진입점.

## 먼저 읽기

1. [`storyline.md`](storyline.md) — 연구 서사. tiny-front regime 근원에서 내려오는 **연구 기여 4개**
   (tier routing / dispatch scheduling / TC trailing / front coarsening)와 ablation 토글 매핑, 그리고
   텐서코어 기여의 정직한 분해(+6~9%).
2. [`optimal-configuration.md`](optimal-configuration.md) — 현재 최적 경로의 빌드/런타임 설정과
   토글↔메소드 매핑표, 검증 수치.

## 폴더 구조

| 폴더 | 역할 |
|---|---|
| [`01-orientation/`](01-orientation/) | API·빌드, 연구 포지셔닝, 코드 lineage, 코드 구조/관례. |
| [`02-design-analysis/`](02-design-analysis/) | 왜 빠른지, no-pivot 안전성, STRUMPACK 과의 설계 차이, GEMM/TC ceiling 근거. |
| [`03-optimization-notes/`](03-optimization-notes/) | kernel-engineering 종합, TF32/FP16 trailing GEMM, 텐서코어 조사. dead-end 는 `archive/`. |
| [`04-benchmarks-profiling/`](04-benchmarks-profiling/) | 논문 재현, STRUMPACK 대비, GEMM/front 분포, 멀티스트림 영향. |
| [`05-reports/`](05-reports/) | canonical 최종 보고서·comprehensive sweep, cuDSS 비교, factorize 진행. |
| `2606012_lab_meeting/` | lab meeting 용 case/level front-size 분포 CSV·요약 (별도 관리, 미수정). |

## 문서 목록

### 01. Orientation
- [`01-api-and-build-design.md`](01-orientation/01-api-and-build-design.md) — cuDSS-like phase API, 빌드 옵션, 소스 인벤토리.
- [`02-related-work-and-novelty.md`](01-orientation/02-related-work-and-novelty.md) — GPU sparse direct solver landscape 와 novelty/한계.
- [`03-lineage-strumpack.md`](01-orientation/03-lineage-strumpack.md) — 코드 lineage 정정, STRUMPACK 한계 L1–L9 대응 매핑.
- [`04-code-structure.md`](01-orientation/04-code-structure.md) — `src/factorize/`·`src/solve/` 파일 레이아웃(old→new 매핑) + 네이밍/스타일/SRP 관례.

### 02. Design Analysis
- [`01-why-custom-fast.md`](02-design-analysis/01-why-custom-fast.md) — D1–D8 설계 분해 + 실제 성능 기여 순위(CUDA Graph·alloc ~80%).
- [`02-no-pivoting-proof.md`](02-design-analysis/02-no-pivoting-proof.md) — power-grid NR Jacobian 에서 pivoting 없이 정확한 이유와 실패 경계.
- [`03-multifrontal-vs-strumpack.md`](02-design-analysis/03-multifrontal-vs-strumpack.md) — front layout·level batching·extend-add·solve 경로의 소스 레벨 비교.
- [`04-gemm-fraction-tc-ceiling.md`](02-design-analysis/04-gemm-fraction-tc-ceiling.md) — **canonical TC ceiling**: 이론 FLOP vs 실측 wall 비중, TC 가속 상한(~15% wall).
- [`05-tier-thresholds.md`](02-design-analysis/05-tier-thresholds.md) — SMALL=32 / MID=128 임계값의 HW 근거, sm_90 여지.

### 03. Optimization Notes
- [`01-kernel-engineering.md`](03-optimization-notes/01-kernel-engineering.md) — substrate 미세최적화·병목진단·결정로그(3단 tier 커널, 동기화, 디스패치, staging, "sync≠wall").
- [`02-tf32-trailing-gemm.md`](03-optimization-notes/02-tf32-trailing-gemm.md) — V9h PTX trailing GEMM 스택, 폐기 매크로, 영구 교훈, FP16 PTX default.
- [`03-tensor-core-investigation.md`](03-optimization-notes/03-tensor-core-investigation.md) — 텐서코어 조사 통합: large-case 성공, 8387/13K dead-end 매트릭스, Ozaki 정확도, **honest ~1.1× 정정**.
- [`archive/`](03-optimization-notes/archive/) — dead-end R&D 로그(TC dedicated, symbolic GEMM, tree restructuring, 폐기 실험). 재시도 시 회피용.

### 04. Benchmarks and Profiling
- [`01-strumpack-paper-reproduction.md`](04-benchmarks-profiling/01-strumpack-paper-reproduction.md) — STRUMPACK 논문 행렬 RTX 3090 재현 + 도메인 검증.
- [`02-strumpack-vs-custom-case8387.md`](04-benchmarks-profiling/02-strumpack-vs-custom-case8387.md) — 같은 multifrontal 인데 custom 이 빠른 이유(ncu/front-size).
- [`03-gemm-fraction-front-distribution.md`](04-benchmarks-profiling/03-gemm-fraction-front-distribution.md) — GEMM wall 비중 + front fsz/nc/uc 분포 + WMMA padding.
- [`04-multistream-impact.md`](04-benchmarks-profiling/04-multistream-impact.md) — subtree 멀티스트림 A/B(Hyper-Q) 영향.

### 05. Reports
- [`01-final-report.md`](05-reports/01-final-report.md) — **canonical** 최종 요약, 최적 dispatch, 권장 mode, 코드 변경.
- [`02-comprehensive-sweep.md`](05-reports/02-comprehensive-sweep.md) — **canonical** 5 cases × 3 modes × 5 batch sizes full table.
- [`03-bench-vs-cudss.md`](05-reports/03-bench-vs-cudss.md) — cuDSS 대비(raw + ubatch+mt-auto) sweep, 정확도, analyze.
- [`04-factorize-progress.md`](05-reports/04-factorize-progress.md) — B=1/non-GEMM factorize 가속 진행·채택 변경·tier-split gate·구조적 한계.
- [`05-tf32-reproduction-2026-06-10.md`](05-reports/05-tf32-reproduction-2026-06-10.md) — 대표 5케이스 fp32 vs tf32 **정확도-매칭 best-vs-best 재현** (honest ~1.1× 확증).

## 정직성 노트

TC(텐서코어) 가속의 헤드라인 수치(1.2~1.3×)는 일부 **baseline cap inflation** 을 포함한다. best-vs-best
공정 통제 시 정직한 천장은 large-case +6~16%, low-fill net≈0 이다(`storyline.md` §5,
`03-optimization-notes/03-tensor-core-investigation.md` §7).
