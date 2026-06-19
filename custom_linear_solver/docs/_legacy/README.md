# _legacy — 이전 문서 트리 (역사 보존)

2026-06-18 문서 재작성 때 **이전 living 문서 전체**를 여기로 내렸다. 현재 사실의 기준은 상위
[`../README.md`](../README.md) 와 `src/`(특히 `src/internal/types.hpp`)다. 이 폴더는 측정 원자료·실험 로그·
문헌 판정의 상세 근거를 찾을 때만 본다.

> **읽기 주의**: 날짜 박힌 로그와 옛 설계 문서는 **작성 시점 사실**이다. tier 경계가 `33–159/≥160` 또는
> `MID_THRESH=128`, panel-resident/large tier, best-of-k/GPU-ND 가 현재처럼 보이면 그건 통합·제거 **이전**
> 표기다. 현재 3-tier(small≤32 / mid 33–64 / big >64)·METIS-ND-only 가 정답.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `main-report.md` | 옛 통합 캐논 리포트(서사·기여 분석). 현재 요약은 상위 README. |
| `optimal-configuration.md` | 옛 빌드/런타임 설정·토글 매핑. |
| `08-small-front-demand-and-concept-comparison.md` | small-front 수요 + STRUMPACK 개념 비교(축1/2/3). |
| `01-orientation/` | API·빌드, related-work·novelty, 코드 lineage. |
| `02-design-analysis/` | 왜 빠른지, no-pivot 증명, multifrontal 비교, GEMM/TC ceiling. |
| `03-optimization-notes/` | kernel-engineering, TF32 trailing, 텐서코어 조사, solve 최적화, B=1/배치 factorize, ordering 실험. dead-end 는 `03-optimization-notes/archive/`. |
| `04-benchmarks-profiling/` | STRUMPACK 논문 재현, case8387 대비, GEMM/front 분포, 멀티스트림. |
| `05-reports/` | 종합 sweep, **cuDSS 병합 리포트**, head-to-head, 일반화, fair-STRUMPACK, ncu 기전, tier 통합. |
| `20260612_lab_meeting/` | 2026-06-12 동결 스냅샷(CSV·figure·요약, 통합 이전 비닝). |
| `history/` | B=1 단일 시스템 최적화 로그. |
| `novelty.html`, `batch_extension_methodology.html`, `20260618_batch_cupf_benchmark.md` | 발표/방법론 자료. |

## cuDSS 리포트 병합

옛 `03-bench-vs-cudss.md` + `06-cudss-vs-custom-sweep-2026-06-10.md` + `07-cupf-backend-comparison-2026-06-11.md`
는 [`05-reports/bench-vs-cudss-merged.md`](05-reports/bench-vs-cudss-merged.md) 한 문서로 합쳤다(세 setup —
raw B=1 / 공정 ubatch+mt / 전체 NR — 을 분리 보존). 원본 `.md` 는 제거했고, 측정 원자료(`*/sweep*.tsv`,
`*/operator_ms_full.tsv`, `*.nsys-rep`)는 각 데이터 폴더에 그대로 남아 병합 문서가 가리킨다.

## 전제가 obsolete 한 문서 (현재 코드와 정면 충돌 → 본문에서 제외됐던 것)

| 파일 | 왜 obsolete | 현재 사실 |
|---|---|---|
| `05-tier-thresholds-OBSOLETE-mid128.md` | 전체가 `MID_THRESH=128`, `MID_SHARED_BUDGET=96KB`, `src/multifrontal.cu`, `fsz>128 big` 전제 | mid\|big=**64**, small\|mid=32. 159/111 은 big 커널 내부 staging 상한일 뿐 티어 경계 아님 |
| `04-code-structure-OBSOLETE-layout.md` | `src/factorize/` 를 `phases/kernels/dispatch/scatter.cuh` 4파일 + 소문자 `factor_mid`/`factor_big` 로 기술 | 실제는 티어별 `small/mid/big/single.cuh` + `front_ops/assemble/schedule.cuh` + `factorize.cu`, 심볼 `FactorSmall/FactorMid/FactorBig` |
| `01-final-report-2026-06-10-superseded.md` | 2026-06-10 옛 canonical. `MID_THRESH=128`·옛 4-tier 의사코드 | 상위 `README.md` 로 대체 |

> 2026-06-18 tier 통합(4→3) 경위: [`05-reports/10-tier-consolidation-2026-06-18.md`](05-reports/10-tier-consolidation-2026-06-18.md).
