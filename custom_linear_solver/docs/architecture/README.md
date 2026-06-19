# 아키텍처 문서

> **상태**: canonical   **갱신**: 2026-06-19
> **한 줄**: `custom_linear_solver`(small-front 전력망 Jacobian용 배치 GPU multifrontal 직접 솔버)의 **코드 기반
> 동작 설명서**. 각 문서는 **추상(개념) → 상세(코드)** 두 층으로 쓰여 있고, 모든 주장은 `src/...` 심볼/파일로 앵커한다.

상위 진입점은 [`../README.md`](../README.md)(개요·API 요약·정밀도·성능). 과거 설계·실험·벤치 로그는
[`../_legacy/`](../_legacy/). 이 폴더는 "지금 코드가 어떻게 동작하는가"를 표준 선형대수·GPU 용어로 분해한다.

## 읽는 순서

처음이면 위에서 아래로. 특정 주제만 필요하면 표에서 골라 읽으면 된다.

| # | 문서 | 다루는 것 | 층위 |
|---|---|---|---|
| 01 | [`01-overview.md`](01-overview.md) | 정신 모델·핵심 가정·end-to-end 파이프라인·용어집·작은 예제 1개 | 추상 ↑ |
| 02 | [`02-source-layout.md`](02-source-layout.md) | **소스 코드 구조** — 디렉터리 맵, 파일별 책임, 모듈 간 데이터 흐름, 네이밍 관례 | 구조 |
| 03 | [`03-api-config-build.md`](03-api-config-build.md) | 공개 API(phase 모델)·`SolverConfig`·CMake 옵션·CLI·컴파일타임 매크로·실행법 | 사용 |
| 04 | [`04-memory-layout.md`](04-memory-layout.md) | plan arena vs 수치 front, `MultifrontalPlan`/`State`, front 저장 포맷, 배치 striding, 소유권 | 상세 |
| 05 | [`05-front-tiers.md`](05-front-tiers.md) | `ClassifyFrontTier` 경계(32/64)와 HW 근거, tier→커널 매핑, staging 상한 | 상세 |
| 06 | [`06-factorization.md`](06-factorization.md) | multifrontal 알고리즘, per-front 4단계(panel LU/U-solve/Schur/extend-add), 3 tier 커널, 점유 게이트 | 추상+상세 |
| 07 | [`07-solve.md`](07-solve.md) | 전진/후진 삼각 대입, gather/scatter 치환, 배치 디스패치, B=1 selinv(병렬 GEMV) | 추상+상세 |
| 08 | [`08-runtime-and-batching.md`](08-runtime-and-batching.md) | analyze 파이프라인, `Setup`/state 할당, CUDA graph 캡처, 멀티스트림, **B=1 vs B>1** | 상세 |
| 09 | [`09-precision-and-tensor-cores.md`](09-precision-and-tensor-cores.md) | FP64/FP32/TF32 매트릭스, TF32 PTX mma + Ozaki, `if constexpr` TC 분리(레지스터 0), TC 적격 | 상세 |

## 한눈에 (TL;DR)

- **무엇**: 고정 sparsity 비대칭 희소행렬 `A`를 multifrontal LU로 GPU에서 분해/solve. 타깃은 NR 전력조류 Jacobian.
- **왜 특수한가**: front(dense 부분행렬)가 극단적으로 작다(대부분 `fsz ≤ 16`). 일반 GPU 직접 솔버는 큰 dense front를
  가정하므로 이 영역에서 점유가 굶는다. 이 솔버는 그 분포에 맞춰 커널을 짠다.
- **3-tier**: front 크기 함수로 결정적 라우팅 — small(≤32) / mid(33–64) / big(>64). [05](05-front-tiers.md).
- **두 체제**: `B==1`(단일 시스템, latency 최적, selinv) vs `B>1`(배치, 처리량). [08](08-runtime-and-batching.md).
- **정밀도**: FP64(~1e-13) / FP32(~1e-4) / TF32(Ozaki mma, ~1e-4, 권장). [09](09-precision-and-tensor-cores.md).
