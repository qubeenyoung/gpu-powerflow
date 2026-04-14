# 2026-04-10 TODO

`exp/20260410/` 후속 작업 목록이다.  
현재 benchmark 결과 해석에서 특히 중요한 항목은 CUDA `analyze`와 Newton 반복 내부 연산 분해다.

## 우선순위

1. CUDA `analyze` 분석
2. Newton 반복 내부 residual 추적
3. `REFACTORIZATION` vs `SOLVE` 병목 분석
4. Nsight 분석 가능하도록 수정

## 1. CUDA에서 `analyze` 분석

### 목적

- CUDA `analyze` 시간이 왜 큰지 구조적으로 분해한다.
- 현재 `analyze`에 들어 있는 비용이 무엇인지 명확히 구분한다.
- `cold setup`과 `steady-state solve`를 나눠서 설명할 수 있게 만든다.

### 확인할 것

- `NewtonSolver::analyze()` 내부에서 CPU 공통 Jacobian 분석이 차지하는 비중
- `CudaNewtonSolverBackend::analyze()` 내부의
  - host-side 전처리
  - `cudaMalloc`
  - `cudaMemcpy`
  - cuSPARSE descriptor 생성
  - cuDSS handle/config/data 생성
  - `CUDSS_PHASE_ANALYSIS`
  - 초기 `CUDSS_PHASE_FACTORIZATION`
- 첫 실행과 이후 실행의 차이
- 작은 케이스와 큰 케이스에서 `analyze` scaling이 어떻게 달라지는지

### 산출물

- `analyze` 세부 단계 표
- 단계별 시간 합계와 비율
- `cold analyze`와 `warm analyze` 해석 메모

## 2. 솔버 안의 Newton 반복에서 residual 추적

### 목적

- 각 iteration에서 residual이 어떻게 줄어드는지 추적한다.
- CPU optimized와 CUDA edge의 수렴 경로가 같은지 확인한다.
- iteration 수 차이가 생기는 케이스의 원인을 설명할 수 있게 만든다.

### 확인할 것

- iteration index별 residual norm
- 초기 residual과 종료 residual
- 종료 판정 직전 residual 감소 패턴
- CPU optimized / CUDA edge / CUDA vertex 간 residual trajectory 차이
- `793_goc`처럼 iteration 수가 살짝 흔들리는 케이스의 원인

### 구현 방향

- `NR.computeMismatch` 시점에서 residual norm을 run별/iteration별로 기록
- raw 결과에 iteration trace를 넣거나 별도 JSON/CSV로 저장
- plotting 가능하도록 case별 trace 포맷 통일

### 산출물

- case별 residual trace
- backend별 residual comparison plot 또는 CSV
- iteration 차이 설명 메모

## 3. `REFACTORIZATION`, `SOLVE` 중 뭐가 더 병목인지 분석

### 목적

- CUDA 선형계 단계에서 병목이 정확히 어디인지 분리한다.
- 현재 `NR.solveLinearSystem` 안에 묶여 있는 시간을 더 잘게 나눠서 본다.

### 확인할 것

- `CUDSS_PHASE_REFACTORIZATION` 시간
- `CUDSS_PHASE_SOLVE` 시간
- 필요한 경우 RHS cast / negate 시간
- 케이스 크기에 따라 두 구간의 비율이 어떻게 바뀌는지
- CPU optimized의 KLU numeric factorization과 비교했을 때 병목 의미가 어떻게 다른지

### 구현 방향

- `CudaNewtonSolverBackend::solveLinearSystem()`에서
  - RHS 준비
  - `REFACTORIZATION`
  - `SOLVE`
  를 별도 타이머로 분리
- 가능하면 CPU optimized도
  - Jacobian fill
  - KLU factorize
  - triangular solve
  로 더 쪼갠다

### 산출물

- `solveLinearSystem` 세부 breakdown 표
- case별 병목 비율
- `REFACTORIZATION` 우세 / `SOLVE` 우세 케이스 분류

## 4. Nsight 분석 가능하도록 수정

### 목적

- Nsight Systems / Nsight Compute로 CUDA 경로를 제대로 볼 수 있게 한다.
- 커널, memcpy, cuDSS 호출, solver phase 경계를 프로파일링 도구에서 식별 가능하게 만든다.

### 확인할 것

- 프로파일링 시 불필요한 로그/동기화가 결과를 왜곡하지 않는지
- 관심 구간만 캡처할 수 있는지
- phase별 이름이 Nsight timeline에서 구분되는지

### 구현 방향

- NVTX range 추가
  - `analyze`
  - `computeMismatch`
  - `updateJacobian`
  - `solveLinearSystem`
  - `REFACTORIZATION`
  - `SOLVE`
  - `downloadV`
- benchmark CLI에 profiling-friendly 옵션 추가 검토
  - single-case only
  - single-repeat only
  - warmup count 제어
- 필요 시 timing build와 profiling build를 분리

### 산출물

- Nsight 실행 절차 문서
- NVTX가 포함된 trace screenshot 또는 분석 메모
- profile용 권장 커맨드

## 권장 진행 순서

1. CUDA `analyze` 세부 분해
2. `REFACTORIZATION` / `SOLVE` 분해
3. residual trace 추가
4. Nsight-friendly instrumentation 추가

## 메모

- 1번과 3번은 현재 “CUDA total이 큰 이유”를 설명하는 데 직접 연결된다.
- 2번과 4번은 실제 CUDA solver 최적화 작업의 출발점이다.
- 재측정이 필요하더라도, 먼저 어떤 값을 보고 싶은지 정의하고 나서 벤치마크 러너를 수정하는 편이 낫다.
