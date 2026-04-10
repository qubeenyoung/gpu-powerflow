
## 현재 상태 요약

- CPU 경로는 FP64 기반 기본 동작 경로로 사용할 수 있다.
- CUDA 경로는 기본적으로 혼합 정밀도와 희소 선형계 가속을 목표로 구성되어 있고, 별도 정밀도 선택 구조는 아직 정리 중이다.
- Python 모듈은 `cupf` 이름으로 빌드된다.
- 문서와 헤더 일부에는 batch 및 precision 확장 방향이 남아 있지만, 현재 작업 기준으로는 single-case 경로만 우선 지원한다.
- 정밀도 관련 수정 중에 구조 변경의 필요성을 느껴서 일단 푸시함.


# cuPF

`cuPF`는 희소 `Ybus`를 입력으로 받아 Newton-Raphson 방식으로 전력 조류(power flow)를 계산하는 C++/CUDA 기반 라이브러리다. 같은 위상 구조에서 반복적으로 재사용되는 Jacobian 희소 패턴을 `analyze()` 단계에서 미리 구성하고, `solve()` 단계에서는 mismatch 계산, Jacobian 갱신, 선형계 풀이, 전압 갱신만 반복하도록 분리해 두었다.

프로젝트 이름의 `cu`는 CUDA 가속 경로를 의미하지만, CPU 전용 빌드도 가능하다. CPU 백엔드는 Eigen 기반 FP64 경로를 사용하고, CUDA 백엔드는 기본적으로 Jacobian은 FP32로, 전압 상태와 mismatch 관련 계산은 FP64로 처리하는 혼합 정밀도 경로를 사용한다. 정밀도 선택과 연산 단계 분리(operator pipeline) 리팩터링은 아직 진행 중이다.

## 핵심 특징

- 희소 `Ybus` CSR 입력을 직접 받아 zero-copy에 가까운 인터페이스를 제공한다.
- `analyze()`와 `solve()`를 분리해 동일한 네트워크 구조에서 반복 계산 비용을 줄인다.
- 공통 `NewtonSolver` API 위에 CPU/Eigen 백엔드와 CUDA/cuSPARSE/cuDSS 백엔드를 교체 가능하게 구성했다.
- Python 바인딩을 통해 NumPy 배열 기반으로 동일한 기능을 사용할 수 있다.
- 테스트용 dump case 로더와 smoke test 실행 파일이 포함되어 있다.

## 현재 버전의 문제

- 정밀도 선택 리팩터링이 진행 중이다. C++ 코어에는 FP32/Mixed/FP64 single-case 경로가 들어가고 있지만, README/세부 문서/Python 바인딩/벤치마크 하니스는 아직 완전히 정렬되지 않았다.
- multi-batch는 임시로 범위 밖이다. 현재는 `n_batch == 1` single-case만 지원하고, `solve_batch()`는 예외를 던지도록 막아 둔 상태다.
- Python 바인딩은 아직 legacy FP64(`complex128`) 인터페이스만 지원한다. `PrecisionMode`, FP32 public API, stage/kernel variant 선택은 바인딩에 노출되지 않았다.
- 실험용 CUDA kernel을 계속 추가하기 좋은 구조는 아직 아니다. `MismatchOp/JacobianOp/LinearSolveOp/VoltageUpdateOp` 기반 operator pipeline refactor 전이어서, 새 kernel variant 실험은 여전히 backend 내부 코드 수정이 필요하다.
- 현재 smoke 테스트가 주로 덮는 것은 CPU FP64와 CUDA single-case 기본 경로다. 새 precision path와 future batch path는 추가 검증이 더 필요하다.

## 아키텍처

```text
Python / C++ 사용자
        │
        ▼
  NewtonSolver
   ├── JacobianBuilder
   │    ├── JacobianMaps
   │    └── JacobianStructure
   └── INewtonSolverBackend
        ├── CpuNewtonSolverBackend
        └── CudaNewtonSolverBackend
```

- `NewtonSolver`: 공개 API 진입점이다. NR 루프를 조율하고 백엔드 호출 순서를 관리한다.
- `JacobianBuilder`: `Ybus`, `pv`, `pq`를 바탕으로 Jacobian 희소 구조와 매핑 테이블을 만든다.
- `INewtonSolverBackend`: 실제 수치 연산을 수행하는 백엔드 공통 인터페이스다.
- `CpuNewtonSolverBackend`: Eigen SparseLU 기반 FP64 경로다.
- `CudaNewtonSolverBackend`: cuSPARSE SpMV와 cuDSS 선형 해법을 사용하는 GPU 경로다.

## 계산 흐름

```text
analyze(ybus, pv, pq)
  1. JacobianBuilder가 JacobianStructure / JacobianMaps 생성
  2. backend.analyze()가 메모리 레이아웃과 해석 준비 수행

solve(ybus, sbus, V0, ...)
  1. backend.initialize()
  2. 반복:
     - computeMismatch()
     - updateJacobian()
     - solveLinearSystem()
     - updateVoltage()
  3. backend.downloadV()
```

`analyze()`는 위상 구조가 바뀌지 않는 동안 한 번만 호출하면 된다. 이후 `solve()`는 초기 전압과 주입 전력만 바꿔 여러 번 실행할 수 있다.

## 정밀도 전략

| 백엔드 | Jacobian | 전압 상태 / 결과 |
|---|---|---|
| CPU | FP64 | FP64 |
| CUDA | FP32 | FP64 |

CUDA 경로는 Jacobian 인수분해 비용을 낮추기 위해 FP32를 사용하고, 전압 상태와 최종 결과는 FP64로 유지한다.

## 디렉터리 구조

```text
cuPF/
├── cpp/
│   ├── inc/newton_solver/
│   │   ├── core/
│   │   └── backend/
│   ├── inc/utils/
│   └── src/newton_solver/
│       ├── core/
│       └── backend/
│           ├── cpp/
│           └── cuda/
├── bindings/
├── docs/
└── tests/
```

- `cpp/inc/newton_solver/core`: 공개 타입, `NewtonSolver`, `JacobianBuilder` 헤더
- `cpp/inc/newton_solver/backend`: 백엔드 인터페이스와 클래스 선언
- `cpp/src/newton_solver/core`: 공개 API 구현
- `cpp/src/newton_solver/backend/cpp`: CPU 백엔드 구현
- `cpp/src/newton_solver/backend/cuda`: CUDA 백엔드 구현
- `bindings/pybind_module.cpp`: pybind11 기반 Python 모듈
- `docs/`: 설계 문서
- `tests/cpp`: dump case 기반 smoke test

## 주요 타입과 API

### 입력 데이터

- `YbusView`: `indptr`, `indices`, `data` 포인터를 감싸는 CSR 뷰
- `pv`, `pq`: PV/PQ 버스 인덱스 배열
- `sbus`: 복소 전력 주입 벡터
- `V0`: 초기 버스 전압 벡터

### 설정과 결과

- `NRConfig`: `tolerance`, `max_iter`
- `NRResult`: 최종 전압 `V`, 반복 횟수, 최종 mismatch, 수렴 여부
- `NewtonOptions`: 백엔드 종류와 Jacobian 빌더 방식 선택

### 공개 API

```cpp
NewtonSolver solver({.backend = BackendKind::CPU});

solver.analyze(ybus, pv, n_pv, pq, n_pq);

NRResult result;
solver.solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, result);
```

`solve_batch()`는 현재 precision-selection refactor 동안 임시로 비활성화되어 있으며, 호출 시 예외를 던진다.

## 빌드

기본 빌드는 CPU 전용 정적 라이브러리다.
모든 빌드 산출물은 `/workspace/cuPF/build/<profile>` 아래에 둔다.

### 요구 사항

- CMake 3.22+
- C++17 컴파일러
- Eigen3
- 선택 사항: `spdlog`
- CUDA 빌드 시:
  - CUDA Toolkit
  - cuDSS
- Python 바인딩 빌드 시:
  - Python3 Development.Module
  - pybind11

### CPU 빌드

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release
```

### CUDA 빌드

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
```

cuDSS가 기본 경로에 없으면 `CUDSS_INCLUDE_DIR`, `CUDSS_LIBRARY`를 직접 지정해야 한다.

### CUDA 타이밍 빌드

```bash
cmake --preset cuda-timing
cmake --build --preset cuda-timing
```

`cuda-timing` 프리셋은 `CUPF_ENABLE_LOG`, `CUPF_ENABLE_TIMING`를 함께 켠다.

### Python 바인딩 빌드

```bash
cmake --preset py-release
cmake --build --preset py-release
```

`vertex_based` Jacobian은 별도 빌드 프로필이 아니라 런타임 옵션이다.

## 사용 예시

### C++

```cpp
#include "newton_solver/core/newton_solver.hpp"

NewtonOptions options;
options.backend = BackendKind::CPU;
options.jacobian = JacobianBuilderType::EdgeBased;

NewtonSolver solver(options);
solver.analyze(ybus, pv.data(), pv.size(), pq.data(), pq.size());

NRResult result;
solver.solve(
    ybus,
    sbus.data(),
    v0.data(),
    pv.data(), pv.size(),
    pq.data(), pq.size(),
    NRConfig{1e-8, 50},
    result
);
```

### Python

```python
import cupf

solver = cupf.NewtonSolver(backend="cpu", jacobian="edge_based")
solver.analyze(indptr, indices, data, rows, cols, pv, pq)

result = solver.solve(
    indptr, indices, data, rows, cols,
    sbus, V0, pv, pq,
    tolerance=1e-8,
    max_iter=50,
)
```

Python 바인딩은 다음 자료형을 기대한다.

- `indptr`, `indices`, `pv`, `pq`: `int32`
- `data`, `sbus`, `V0`: `complex128`

## 테스트

CTest가 활성화되면 `cupf_case_smoke` 실행 파일이 생성된다. 기본 smoke test는 `/workspace/v1/core/dumps/case30_ieee` dump case를 사용한다.

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release
```

직접 실행할 수도 있다.

```bash
/workspace/cuPF/build/cpu-release/cupf_case_smoke \
  --case-dir /workspace/v1/core/dumps/case30_ieee \
  --backend cpu \
  --jacobian edge_based
```

## 문서

세부 설계는 `docs/` 아래 문서에 정리되어 있다.

- `docs/overview.md`: 전체 구조 요약
- `docs/core/types.md`: 공용 타입
- `docs/core/jacobian_builder.md`: Jacobian 희소 패턴 분석
- `docs/core/newton_solver.md`: 공개 API와 NR 루프
- `docs/backend/interface.md`: 백엔드 인터페이스
- `docs/backend/cpu.md`: CPU 백엔드 구현
- `docs/backend/cuda.md`: CUDA 백엔드 구현
- `docs/bindings.md`: Python 바인딩

