# cuPF Overview

GPU-accelerated Newton-Raphson power flow solver. CSR 형식의 Ybus를 입력받아 수렴된 버스 전압을 반환한다.

---

## 아키텍처

```
Python / C++ 사용자
        │
        ▼
  NewtonSolver          ← 공개 API. NR 루프 진행자.
   ├── JacobianBuilder  ← 1회성 희소 패턴 분석
   └── INewtonSolverBackend  ← 백엔드 추상 인터페이스
        ├── CpuNewtonSolverBackend   (Eigen + SparseLU)
        └── CudaNewtonSolverBackend  (cuSPARSE + cuDSS)
```

---

## NR 루프 흐름

```
analyze(ybus, pv, pq)          ← 위상 분석, 한 번만 호출
    └── JacobianBuilder → JacobianMaps + JacobianStructure
    └── backend.analyze(...)   ← GPU 메모리 할당, cuDSS 심볼릭 분석

solve(ybus, sbus, V0, ...)
    └── backend.initialize()   ← V0, Sbus 업로드
    └── while iter < max_iter:
          computeMismatch()    ← Ibus = Ybus·V, F 계산
          if ||F|| < tol: break
          updateJacobian()     ← JacobianMaps로 J 값 채우기
          solveLinearSystem()  ← J·dx = -F
          updateVoltage()      ← Va, Vm 보정, V 재구성
    └── downloadV()            ← 최종 전압 반환
```

---

## 정밀도

| 백엔드 | Jacobian | 전압 상태 |
|--------|----------|-----------|
| CPU    | FP64     | FP64      |
| CUDA   | FP32     | FP64      |

CUDA는 FP32 Jacobian으로 인수분해 속도를 높이고, 전압은 FP64로 정확도를 유지하는 혼합 정밀도 방식.

---

## 파일 구조

```
cuPF/
├── cpp/
│   ├── inc/newton_solver/
│   │   ├── core/               ← 공개 헤더 (공용 타입, NewtonSolver, JacobianBuilder)
│   │   └── backend/            ← 공개 헤더 (인터페이스, 백엔드 클래스 선언)
│   ├── inc/utils/              ← 유틸리티 헤더 (logger, timer, dump, cuda_utils)
│   └── src/newton_solver/
│       ├── core/               ← NewtonSolver, JacobianBuilder 구현
│       └── backend/
│           ├── cpp/            ← CPU 백엔드 구현
│           └── cuda/           ← CUDA 백엔드 구현 (.cu 포함)
├── bindings/
│   └── pybind_module.cpp       ← Python 바인딩
├── docs/                       ← 이 문서들
└── tests/
```

---

## 문서 목차

| 문서 | 내용 |
|------|------|
| [core/types.md](core/types.md) | 공용 데이터 타입 (CSRView, JacobianMaps, NRResult 등) |
| [core/jacobian_builder.md](core/jacobian_builder.md) | 희소 패턴 분석 및 매핑 테이블 생성 |
| [core/newton_solver.md](core/newton_solver.md) | 공개 API, NR 루프 |
| [backend/interface.md](backend/interface.md) | 백엔드 추상 인터페이스 |
| [backend/cpu.md](backend/cpu.md) | CPU 백엔드 (Eigen + SparseLU) |
| [backend/cuda.md](backend/cuda.md) | CUDA 백엔드 (cuSPARSE + cuDSS) |
| [backend/schur_complement.md](backend/schur_complement.md) | Schur complement 관점에서 본 Jacobian 블록 구조와 cuDSS sample 해석 |
| [utils/overview.md](utils/overview.md) | logger, timer, dump, cuda_utils |
| [bindings.md](bindings.md) | Python 바인딩 |
