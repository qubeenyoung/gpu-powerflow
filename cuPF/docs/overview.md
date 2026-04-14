# cuPF — 설계 개요

## 목적

cuPF는 GPU 가속 Newton-Raphson 전력조류(Power Flow) solver 라이브러리다.
CPU와 CUDA 백엔드를 모두 지원하며, FP64 및 혼합 정밀도(Mixed) 계산 정책을 선택할 수 있다.

---

## 전체 디렉터리 구조

```
cuPF/
├── CMakeLists.txt                 # 빌드 설정 (CUDA, cuDSS, Python 바인딩 등 옵션 포함)
├── cpp/
│   ├── inc/newton_solver/         # 공개 헤더 (사용자 인터페이스)
│   │   ├── core/                  # Solver·Builder·Context·Plan 인터페이스
│   │   └── ops/                   # Op 추상 인터페이스
│   └── src/newton_solver/         # 구현
│       ├── core/                  # NewtonSolver, JacobianBuilder, PlanBuilder
│       ├── ops/                   # 4-stage Op 구현체 (CPU + CUDA)
│       ├── storage/               # 버퍼·핸들 관리 (CPU + CUDA)
│       └── reference/             # 정확도 검증용 참조 구현 (SuperLU, naive)
├── bindings/                      # Python 바인딩 (pybind11)
├── tests/                         # 단위·통합 테스트
├── benchmarks/                    # 성능 측정
└── docs/                          # 설계 문서 (이 파일 포함)
```

---

## 핵심 설계 원칙

### 1. 생성 시점에 모든 결정을 완료한다

`NewtonSolver` 생성자가 `PlanBuilder::build(options)`를 호출해 `ExecutionPlan`을 조립한다.
이후 `analyze()`·`solve()` hot path에서는 인터페이스 가상 함수만 호출하며, backend/precision 분기가 없다.

### 2. Storage + Op 분리

`IStorage`는 메모리와 라이브러리 핸들만 소유한다. 계산 로직은 `IMismatchOp`, `IJacobianOp`, `ILinearSolveOp`, `IVoltageUpdateOp` 네 개의 Op 클래스에 위임한다.
이 분리 덕분에 새로운 backend를 추가할 때 Storage와 4개의 Op만 구현하면 된다.

### 3. public I/O는 항상 FP64

`analyze()`, `solve()`의 인자와 `NRResultF64`는 항상 `double` / `std::complex<double>`다.
Mixed 모드에서 내부적으로 FP32가 사용되더라도 사용자는 이를 직접 알 필요가 없다.

### 4. 분석 단계를 한 번만 실행한다

`analyze()`는 Ybus 희소 구조가 바뀌지 않는 한 한 번만 호출한다.
Jacobian 희소 패턴·산포 맵·linear solver symbolic 분석 결과가 모두 재사용된다.

---

## 사용 흐름

```
NewtonSolver solver(options);   // ExecutionPlan 조립

solver.analyze(ybus, pv, n_pv, pq, n_pq);   // 한 번만

for (각 시나리오) {
    solver.solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, result);
}
```

---

## 실행 프로파일 (variants)

| 프로파일         | backend | compute | Jacobian kernel  | Linear solver   |
|----------------|---------|---------|-----------------|-----------------|
| CPU FP64       | CPU     | FP64    | Edge-based (CPU) | KLU (Eigen)     |
| CUDA FP64      | CUDA    | FP64    | Edge 또는 Vertex | cuDSS FP64      |
| CUDA Mixed     | CUDA    | Mixed   | Edge 또는 Vertex | cuDSS FP32      |

Mixed 프로파일은 고정 구성이다. Mismatch·Voltage는 FP64, Jacobian·Solve는 FP32를 사용한다.

---

## NR 반복 루프 (hot path)

```
for iter in range(max_iter):
    mismatch.run(ctx)      # F = S_calc - S_spec, normF 계산
    if converged: break

    jacobian.run(ctx)      # J.values ← Ybus 원소에서 scatter
    linear_solve.run(ctx)  # J·dx = -F 풀기 (factorize + solve)
    voltage_update.run(ctx) # Va, Vm에 dx 적용 → V 재구성
```

각 stage에 NVTX range와 ScopedTimer가 감싸져 있어 Nsight와 wall-clock 프로파일링을 동시에 지원한다.

---

## 세부 문서 목차

| 문서 | 내용 |
|------|------|
| [math.md](math.md) | Newton-Raphson 전력조류 수학 배경 |
| [core/README.md](core/README.md) | NewtonSolver, JacobianBuilder, ExecutionPlan, Context |
| [ops/README.md](ops/README.md) | 4-stage Op 인터페이스 및 구현체 |
| [storage/README.md](storage/README.md) | IStorage 및 각 backend 스토리지 |
| [variants/README.md](variants/README.md) | 실행 프로파일별 구성과 선택 기준 |
| [RULE.md](RULE.md) | 타이머·NVTX·Python 바인딩 규칙 |
