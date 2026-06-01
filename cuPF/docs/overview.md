# cuPF — 설계 개요

## 목적

cuPF는 GPU 가속 Newton-Raphson 전력조류(Power Flow) solver 라이브러리다.
CPU와 CUDA 백엔드를 모두 지원하며, FP64·FP32·혼합 정밀도(Mixed) 계산 정책을 선택할 수 있다.

---

## 전체 디렉터리 구조

```
cuPF/
├── CMakeLists.txt                 # 빌드 설정 (CUDA, cuDSS, Python 바인딩 등 옵션 포함)
├── cpp/
│   ├── inc/                       # 공개 헤더 (사용자 인터페이스)
│   │   ├── newton_solver/core/    # newton_solver.hpp, newton_solver_types.hpp
│   │   └── utils/                 # cuda_utils, dump, logger, nvtx_trace, timer
│   └── src/newton_solver/         # 구현
│       ├── core/                  # NewtonSolver, SolverPipeline(variant), contexts, adjoint
│       ├── ops/                   # stage 연산 구현체 (CPU + CUDA)
│       │   ├── ibus/              # Ibus = Ybus * V
│       │   ├── mismatch/          # F = S_calc - S_spec, normF
│       │   ├── jacobian/          # Jacobian 분석 + fill
│       │   ├── linear_solve/      # KLU(CPU) / cuDSS(CUDA) / custom(gated)
│       │   └── voltage_update/    # Va/Vm 갱신
│       └── storage/               # 버퍼·핸들 관리 (cpu/, cuda/)
├── bindings/                      # Python 바인딩 (pybind11, torch extension)
├── python/cupf/                   # Python 패키지 (torch autograd 래퍼 포함)
├── tests/                         # 단위·통합 테스트 + native evaluator
│   ├── cpp/cupf_cpp_evaluate.cpp  # C++ benchmark/evaluation executable (BUILD_EVALUATORS)
│   └── python/                    # pybind/torch unit and smoke tests only
└── docs/                          # 설계 문서 (이 파일 포함)
```

> 참고: 사용자-facing benchmark runner/evaluator는 워크스페이스 루트의
> `python/tests/`에 둔다. `cuPF/python/`에는 패키지 코드만, `cuPF/tests/python/`에는
> cuPF 자체 unit/smoke 테스트만 둔다.

---

## 핵심 설계 원칙

### 1. 생성 시점에 backend/precision을 확정한다

`NewtonSolver` 생성자가 `NewtonOptions`(backend·compute)를 보고 대응하는 **pipeline**을
조립한다. pipeline은 가상 인터페이스가 아니라 `SolverPipeline`이라는 `std::variant`에
담긴 구체 struct다. 이후 `initialize()`·`solve()` hot path에서는 `std::visit`로
선택된 한 pipeline만 실행하므로 매 iteration마다 backend/precision 분기가 없다.

> cuPF는 전역 네임스페이스를 사용한다(`cupf::` 네임스페이스 없음).

### 2. Storage + Op 분리 (정적 디스패치)

각 pipeline은 한 개의 Storage struct와 stage별 Op struct를 값으로 소유한다.
Storage(`CpuFp64Storage`, `CudaFp64Storage`, `CudaFp32Storage`, `CudaMixedStorage`)는
메모리와 라이브러리 핸들만 소유한다. 계산 로직은 stateless Op struct
(`Cpu*Op`/`Cuda*Op`)에 위임한다. 가상 함수 테이블이나 `IStorage`/`I*Op` 추상
인터페이스는 없다. 새 backend를 추가하려면 Storage 하나와 stage Op들을 구현하고
새 pipeline struct + variant 항목을 추가한다.

### 3. public I/O는 항상 FP64

`initialize()`, `solve()`의 인자와 `NRResult`는 항상 `double` / `std::complex<double>`다.
FP32·Mixed 모드에서 내부적으로 FP32가 사용되더라도 사용자는 이를 직접 알 필요가 없다.

### 4. 분석 단계를 한 번만 실행한다

`initialize()`는 Ybus 희소 구조가 바뀌지 않는 한 한 번만 호출한다.
Jacobian 희소 패턴·산포 맵·linear solver symbolic 분석 결과가 모두 재사용된다.

---

## 사용 흐름

```
NewtonSolver solver(options);   // options에 맞는 pipeline(variant) 조립

solver.initialize(ybus, pv, n_pv, pq, n_pq);   // 한 번만

for (각 시나리오) {
    solver.solve(ybus, sbus, V0, pv, n_pv, pq, n_pq, config, solve_options, result);
}
```

기존 `solve()`는 내부적으로 `batch_size = 1`인 경로이고,
CUDA FP32/Mixed 경로는 `solve_batch()`로 `B > 1` 실행을 지원한다.

---

## 실행 프로파일 (variants)

| 프로파일       | pipeline struct          | backend | compute | Jacobian        | Linear solver   |
|--------------|--------------------------|---------|---------|-----------------|-----------------|
| CPU FP64     | `CpuFp64Pipeline`        | CPU     | FP64    | Edge-based (CPU) | KLU (Eigen)     |
| CUDA FP64    | `CudaFp64Pipeline`       | CUDA    | FP64    | Edge one-pass    | cuDSS FP64      |
| CUDA FP32    | `CudaFp32Pipeline`       | CUDA    | FP32    | Edge one-pass    | cuDSS FP32      |
| CUDA Mixed   | `CudaMixedPipeline`      | CUDA    | Mixed   | Edge one-pass    | cuDSS FP32      |
| CUDA FP64(custom) | `CudaFp64CustomPipeline` | CUDA | FP64 | Edge one-pass | 자체 CUDA solver |

`CudaFp64CustomPipeline`은 `CUPF_ENABLE_CUSTOM_SOLVER=ON`일 때만 variant에
포함되는 선택적 경로다(기본 `OFF`). Mixed 프로파일은 고정 구성이다. `Ybus`, `Va/Vm`
전압 상태, `V_re/V_im` cache, `Sbus`, `Ibus`, `F`는 FP64로 유지하고, `J/dx`는 FP32로 둔다.

---

## NR 반복 루프 (hot path)

```
for iter in range(max_iter):
    ibus.run(ctx)           # Ibus = Ybus * V
    mismatch.run(ctx)       # F = S_calc - S_spec
    mismatch_norm.run(ctx)  # normF 계산, 수렴 판정
    if converged: break

    jacobian.run(ctx)       # J.values ← Ybus 원소에서 scatter
    linear_solve.prepare_rhs(ctx)
    linear_solve.factorize(ctx)
    linear_solve.solve(ctx) # 선형계 풀이 (J·dx = F)
    voltage_update.run(ctx) # Va, Vm에 dx를 빼고 V cache 재구성
```

개념적으로는 mismatch / jacobian / linear_solve / voltage_update 4-stage이며,
`ibus`와 `mismatch_norm`은 mismatch stage를 떠받치는 substage다.
각 stage에 NVTX range와 ScopedTimer가 감싸져 있어 Nsight와 wall-clock 프로파일링을 동시에 지원한다.

---

## 세부 문서 목차

| 문서 | 내용 |
|------|------|
| [math.md](math.md) | Newton-Raphson 전력조류 수학 배경 |
| [core/README.md](core/README.md) | NewtonSolver, Jacobian analysis, SolverPipeline, Context |
| [ops/README.md](ops/README.md) | stage Op 구현체 |
| [storage/README.md](storage/README.md) | Storage struct와 각 backend 버퍼 |
| [variants/README.md](variants/README.md) | 실행 프로파일별 구성과 선택 기준 |
| [gpu_batch_improvement_plan.md](gpu_batch_improvement_plan.md) | CUDA 기본 batch 실행과 kernel 개선 계획 |
| [implementation/README.md](implementation/README.md) | CUDA batch refactor 단계별 구현 계획 |
| [implementation/docs_src_reconciliation.md](implementation/docs_src_reconciliation.md) | docs↔src 대조표·리팩토링 요약 (cycle 5 / s5) |
| [RULE.md](RULE.md) | 타이머·NVTX·Python 바인딩 규칙 |
