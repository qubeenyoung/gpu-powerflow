# Phase 00 — Solver Pipeline and File Boundaries

이 단계의 목적은 구현 전에 outer solver 구조와 CUDA 내부 실행 스케줄을 고정하는 것이다.

---

## 목표

1. 바깥 Newton loop는 지금처럼 4-stage를 유지한다.
2. CUDA op 내부는 substage schedule로 분해한다.
3. 한 파일은 한 개의 main kernel 또는 한 개의 얇은 orchestrator만 가진다.
4. storage validation framework는 추가하지 않는다.

---

## 유지할 outer 구조

대상 파일:
- [cpp/inc/newton_solver/core/execution_plan.hpp](../../cpp/inc/newton_solver/core/execution_plan.hpp)
- [cpp/src/newton_solver/core/newton_solver.cpp](../../cpp/src/newton_solver/core/newton_solver.cpp)
- [cpp/inc/newton_solver/ops/op_interfaces.hpp](../../cpp/inc/newton_solver/ops/op_interfaces.hpp)

유지할 형태:

```text
mismatch.run(ctx)
if converged: break
jacobian.run(ctx)
linear_solve.run(ctx)
voltage_update.run(ctx)
```

이 단계에서는 `IIterationPipeline` 같은 새 최상위 추상화는 만들지 않는다.

---

## CUDA inner schedule

### Mismatch

`CudaMismatchOp::run()` 내부에서 다음 순서를 따른다.

```text
launch_compute_ibus()
launch_compute_mismatch()
launch_reduce_norm()
```

### Jacobian

`CudaJacobianOp::run()` 내부에서 다음 순서를 따른다.

```text
launch_fill_offdiag()
launch_fill_diag_from_ibus()
```

### Linear solve

`CudaLinearSolveOp::run()`은 여전히 factorize + solve를 담당한다.

### Voltage update

`CudaVoltageUpdateOp::run()`은 `Va/Vm` 갱신과 `V_re/V_im` cache 재구성을 담당한다.

---

## 파일 분할 원칙

### 허용하는 형태

1. 얇은 orchestrator file

```text
cuda_f64.cu
  - CudaMismatchOp::run()
  - helper launch function declarations
  - no unrelated heavy kernel bodies
```

2. main kernel file

```text
compute_ibus_batch_fp32.cu
  - one main kernel family
  - small helper device functions allowed
```

### 피해야 하는 형태

- mismatch, jacobian, reduction, update kernel을 하나의 `.cu` 파일에 계속 추가하는 구조
- single-case kernel과 batch kernel을 별도 디렉터리 트리로 복제하는 구조
- `CudaBatch*`용 별도 source tree

---

## 구현 지침

1. outer 4-stage를 먼저 고정한다.
2. 그 다음 각 CUDA op에 대해 "wrapper file"과 "kernel files" 경계를 정한다.
3. 파일 이름만 봐도 어떤 kernel인지 알 수 있게 한다.
4. orchestration은 op wrapper에서, 수치 계산은 kernel file에서 담당한다.

---

## 완료 조건

- NR loop가 여전히 4-stage다.
- CUDA 내부 스케줄이 문서로 고정되었다.
- "한 파일 = 한 main kernel" 원칙에 맞는 파일 분할안이 정리되었다.
- storage validation framework를 추가하지 않는다는 결정이 문서에 명시되었다.
