# CUDA Batch Refactor Implementation Plan

이 문서는 `docs/gpu_batch_improvement_plan.md`의 설계 결정을 실제 구현 순서로 옮긴 상위 계획 문서다.
설계 문서는 "어떻게 되어야 하는가"를 설명하고, 이 문서는 "어떤 순서로 옮길 것인가"를 설명한다.

기준 설계 문서:
- [../gpu_batch_improvement_plan.md](../gpu_batch_improvement_plan.md)
- 구현 차이 기록:
- [deviations.md](deviations.md)

---

## 고정 원칙

아래 원칙은 전체 단계에서 흔들리지 않는다.

1. CUDA backend의 기본 실행 모델은 batch다.
   single-case solve는 별도 구현이 아니라 `B=1`인 같은 실행 경로다.
2. 바깥 Newton solver 추상화는 유지한다.

```text
ExecutionPlan
  storage
  mismatch
  jacobian
  linear_solve
  voltage_update
```

3. CUDA op 내부는 finer-grained substage schedule로 분해한다.

```text
CudaMismatchOp::run()
  launch_compute_ibus()
  launch_compute_mismatch()
  launch_reduce_norm()

CudaJacobianOp::run()
  launch_fill_offdiag()
  launch_fill_diag_from_ibus()
```

4. `cuda_batch`용 별도 storage/op/plan class를 만들지 않는다.
   기존 CUDA storage/op를 batch-aware로 확장한다.
5. 한 개의 파일은 한 개의 main kernel 또는 한 개의 얇은 orchestrator 책임만 가진다.
   서로 다른 성격의 kernel을 한 `.cu` 파일에 계속 누적하지 않는다.
6. generic storage validation framework는 추가하지 않는다.
   필요한 상태 계약은 storage 주석, field 이름, stage별 불변식으로 충분히 표현한다.
7. authoritative voltage state는 `Va/Vm` FP64, derived cache인 `V_re/V_im`도 FP64다.
8. `Ibus = Ybus * V`는 cuSPARSE 기본 경로가 아니라 custom batch CSR kernel이 기본이다.
9. 부호 convention은 `F = S_calc - S_spec`, `J * dx = F`, `state -= dx`로 통일한다.

---

## 단계 문서

1. [phase_00_solver_pipeline.md](phase_00_solver_pipeline.md)
   outer 4-stage 유지, inner CUDA substage schedule, 파일 책임 분할
2. [phase_01_batch_api_and_types.md](phase_01_batch_api_and_types.md)
   batch-aware public/core API와 타입 확장
3. [phase_02_cuda_storage_layout.md](phase_02_cuda_storage_layout.md)
   batch-major storage, dtype layout, upload/download 경계
4. [phase_03_voltage_update_and_sign.md](phase_03_voltage_update_and_sign.md)
   `Va/Vm` authoritative update와 부호 convention 전환
5. [phase_04_mismatch_and_ibus.md](phase_04_mismatch_and_ibus.md)
   custom `Ibus` kernel, mismatch 계산, norm reduction
6. [phase_05_jacobian_split.md](phase_05_jacobian_split.md)
   off-diagonal/diagonal 분리, `Ibus` 재사용, atomic 제거 방향
7. [phase_06_cudss_uniform_batch.md](phase_06_cudss_uniform_batch.md)
   cuDSS uniform batch factorize/solve 경로 이식
8. [phase_07_validation_and_cleanup.md](phase_07_validation_and_cleanup.md)
   `B=1` 회귀, batch 검증, 주석/가독성/문서 정리

---

## 권장 구현 순서

구현은 반드시 아래 순서를 따른다.

1. outer solver 구조와 파일 분할 정책을 먼저 고정한다.
2. batch-aware API와 result/context를 추가한다.
3. CUDA storage를 batch-major로 바꾼다.
4. voltage update와 부호 convention을 먼저 전환한다.
5. mismatch 안에 `Ibus` custom kernel과 residual pipeline을 넣는다.
6. Jacobian을 `Ibus` 재사용 구조로 바꾼다.
7. 마지막에 cuDSS uniform batch를 붙인다.
8. 끝나면 `B=1` 회귀와 batch 성능/수렴 검증을 수행한다.

이 순서를 따르는 이유는 `B=1` 기준의 correctness를 각 단계에서 계속 확인할 수 있기 때문이다.

---

## 전체 점검 항목

모든 단계가 끝나면 아래 항목을 한 번에 점검한다.

### 가독성

- 파일 책임이 명확한가
- 한 파일에 여러 "메인 kernel"이 뒤섞여 있지 않은가
- batch-aware path와 single-case wrapper 관계가 코드만 읽어도 드러나는가
- `PlanBuilder`가 여전히 profile 조립만 담당하는가

### 주석

- `F = S_calc - S_spec` 부호 설명이 코드와 문서에서 일관적인가
- `J * dx = F`, `state -= dx` 설명이 update/solve 주석과 맞는가
- `Va/Vm` authoritative, `V_re/V_im` derived cache 설명이 storage와 update 주석에 반영되었는가
- `Ibus`가 mismatch-produced reusable state라는 설명이 mismatch/jacobian/storage 주석에 반영되었는가

### 구조

- `cuda_batch` 별도 구현 경로가 생기지 않았는가
- `CudaBatch*`, `BatchExecutionPlan`, `BatchPlanBuilder`가 새로 추가되지 않았는가
- `B=1`이 특수 분기 없이 같은 CUDA path를 타는가
- cuDSS는 uniform batch descriptor를 쓰고 있는가

### 성능/동작

- `B=1` 결과가 기존 single-case와 수치적으로 맞는가
- mismatch norm을 위해 전체 `d_F`를 host로 내리지 않는가
- FP64 path에서 RHS `-F` host roundtrip이 사라졌는가
- edge off-diagonal atomic이 제거되었는가

### 문서

- `docs/gpu_batch_improvement_plan.md`와 구현 세부가 어긋나지 않는가
- `docs/overview.md`, `docs/ops/README.md`, `docs/storage/README.md`, `docs/variants/README.md`의 설명이 현재 구현과 맞는가
- `cpp/src/newton_solver/ops/cuda_batch/TODO.md`가 더 이상 별도 batch 경로를 유도하지 않는가
