# Phase 03 — Voltage Update and Sign Convention

이 단계의 목적은 voltage update를 `Va/Vm` authoritative 구조로 바꾸고, Newton sign convention을 새 기준으로 통일하는 것이다.

---

## 목표

1. `Va/Vm`은 FP64 authoritative state다.
2. `V_re/V_im`은 FP64 derived cache다.
3. 부호 convention은 `F = S_calc - S_spec`, `J * dx = F`, `state -= dx`로 통일한다.

---

## 대상 파일

- [cpp/src/newton_solver/ops/voltage_update/cuda_mixed.cu](../../cpp/src/newton_solver/ops/voltage_update/cuda_mixed.cu)
- [cpp/src/newton_solver/ops/voltage_update/cuda_fp64.cu](../../cpp/src/newton_solver/ops/voltage_update/cuda_fp64.cu)
- [cpp/src/newton_solver/ops/linear_solve/cuda_cudss32.cpp](../../cpp/src/newton_solver/ops/linear_solve/cuda_cudss32.cpp)
- [cpp/src/newton_solver/ops/linear_solve/cuda_cudss64.cpp](../../cpp/src/newton_solver/ops/linear_solve/cuda_cudss64.cpp)
- 관련 주석이 있는 core/ops 문서와 코드

---

## 작업 내용

### 1. decompose kernel 제거

기존:

```text
V -> Va/Vm
apply dx
Va/Vm -> V
```

목표:

```text
Va/Vm -= dx
V_re/V_im cache 재구성
```

### 2. V cache 재구성

update 이후 `Va/Vm`에서 FP64 `V_re/V_im` cache를 재구성한다.

### 3. 결과 복원은 FP64 authoritative state에서

public result는 `Va/Vm`에서 재구성해 만든다.

### 4. linear solve RHS convention 변경

host-side `-F` 준비를 제거하는 방향으로 solver 주석과 구현을 바꾼다.

---

## 완료 조건

- CUDA update가 `Va/Vm` authoritative state를 직접 갱신한다.
- `V_re/V_im` cache는 update 이후 유효한 FP64 복소 cache다.
- 코드와 문서의 부호 설명이 모두 새 convention과 일치한다.
