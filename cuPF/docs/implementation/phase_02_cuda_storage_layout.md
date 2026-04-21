# Phase 02 — CUDA Storage Layout

이 단계의 목적은 기존 CUDA storage를 batch-major layout과 새 dtype 정책에 맞게 바꾸는 것이다.

---

## 목표

1. 별도 batch storage class를 만들지 않는다.
2. 기존 CUDA storage를 batch-aware로 확장한다.
3. authoritative state와 derived cache의 dtype을 고정한다.

---

## 대상 파일

- [cpp/src/newton_solver/storage/cuda/cuda_mixed_storage.hpp](../../cpp/src/newton_solver/storage/cuda/cuda_mixed_storage.hpp)
- [cpp/src/newton_solver/storage/cuda/cuda_mixed_storage.cpp](../../cpp/src/newton_solver/storage/cuda/cuda_mixed_storage.cpp)
- 필요 시 [cpp/src/newton_solver/storage/cuda/cuda_fp64_storage.hpp](../../cpp/src/newton_solver/storage/cuda/cuda_fp64_storage.hpp)
- 필요 시 [cpp/src/newton_solver/storage/cuda/cuda_fp64_storage.cpp](../../cpp/src/newton_solver/storage/cuda/cuda_fp64_storage.cpp)

---

## 고정 레이아웃

```text
Va/Vm:      double [B * n_bus]
V_re/V_im:  double [B * n_bus]
Ibus_re/im: double [B * n_bus]
F:          double [B * dimF]
dx:         float  [B * dimF]
J_values:   float  [B * nnz_J]
```

Ybus와 Sbus:

```text
Ybus_re/im: float [nnz_Y] or [B * nnz_Y]
Sbus_re/im: double [B * n_bus]
```

---

## 작업 내용

### 1. upload 경계에서 dtype 변환

입력이 FP64 complex여도 내부 optimized profile은 upload 시 FP32/FP64 내부 레이아웃으로 변환한다.

### 2. public 결과는 FP64

최종 `NRResultF64.V`는 `Va/Vm`에서 재구성한다.
`V_re/V_im` cache도 FP64이지만, public 결과는 authoritative `Va/Vm`에서 재구성한다.

### 3. batch-major indexing helper

storage 내부에 `offset(batch, idx)` 또는 equivalent helper를 둬서 인덱싱 실수를 줄인다.

### 4. no generic validity framework

storage에 별도 validation state machine을 넣지 않는다.
필요한 stage 계약은 주석과 field 의미로 표현한다.

---

## 완료 조건

- storage가 `B=1`과 `B>1`을 같은 레이아웃으로 처리한다.
- authoritative `Va/Vm`과 derived `V_re/V_im`/`Ibus` cache가 분리된다.
- upload/download에서 public FP64와 internal mixed dtype 변환이 일관된다.
