# Phase 06 — cuDSS Uniform Batch Integration

이 단계의 목적은 batch-aware storage와 Jacobian layout 위에 cuDSS uniform batch factorize/solve를 얹는 것이다.

---

## 목표

1. cuDSS는 uniform batch descriptor를 사용한다.
2. single-case는 `B=1`인 같은 path를 탄다.
3. host-side RHS roundtrip을 제거한다.

---

## 대상 파일

- [cpp/src/newton_solver/ops/linear_solve/cuda_cudss32.cpp](../../cpp/src/newton_solver/ops/linear_solve/cuda_cudss32.cpp)
- [cpp/src/newton_solver/ops/linear_solve/cuda_cudss64.cpp](../../cpp/src/newton_solver/ops/linear_solve/cuda_cudss64.cpp)
- [cpp/src/newton_solver/ops/linear_solve/cudss_config.hpp](../../cpp/src/newton_solver/ops/linear_solve/cudss_config.hpp)

---

## 고정 레이아웃

```text
J_row_ptr: [dimF + 1]
J_col_idx: [nnz_J]
J_values:  [B * nnz_J]
F/rhs:     [B * dimF]
dx:        [B * dimF]
```

`row_ptr`와 `col_idx`는 batch 공통이고, values/RHS/solution만 batch-strided다.

---

## 작업 내용

### 1. FP32 path

- `J_values`는 FP32 uniform batch
- `dx`는 FP32 uniform batch
- `F`는 FP64이므로 device cast buffer를 통해 RHS FP32를 만든다

### 2. FP64 path

- 가능하면 `d_F`를 직접 RHS로 사용한다
- host-side `-F` 준비를 제거한다

### 3. refactorization reuse

기존 symbolic/factorization reuse 정책을 batch path에서도 유지한다.

---

## 주의사항

- 이 단계 전에는 batch-aware storage, mismatch, Jacobian이 이미 안정화되어 있어야 한다.
- cuDSS 경로를 먼저 바꾸지 않는다.

---

## 완료 조건

- cuDSS uniform batch factorize/solve가 동작한다.
- `B=1`은 기존 single-case와 같은 결과를 낸다.
- host-side RHS roundtrip이 제거되거나 최소 device-cast 수준으로 줄어든다.
