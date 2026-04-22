# Phase 05 — Jacobian Edge One-Pass

이 단계의 목적은 Jacobian fill을 `Ibus` 재사용 구조로 바꾸고, edge path의 atomic을 없애는 것이다.
현재 CUDA Jacobian fill은 `fill_jacobian_gpu.cu`의 edge one-pass 커널로 통합되어 있다.

---

## 목표

1. off-diagonal fill은 direct write로 간다.
2. mixed edge path는 diagonal Ybus entry thread가 `Ibus` correction까지 한 번에 처리한다.
3. edge off-diagonal atomic을 제거한다.

---

## 대상 파일

- [cpp/src/newton_solver/ops/jacobian/fill_jacobian_gpu.cu](../../cpp/src/newton_solver/ops/jacobian/fill_jacobian_gpu.cu)
- [cpp/src/newton_solver/ops/jacobian/fill_jacobian.cpp](../../cpp/src/newton_solver/ops/jacobian/fill_jacobian.cpp)

---

## 내부 schedule

`CudaJacobianOp::run()`은 다음 순서를 따른다.

```text
edge mixed:
launch_fill_jacobian_gpu<float>()

fp64 cuda:
launch_fill_jacobian_gpu<double>()
```

### Off-diagonal

조건:

```text
mapJ**[k]가 고유한 J 위치를 가리킨다
```

이 조건이 성립하면 direct write를 사용한다.

### Diagonal

mixed edge path에서는 diagonal Ybus entry의 thread가 self term을 direct write한 뒤
FP64 `d_Ibus_*`를 읽어 FP32로 변환한 correction을 같은 kernel에서 더한다.

---

## 파일 분할 원칙

Jacobian fill은 CPU `fill_jacobian.cpp`, CUDA `fill_jacobian_gpu.cu`로 둔다.
CUDA 파일은 `T=float/double` 템플릿으로 mixed/FP64를 함께 처리한다.

---

## 주의사항

- `d_J_values.memsetZero()`는 모든 J entry가 정확히 한 번 write된다는 것이 확인되기 전까지 바로 없애지 않는다.
- 먼저 correctness를 맞추고, 그 다음 full memset 제거 또는 slot-local zeroing으로 넘어간다.

---

## 완료 조건

- Jacobian fill이 `Ibus` 재사용 구조로 바뀐다.
- edge off-diagonal atomic이 제거된다.
- edge path는 one-pass로 정리된다.
- `B=1` Jacobian 수치가 기존 구현과 맞는다.
