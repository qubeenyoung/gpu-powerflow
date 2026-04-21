# Phase 05 — Jacobian Off-Diagonal and Diagonal Split

이 단계의 목적은 Jacobian fill을 `Ibus` 재사용 구조로 바꾸고, off-diagonal과 diagonal 책임을 분리하는 것이다.

---

## 목표

1. off-diagonal fill은 direct write로 간다.
2. diagonal fill은 `Ibus` 기반 별도 경로로 계산한다.
3. edge off-diagonal atomic을 제거한다.

---

## 대상 파일

- [cpp/src/newton_solver/ops/jacobian/cuda_edge_fp32.cu](../../cpp/src/newton_solver/ops/jacobian/cuda_edge_fp32.cu)
- [cpp/src/newton_solver/ops/jacobian/cuda_vertex_fp32.cu](../../cpp/src/newton_solver/ops/jacobian/cuda_vertex_fp32.cu)
- 필요 시 FP64 대응 파일
- 새 kernel file들
  - `fill_jacobian_offdiag_*.cu`
  - `fill_jacobian_diag_from_ibus_*.cu`

---

## 내부 schedule

`CudaJacobianOp::run()`은 다음 순서를 따른다.

```text
launch_fill_offdiag()
launch_fill_diag_from_ibus()
```

### Off-diagonal

조건:

```text
mapJ**[k]가 고유한 J 위치를 가리킨다
```

이 조건이 성립하면 direct write를 사용한다.

### Diagonal

대각은 `Ibus`, `V`, `Ydiag` 기반 별도 kernel에서 계산한다.
이렇게 하면 edge path의 diagonal atomic을 줄일 수 있다.

---

## 파일 분할 원칙

한 파일은 한 개의 main kernel만 가진다.

예:

```text
cuda_edge_fp32.cu                    // thin edge op wrapper
fill_jacobian_edge_offdiag_fp32.cu   // main offdiag kernel
fill_jacobian_diag_from_ibus_fp32.cu // main diag kernel
```

vertex path도 같은 원칙을 따른다.

---

## 주의사항

- `d_J_values.memsetZero()`는 모든 J entry가 정확히 한 번 write된다는 것이 확인되기 전까지 바로 없애지 않는다.
- 먼저 correctness를 맞추고, 그 다음 full memset 제거 또는 slot-local zeroing으로 넘어간다.

---

## 완료 조건

- Jacobian fill이 `Ibus` 재사용 구조로 바뀐다.
- edge off-diagonal atomic이 제거된다.
- diagonal path가 분리된다.
- `B=1` Jacobian 수치가 기존 구현과 맞는다.
