# Phase 04 — Mismatch and Ibus Pipeline

이 단계의 목적은 mismatch stage 안에 custom `Ibus` kernel과 residual pipeline을 넣고, `Ibus`를 Jacobian에서 재사용할 수 있게 하는 것이다.

---

## 목표

1. `Ibus = Ybus * V`는 custom batch CSR kernel이 기본이다.
2. mismatch stage가 `Ibus` 생성과 residual 계산을 함께 담당한다.
3. norm reduction은 device에서 수행한다.

---

## 대상 파일

- [cpp/src/newton_solver/ops/mismatch/cuda_f64.cu](../../cpp/src/newton_solver/ops/mismatch/cuda_f64.cu)
- 새 kernel file들
  - `compute_ibus_batch_fp32*.cu`
  - `compute_mismatch_batch_f64*.cu`
  - `reduce_norm_batch_f64*.cu`
- storage header/cpp (Ibus buffer 추가)

---

## 내부 schedule

`CudaMismatchOp::run()`은 다음 순서를 따른다.

```text
launch_compute_ibus()
launch_compute_mismatch()
launch_reduce_norm()
```

### compute_ibus

```text
Ibus64 = Ybus32 * V64
```

기본 정책:
- warp-per-(batch, bus)
- row length histogram을 본 뒤 short/long row variant 추가 검토

### compute_mismatch

```text
F64 = V64 * conj(Ibus64) - Sbus64
```

### reduce_norm

batch별 `normF` 또는 convergence flag를 device에서 구한다.

---

## 파일 분할 원칙

한 파일에 여러 main kernel을 쌓지 않는다.

예:

```text
cuda_f64.cu                     // thin mismatch op wrapper
compute_ibus_batch_fp32.cu      // main Ibus kernel
compute_mismatch_batch_f64.cu   // main mismatch kernel
reduce_norm_batch_f64.cu        // main reduction kernel
```

---

## 주의사항

- cuSPARSE SpMM은 이 단계의 기본 경로가 아니다.
- `Ibus`는 mismatch-produced reusable state다.
- storage validation framework를 추가하지 않는다.

---

## 완료 조건

- mismatch stage 안에서 `Ibus`가 계산되고 storage에 남는다.
- `F`와 `normF`가 host roundtrip 없이 만들어진다.
- `B=1`에서 기존 residual과 수치적으로 맞는다.
