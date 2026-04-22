# Phase 04 — Mismatch and Ibus Pipeline

이 단계의 목적은 mismatch stage 안에 `Ibus` 계산과 residual pipeline을 분리하고,
`Ibus`를 Jacobian에서 재사용할 수 있게 하는 것이다.

---

## 목표

1. `Ibus = Ybus * V`는 독립된 ibus stage에서 계산한다.
2. mismatch stage가 `Ibus` 생성과 residual 계산을 함께 담당한다.
3. CUDA norm reduction은 device에서 수행한다.

---

## 대상 파일

- [cpp/src/newton_solver/ops/ibus/compute_ibus.cpp](../../cpp/src/newton_solver/ops/ibus/compute_ibus.cpp)
- [cpp/src/newton_solver/ops/ibus/compute_ibus.cu](../../cpp/src/newton_solver/ops/ibus/compute_ibus.cu)
- [cpp/src/newton_solver/ops/mismatch/cuda_mismatch.cu](../../cpp/src/newton_solver/ops/mismatch/cuda_mismatch.cu)
- 새 kernel file들
  - `../ibus/compute_ibus*.cu`
  - `compute_mismatch_from_ibus*.cu`
  - `reduce_mismatch_norm*.cu`
- storage header/cpp (Ibus buffer 추가)

---

## 내부 schedule

`CudaMismatchOp::run()`은 FP64와 Mixed 모두 `Ibus`를 만든 뒤 residual과 norm을 계산한다.

```text
launch_compute_ibus()
launch_compute_mismatch_from_ibus()
launch_reduce_mismatch_norm()
```

### compute_ibus

```text
Ibus = Ybus * V
Mixed: Ibus remains FP64 and is reused by mismatch and Jacobian
```

기본 정책:
- tiled row × batch launch: blockIdx.x = bus, blockIdx.y = batch tile
- row length histogram을 본 뒤 short/long row variant 추가 검토

### compute_mismatch_from_ibus

```text
F64 = V64 * conj(Ibus64) - Sbus64
```

### reduce_mismatch_norm

batch별 `normF` 또는 convergence flag를 device에서 구한다.

---

## 파일 분할 원칙

한 파일에 여러 main kernel을 쌓지 않는다.

예:

```text
cuda_mismatch.cu                // thin mismatch op schedule
../ibus/compute_ibus.cu         // Ybus + V -> Ibus
compute_mismatch_from_ibus.cu   // Ibus + V + Sbus -> F
reduce_mismatch_norm.cu         // F -> norm
```

---

## 주의사항

- cuSPARSE SpMM은 이 단계의 기본 경로가 아니다.
- `Ibus`는 mismatch-produced reusable state다. Jacobian도 FP64 `Ibus`를 읽고
  커널 안에서 FP32로 변환한다.
- storage validation framework를 추가하지 않는다.

---

## 완료 조건

- mismatch stage 안에서 `Ibus`가 계산되고 storage에 남는다.
- `F`와 `normF`가 host roundtrip 없이 만들어진다.
- `B=1`에서 기존 residual과 수치적으로 맞는다.
