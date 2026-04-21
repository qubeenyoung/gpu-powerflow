# Phase 07 — Validation, Readability, and Cleanup

이 단계의 목적은 구현 완료 후 correctness, 가독성, 주석, 문서를 한 번에 정리하는 것이다.

---

## 목표

1. `B=1` 회귀가 통과한다.
2. batch 수렴/성능이 확인된다.
3. 주석과 문서가 실제 구현과 맞는다.

---

## 대상 파일

- [tests/](../../tests)
- [benchmarks/](../../benchmarks)
- [docs/](../../docs)
- CUDA storage/op/linear solve 구현 파일 전반

---

## 검증 항목

### Correctness

- `B=1` 결과가 기존 single-case 결과와 수치적으로 맞는가
- `B>1`에서 batch별 수렴 여부와 final mismatch가 합리적인가
- FP32 `Ybus/J/dx` + FP64 Sbus/Ibus/voltage profile에서 실패 케이스가 정리되는가

### Jacobian

- edge/vertex 결과가 수치적으로 맞는가
- off-diagonal direct write가 충돌 없이 동작하는가
- diagonal split 이후에도 기존 Jacobian과 일치하는가

### Linear solve

- `J * dx = F`, `state -= dx` 부호 convention이 일관적인가
- RHS host roundtrip이 사라졌는가

### Performance

- `B=1` latency 회귀가 없는가
- small/medium batch throughput 숫자가 수집되었는가
- stream 지원 시 overlap이 실제로 생기는가

---

## 가독성/주석 점검

다음 항목을 반드시 손으로 훑는다.

- stale comment 제거
- `cuSPARSE SpMV` 같은 예전 설명 제거
- `F = S_calc - S_spec`로 주석 통일
- `Va/Vm` authoritative, `V_re/V_im` derived cache 설명 통일
- `Ibus`가 mismatch-produced reusable state라는 설명 통일
- 파일 이름과 실제 main kernel 책임이 맞는지 확인

---

## 문서 정리

반드시 업데이트할 문서:

- [../gpu_batch_improvement_plan.md](../gpu_batch_improvement_plan.md)
- [../overview.md](../overview.md)
- [../ops/README.md](../ops/README.md)
- [../storage/README.md](../storage/README.md)
- [../variants/README.md](../variants/README.md)
- [../../cpp/src/newton_solver/ops/cuda_batch/TODO.md](../../cpp/src/newton_solver/ops/cuda_batch/TODO.md)

---

## 완료 조건

- 회귀/배치 검증 결과가 남아 있다.
- 주요 benchmark와 test가 새 path를 다룬다.
- 주석과 문서가 코드와 맞는다.
- 별도 batch 구현 경로를 유도하는 stale TODO가 남아 있지 않다.

---

## 2026-04-21 수행 기록

- `cupf_case_benchmark`에 `--batch-size`를 추가했다. `B>1`은 CUDA Mixed 기본 경로에서만 허용하고,
  modified/hybrid ablation 경로는 명시적으로 `B=1`만 받는다.
- CUDA Mixed batch smoke test를 추가했다. dump dataset이 있으면 `B=2` 동일 입력을 `solve_batch()`로 실행해
  batch별 convergence, final mismatch, batch 간 전압 일치를 확인한다.
- 수동 CUDA benchmark path의 `SolveContext`에 batch metadata와 stride를 명시했다.
- hybrid ablation에서 CUDA 전압을 CPU로 복사할 때 FP32 `V_re/V_im` cache가 아니라 FP64 `Va/Vm`에서 재구성하도록 바꿨다.
- cuDSS FP32 uniform batch dense RHS/solution descriptor의 column count를 `dimF`가 아니라 `1`로 분리했다.
- v1 브랜치의 cuDSS uniform batch 구현을 참고해 `cudssMatrixCreateBatchCsr` 방식에서
  `CUDSS_CONFIG_UBATCH_SIZE + 단일 CSR/Dn descriptor + flat batch-major buffer` 방식으로 교체했다.
- CUDA linear diagnostics residual을 현재 convention에 맞게 `J dx - F`로 정정했다.
  `B>1` dump에서는 batch 0 slice를 진단 대상으로 삼는다.
- 문서와 주석에서 예전 cuSPARSE/RHS `-F`/FP64-only mismatch 설명을 현재 CUDA Mixed 설계에 맞췄다.
- Texas Univ `case_ACTIVSg200` B=1 수렴성 점검 결과, FP32 `V_re/V_im` cache는 final mismatch가
  `6e-5` 수준에서 멈췄다. 전압 cache를 FP64로 되돌린 뒤 `7.254e-8`까지 개선되었지만,
  `1e-8` tolerance는 아직 통과하지 못했다. 따라서 당시 남은 수렴성 원인은 FP32 `Sbus/Ibus/J/dx` 또는
  FP32 cuDSS solve 쪽을 후속으로 좁힌다.

남은 검증:

- 현재 작업 환경에는 외부 dump dataset과 GoogleTest dependency가 없어 runtime convergence test는 빌드 후 skip될 수 있다.
- `d_J_values.memsetZero()` 제거는 아직 하지 않았다. poison/coverage 검증 후 별도 단계로 진행한다.
- stream-aware `DeviceBuffer`와 kernel launch는 아직 구현 전이다.

빌드 검증:

- `cmake -S . -B build-phase7-cpu -DWITH_CUDA=OFF -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_BENCHMARKS=ON`
- `cmake --build build-phase7-cpu --target cupf cupf_case_benchmark -j2`
- `cmake -S . -B build-phase7-cuda -DWITH_CUDA=ON -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_BENCHMARKS=ON`
- `cmake --build build-phase7-cuda --target cupf cupf_case_benchmark -j2`
- `python3 -m py_compile benchmarks/run_benchmarks.py`
- `git diff --check`

검증 결과:

- CPU-only build와 CUDA build 모두 통과했다.
- `ctest`는 GoogleTest 미탑재로 테스트가 생성되지 않았다.
- `/workspace/datasets/cuPF_benchmark_dumps/case30_ieee`와 `/workspace/v1/core/dumps/case30_ieee`가 없어 runtime smoke는 실행하지 못했다.
