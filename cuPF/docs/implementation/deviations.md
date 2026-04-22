# Implementation Deviations Log

이 문서는 실제 구현이 원래 단계 문서와 달라진 부분만 기록한다.
차이가 없으면 "없음"을 유지한다.

---

## 2026-04-21

### Phase 01

- 계획:
  batch-aware public/core API를 추가하고 single-case는 `B=1` wrapper로 유지한다.
- 실제 구현:
  C++ core에 `solve_batch()`와 `NRBatchResult` scaffolding을 추가했다.
- 차이:
  `batch_size > 1` 실제 실행은 아직 막아 두었다.
  이유는 storage와 CUDA kernel이 아직 batch-major layout을 처리하지 못하기 때문이다.
  이는 설계 변경이 아니라 구현 순서를 명시적으로 드러낸 것이다.

- 계획:
  batch-aware API는 향후 Python에서도 노출될 수 있다.
- 실제 구현:
  이번 단계에서는 Python binding은 건드리지 않았다.
- 차이:
  현재 pybind layer는 아직 `solve()` wrapper 자체를 public으로 노출하지 않기 때문에,
  batch binding 추가는 storage/kernel phase 이후에 묶어서 진행한다.

### Phase 02

- 계획:
  `CudaMixedStorage`를 batch-major layout과 새 dtype 정책으로 전환한다.
- 실제 구현:
  `CudaMixedStorage`의 내부 버퍼를 새 optimized mixed profile에 맞췄다.
  현재 정책은 `J/dx`만 FP32, `Ybus/Sbus/Ibus/Va/Vm/V_re/V_im/F`는 FP64다.
  `V_re/V_im`은 FP64 derived cache이고, single-case 결과 다운로드는 FP64 `Va/Vm`에서 재구성한다.
- 차이:
  `batch_size > 1` 실제 실행은 아직 `NewtonSolver::solve_batch()`에서 막혀 있다.
  storage에는 batch-major 크기 산정과 offset helper를 넣었지만,
  cuDSS uniform batch descriptor와 kernel batch indexing은 아직 Phase 04~06 대상이다.

- 계획:
  Phase 02는 storage layout 중심으로 진행한다.
- 실제 구현:
  초기 구현에서는 `V_re/V_im`을 FP32 cache로 낮추면서 mixed CUDA mismatch/Jacobian 커널 시그니처도
  FP32 입력을 읽도록 함께 맞췄다. 이후 Texas Univ 수렴성 점검을 반영해 `V_re/V_im`을 FP64 cache로
  되돌렸으나, `Ybus32 * V32 -> Ibus64` 실험을 위해 다시 FP32 cache로 낮췄다.
  해당 실험이 크게 악화되어 이후 `Ybus64 * V64 -> Ibus64` 조합으로 다시 FP64 cache를 사용했다.
- 차이:
  이는 storage dtype 변경을 빌드 가능한 형태로 유지하기 위한 연결 작업이다.
  `Ibus64 = Ybus64 * V64` custom CSR kernel 분리는 아직 Phase 04에 남아 있으며,
  그 전까지 mismatch는 임시로 typed inline SpMV를 사용한다.

### Phase 03

- 계획:
  `Va/Vm`을 authoritative state로 유지하고 `J * dx = F`, `state -= dx` convention으로 바꾼다.
- 실제 구현:
  CUDA FP64/Mixed voltage update에서 `V -> Va/Vm` decompose kernel launch를 제거했다.
  cuDSS32/64 RHS 준비는 `-F`가 아니라 `F`를 사용하고, CUDA voltage update는 `dx`를 뺀다.
- 차이:
  CPU FP64 경로도 `J * dx = F`, `state -= dx` convention으로 맞췄다.
  backend 간 linear solve/update 부호 차이는 더 이상 남겨두지 않는다.

### Phase 04

- 계획:
  mismatch stage 안에서 `launch_compute_ibus()`,
  `launch_compute_mismatch_from_ibus()`,
  `launch_reduce_mismatch_norm()` 순서의 substage schedule을 사용한다.
- 실제 구현:
  CUDA Mixed path에 `Ibus64 = Ybus64 * V64` custom CSR kernel,
  `F64 = V64 * conj(Ibus64) - Sbus64` kernel, batch별 norm reduction kernel을 추가했다.
- 차이:
  `ctx.normF`는 batch별 norm vector를 device에서 만든 뒤 host로 norm vector만 복사해
  max 값을 계산한다. 즉 `F` 전체 host roundtrip은 제거했지만, iteration control을 위해
  작은 host copy는 남아 있다.

- 계획:
  norm reduction은 device에서 수행한다.
- 실제 구현:
  정상 경로는 device reduction만 사용한다.
- 차이:
  `ENABLE_DUMP`가 켜진 진단 경로에서는 residual dump를 위해 `F` 전체를 host로 복사한다.
  이는 hot path가 아니라 dump-only path다.

### Phase 05

- 계획:
  Jacobian fill은 offdiag direct write와 `Ibus` 기반 diagonal fill로 분리한다.
- 실제 구현:
  CUDA Mixed edge/vertex FP32 op를 thin orchestrator로 바꾸고,
  offdiag/self mapped term kernel과 `Ibus` 기반 diagonal correction kernel을 분리했다.
- 차이:
  정확한 diagonal 값을 위해 self Ybus entry는 offdiag/self kernel에서 direct write하고,
  diagonal kernel이 `j * V_i * conj(I_i)` 및 `conj(I_i) * Vhat_i` correction을 더한다.
  이는 CPU Jacobian의 `self term write + Ibus diagonal +=` 구조와 같은 convention이다.

- 계획:
  `d_J_values.memsetZero()`는 모든 J entry write가 검증되기 전까지 제거하지 않는다.
- 실제 구현:
  memset은 아직 유지했다.
- 차이:
  `/workspace/gpu-powerflow/exp/20260420/jac_asm` 실험도 edge 경로가 vertex보다 빠르며
  benchmark에서 memset을 사용하고 있었다. memset 제거는 poison/coverage 검증 이후 별도 단계로 남긴다.

- 참고 실험:
  `exp/20260420/jac_asm/results/jac_asm_gpu1_summary.md` 기준 모든 케이스에서 edge fill이
  vertex fill보다 빨랐다. 따라서 현재 구현은 edge path를 기본/우선 경로로 유지하고,
  vertex path는 옵션 호환 경로로 남긴다.

### Phase 06

- 계획:
  cuDSS는 uniform batch descriptor를 사용하고 single-case도 `B=1`인 같은 path를 탄다.
- 실제 구현:
  CUDA Mixed FP32 linear solve는 v1 구현과 동일하게 `CUDSS_CONFIG_UBATCH_SIZE`를 설정하고,
  단일 `cudssMatrixCreateCsr`/`cudssMatrixCreateDn` descriptor에 flat batch-major
  `J/rhs/dx` buffer를 연결한다. `B=1`도 같은 UBATCH path를 사용한다.
- 차이:
  초기 구현은 `cudssMatrixCreateBatchCsr`/`cudssMatrixCreateBatchDn` API를 사용했지만,
  현재 설치된 cuDSS가 `CUDSS_STATUS_NOT_SUPPORTED`를 반환했다. v1 브랜치의 동작 코드가
  `CUDSS_CONFIG_UBATCH_SIZE + 단일 CSR/Dn descriptor` 방식이므로 그 방식으로 교체했다.
  descriptor 생성은 `initialize()`가 아니라 첫 `factorize()` 시점으로 지연했다.
  이유는 `initialize()` 시점에는 `solve_batch()`의 `batch_size`를 아직 알 수 없고,
  `upload()` 이후 `J_values/F/dx` buffer가 batch-major 크기로 확정되기 때문이다.

- 계획:
  host-side RHS roundtrip을 제거한다.
- 실제 구현:
  CUDA Mixed FP32 path는 `d_F(double)`에서 `rhs(float)`로 device cast kernel을 사용한다.
  CUDA FP64 path는 `d_F`를 cuDSS RHS dense matrix에 직접 연결한다.
- 차이:
  dump-only 진단 경로에서는 linear system dump를 위해 `F/J/dx`를 host로 복사한다.

- 계획:
  기본 실행 모델은 batch이고 single-case는 `B=1`이다.
- 실제 구현:
  `solve_batch(batch_size > 1)`은 CUDA Mixed path에서만 열었다.
- 차이:
  CPU FP64와 CUDA FP64 storage는 아직 single-case compatibility path다.
  또한 active mask/active compaction은 아직 없어서 모든 batch가 같은 iteration count로 진행되고,
  `ctx.converged`는 batch별 norm의 max 기준으로 판정한다.

### Phase 07

- 계획:
  correctness와 benchmark로 `B=1` 회귀 및 `B>1` 수렴을 확인한다.
- 실제 구현:
  benchmark에 `--batch-size`를 추가했고, CUDA Mixed 기본 경로에서 `solve_batch()`를 직접 실행하게 했다.
  CUDA Mixed `B=2` smoke test도 추가했다.
- 차이:
  현재 작업 환경에서는 외부 dump dataset과 GoogleTest dependency가 없어 runtime convergence 결과를
  이 단계에서 수집하지 못할 수 있다. 대신 test/benchmark hook을 추가하고 빌드 검증 대상으로 삼았다.
  CPU-only 및 CUDA build는 통과했고, `ctest`는 GoogleTest 미탑재로 생성된 테스트가 없었다.

- 계획:
  모든 CUDA 경로가 기본 batch path를 탄다.
- 실제 구현:
  `B>1` benchmark 실행은 CUDA Mixed 기본 경로에서만 허용했다.
- 차이:
  modified schedule 및 hybrid ablation benchmark는 수동 stage 조합이라 batch-aware execution plan을
  우회한다. 이 경로들은 이번 단계에서 명시적으로 `B=1`만 받도록 막았다.

- 계획:
  `Va/Vm` authoritative state 설명과 구현을 통일한다.
- 실제 구현:
  hybrid ablation의 CUDA→CPU 전압 복사를 FP32 `V_re/V_im` cache가 아니라 FP64 `Va/Vm` 기반
  재구성으로 바꿨다.
- 차이:
  없음. 이는 원래 전압 상태 계획과 맞춘 cleanup이다.

- 계획:
  cuDSS uniform batch descriptor는 Jacobian batch와 dense RHS/solution batch를 올바른 shape로 표현한다.
- 실제 구현:
  Phase 07 점검 중 FP32 batch dense descriptor의 column count가 `dimF`로 재사용되던 것을
  별도 `1` column 배열로 분리했다.
- 차이:
  원래 계획과 다른 방향은 아니며, Phase 06 구현의 shape 정합성 보정이다.

- 계획:
  CUDA linear solve diagnostics는 `J * dx = F` convention과 일치해야 한다.
- 실제 구현:
  dump-only linear residual 계산을 `J dx + F`에서 `J dx - F`로 수정했다.
  CUDA Mixed `B>1` dump에서는 linear residual 진단을 batch 0 slice에 대해 남긴다.
- 차이:
  원래 계획과 다른 방향은 아니며, Phase 03 sign convention 전환 후 남은 진단 경로 cleanup이다.
  batch별 전체 linear diagnostics dump는 파일 naming/layout을 정해야 하므로 후속으로 남겼다.

- 계획:
  전압 표현은 한때 `Va/Vm` FP64, `V_re/V_im` FP32 cache로 낮춰 memory bandwidth를 줄인다.
- 실제 구현:
  `V_re/V_im` cache를 FP64로 되돌렸다가, 후속 `Ybus32 * V32 -> Ibus64` 실험에서 다시 FP32 cache로 낮췄다.
  해당 실험 후 다시 FP64 cache로 복구했다. 현재 CUDA Mixed 전압 표현은 `Va/Vm`과 `V_re/V_im` 모두 FP64다.
- 차이:
  Texas Univ `case_ACTIVSg200` B=1에서 FP32 cache는 final mismatch가 약 `6e-5`에서 멈췄고,
  FP64 cache로 바꾼 뒤 약 `7.254e-8`까지 개선되었다. 다만 `1e-8` tolerance는 아직 실패하므로,
  당시 남은 수렴성 차이는 FP32 `Ibus/J/dx` 또는 FP32 cuDSS solve 영향을 별도로 분리해야 했다.

- 계획:
  초기 mixed profile은 `Sbus`도 FP32로 낮춰 bandwidth를 줄인다.
- 실제 구현:
  `Sbus`를 FP64로 되돌렸다. 현재 mismatch는 `F64 = V64 * conj(Ibus64) - Sbus64`로 계산한다.
- 차이:
  Texas Univ B=1 전체 sweep에서 모든 케이스가 `1e-8` 기준 50 iteration까지 미수렴했고,
  final mismatch가 `1e-7~2.5e-6` 범위에 걸렸다. 지정 전력 quantization 영향을 분리하기 위해
  `Sbus`를 FP64로 올린다. 이후 `Ibus`도 FP64로 올렸고, 현재 정책에서는 `Ybus`도 FP64로 유지한다.
- 검증:
  `texas_batch1_sbus_fp64_20260421_r3` sweep에서 `B=1`, `tol=1e-8`, `max_iter=50`,
  `warmup=1`, `repeats=3` 조건으로 다시 측정했다. 모든 12개 케이스는 여전히 50 iteration까지
  미수렴했다. `case_ACTIVSg200`, `case_ACTIVSg2000`, `case_ACTIVSg25k` 등 일부 final mismatch는
  개선됐지만, `Base_MIOHIN_76GW`, `Base_West_Interconnect_121GW` 등은 악화됐다.
  따라서 해당 시점의 수렴 바닥은 `Sbus` FP32 단독보다는 FP32 `Ybus/Ibus/J/dx` 또는
  FP32 cuDSS solve 쪽에 남아 있는 것으로 본다.

- 계획:
  `Ybus32`, `V_re/V_im32` 입력을 사용해 FP32 CUDA core 경로로 `Ibus`를 계산하되, `Ibus` 저장은 FP64로 둔다.
- 실제 구현:
  `d_V_re/im`을 FP32 cache로 낮추고, `d_Ibus_re/im`을 FP64 buffer로 올렸다.
  `compute_ibus_kernel`은 FP32 곱셈/누산과 warp reduction을 수행한 뒤 lane 0에서
  FP64 `Ibus`로 저장한다. mismatch와 Jacobian diagonal은 FP64 `Ibus`를 읽는다.
- 차이:
  이 `Ibus64`는 FP64 곱셈/누산 결과가 아니라 FP32로 계산된 값을 FP64 buffer에 저장한 것이다.
  따라서 이 실험은 Ibus 저장/후속 사용 precision과 FP32 core 경로의 영향을 보는 ablation이지,
  full FP64 residual과 같은 정밀도 검증은 아니다.
- 검증:
  `cuda_mixed_edge`, `B=1`, `warmup=0`, `repeats=1`, `tol=1e-8`, `max_iter=50`으로 Texas Univ
  12개 케이스를 한 번씩 실행했다. 모든 케이스가 50 iteration까지 미수렴했고, final mismatch는
  `case_ACTIVSg200`도 `1e-4` 수준으로 악화됐다. 즉 이 조합에서는 `Ibus`를 FP64 buffer로 저장하는
  이득보다 `V_re/V_im` FP32 cache 오차가 더 크게 보인다.

- 계획:
  직전 실험에서 악화된 `V_re/V_im` FP32 cache를 FP64로 되돌려 `Ybus32 * V64 -> Ibus64` 조합을 확인한다.
- 실제 구현:
  `d_V_re/im`을 다시 FP64 buffer로 바꾸고, Ibus/mismatch/Jacobian kernel은 FP64 V cache를 읽는다.
  `Ibus`와 `Sbus`는 FP64로 유지한다.
- 차이:
  이 조합은 FP32 CUDA core만 타는 Ibus 실험은 아니다. `Ybus` 값은 FP32지만, V cache와 누산/저장은
  FP64이므로 수렴성 회복 여부를 우선 확인하는 ablation이다.
- 검증:
  `cuda_mixed_edge`, `B=1`, `warmup=0`, `repeats=1`, `tol=1e-8`, `max_iter=50`으로 Texas Univ
  12개 케이스를 한 번씩 실행했다. 모든 케이스가 3~7 iteration 안에 수렴했고, final mismatch는
  `6e-12~3.2e-9` 범위였다. 따라서 직전 실패의 핵심 원인은 `Ibus` 저장 precision보다
  `V_re/V_im` FP32 cache 영향이 컸고, 당시 조합(`Ybus32`, `V64`, `Ibus64`, `J/dx32`)은
  `1e-8` 기준 수렴성을 회복한다.

- 계획:
  mixed profile의 `Ybus`도 FP64로 유지한다.
- 실제 구현:
  `CudaMixedBuffers::d_Ybus_re/im`은 FP64 buffer로 유지하고, mixed `Ibus` kernel은
  `Ybus64 * V64 -> Ibus64`를 계산한다. Jacobian fill은 `JScalar`와 `YbusScalar`를 분리해
  mixed에서 `Ybus64/V64/Ibus64`를 읽은 뒤 FP32 산술로 `J32`를 쓴다.
- 차이:
  현재 mixed profile에서 FP32로 남는 hot buffer는 `J_values`, `dx`, cuDSS RHS copy다.

- 계획:
  Jacobian의 `v_re/v_im` 입력을 FP32로 내려 Jacobian 입출력을 FP32 중심으로 맞춘다.
- 실제 구현:
  별도 FP32 `V/Ibus` cache를 두는 중간안을 제거했다. 당시 storage 경계의 `V_re/V_im`,
  `Vm`, `Ibus`는 FP64로 유지하고, edge/vertex/diag Jacobian kernel 내부에서 load 직후
  `float`로 cast해 FP32 산술로 `d_J_values`를 채운다.
- 차이:
  사용자 피드백을 반영해 “입출력은 FP64 허용, 내부 계산은 FP32”로 계획을 조정했다.
  이 방식은 mismatch residual의 FP64 입력을 오염시키지 않고 Jacobian 산술만 FP32 core 경로로
  제한한다.
- 검증:
  `cmake --build build/bench-end2end --target cupf_case_benchmark -j2`와 `git diff --check`를 통과했다.
  `case_ACTIVSg200`, `B=1`, `tol=1e-8`, `max_iter=50` smoke에서 `cuda_mixed_edge`는 3 iteration,
  final mismatch `3.219e-12`, 당시 vertex 프로파일은 3 iteration, final mismatch `2.971e-12`로 수렴했다.

- 계획:
  Jacobian 전용 FP32 cache를 두고, edge-based mixed Jacobian은 `Ibus` cache를 쓰면서 이전처럼
  한 kernel에서 조립한다.
- 실제 구현:
  `d_J_Ibus_re/im` FP32 mirror는 제거했다.
  voltage update는 FP64 `V_re/V_im` cache만 갱신하고,
  `compute_ibus`는 `Ibus64`만 기록한다.
  mixed edge kernel은 Ybus entry별 direct write를 수행하며, `i == j`인 diagonal Ybus entry에서
  self term write 직후 FP64 `Ibus`를 읽어 FP32로 변환한 diagonal correction을 더한다.
  mixed edge op는 더 이상 별도 diagonal correction kernel을 호출하지 않는다.
  mixed vertex offdiag/self kernel은 warp-per-bus에서 thread-per-bus로 바꿔, thread 하나가
  active bus 하나의 CSR row를 순회하게 했다.
- 차이:
  FP32 Ibus mirror의 추가 메모리와 write traffic을 피하고, FP64 Ibus 하나를 mismatch/Jacobian이 공유한다.
  Jacobian diagonal correction은 FP64 Ibus read 비용을 감수하고 kernel 내부 cast로 처리한다.
  다만 vertex path는 여전히 offdiag/self fill과 diagonal correction kernel을 분리한다.
- 검증:
  `cmake --build build/bench-end2end --target cupf_case_benchmark -j2`를 통과했다.
  `case_ACTIVSg200`, `B=1`, `tol=1e-8`, `max_iter=50` smoke에서 `cuda_mixed_edge`는 3 iteration,
  final mismatch `3.715e-12`, 당시 vertex 프로파일은 3 iteration, final mismatch `3.696e-12`로 수렴했다.
  같은 조건으로 Texas Univ 12개 케이스의 `cuda_mixed_edge` B=1 sweep을 실행했고, 모두 3~7 iteration 안에
  수렴했다. final mismatch 범위는 `4.284e-12`~`6.906e-09`였다.
  `case_ACTIVSg200`, `B=4` smoke에서도 edge는 3 iteration, final mismatch `4.479e-12`,
  vertex는 3 iteration, final mismatch `3.664e-12`로 수렴했다.
