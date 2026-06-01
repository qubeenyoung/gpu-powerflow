# core — 파이프라인·컨텍스트·adjoint·torch bridge

`core/`는 `NewtonSolver` 드라이버([newton_solver.md](newton_solver.md))를 제외한 내부
오케스트레이션을 담는다: 파이프라인 프로파일, 단계 컨텍스트, adjoint(backward) 드라이버,
torch zero-copy bridge, 그리고 두 개의 작은 수학/구조 헬퍼.

**소스**: `cpp/src/newton_solver/core/` (`pipeline.hpp`, `solver_contexts.hpp`,
`newton_solver_adjoint.{hpp,cpp}`, `adjoint_math.{hpp,cpp}`, `csr_transpose.{hpp,cpp}`,
`torch_bridge.cpp`, `newton_solver_cuda_bridge.hpp`).

---

## 1. 파이프라인 프로파일 (`pipeline.hpp`)

각 "프로파일"은 **storage + linear solver + adjoint 캐시**를 묶고, 고정된 NR 단계
메서드 집합을 노출한다:

```
initialize / upload / download_batch
ibus → mismatch → mismatch_norm → jacobian → prepare_rhs → factorize → solve → voltage_update
```

5개 프로파일: `CpuFp64Pipeline`, `CudaFp64Pipeline`, `CudaFp32Pipeline`,
`CudaMixedPipeline`, (`CUPF_ENABLE_CUSTOM_SOLVER` 시) `CudaFp64CustomPipeline`.

- 추상 가상 인터페이스는 **없다**. `SolverPipeline`은 이들의 `std::variant`이고,
  드라이버가 `std::visit`로 활성 프로파일의 메서드를 정적 디스패치한다. 프로파일은
  구조적으로 호환되며 정밀도/저장소/솔버 백엔드만 다르다.
- `static constexpr bool batch_supported` 가 batch_size>1 허용 여부를 표시한다
  (CUDA 3종 true, CPU·Custom false).
- `CpuLinearSolveAny`는 KLU/UMFPACK을 다시 variant로 감싸 CPU 선형 솔버를 런타임 선택한다.

## 2. 컨텍스트 (`solver_contexts.hpp`)

단계 메서드가 주고받는 POD 컨텍스트:

- **`InitializeContext`**: Ybus view, Jacobian 패턴/scatter map, 버스 인덱스 — `prepare()`로 전달(1회).
- **`SolveContext`**: per-solve 입력(Ybus/Sbus/V0 포인터, `batch_size`, stride,
  `ybus_values_batched`) — `upload()`로 전달.
- **`IterationContext`**: NR 루프 공유 상태(`iter`, `normF`, `converged`,
  `jacobian_updated_this_iter` 등). pv/pq와 config 참조.
- **`AdjointCache`**: 수렴 상태에서 캐시한 factorization 기록. "재사용 가능 여부" 플래그
  (`has_adjoint_cache`, `adjoint_cache_matches_final_state`,
  `factorization_supports_transpose_solve` 등)와 provenance/진단 플래그
  (`used_explicit_transpose`, `jt_*` 등), 차원/배치/backend 이름으로 구성. forward
  시작마다 리셋되고 `prepare_adjoint_cache()`가 채운다.

## 3. NR 루프 (`newton_solver.cpp::run_iteration_stages`)

반복마다: `ibus → mismatch → mismatch_norm`으로 잔차와 norm을 구하고, **루프 시작에서
수렴이면 jacobian/solve 전에 종료**한다(수렴 상태를 한 스텝 더 흔들지 않고, 비싼
factorize/solve를 건너뛰기 위함). 미수렴이면 `jacobian → prepare_rhs → factorize →
solve → voltage_update`. 각 단계는 `StageScope`로 NVTX+타이밍 래핑([utils.md](utils.md)).

## 4. Adjoint 드라이버 (`newton_solver_adjoint.cpp`)

수렴 상태에서 `J^T λ = dL/dx`를 푼다. `solve_adjoint_pipeline` 오버로드가 프로파일별로
있고, CUDA 3종은 템플릿 `solve_adjoint_cuda_pipeline<Pipeline, ValueT>`를 공유한다.

- **CPU(KLU/UMFPACK)**: 같은 LU로 `solve_transpose`(native J^T) — explicit transpose 불필요.
- **CUDA(cuDSS)**: native transpose solve가 없어 **explicit J^T**를 쓴다. forward 때
  `prepare_adjoint_cache`가 J^T 값을 device에서 scatter해 factorize·캐시하고, backward는
  삼각 solve만 한다. 캐시가 없고 옵션이 허용하면 backward에서 재구성(refactorize)한다.
- per-case 잔차/스텝은 batch-major. `cuda_storage_batch_size`/`cuda_storage_nnz_j`
  (storage 공용 accessor, [storage.md](storage.md))로 배치/nnz를 얻는다.

## 5. adjoint 수학 헬퍼 (`adjoint_math.cpp`)

순수 호스트 수학(솔버/CUDA 상태 없음, 단위테스트 친화):

- `validate_adjoint_args`: 차원·배치·널·stride·`dimF=n_pv+2·n_pq` 검증.
- `build_grad_state`: 버스별 dL/dVa·dL/dVm를 `[dVa@pv | dVa@pq | dVm@pq]` RHS로 패킹.
- `project_load_gradients`: λ를 버스별 부하 gradient로 사영(= −λ).
- `relative_residual_norm_csc`: CSC J로 `‖J^T λ − rhs‖/‖rhs‖`(long double 누산).

GPU에서 같은 패킹/사영을 하는 device 커널은 [ops/linear_solve.md](ops/linear_solve.md)
(torch bridge 경로).

## 6. csr_transpose.cpp

`build_transpose_pattern`: J의 CSR 패턴으로 J^T의 `row_ptr/col_idx`와
`src_to_transpose_pos`(원본 nz → J^T 슬롯 맵)를 counting sort로 1회 계산. cuDSS adjoint가
이 맵으로 값만 scatter해 J^T를 재구성한다.

## 7. torch bridge (`torch_bridge.cpp`)

torch CUDA 텐서의 raw device 포인터를 받아 호스트 복사 없이 forward/backward를 실행하는
`cupf::torch_api` 자유 함수들. `ensure_cuda_tensor_batch`로 배치 크기에 맞춰 버퍼를
resize하고, `set_pf_inputs_from_load`로 base power+load 텐서에서 Sbus/V0를 만들고, NR
루프/adjoint를 돌린 뒤 출력 텐서에 직접 쓴다. FP32/Mixed/FP64 모두 batch>1 지원. 이
파일은 항상 컴파일되지만(raw 포인터 표면), 파이썬 노출은 `CUPF_WITH_TORCH`에서만
(`torch_cupf_extension.cpp`). dtype은 프로파일과 일치해야 한다(FP64→float64,
FP32/Mixed→float32).
