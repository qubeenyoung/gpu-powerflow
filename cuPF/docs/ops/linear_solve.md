# ops/linear_solve — 직접 희소 선형 솔버

NR 스텝 `J·dx = F`(forward)와 adjoint `J^T λ = b`를 푼다. CUDA는 cuDSS, CPU는
KLU/UMFPACK, (선택) 외부 custom 솔버.

**소스**: `cpp/src/newton_solver/ops/linear_solve/` — `cuda_cudss.{cpp,hpp}`,
`cudss_config.hpp`, `cpu_klu.{cpp,hpp}`, `cpu_umfpack.{cpp,hpp}`,
`cuda_custom_solver.{cpp,hpp}`(gated), `cuda_linear_solve_kernels.hpp`,
`linear_solve_kernels.cu`, `torch_bridge_kernels.cu`.

---

## 1. cuDSS (CUDA) — `cuda_cudss.cpp`

cuDSS 직접 LU. 한 NR 반복은 4단계로 진행: **ANALYSIS**(패턴만, 1회 심볼릭) →
**FACTORIZATION**(첫 수치 LU) → 이후 **REFACTORIZATION**(구조 재사용, 값만 갱신) →
**SOLVE**(삼각 solve로 dx).

- 모든 cuDSS 핸들/디스크립터는 PIMPL `State`에 격리(공개 헤더는 cuDSS 타입 무관).
- **배치**: 전체 배치가 하나의 uniform-batch 문제. `cudss_config.hpp::configure_solver`가
  `CUDSS_CONFIG_UBATCH_SIZE`를 설정하고, batch-major 연속 버퍼가 그대로 입력 — B=1/B>1
  동일 경로. 디스크립터는 (batch, dimF, nnz_J)로 캐시하고 그 값이 바뀔 때만 재생성.
- **정밀도/RHS 선택**: double 및 (float+FP32)는 `buf.d_F`를 in-place로 풀고,
  (float+Mixed)는 FP64 잔차를 FP32 RHS로 down-cast(`prepare_rhs`)해서 푼다.
- 배치/nnz는 storage 공용 accessor `cuda_storage_batch_size`/`cuda_storage_nnz_j`
  ([storage.md](../storage.md))로 얻는다.

### adjoint (explicit J^T)

cuDSS는 이 구성에서 native transpose solve가 없다(`supports_transpose_solve()==false`).
그래서 J^T를 **명시적으로** 만든다: J^T sparsity는 `initialize`에서 1회 계산한 값-순열
맵([core.md](../core.md) csr_transpose)이고, 매 pass는 현재 J 값을 그 맵으로 device에서
scatter해 J^T 값을 채운 뒤 factorize·캐시한다. 캐시 후 반복 adjoint solve는 삼각 solve만
부담(`solve_adjoint_explicit_transpose_cached`); 호스트 RHS 경로(`..._host`)도 있다.

### cuDSS 옵션 (`cudss_config.hpp`)

reordering 알고리즘(빌드: `CUPF_CUDSS_REORDERING_ALG`), matching 전처리/알고리즘
(런타임 `CuDSSOptions.use_matching`/`matching_alg`), pivot epsilon, host MT
(`CUPF_CUDSS_ENABLE_MT`/`HOST_NTHREADS`), nested-dissection 레벨을 cudssConfig에 연결.

## 2. CPU — KLU / UMFPACK

둘 다 SuiteSparse CSC 직접 LU이며 동일한 `initialize → factorize → solve` 인터페이스.
symbolic은 sparsity만 의존(1회), numeric은 매 반복. **native transpose solve**를
지원(`klu_tsolve` / `UMFPACK_At`)해 adjoint를 **같은 LU**로 풀므로 explicit J^T가 불필요
(cuDSS와 대비). UMFPACK은 transpose solve 때 행렬 배열을 다시 넘겨야 해 factorize 시점의
포인터(`ap_/ai_/ax_`)를 캐시한다. `CpuLinearSolverKind`로 선택.

## 3. custom (선택)

`CUPF_ENABLE_CUSTOM_SOLVER` 빌드 시 외부 `custom_linear_solver` 라이브러리를 같은
analyze→factorize→solve 인터페이스로 감싸는 어댑터(FP64 단일 케이스). adjoint는 미구현
(던짐).

## 4. 지원 device 커널

선언은 `cuda_linear_solve_kernels.hpp`, 정의는 두 TU로 분리:

- **`linear_solve_kernels.cu`** (cuDSS 경로): `launch_prepare_rhs`(FP64→FP32 RHS
  down-cast), `launch_transpose_csr_values`(J 값을 J^T 순서로 scatter).
- **`torch_bridge_kernels.cu`** (torch 경로 I/O): `gather_adjoint_rhs`(버스별 dL/dx를
  dimF RHS로 패킹), `project_load_gradients`(λ→부하 gradient scatter; pv/pq disjoint를
  이용한 O(n_bus) 산포 — 버스당 리스트 선형 스캔 제거), `set_pf_inputs_from_load`
  (base power+load→Sbus/V), `copy_voltage_outputs`(Va/Vm 출력 캐스팅).
