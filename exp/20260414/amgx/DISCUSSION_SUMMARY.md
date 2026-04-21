# AMGX JFNK 논의 요약

Date: 2026-04-14

## 작업 맥락

- 현재 작업 브랜치는 `lsw_exp`이고, 별도 로컬 worktree로 `/workspace/gpu-powerflow-master`에 `master`를 같이 보이게 했다.
- `lsw_exp` 쪽에는 `cuPF`가 없어서 `/workspace/gpu-powerflow/cuPF -> /workspace/gpu-powerflow-master/cuPF` 심볼릭 링크로 연결했다.
- `nvidia-cudss-cu12`를 pip로 설치했고, CMake가 cuDSS를 찾을 수 있도록 `libcudss.so -> libcudss.so.0` 형태의 링크를 추가했다.
- `datasets/texas_univ_cases.zip`을 풀어서 12개 Texas University case의 cuPF dump를 `exp/20260414/amgx/cupf_dumps`에 만들었다.
- 실험 대상 실행 파일은 `exp/20260414/amgx/build/amgx_jfnk_probe`이다.

## 현재 코드의 큰 구조

- 비선형 문제는 Newton-Krylov 형태로 풀고, 외부 수렴 판정은 nonlinear mismatch `||F(x)|| <= --tolerance` 기준이다. 현재 주요 실험의 `--tolerance`는 `1e-8`이다.
- 내부 선형계는 FGMRES로 풀며, `--linear-tol=1e-2`는 선형계 상대 잔차 기준이다. 이것은 Newton-Raphson 비선형 수렴 기준과 별개다.
- JFNK matvec은 명시적 analytic Jacobian을 곱하지 않고 GPU에서 `F(x + eps v)`를 다시 계산해 `(F(x + eps v) - F(x)) / eps`를 만든다.
- `amg_fd` 전처리기는 coloring된 finite-difference Jacobian 값을 GPU에서 조립한 뒤 AMGX에 넘겨 FGMRES의 right preconditioner로 쓴다.
- 새로 추가한 `bus_block_jacobi_fd`는 같은 coloring FD Jacobian 값을 만들되 AMGX 대신 bus-local block inverse를 만들어 적용한다.

## 추가된 관측 기능

- `--trace-residuals PATH`
  - CSV 컬럼: `case,outer_iteration,phase,inner_iteration,residual_abs,residual_rel,rhs_norm`
  - 기록 phase: `outer_mismatch`, `linear_initial`, `linear_estimate`, `linear_restart`
- `--jacobian-error-csv PATH`
  - FD/coloring 전처리기 Jacobian 값과 analytic CUDA Jacobian 값을 같은 상태에서 비교한다.
  - CSV 컬럼: `case,outer_iteration,dim,nnz,fd_norm,exact_norm,diff_norm,relative_fro_error,max_abs_error,max_relative_error,max_abs_index,fd_at_max,exact_at_max`

## 수렴 판정에 대한 정리

- 지금까지 `converged=true`는 코드상 nonlinear residual이 `1e-8` 이하라는 뜻이다.
- 다만 `strict` 모드에서는 내부 FGMRES가 `linear_tol=1e-2`를 만족하지 못하면 Newton update를 적용하지 않고 멈춘다.
- `continue` 모드는 내부 선형계가 `linear_tol`에 도달하지 못해도 현재까지 얻은 GMRES 해로 Newton update를 적용하고 다음 외부 iteration으로 진행한다.
- 따라서 `continue=true`에서 "linear failure"가 있어도 nonlinear residual이 계속 줄어 최종적으로 `1e-8` 이하가 되면 비선형 문제 관점에서는 수렴으로 볼 수 있다.
- 반대로 `linear_tol=1e-2`만 만족했다고 해서 Newton-Raphson 수렴이라고 부를 수는 없다. 최종 판단은 nonlinear residual 기준이어야 한다.

## Base_Texas_66GW에서 확인한 사실

- 초기 nonlinear mismatch는 독립 Python 계산에서도 크게 나왔다.
  - `norm_inf = 99.0262319344648`
  - `P max = 13.499817519258048`
  - `Q max = 99.0262319344648`
- 따라서 큰 초기 residual은 CUDA mismatch kernel의 단순 오류 때문이 아니라 현재 `V0/Sbus/Ybus` 조합에서 실제로 큰 값으로 시작하는 것이다.
- `amg_fd` strict 모드에서는 첫 선형계가 실패하면서 update가 적용되지 않아 nonlinear residual이 그대로 `99.026...`에 머물렀다.
- `amg_fd + DENSE_LU_SOLVER + continue`에서는 residual이 `99.026 -> 2.070 -> ... -> 1.071` 정도까지 줄었지만 20 outer iteration 안에 `1e-8`에 도달하지 못했다.
- GMRES restart를 `30, 60, 100, 200`으로 키우면 첫 선형계 상대 잔차는 조금 좋아졌지만 모두 `1e-2`에는 못 미쳤다.

## Jacobian 오차 확인

`Base_Texas_66GW`에서 FD/coloring Jacobian과 analytic CUDA Jacobian을 비교했다.

| metric | value |
|---|---:|
| dim | 14431 |
| nnz | 104763 |
| `||J_fd||_F` | `1.069673148266e+05` |
| `||J_exact||_F` | `1.069673143565e+05` |
| `||J_fd - J_exact||_F` | `1.545206784142e-03` |
| relative Frobenius error | `1.444559764296e-08` |
| max absolute error | `2.009495647144e-04` |
| max relative error | `5.697468182237e-02` |

해석: FD/coloring으로 조립한 전처리기 행렬은 analytic Jacobian과 매우 가깝다. 현재 큰 수렴 문제의 주원인을 "Jacobian 조립 오차"로 보기는 어렵다.

## Graph coloring의 의미

- Coloring은 Jacobian의 여러 column을 동시에 finite difference로 추정하기 위한 압축 기법이다.
- 같은 row에서 충돌하지 않는 column끼리는 같은 색으로 묶을 수 있다.
- 색 수가 적을수록 `dim`번의 Jv가 아니라 `colors`번의 Jv만으로 sparse Jacobian 값을 얻을 수 있다.
- `Base_Texas_66GW`에서는 `dim=14431`, `colors=24`라서 약 `601x` 압축 효과가 있었다.
- 이 문제에서는 전력망 Jacobian의 sparsity 때문에 독립적으로 동시에 perturb할 수 있는 column이 많고, coloring이 매우 의미가 있다.

## 전처리기 구현 정리

- `none`
  - 전처리기 없이 identity 적용.
- `amg_fd`
  - coloring FD로 sparse Jacobian 값을 조립한다.
  - AMGX scalar CSR AMG를 right preconditioner로 적용한다.
  - coarse solver는 `DENSE_LU_SOLVER`로 바꿔 실험했다.
- `bus_block_jacobi_fd`
  - coloring FD로 만든 sparse Jacobian 값에서 bus-local diagonal block만 뽑아 inverse를 만든다.
  - PV bus는 `dP/dtheta` 1x1 block을 쓴다.
  - PQ bus는 `[dP/dtheta dP/dVm; dQ/dtheta dQ/dVm]` 2x2 block을 쓴다.
  - 이것은 AMG는 아니고, 물리적으로 한 bus 안의 theta/Vm 결합을 국소 block으로 반영한 block Jacobi preconditioner다.

## 현재 결론

- scalar CSR AMGX AMG는 이 전력 조류 Jacobian ordering/구조에서 큰 case의 선형계 수렴을 충분히 개선하지 못했다.
- FD/coloring Jacobian 조립 자체는 정확해 보인다.
- `bus_block_jacobi_fd`는 단순하지만 `Base_Texas_66GW`와 여러 case에서 nonlinear residual을 훨씬 잘 줄였다.
- 다만 `continue` 모드가 필요했던 case가 많으므로, 내부 선형계 실패와 외부 Newton update의 관계는 별도 정책으로 더 정리해야 한다.
