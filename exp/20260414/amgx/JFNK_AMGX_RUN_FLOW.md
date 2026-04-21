# JfnkLinearSolveAmgx::run 흐름 정리

Date: 2026-04-14

대상 코드: `exp/20260414/amgx/cpp/jfnk_amgx_gpu.cu`

대상 메소드: `void JfnkLinearSolveAmgx::run(IterationContext& ctx)`

이 문서는 예외 처리와 방어 코드의 세부 분기를 제외하고, GPU JFNK 선형해법이 한 outer Newton iteration 안에서 어떤 순서로 동작하는지 추상화해 정리한다.

## 한 줄 요약

`run()`은 현재 mismatch `F(x)`로부터 선형계 `J dx = -F(x)`의 우변을 만들고, 필요하면 finite-difference Jacobian 기반 전처리기를 구성한 뒤, GPU에서 JFNK 방식의 Jacobian-vector product와 FGMRES 반복으로 Newton update `dx`를 계산해 `storage_.d_dx`에 저장한다.

## 입력과 출력

입력 상태:

- `ctx.iter`: 현재 outer Newton iteration 번호. trace 기록에 사용된다.
- `storage_`: 현재 전압, mismatch, Ybus, PV/PQ bus index, diagonal Jacobian 정보 등 GPU resident 데이터를 가진다.
- `options_`: FGMRES 반복 수, restart 크기, 선형 허용오차, 전처리기 종류, permutation, residual trace 옵션 등을 가진다.
- `analyze()`에서 저장해 둔 host-side Ybus graph와 PV/PQ index: finite-difference sparse pattern 생성에 사용된다.

출력 상태:

- `storage_.d_dx`: 이번 선형해법이 계산한 Newton update.
- `stats_`: 이번 solve의 성공 여부, 내부 반복 수, Jv 호출 수, 추정 상대 잔차, epsilon, 실패 사유, 누적 시간 통계.
- 선택적으로 residual trace와 Jacobian error CSV.

## 전체 흐름

1. 선형 solve 단위 상태를 초기화한다.
2. 우변 `rhs = -F(x)`와 초기해 `x = 0`을 만든다.
3. JFNK용 `apply_jv(direction, output)` 연산을 정의한다.
4. 옵션에 따라 finite-difference Jacobian 기반 전처리기를 구성한다.
5. 전처리기 적용 함수 `apply_preconditioner(input, output)`을 정의한다.
6. 우변 norm과 초기 residual을 계산하고, 이미 충분히 작으면 `dx = 0`으로 종료한다.
7. FGMRES restart cycle을 반복하며 Krylov basis를 만들고 least-squares 문제를 풀어 update를 누적한다.
8. 성공 또는 반복 한도 도달 후, 최종 `x`를 `storage_.d_dx`로 복사하고 통계를 갱신한다.

## 1. Solve 상태와 작업 버퍼 준비

메소드 초반에는 현재 solve에 대한 통계를 초기화한다.

- `last_success = false`
- `last_iterations = 0`
- `last_jv_calls = 0`
- `last_estimated_error = NaN`
- `last_epsilon = 0`
- `last_failure_reason = ""`

그 다음 cuBLAS handle과 GPU 작업 버퍼를 만든다.

주요 벡터 버퍼:

- `d_base_f`: 현재 상태의 mismatch `F(x)` 복사본
- `d_rhs`: 선형계 우변 `-F(x)`
- `d_x`: FGMRES가 누적하는 선형해, 최종적으로 `dx`가 된다.
- `d_r`: 현재 residual `rhs - J x`
- `d_w`: Arnoldi 과정에서 사용하는 작업 벡터
- `d_ax`: `J x` 또는 보정 residual 계산용 작업 벡터
- `d_seed`, `d_fd_jv`: coloring finite-difference Jacobian 조립용 seed/Jv 버퍼
- voltage scratch 버퍼들: perturb된 전압 상태에서 mismatch를 다시 계산하기 위한 임시 버퍼
- `d_basis`: FGMRES basis `v_j`
- `d_z_basis`: right preconditioned basis `z_j = M^{-1} v_j`

초기 선형계는 다음처럼 잡는다.

```text
base_f = F(x_current)
rhs    = -base_f
x      = 0
r      = rhs
```

## 2. JFNK Jacobian-vector product 정의

`apply_jv(direction, output)`은 명시적 analytic Jacobian을 곱하지 않는다. 대신 finite difference로 다음 값을 만든다.

```text
output = (F(x + eps * direction) - F(x)) / eps
```

추상 흐름:

1. `direction`의 infinity norm을 측정한다.
2. `options_.auto_epsilon`이면 direction 크기에 맞춰 `eps`를 정하고, 아니면 `fixed_epsilon`을 사용한다.
3. 현재 복소 전압 `V`를 `Va`, `Vm`으로 분해한다.
4. `direction`을 PV/PQ 변수 배치에 맞춰 `Va`, `Vm` perturbation으로 적용한다.
5. perturb된 `Va`, `Vm`에서 복소 전압을 재구성한다.
6. perturb된 전압으로 power mismatch `F(x + eps direction)`을 다시 계산한다.
7. `base_f = F(x)`와 차분해 Jv 결과를 만든다.
8. Jv 호출 수와 Jv 관련 시간 통계를 누적한다.

이 함수는 FGMRES의 matvec뿐 아니라 전처리기 구성, multiplicative 전처리기 조합에서도 재사용된다.

## 3. Finite-difference 전처리기 구성

전처리기가 `amg_fd` 또는 `bus_block_jacobi_fd`이면 sparse finite-difference Jacobian 값을 먼저 만든다.

공통 조립 흐름:

1. `analyze()`에서 저장한 Ybus graph와 PV/PQ index로 sparse FD pattern을 만든다.
2. pattern을 GPU로 업로드한다.
3. graph coloring의 각 color에 대해:
   - 같은 color의 column들을 `d_seed`에 동시에 set한다.
   - `apply_jv(d_seed, d_fd_jv)`를 호출한다.
   - 충돌 없는 column들의 결과를 CSR value 배열 `d_fd_values`에 scatter한다.
4. 옵션이 켜져 있으면 FD Jacobian과 analytic CUDA Jacobian의 오차를 CSV로 기록한다.

이 단계가 끝나면 `d_fd_values`에는 전처리기용 sparse Jacobian 값이 들어 있다.

### amg_fd 경로

`preconditioner == "amg_fd"`인 경우, FD Jacobian CSR을 AMGX preconditioner에 올린다.

- `permutation == "none"`이면 원래 CSR row/column ordering 그대로 AMGX에 전달한다.
- `permutation != "none"`이면 host에서 matrix를 재배열한 뒤, permutation된 CSR을 AMGX에 전달한다.
- 이후 전처리기 적용 시에도 rhs를 old ordering에서 new ordering으로 permute하고, AMGX solve 결과를 다시 original ordering으로 되돌린다.

### bus_block_jacobi_fd 경로

`preconditioner == "bus_block_jacobi_fd"`인 경우, 같은 FD Jacobian 값에서 bus-local diagonal block inverse를 만든다.

- PV bus는 `dP/dtheta` 1x1 block inverse를 쓴다.
- PQ bus는 `[dP/dtheta, dP/dVm; dQ/dtheta, dQ/dVm]` 2x2 block inverse를 쓴다.

`preconditioner_combine`이 `single`이 아니면 bus block Jacobi와 AMGX를 함께 쓰기 위해 AMGX 쪽 CSR과 임시 벡터도 준비한다.

## 4. 전처리기 적용 함수 정의

`apply_preconditioner(input, output)`은 옵션에 따라 right preconditioner `M^{-1}`을 적용한다.

지원되는 추상 동작:

- `amg_fd`: AMGX solve를 적용한다.
- `bus_block_jacobi_fd + single`: bus-local block inverse만 적용한다.
- `bus_block_jacobi_fd + additive`: block 결과와 AMGX 결과를 더한다.
- `bus_block_jacobi_fd + block_then_amg`:
  1. `z1 = M_block^{-1} r`
  2. `r2 = r - J z1`
  3. `z2 = M_amg^{-1} r2`
  4. `z = z1 + z2`
- `bus_block_jacobi_fd + amg_then_block`:
  1. `z1 = M_amg^{-1} r`
  2. `r2 = r - J z1`
  3. `z2 = M_block^{-1} r2`
  4. `z = z1 + z2`
- `none`: 입력을 그대로 출력으로 복사한다.

FGMRES에서는 이 함수로 `z_j = M_j^{-1} v_j`를 만든 뒤, `J z_j`를 Krylov matvec으로 사용한다. 이 구조 때문에 코드의 solver는 left-preconditioned GMRES가 아니라 right-preconditioned FGMRES 형태에 가깝다.

## 5. 초기 residual 평가

우변 norm은 다음처럼 계산된다.

```text
rhs_norm = ||rhs||_2
atol     = linear_tolerance * rhs_norm
```

초기 residual은 `r = rhs`이므로:

```text
residual_norm = ||r||_2
relative      = residual_norm / rhs_norm
```

이 값은 `linear_initial` phase로 trace에 기록된다.

초기 residual이 이미 `atol` 이하라면 선형해는 `x = 0`으로 충분하므로 `storage_.d_dx`에 0을 복사하고 종료한다.

## 6. FGMRES restart cycle

메인 반복은 `max_inner_iterations`에 도달할 때까지 restart 단위로 진행된다.

각 restart cycle의 시작:

1. 현재 residual norm `beta = ||r||_2`를 계산한다.
2. `beta <= atol`이면 수렴으로 처리한다.
3. 이번 cycle의 basis 크기를 `min(gmres_restart, 남은 반복 수)`로 정한다.
4. Hessenberg 행렬 `H`와 우변 `g = [beta, 0, ...]`를 만든다.
5. 첫 basis를 `v_0 = r / beta`로 둔다.

각 inner iteration `j`의 흐름:

1. 전처리기를 적용해 `z_j = M^{-1} v_j`를 만든다.
2. JFNK matvec을 적용해 `w = J z_j`를 만든다.
3. 기존 basis `v_i`들에 대해 modified Gram-Schmidt 직교화를 수행한다.
   - `H(i, j) = dot(v_i, w)`
   - `w = w - H(i, j) v_i`
4. 남은 벡터 norm `h_next = ||w||_2`를 계산하고 `H(j+1, j)`에 저장한다.
5. breakdown이 아니면 다음 basis `v_{j+1} = w / h_next`를 저장한다.
6. 현재까지의 작은 least-squares 문제를 CPU Eigen으로 푼다.

least-squares 문제는 다음 형태다.

```text
min_y || g - H y ||_2
```

이를 통해 현재 cycle의 최선 계수 `best_y`와 residual estimate를 갱신한다. 추정 상대 잔차는 `linear_estimate` phase로 trace에 기록된다.

추정 상대 잔차가 `linear_tolerance` 이하이거나 happy breakdown이면:

```text
x = x + sum_k best_y[k] * z_k
```

를 적용하고 선형 solve를 성공으로 끝낸다.

## 7. Restart 시 residual 재계산

restart cycle 안에서 수렴하지 못했지만 최소 하나 이상의 basis가 만들어졌다면, 지금까지의 best coefficient를 현재 해 `x`에 더한다.

그 다음 실제 residual을 다시 계산한다.

```text
ax = J x
r  = rhs - ax
```

이 값은 `linear_restart` phase로 trace에 기록된다. 이후 다음 restart cycle은 새 residual `r`에서 다시 시작한다.

## 8. 종료 처리

반복이 끝나면 성공 여부와 실패 사유를 정리하고, solve 통계를 누적한다.

최종적으로 FGMRES가 누적한 선형해 `d_x`를 Newton update 저장 위치로 복사한다.

```text
storage_.d_dx = d_x
```

따라서 호출자는 이후 outer Newton 단계에서 이 `dx`를 사용해 전압 상태를 갱신한다.

## 데이터 흐름 요약

```text
현재 GPU 상태
  storage_.d_V, storage_.d_F, storage_.d_Ybus, PV/PQ index
        |
        v
우변 구성
  rhs = -F(x), x = 0, r = rhs
        |
        v
전처리기 준비
  coloring seed -> Jv 반복 -> FD sparse Jacobian values
        |
        +--> AMGX AMG setup
        |
        +--> bus-local block inverse setup
        |
        v
FGMRES 반복
  v_j -> M^{-1} v_j = z_j -> J z_j -> Arnoldi/least-squares
        |
        v
해 누적
  x += sum(y_j z_j)
        |
        v
Newton update 저장
  storage_.d_dx = x
```

## trace phase 의미

- `linear_initial`: solve 시작 직후 초기 residual.
- `linear_estimate`: inner iteration마다 Hessenberg least-squares로 추정한 residual.
- `linear_restart`: restart cycle이 끝난 뒤 실제 `J x`로 다시 계산한 residual.

## 핵심 특징

- Jacobian-vector product는 analytic Jacobian matvec이 아니라 `F(x + eps v)` 기반 finite difference다.
- 전처리기용 행렬도 같은 Jv 연산을 coloring seed에 적용해 sparse FD Jacobian으로 조립한다.
- AMGX는 선형 solver 본체가 아니라 FGMRES의 preconditioner로 쓰인다.
- `bus_block_jacobi_fd`는 FD Jacobian의 bus-local diagonal block만 이용하는 경량 전처리기다.
- `preconditioner_combine` 옵션은 bus block Jacobi와 AMGX를 additive 또는 multiplicative correction 형태로 결합한다.
- 선형 solver의 최종 산출물은 `storage_.d_dx`이며, nonlinear convergence 자체는 이 메소드 바깥의 outer Newton 루프가 판단한다.
