# 다음 3개 실험 계획

Date: 2026-04-14

## 목표

현재 관측은 두 가지를 동시에 말한다.

- FD/coloring Jacobian 조립은 analytic Jacobian과 거의 일치한다.
- AMGX scalar AMG는 현재 ordering의 전력 조류 Jacobian에서 약하고, bus-local block Jacobi는 의미 있는 개선을 보인다.

따라서 다음 실험은 "AMGX가 더 좋은 그래프를 보게 만들기", "bus-local block 정보를 AMGX 앞뒤에 붙이기", "고정 2x2 block system으로 재정의하기" 순서로 진행한다.

## Experiment 1: Scalar CSR row/column permutation

우선순위: 높음

가설:

- 수학적 선형계는 그대로 두고 AMGX에 올리기 직전 row/column permutation만 적용하면, 같은 bus의 theta/Vm 자유도가 그래프상 가까워진다.
- AMGX generic aggregation이 bus-local coupling을 더 자연스럽게 볼 가능성이 있다.

구현:

- 기존 `A_fd` scalar CSR 조립 경로는 유지한다.
- AMGX 업로드 직전 permutation vector `p`와 inverse permutation `pinv`를 만든다.
- 후보 ordering:
  - `bus_local`: `bus 0 local vars, bus 1 local vars, ...`
  - `pv_then_pq_interleaved`: `[PV theta] [PQ theta, PQ Vm interleaved]`
- right preconditioner 적용 시에는 `z = P^T M_perm^{-1} P r` 형태가 되도록 rhs와 solution을 permutation/unpermutation한다.
- FGMRES/JFNK matvec의 원래 변수 ordering은 유지한다.

측정:

- 대상: 12개 전체 case
- 비교 기준: `amg_fd` 기존 ordering, `bus_block_jacobi_fd`
- 기록: nonlinear residual trace, linear residual trace, total inner, linear failures, wall time, convergence count

성공 기준:

- 기존 `amg_fd`보다 수렴 case 수가 증가하거나, 실패 case의 final nonlinear residual/inner iteration이 명확히 개선된다.
- 구현 비용 대비 개선이 없으면 다음 실험으로 넘긴다.

## Experiment 2: Bus-local block Jacobi와 scalar AMGX 결합

우선순위: 높음

가설:

- bus block Jacobi가 local theta/Vm coupling을 잘 잡고, AMGX가 남은 global graph error를 일부 줄일 수 있다.
- 단독 `bus_block_jacobi_fd`와 단독 `amg_fd`보다 결합 전처리기가 더 안정적일 수 있다.

구현 후보:

- additive 형태: `M^{-1} r = M_block^{-1} r + M_amgx^{-1} r`
- multiplicative 형태: `z1 = M_block^{-1} r`, `r2 = r - A z1`, `z = z1 + M_amgx^{-1} r2`
- 반대 순서 multiplicative 형태: `z1 = M_amgx^{-1} r`, `r2 = r - A z1`, `z = z1 + M_block^{-1} r2`

주의:

- multiplicative 형태는 전처리기 적용 중 sparse matvec `A z1`이 필요하므로 비용이 증가한다.
- FGMRES는 iteration마다 preconditioner가 변해도 허용되지만, 실험 로그에서는 어떤 조합을 썼는지 명확히 남겨야 한다.
- 먼저 `Base_Texas_66GW`, `Base_West_Interconnect_121GW`, `case_ACTIVSg25k`, `case_ACTIVSg70k`, `Base_Eastern_Interconnect_515GW` 위주로 본다.

측정:

- `strict`와 `continue`를 둘 다 측정한다.
- 특히 `continue`에서 nonlinear residual은 줄지만 linear failure가 많은 case의 linear failure 수가 줄어드는지 본다.

성공 기준:

- `bus_block_jacobi_fd + continue`의 8/12 수렴보다 개선된다.
- 또는 4개 실패 case 중 하나 이상이 `1e-8`에 도달한다.
- wall time이 크게 증가한다면 residual 개선폭과 함께 판단한다.

## Experiment 3: Uniform 2x2 block CSR system

우선순위: 낮음

가설:

- AMGX에 scalar CSR이 아니라 block size 2의 시스템으로 올리면 bus의 theta/Vm 결합을 AMG가 더 직접적으로 볼 수 있다.

어려운 점:

- AMGX block CSR API는 고정 block size를 요구한다.
- PV bus는 실제로 theta만 미지수인 1자유도이므로, uniform 2x2 block system을 만들려면 padding이나 보조 변수/보조 식이 필요하다.
- padding 방식에 따라 conditioning과 null/identity row 처리가 solver에 영향을 줄 수 있다.

구현 후보:

- 모든 non-slack bus를 2자유도 block으로 맞춘다.
- PQ bus는 기존 theta/Vm 2개를 그대로 쓴다.
- PV bus는 Vm에 해당하는 보조 unknown/equation을 추가한다.
- 보조 equation은 identity 또는 강한 diagonal regularization으로 구성하되, 실제 Newton update에는 theta 성분만 반영한다.

측정:

- 먼저 작은 case `case_ACTIVSg200`, `case_ACTIVSg500`, `case_ACTIVSg2000`에서 correctness를 확인한다.
- 이후 `Base_Texas_66GW`와 실패 대형 case로 확장한다.
- scalar reordered AMGX와 bus block Jacobi 결합 실험보다 나을 때만 전체 12개를 돈다.

성공 기준:

- block CSR 구현이 원래 문제와 같은 Newton step을 안정적으로 재현한다.
- scalar CSR AMGX보다 linear residual 감소가 뚜렷하다.
- padding으로 인한 부작용이 없거나 관리 가능하다.

## 권장 순서

1. Experiment 1: scalar CSR permutation
2. Experiment 2: bus block Jacobi + scalar AMGX 결합
3. Experiment 3: uniform 2x2 block CSR

1번은 구현 비용이 낮고 현재 의견과 가장 잘 맞는다. 2번은 이미 효과가 확인된 bus block Jacobi를 활용하므로 실용적이다. 3번은 가능하지만 설계 변경이 커서 앞의 두 실험 결과를 보고 진행하는 것이 낫다.
