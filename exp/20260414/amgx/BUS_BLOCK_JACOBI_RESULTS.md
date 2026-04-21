# Bus Block Jacobi 실험 결과

Date: 2026-04-14

## 실험 설정

- 실행 파일: `exp/20260414/amgx/build/amgx_jfnk_probe`
- 대상: `exp/20260414/amgx/cupf_dumps`의 12개 Texas University case
- solver: `fgmres`
- preconditioner: `bus_block_jacobi_fd`
- nonlinear tolerance: `1e-8`
- linear tolerance: `1e-2`
- inner max iterations: `500`
- restart: `200`
- max outer iterations: `20`
- mode: `continue-on-linear-failure`
- 결과 파일: `exp/20260414/amgx/results/bus_block_jacobi_continue_all_inner500_restart200.csv`

## Base_Texas_66GW 단일 case

| mode | converged | outer | final nonlinear residual | linear failures | sec |
|---|---:|---:|---:|---:|---:|
| strict | false | 4 | `1.520871e-03` | 1 | 2.952 |
| continue | true | 10 | `7.521689e-10` | 6 | 8.482 |

해석:

- strict에서는 outer 4에서 내부 선형계가 `linear_tol=1e-2`에 도달하지 못해 멈췄다.
- 이때 nonlinear residual은 이미 `99.026 -> 2.170 -> 0.1645 -> 1.521e-3`까지 줄어 있었다.
- continue에서는 선형계 실패 후에도 update를 적용했고, residual이 `5.972e-5 -> 5.894e-6 -> 1.390e-6 -> 2.294e-7 -> 1.654e-8 -> 7.522e-10`까지 내려가 비선형 수렴했다.

## 12개 case continue 결과

| case | converged | outer | final nonlinear residual | linear failures | total inner | sec | failure reason |
|---|---:|---:|---:|---:|---:|---:|---|
| case_ACTIVSg200 | true | 4 | `3.007629e-09` | 0 | 209 | 0.444 | none |
| case_ACTIVSg500 | true | 5 | `9.166887e-09` | 0 | 423 | 0.590 | none |
| MemphisCase2026_Mar7 | true | 4 | `5.723269e-09` | 0 | 298 | 0.461 | none |
| case_ACTIVSg2000 | true | 6 | `1.834679e-09` | 0 | 544 | 0.989 | none |
| Base_Florida_42GW | true | 12 | `1.790322e-09` | 9 | 4570 | 9.940 | none |
| Texas7k_20220923 | true | 6 | `6.680310e-09` | 2 | 1719 | 3.724 | none |
| Base_Texas_66GW | true | 10 | `7.521689e-10` | 6 | 3717 | 8.169 | none |
| Base_MIOHIN_76GW | true | 8 | `3.135854e-09` | 4 | 2380 | 5.155 | none |
| Base_West_Interconnect_121GW | false | 20 | `4.324753e-04` | 18 | 9219 | 21.120 | max_outer_iterations |
| case_ACTIVSg25k | false | 20 | `3.137743e+00` | 20 | 10000 | 23.037 | max_outer_iterations |
| case_ACTIVSg70k | false | 20 | `3.352852e+00` | 20 | 10000 | 28.908 | max_outer_iterations |
| Base_Eastern_Interconnect_515GW | false | 20 | `5.583335e-03` | 18 | 9197 | 28.345 | max_outer_iterations |

요약:

- 12개 중 8개 case가 nonlinear tolerance `1e-8`에 도달했다.
- 총 실행 시간은 약 `130.882 s`였다.
- 실패한 4개 case도 일부는 nonlinear residual이 크게 감소했지만, `max_outer_iterations=20`에 걸렸다.
- 기존 scalar AMGX AMG 대비 `bus_block_jacobi_fd`가 훨씬 강한 수렴 신호를 보였다.

## 기존 AMGX AMG와의 비교 포인트

- 초기 재현 run 기준 `amg_fd`는 12개 중 4개만 최종적으로 nonlinear 수렴했다.
- `Base_Texas_66GW`에서는 `amg_fd` strict가 첫 선형계 실패로 update를 적용하지 못했고, `continue`로도 `1e-8`까지 가지 못했다.
- 반면 `bus_block_jacobi_fd + continue`는 같은 case에서 nonlinear residual `7.52e-10`까지 내려갔다.
- 핵심 차이는 AMGX scalar AMG가 그래프/스케일/변수 ordering을 일반적인 scalar 행렬로만 보는 반면, bus block Jacobi는 PV/PQ bus 내부의 물리적 local coupling을 직접 반영한다는 점이다.
