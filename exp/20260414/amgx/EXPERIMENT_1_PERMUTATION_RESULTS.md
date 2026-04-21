# Experiment 1: Scalar CSR Permutation Results

Date: 2026-04-14

## 조건

- solver: `fgmres`
- preconditioner: `amg_fd`
- AMGX coarse solver: `DENSE_LU_SOLVER`
- nonlinear tolerance: `1e-8`
- linear tolerance: `1e-2`
- inner max iterations: `500`
- GMRES restart: `200`
- max outer iterations: `20`
- mode: `continue-on-linear-failure`
- 실행 방식: GPU process 1개만 사용해 순차 실행

비교한 ordering:

- `none`: 기존 scalar CSR ordering
- `bus_local`: bus 순서대로 local variables를 인접 배치
- `pq_interleaved`: `[PV theta] [PQ theta, PQ Vm interleaved]`

결과 파일:

- `exp/20260414/amgx/results/proper_exp1_baseline_none_all.csv`
- `exp/20260414/amgx/results/proper_exp1_bus_local_all.csv`
- `exp/20260414/amgx/results/proper_exp1_pq_interleaved_all.csv`

## 전체 요약

| ordering | converged / 12 | total sec | note |
|---|---:|---:|---|
| none | 5 | 220.019 | baseline |
| bus_local | 5 | 222.722 | convergence count unchanged |
| pq_interleaved | 5 | 223.270 | convergence count unchanged |

## Case별 final nonlinear residual

| case | none | bus_local | pq_interleaved |
|---|---:|---:|---:|
| case_ACTIVSg200 | `2.498509e-09` | `2.498811e-09` | `2.499541e-09` |
| case_ACTIVSg500 | `8.681753e-11` | `8.682453e-11` | `8.683104e-11` |
| MemphisCase2026_Mar7 | `2.324952e-09` | `2.325122e-09` | `2.325392e-09` |
| case_ACTIVSg2000 | `6.476702e-10` | `7.056974e-10` | `6.453946e-10` |
| Base_Florida_42GW | `1.563897e+00` | `1.562825e+00` | `1.563084e+00` |
| Texas7k_20220923 | `5.988874e-09` | `6.178439e-09` | `5.286654e-09` |
| Base_Texas_66GW | `1.071163e+00` | `1.076427e+00` | `1.071989e+00` |
| Base_MIOHIN_76GW | `8.434259e-01` | `8.425825e-01` | `8.436244e-01` |
| Base_West_Interconnect_121GW | `6.586120e-01` | `6.561115e-01` | `6.549061e-01` |
| case_ACTIVSg25k | `1.518774e+01` | `1.507328e+01` | `1.549079e+01` |
| case_ACTIVSg70k | `9.334905e+01` | `9.724037e+01` | `9.401858e+01` |
| Base_Eastern_Interconnect_515GW | `6.400198e+00` | `6.404323e+00` | `6.405996e+00` |

## 판단

1번 실험의 결론은 부정적이다. Scalar CSR row/column permutation만으로는 수렴 case 수가 늘지 않았고, 전체 runtime도 약간 증가했다.

일부 실패 case에서 final residual이 아주 조금 좋아진 항목은 있지만, 효과 크기가 작고 일관되지 않다. `Base_Texas_66GW`처럼 우리가 집중해서 보던 case도 `none=1.071163e+00`, `bus_local=1.076427e+00`, `pq_interleaved=1.071989e+00`로 사실상 개선이 없다.

따라서 "ordering만 바꾸면 AMGX generic aggregation이 bus-local coupling을 더 잘 볼 수 있다"는 가설은 이 설정에서는 지지되지 않는다. 다음 단계로 1번을 더 파려면 permutation 자체보다 AMGX aggregation/smoother 설정, scaling, 또는 bus-block 정보를 직접 주는 방향을 봐야 한다.
