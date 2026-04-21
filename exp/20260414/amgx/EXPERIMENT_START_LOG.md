# 3개 실험 preliminary 시작 로그

Date: 2026-04-14

주의: 이 문서는 "완료된 엄밀한 실험 결과"가 아니라 구현 smoke와 대표 case preliminary run이다. 처음 일부 run을 GPU에서 병렬로 실행해 시간 비교가 오염됐고, 이후 일부 결과만 순차 실행으로 다시 확인했다. 정식 비교는 별도 결과 파일과 문서에서 같은 조건/순차 실행으로 정리한다.

## 공통 변경

새 CLI 옵션을 추가했다.

- `--amg-permutation none|bus_local|pq_interleaved`
- `--preconditioner-combine single|additive|block_then_amg|amg_then_block`

`--amg-permutation`은 `--preconditioner amg_fd`에서만 사용한다. 기존 scalar CSR `A_fd`를 만든 뒤 AMGX 업로드 직전에 row/column/value를 permutation하고, preconditioner apply 때 rhs와 solution을 permutation/unpermutation한다.

`--preconditioner-combine`은 `--preconditioner bus_block_jacobi_fd`에서만 사용한다. `single`은 기존 bus block Jacobi 단독이고, 나머지는 AMGX와 결합한다.

## Experiment 1: scalar CSR permutation

Smoke:

| case | permutation | outer | converged | final nonlinear residual | total inner | sec |
|---|---|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | bus_local | 4 | true | `2.498811e-09` | 72 | 0.557 |

`Base_Texas_66GW`, max outer 5:

| permutation | outer | converged | final nonlinear residual | total inner | linear failures | sec |
|---|---:|---:|---:|---:|---:|---:|
| bus_local | 5 | false | `1.135921e+00` | 2500 | 5 | 7.165 |
| pq_interleaved | 5 | false | `1.136533e+00` | 2500 | 5 | 7.155 |

`Base_Texas_66GW`, max outer 20:

| permutation | outer | converged | final nonlinear residual | total inner | linear failures | sec |
|---|---:|---:|---:|---:|---:|---:|
| bus_local | 20 | false | `1.076427e+00` | 10000 | 20 | 27.567 |

초기 판단:

- `bus_local` permutation만으로는 기존 scalar AMGX AMG 대비 뚜렷한 개선이 없다.
- `Base_Texas_66GW` 기준으로는 단독 `bus_block_jacobi_fd`보다 훨씬 약하다.
- `pq_interleaved`도 outer 5 기준 `bus_local`과 거의 같다.

## Experiment 2: bus block Jacobi + scalar AMGX 결합

Smoke:

| case | combine | outer | converged | final nonlinear residual | total inner | sec |
|---|---|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | additive | 4 | true | `3.766061e-09` | 90 | 0.562 |

`Base_Texas_66GW`:

| combine | max outer | converged | final nonlinear residual | total inner | linear failures | sec |
|---|---:|---:|---:|---:|---:|---:|
| additive | 20 | false | `1.474203e-01` | 9672 | 19 | 235.314 |
| block_then_amg | 5 | false | `6.330312e-02` | 2107 | 4 | 6.145 |
| amg_then_block | 5 | false | `8.391379e-02` | 2076 | 4 | 6.075 |

초기 판단:

- additive는 작은 case에서는 동작하지만 `Base_Texas_66GW`에서 너무 느리고 수렴도 약하다.
- multiplicative 후보 둘은 additive보다 outer 5 residual은 낫지만, 기존 `bus_block_jacobi_fd + continue`의 `Base_Texas_66GW` residual 감소보다 훨씬 약하다.
- AMGX를 단순히 덧붙이는 방식은 현재로서는 좋은 방향으로 보이지 않는다.

## Experiment 3: uniform 2x2 block CSR feasibility

아직 AMGX block CSR solver path는 구현하지 않았다. 먼저 padding 규모를 확인했다.

| case | nbus | PV | PQ | original dim | uniform 2x2 dim | padding |
|---|---:|---:|---:|---:|---:|---:|
| case_ACTIVSg200 | 200 | 37 | 162 | 361 | 398 | 37 |
| case_ACTIVSg500 | 500 | 55 | 444 | 943 | 998 | 55 |
| MemphisCase2026_Mar7 | 993 | 187 | 805 | 1797 | 1984 | 187 |
| case_ACTIVSg2000 | 2000 | 391 | 1608 | 3607 | 3998 | 391 |
| Base_Florida_42GW | 5658 | 104 | 5553 | 11210 | 11314 | 104 |
| Texas7k_20220923 | 6717 | 589 | 6127 | 12843 | 13432 | 589 |
| Base_Texas_66GW | 7336 | 239 | 7096 | 14431 | 14670 | 239 |
| Base_MIOHIN_76GW | 10189 | 190 | 9998 | 20186 | 20376 | 190 |
| Base_West_Interconnect_121GW | 20758 | 766 | 19991 | 40748 | 41514 | 766 |
| case_ACTIVSg25k | 25000 | 2752 | 22247 | 47246 | 49998 | 2752 |
| case_ACTIVSg70k | 70000 | 5894 | 64105 | 134104 | 139998 | 5894 |
| Base_Eastern_Interconnect_515GW | 78478 | 2038 | 76439 | 154916 | 156954 | 2038 |

초기 판단:

- uniform 2x2로 가면 padding 수는 PV bus 수와 같다.
- padding 비율 자체는 크지 않지만, PV Vm에 대한 보조 변수/보조 식을 어떤 식으로 넣을지가 solver behavior를 좌우한다.
- 다음 구현 단계는 작은 case부터 identity-padded block CSR을 만들어 AMGX upload가 되는지 확인하는 것이다.
