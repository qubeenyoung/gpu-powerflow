# 2026-05-07 보조 실험 계획

Date: 2026-05-07

## 목적

이번 보조실험의 목적은 현재 cuPF GPU 전력조류 계산을 단순히 빠르게 만드는 것이 아니라, 올해와 내년 과제 목표를 연결할 실험적 논리를 확보하는 것이다.

- 올해 목표: cuITER 기반 반복 선형 솔버를 Newton 전력조류 루프에 넣을 수 있는 근거 확보
- 내년 목표: 다중 GPU와 Tensor Core 활용을 고려한 계산 구조 탐색
- 최종 방향: 반복 선형 솔버를 cuDSS의 완전한 대체재로만 보지 않고, cuDSS와 함께 쓰는 하이브리드 구조를 설계

핵심 논리는 다음과 같다.

```text
Newton 반복의 모든 중간 선형계를 항상 정확하게 풀 필요는 없다.
초기/중간 반복에서는 mismatch를 줄이는 방향만 충분히 맞으면 되고,
마지막 수렴 구간에서만 cuDSS 또는 더 엄격한 solve가 필요할 수 있다.
```

따라서 이번 실험은 세 가지 질문에 답하도록 구성한다.

1. 배치 크기가 GPU utilization과 처리량에 어떤 영향을 주며, 다중 GPU DP/PP에서 micro-batch로 쓸 만한 크기는 어디인가?
2. batch 1에서 모든 case의 mismatch 벡터는 Newton 반복마다 크기와 방향이 어떻게 변하며, 중간 반복을 부정확하게 계산해도 되는 근거가 있는가?
3. 어떤 operator가 정밀도에 민감하며, Tensor Core 또는 FP32 CUDA core로 옮길 수 있는 연산은 무엇인가?

## 현재까지의 관련 관측

기존 결과에서 이번 실험과 직접 연결되는 신호는 다음과 같다.

- `exp/20260423/paper/speed_matpower_final/matpower_comparison_mt_20260423/solve_bin_summary.md`에서 MATPOWER case는 batch 4/16/64/256 중 모든 bus-size bin에서 batch 256이 best multibatch로 선택되었다. 이는 batch가 GPU launch overhead와 low utilization을 숨기는 데 효과적이라는 신호다.
- `exp/20260423/paper/precision_matpower_final/matpower_precision_fp32_mixed_fp64_fair_analysis_20260423/fair_precision_bin_summary.md`에서 FP32는 모든 bus-size bin에서 `success 0/N`이었고, Mixed와 FP64는 모두 수렴했다. 단, Mixed의 solve/update 시간은 FP64보다 작았다. 이는 "전부 FP64"가 아니라 "정밀도가 필요한 부분만 FP64 또는 보정"하는 방향이 유효하다는 신호다.
- 현재 `exp/20260420/newton_solver` 경로는 voltage와 mismatch 벡터를 double로 유지하면서, `sbus`, Jacobian value, `dx`, cuDSS solve는 float 기반으로 구성되어 있다. 즉 이미 혼합 정밀도 구조를 일부 갖고 있으며, 어떤 operator가 실패를 유발하는지 분해할 필요가 있다.
- `exp/20260428/operator_profile`에는 operator timing 측정 경로가 있고, Nsight Systems/Compute로 들어갈 수 있는 command hint가 이미 정리되어 있다. 이번 실험은 여기에 utilization, per-iteration mismatch dump, precision toggle을 붙이는 형태가 자연스럽다.

## 실험 1: 배치 크기에 따른 GPU utilization 분석

### 질문

Power-flow workload에서 batch size를 키울 때 GPU가 언제부터 충분히 채워지는가? 다중 GPU에서 DP 또는 PP로 나눌 때 micro-batch 크기를 얼마로 잡아야 throughput과 latency의 균형이 좋은가?

### 가설

- 작은 case와 batch 1은 kernel launch, cuDSS setup, host-device synchronization overhead의 비중이 커서 GPU utilization이 낮다.
- batch가 커질수록 mismatch/Jacobian/update 같은 custom CUDA kernel은 grid가 커져 SM occupancy가 좋아진다.
- cuDSS factor/solve는 batch 증가에 따라 이득이 있지만, matrix size와 memory footprint에 따라 특정 지점 이후 포화 또는 메모리 병목이 생긴다.
- micro-batch 후보는 단순히 가장 빠른 batch가 아니라, GPU utilization 포화점 이후의 첫 번째 안정 지점이어야 한다.

### 대상 case

MATPOWER 전체 case를 모두 돌리되, 분석 표는 bus-size bin별로 묶는다.

```text
<100
100-999
1k-9,999
10k-49,999
>=50k
```

추가로 대표 case를 고정 추적한다.

```text
case118
case1354pegase
case9241pegase
case_ACTIVSg25k
case_ACTIVSg70k
case_SyntheticUSA
```

### 배치 sweep

초기 sweep:

```text
batch = 1, 2, 4, 8, 16, 32, 64, 128, 256
```

메모리가 허용되면 확장:

```text
batch = 512, 1024
```

단, 큰 case는 OOM 또는 cuDSS workspace 급증 가능성이 있으므로 case별 가능한 최대 batch를 별도로 기록한다.

### 측정 항목

기본 성능:

| Metric | 의미 |
| --- | --- |
| `elapsed_ms` | end-to-end 또는 solve/update 전체 시간 |
| `ms_per_scenario` | `elapsed_ms / batch_size` |
| `scenario_per_sec` | throughput |
| `iterations` | 수렴 반복 수 변화 여부 |
| `final_mismatch` | batch 변화로 정답성이 바뀌는지 확인 |
| `success` | convergence success |

GPU utilization:

| Metric | 도구 |
| --- | --- |
| SM utilization | Nsight Compute 또는 `nvidia-smi dmon` |
| achieved occupancy | Nsight Compute |
| DRAM throughput | Nsight Compute |
| L2 throughput/cache hit | Nsight Compute |
| kernel launch count/time | Nsight Systems |
| host idle / sync gap | Nsight Systems |
| cuDSS phase별 시간 | benchmark timer 또는 NVTX |

cuDSS phase breakdown:

```text
analysis
factorization/refactorization
solve
```

custom kernel breakdown:

```text
mismatch
jacobian_fill
rhs_pack
voltage_update
norm/reduction
```

### 산출물

```text
exp/20260507/results/batch_util/summary.csv
exp/20260507/results/batch_util/operator_breakdown.csv
exp/20260507/results/batch_util/gpu_util_samples.csv
exp/20260507/results/batch_util/microbatch_recommendation.md
exp/20260507/results/batch_util/figures/
```

추천 micro-batch 표는 다음 형태로 만든다.

| bus-size bin | latency 우선 | throughput 우선 | DP 추천 | PP 추천 | 근거 |
| --- | ---: | ---: | ---: | ---: | --- |
| `<100` | TBD | TBD | TBD | TBD | utilization 포화점 |
| `100-999` | TBD | TBD | TBD | TBD | utilization 포화점 |
| `1k-9,999` | TBD | TBD | TBD | TBD | utilization 포화점 |
| `10k-49,999` | TBD | TBD | TBD | TBD | memory/solve 포화 |
| `>=50k` | TBD | TBD | TBD | TBD | memory/solve 포화 |

### 성공 기준

- bus-size bin별로 "batch를 더 키워도 `ms_per_scenario` 개선이 작아지는 지점"을 찾는다.
- DP micro-batch는 throughput 기준, PP micro-batch는 stage balance와 activation/workspace memory 기준으로 추천한다.
- batch size가 정답성, iteration 수, final mismatch에 영향을 주지 않는지 확인한다.

## 실험 2: batch 1에서 mismatch 벡터의 크기와 방향 추세

### 질문

모든 case에서 Newton 반복 중 mismatch 벡터 `F_k`는 크기뿐 아니라 방향도 안정적으로 줄어드는가? 반복 선형 솔버가 중간 반복에서 대략적인 방향만 맞춰도 최종 수렴에 도달할 수 있는가?

### 핵심 논리

cuITER를 cuDSS와 함께 쓰려면 다음 중 하나를 보여야 한다.

- 초기/중간 Newton 반복에서는 `dx`가 cuDSS 해와 완전히 같지 않아도 `F_{k+1}`의 크기가 감소한다.
- `F_k`의 normalized direction이 반복 사이에 급격히 흔들리지 않으면, inexact solve가 같은 수렴 방향을 유지할 가능성이 있다.
- 마지막 수렴 구간에서만 cuDSS 또는 엄격 tolerance solve로 전환해도 최종 mismatch는 유지된다.

따라서 단순히 `||F_k||`만 기록하지 않고, 방향성과 component별 패턴을 같이 본다.

### 기준 trajectory dump

batch 1, 모든 case에 대해 baseline trajectory를 저장한다.

```text
precision: Mixed baseline
linear solve: cuDSS
batch: 1
tolerance: 1e-8
max_iter: 10
```

반복마다 저장할 값:

| 값 | 설명 |
| --- | --- |
| `F_k` | mismatch vector |
| `norm_inf(F_k)` | 수렴 판정에 가까운 크기 |
| `norm_2(F_k)` | 전체 에너지 |
| `u_k = F_k / ||F_k||_2` | normalized direction |
| `dx_k` | Newton update |
| `norm_2(dx_k)` / `norm_inf(dx_k)` | update 크기 |
| `J_k dx_k + F_k` | linear solve residual |
| `V_k` | 필요 시 voltage state |

방향성 metric:

| Metric | 의미 |
| --- | --- |
| `cos(F_k, F_{k-1})` | mismatch 방향이 반복 사이에 유지되는지 |
| `cos(dx_k, dx_k_direct)` | inexact update가 cuDSS update와 같은 방향인지 |
| `cos(F_{k+1}^{approx}, F_{k+1}^{direct})` | 다음 mismatch 방향 보존 여부 |
| `topk_overlap(abs(F_k), abs(F_{k-1}))` | mismatch가 큰 bus/component가 유지되는지 |
| `contraction = ||F_{k+1}|| / ||F_k||` | 수렴 감소율 |
| `predicted_reduction = ||F_k + J_k dx_k||` | linear residual이 nonlinear 감소를 설명하는지 |

component 분해:

```text
P mismatch on PV/PQ
Q mismatch on PQ
bus별 top-k mismatch
area 또는 case metadata가 있으면 area별 aggregation
```

### inexact solve 실험

baseline cuDSS trajectory를 만든 뒤, 같은 `J_k, F_k`에 대해 반복 선형 solve 품질을 낮춘다.

후보:

```text
linear relative tolerance = 1e-1, 1e-2, 1e-3, 1e-4
max inner iteration cap = 5, 10, 20, 50
precision = FP32, Mixed
preconditioner = none, block Jacobi, AMGX/cuITER candidate
```

기록:

| Metric | 설명 |
| --- | --- |
| `linear_relres` | 선형계 잔차 |
| `cos(dx_approx, dx_cudss)` | update 방향 보존 |
| `||dx_approx - dx_cudss|| / ||dx_cudss||` | update 크기 오차 |
| `||F_next_approx|| / ||F_next_cudss||` | 다음 mismatch 악화 정도 |
| `accepted` | `||F_next_approx|| < ||F_k||`이면 중간 update로 허용 가능 |
| `cuDSS_recovery_success` | 몇 번 approximate 후 cuDSS로 전환하면 최종 수렴하는지 |

### 하이브리드 정책 후보

실험 결과를 바탕으로 다음 정책을 비교한다.

| 정책 | 설명 |
| --- | --- |
| `all_cudss` | 모든 Newton 반복에서 cuDSS |
| `iter_then_cudss_by_iter` | 초반 N회 cuITER, 이후 cuDSS |
| `iter_then_cudss_by_norm` | `||F_k|| > threshold`이면 cuITER, 작아지면 cuDSS |
| `iter_with_cudss_guard` | cuITER update 후 mismatch 감소가 없으면 cuDSS로 retry |
| `periodic_cudss` | 매 M번째 반복만 cuDSS로 correction |

초기 추천 guard:

```text
if ||F_{k+1}^{approx}|| <= 0.9 * ||F_k||:
    accept approximate update
else:
    recompute dx_k with cuDSS and retry
```

이 기준은 임시값이며, 실험 2 결과로 case-size별 또는 iteration-stage별 threshold를 조정한다.

### 산출물

```text
exp/20260507/results/mismatch_trends/trajectory_summary.csv
exp/20260507/results/mismatch_trends/vector_metrics.csv
exp/20260507/results/mismatch_trends/inexact_solve_sweep.csv
exp/20260507/results/mismatch_trends/hybrid_policy_comparison.csv
exp/20260507/results/mismatch_trends/MISMATCH_DIRECTION_ANALYSIS.md
exp/20260507/results/mismatch_trends/figures/
```

핵심 figure:

- case별 `||F_k||` 감소 곡선
- `cos(F_k, F_{k-1})` heatmap
- `cos(dx_approx, dx_cudss)` vs `linear_relres`
- approximate update accept/reject map
- cuITER N회 후 cuDSS recovery 성공률

### 성공 기준

- 많은 case에서 초기/중간 반복의 approximate update가 mismatch 감소 방향을 유지한다.
- `linear_relres`가 느슨해도 `cos(dx_approx, dx_cudss)`가 충분히 높고, `||F_{k+1}|| < ||F_k||`가 유지되는 구간을 식별한다.
- cuDSS 전환 guard를 넣으면 approximate solve 실패가 최종 수렴 실패로 이어지지 않음을 보인다.

## 실험 3: 정밀도에 민감한 operator 식별

### 질문

FP32에서 수렴 실패가 발생하는 원인은 어느 operator의 오차 때문인가? Tensor Core를 쓸 수 있는 연산과 Tensor Core가 아니더라도 FP32 CUDA core로 안전하게 옮길 수 있는 연산은 무엇인가?

### operator 후보

현재 Newton 루프 기준 후보는 다음과 같다.

| Operator | 현재/후보 정밀도 | 민감도 예상 |
| --- | --- | --- |
| `mismatch`: `Ybus * V`, `S(V) - Sbus` | FP64/Mixed/FP32 | 높음 |
| mismatch norm reduction | FP64/FP32 + compensated reduction | 높음 |
| Jacobian edge fill | FP32/FP64/Mixed | 중간-높음 |
| RHS pack: `-F` cast to solve RHS | FP64 to FP32 | 중간 |
| cuDSS factor/solve | FP32/FP64, iterative refinement 후보 | 높음 |
| voltage update | FP64 state + FP32 dx, 또는 FP32 state | 중간 |
| block/local preconditioner apply | FP32/FP64 | 중간 |
| SpMV/linear residual 계산 | FP32/FP64/TF32 후보 | 높음 |

### one-operator-at-a-time precision toggle

FP64 end-to-end를 reference로 두고, operator 하나씩 정밀도를 바꾼다.

기준:

```text
reference: FP64 all
baseline: current Mixed
bad baseline: FP32 all
```

toggle 예시:

| Config | 목적 |
| --- | --- |
| `mixed_current` | 현재 실용 baseline |
| `fp32_mismatch_only` | mismatch 계산만 FP32로 낮춤 |
| `fp64_mismatch_only` | FP32 전체에서 mismatch만 FP64로 올림 |
| `fp64_norm_only` | norm/reduction만 FP64로 올림 |
| `fp64_jac_only` | Jacobian assembly만 FP64로 올림 |
| `fp64_solve_only` | linear solve만 FP64 또는 refinement 적용 |
| `fp64_voltage_only` | voltage state/update만 FP64 유지 |

각 config에서 기록:

| Metric | 설명 |
| --- | --- |
| `success` | 수렴 여부 |
| `final_mismatch` | 최종 mismatch |
| `iterations` | 반복 수 |
| `operator_abs_error` | FP64 reference 대비 절대 오차 |
| `operator_rel_error` | FP64 reference 대비 상대 오차 |
| `max_component_error` | bus/component별 최대 오차 |
| `argmax_error_bus` | 오차가 집중되는 위치 |
| `time_ms` | 성능 비용 |

### Tensor Core 활용 가능성 분류

Tensor Core는 dense 또는 block/dense tile 형태의 matrix multiply/accumulate에 강하다. 현재 power-flow kernel 중 irregular sparse gather, branch가 많은 bus-type 처리, scalar reduction은 Tensor Core와 직접 맞지 않는다. 따라서 "바로 Tensor Core"와 "구조 변경 후 후보"를 구분한다.

| 분류 | 연산 | 판단 |
| --- | --- | --- |
| 바로 후보 낮음 | mismatch의 row-wise sparse `Ybus * V` batch 1 | irregular SpMV 성격이라 Tensor Core 이점이 작을 가능성 |
| 구조 변경 후보 | batch가 큰 `Ybus * V`를 sparse matrix × dense multi-RHS로 재구성 | micro-batch가 크면 SpMM 형태가 되어 Tensor Core/TF32 검토 가능 |
| 구조 변경 후보 | 여러 scenario의 `J * x` 또는 Krylov basis 연산 | multi-RHS dense block이 생기면 GEMM/SpMM 후보 |
| 후보 낮음 | Jacobian edge fill elementwise formula | transcendental은 없지만 edge별 scalar arithmetic이라 CUDA core 적합 |
| 후보 낮음 | norm/reduction | Tensor Core보다 FP64/compensated reduction 또는 warp reduction 최적화 대상 |
| 후보 가능 | block Jacobi 2x2/작은 block batched solve | Tensor Core보다는 register-level FP32 또는 small batched kernel이 우선 |

### FP32 CUDA core 활용 방안

Tensor Core를 쓰지 않더라도 다음 연산은 FP32 CUDA core로 적극 최적화할 수 있다.

- Jacobian edge fill: FP32 계산 후 민감한 diagonal accumulation만 FP64 또는 compensated accumulation으로 분리
- mismatch의 일부: `Ybus` 값과 `Sbus`는 FP32 유지, voltage state와 최종 mismatch accumulation은 FP64 유지
- Krylov/iterative solver 중간: SpMV, axpy, dot/norm 일부를 FP32로 수행하고 residual check만 FP64로 재평가
- block Jacobi/preconditioner apply: local 2x2 inverse/apply는 FP32로 수행하고 guard residual만 FP64로 계산
- voltage update: `dx`는 FP32, `V` accumulation은 FP64 유지

### 산출물

```text
exp/20260507/results/precision_ops/operator_precision_sweep.csv
exp/20260507/results/precision_ops/operator_error_by_iteration.csv
exp/20260507/results/precision_ops/sensitive_operator_ranking.md
exp/20260507/results/precision_ops/tensorcore_candidate_matrix.md
exp/20260507/results/precision_ops/figures/
```

최종 ranking 표:

| rank | operator | accuracy sensitivity | speed impact | Tensor Core 가능성 | FP32 CUDA core 가능성 | 다음 분석 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD |

### 성공 기준

- FP32 실패를 유발하는 operator를 1-3개 수준으로 좁힌다.
- Mixed precision에서 반드시 FP64로 남겨야 하는 연산과 FP32로 내려도 되는 연산을 구분한다.
- Tensor Core는 "쓸 수 있다/없다"가 아니라, 어떤 재구조화가 있어야 후보가 되는지까지 정리한다.

## 실행 순서

1. Instrumentation 추가
   - per-iteration `F`, `dx`, norm, linear residual dump
   - NVTX range: mismatch, Jacobian fill, cuDSS analysis/factor/solve, voltage update
   - precision toggle option

2. 실험 1 batch utilization
   - 기존 batch benchmark를 재사용
   - 먼저 MATPOWER representative case로 smoke run
   - 이후 전체 case와 bus-size bin summary 생성

3. 실험 2 mismatch direction
   - batch 1 baseline cuDSS trajectory dump
   - inexact solve sweep
   - hybrid policy simulation

4. 실험 3 precision operator
   - FP64 reference 생성
   - one-operator-at-a-time toggle
   - Tensor Core/FP32 CUDA core 후보 분류

5. 종합 문서 작성
   - cuITER + cuDSS hybrid justification
   - DP/PP micro-batch recommendation
   - precision-sensitive operator ranking

## 예상 결론 형태

이번 실험이 끝나면 다음과 같은 주장을 데이터로 만들 수 있어야 한다.

1. "batch size X 이상에서 GPU utilization이 포화되므로, 다중 GPU DP/PP의 micro-batch는 case-size bin별로 X/Y/Z를 쓰는 것이 합리적이다."
2. "Newton 초기/중간 반복에서는 mismatch 방향이 안정적이고 approximate update도 감소 방향을 유지하므로, cuITER를 느슨한 tolerance로 쓰고 cuDSS로 guard/recovery하는 하이브리드 구조가 타당하다."
3. "정밀도 실패는 모든 FP32 연산 때문이 아니라 특정 operator의 accumulation/linear solve/residual check에서 발생하며, 나머지는 FP32 CUDA core 또는 향후 SpMM/GEMM 재구조화를 통한 Tensor Core 후보가 된다."

## 디렉터리 구조

```text
exp/20260507/
  PLAN.md
  scripts/
    run_batch_util_sweep.py
    dump_mismatch_trajectory.py
    run_inexact_solve_sweep.py
    run_precision_operator_sweep.py
  results/
    batch_util/
    mismatch_trends/
    precision_ops/
```

초기에는 `PLAN.md`만 두고, 실제 실행 스크립트와 결과는 각 실험 착수 시 추가한다.
