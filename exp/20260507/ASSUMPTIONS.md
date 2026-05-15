# 2026-05-07 실험 진행 중 임의 판단 기록

Date: 2026-05-07

사용자가 명시하지 않아 진행자가 정한 조건은 여기에 계속 기록한다.

## 실험 1: batch utilization 시작 조건

- 전체 case sweep 전에 대표 case smoke sweep를 먼저 수행한다.
  - 이유: batch 512/1024 또는 큰 case에서 OOM/cuDSS workspace 실패가 날 수 있으므로 실행 경로와 CSV schema를 먼저 검증한다.
- smoke sweep 대표 case는 다음 5개로 시작한다.
  - `case118`: 작은 IEEE case, 기존 결과와 비교하기 쉬움
  - `case1354pegase`: 1k급 case
  - `case9241pegase`: 10k 미만 큰 MATPOWER case
  - `case_ACTIVSg25k`: 25k급 synthetic case
  - `case_ACTIVSg70k`: 70k급 synthetic case
- smoke sweep batch는 `1, 2, 4, 8, 16, 32, 64, 128, 256`으로 시작한다.
  - `512, 1024`는 smoke 결과에서 메모리와 시간이 감당 가능할 때 확장한다.
- smoke sweep timing은 `warmup=1`, `repeats=3`으로 시작한다.
  - 이유: 첫 목표가 최종 수치가 아니라 실행 경로와 batch scaling 경향 확인이기 때문이다.
  - 최종 표는 `warmup=3`, `repeats=10` 이상으로 재측정한다.
- benchmark binary는 기존 prebuilt end-to-end binary를 사용한다.
  - `/workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark`
  - 이유: 20260423 최종 speed 실험과 같은 cuDSS MT auto 경로를 유지하기 위해서다.
- GPU utilization sampling은 `nvidia-smi --query-gpu ... -lms 100`을 benchmark process와 병렬로 실행해 coarse sample로 저장한다.
  - 짧은 run에서는 sampling 해상도가 부족할 수 있으므로 최종 Nsight 측정 전의 1차 지표로만 사용한다.
  - 최종 판단은 Nsight Systems/Compute로 대표 batch를 다시 확인한다.

## 실험 1: smoke 이후 전체 timing sweep 조건

- smoke sweep가 5개 대표 case에서 batch 256까지 모두 성공했으므로, 다음 단계로 MATPOWER dump 전체 78개 case timing sweep를 수행한다.
- 전체 78개 case sweep는 GPU utilization sampling을 끄고 timing만 먼저 수집한다.
  - 이유: `nvidia-smi` sampler를 78 case × 9 batch 조합마다 붙이면 sampling overhead가 커지고, 짧은 case에서는 0% 또는 불안정 sample이 자주 나온다.
  - utilization은 smoke 대표 case coarse sample과 이후 Nsight 대표 측정으로 보강한다.
- 전체 78개 case sweep도 먼저 `warmup=1`, `repeats=3`으로 실행한다.
  - 이유: case별 batch scaling과 실패/OOM 여부를 빠르게 식별하기 위한 1차 pass다.
  - 최종 보고용 수치는 smoke/전체 1차 결과를 보고 필요한 batch 후보만 `warmup=3`, `repeats=10`으로 재측정한다.

## 실험 1: ncu batch sweep 시작 조건

- ncu batch sweep도 batch 축은 반드시 `1, 2, 4, 8, 16, 32, 64, 128, 256` 전체를 사용한다.
- 첫 ncu sweep는 전체 78개 case가 아니라 대표 case에서 시작한다.
  - 이유: ncu `basic` set은 kernel별 metric 수집으로 실행 시간이 크게 늘어나므로, 먼저 batch scaling의 형태와 수집 schema를 검증한다.
- 대표 case는 사용자가 지정한 크기축 `100, 1000, 5000, 10000, 25K, 70K`에 맞춘다.
  - `100`: `case94pi` (`n_bus=94`)
  - `1000`: `case1197` (`n_bus=1197`)
  - `5000`: `case6468rte` (`n_bus=6468`)
  - `10000`: `case_ACTIVSg10k` (`n_bus=10000`)
  - `25K`: `case_ACTIVSg25k` (`n_bus=25000`)
  - `70K`: `case_ACTIVSg70k` (`n_bus=70000`)
- 정확한 5000 bus case는 현재 MATPOWER dump에 없어서, 3k-8k 범위 중 가까운 대형 case인 `case6468rte`를 5k 대표로 둔다.
- 첫 ncu sweep는 cuPF custom kernel만 대상으로 한다.
  - 대상: `compute_ibus_kernel`, `compute_mismatch_from_ibus_kernel`, `reduce_mismatch_norm_kernel`, `fill_jacobian_gpu_kernel`, `prepare_rhs_kernel`, `apply_voltage_update_kernel`, `reconstruct_voltage_kernel`
  - 이유: cuDSS 내부 kernel을 모두 포함하면 factor/solve 내부 kernel 수가 많아 batch sweep 비용이 커진다.
  - cuDSS kernel은 별도 ncu sweep로 분리한다.
- ncu 실행은 `warmup=0`, `repeats=1`로 시작한다.
  - 이유: ncu 자체가 kernel replay를 수행하므로, 첫 단계에서는 metric 수집 가능성과 batch별 상대 경향 확인을 우선한다.

## 실험 1: cuDSS 포함 ncu batch sweep 조건

- 100-bus 대표 case는 `case94pi`에서 `case118`로 변경한다.
  - 이유: 사용자가 `94말고 118`로 지시했으며, `case118`은 기존 IEEE 대표 case로 비교 기준이 많다.
- cuDSS 포함 pass는 ncu kernel filter를 제거해 모든 CUDA kernel을 수집한다.
  - 이유: cuDSS 내부 kernel 중 일부는 `cudss::` namespace가 붙지만, 일부 초기화/graph/offset kernel은 일반 kernel name으로 나타나므로 필터 기반으로는 누락될 수 있다.
- 이 pass도 batch 축은 `1, 2, 4, 8, 16, 32, 64, 128, 256` 전체를 유지한다.
- 실행 조건은 `warmup=0`, `repeats=1`, `ncu set=basic`으로 둔다.
  - 이유: all-kernel ncu는 custom-only보다 훨씬 무거우므로 먼저 대표 case/전체 batch의 1차 profile을 확보한다.

## 실험 2: mismatch trajectory 시작 조건

- batch는 사용자 목적에 맞춰 `1`로 고정한다.
- 대상 case는 MATPOWER dump 전체 78개 case로 둔다.
- profile은 실험 1과 같은 `cuda_mixed_edge`를 사용한다.
  - 이유: cuITER/cuDSS hybrid 논리는 현재 mixed CUDA + cuDSS 경로를 기준으로 이어지기 때문이다.
- benchmark binary는 dump-enabled prebuilt binary를 사용한다.
  - `/workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto-dump/benchmarks/cupf_case_benchmark`
  - 이유: 이미 `--dump-residuals`, `--dump-newton-diagnostics`, `--dump-dir` 옵션이 있고 반복별 `residual_iter*.txt`를 남긴다.
- dump run은 `warmup=0`, `repeats=1`로 둔다.
  - 이유: residual vector 전체를 text로 저장하므로 warmup/repeat를 늘리면 raw dump 크기가 빠르게 커진다.
- 방향성은 같은 case의 연속 Newton 반복 mismatch 벡터 `F_k`, `F_{k-1}` 사이 cosine으로 정의한다.
  - `cos(F_k, F_{k-1}) = dot(F_k, F_{k-1}) / (||F_k||_2 ||F_{k-1}||_2)`
- 크기는 `L_inf`, `L2`, `L1` norm을 모두 기록하되, 수렴 판정과 직접 연결되는 주 지표는 `L_inf`로 둔다.
- 큰 mismatch component 위치 유지성은 top-k overlap으로 본다.
  - `k = min(100, max(10, ceil(0.01 * dimF)))`
  - 이유: 작은 case에서도 최소 10개 component를 보고, 큰 case에서는 파일 크기/해석성을 위해 최대 100개로 제한한다.
- dump 파일 중 `residual_iter*.txt`를 분석 기준으로 사용한다.
  - `residual_before_update_iter*.txt`도 생성되지만 현재 코드상 같은 mismatch_norm 직후 벡터라 raw 보존용으로만 둔다.

## 실험 2: mismatch trajectory 시각화 정리

- 사용자가 선택한 합본 그림은 두 개의 figure로 분리한다.
  - `iteration_count_median_norm.png`: 반복별 mismatch 크기 감소.
  - `iteration_count_median_direction.png`: 반복별 방향 유사도.
- 이전 합본 파일과 다른 figure 파일은 삭제하고, 위 두 figure만 유지한다.
- 두 figure 모두 plot title은 제거한다.
- 방향 유사도 그림의 x축은 `cos(F_k, F_{k-1})`가 계산되는 도착 반복 `k`로 둔다.
  - 따라서 첫 전이는 `k=1`이다.
- 첫 전이는 열린 원, 마지막 전이는 채운 마름모로 강조한다.
  - 이유: 사용자가 보고 싶은 논점이 첫 반복과 마지막 반복에서 방향 유사도가 상대적으로 낮다는 경향성이기 때문이다.
- 첫 전이 위치 `k=1`은 옅은 세로 배경으로 추가 강조한다.
  - 마지막 전이는 수렴 반복 수 그룹마다 위치가 다르므로 공통 세로 배경 대신 마커로 표시한다.
