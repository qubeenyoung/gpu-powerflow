# Mismatch Trajectory 진행 상태

Date: 2026-05-07

## 전체 MATPOWER batch 1 dump

Run:

```text
exp/20260507/results/mismatch_trends/all_matpower_b1_cuda_mixed_dump_20260507
```

조건:

```text
cases = MATPOWER dump 전체 78개
batch = 1
profile = cuda_mixed_edge
tolerance = 1e-8
max_iter = 50
warmup = 0
repeats = 1
binary = bench-end2end-superlu-cudss-mt-auto-dump/cupf_case_benchmark
dump = residual_iter*.txt
```

산출물:

```text
trajectory_summary.csv
vector_metrics.csv
run_summary.csv
raw/<case>/dumps/repeat_00/residual_iter*.txt
MISMATCH_DIRECTION_ANALYSIS.md
figures/*.png
```

결과:

- 총 `78/78` case가 수렴했다.
- 반복별 vector metric row는 `364`개다.
- 연속 반복 전이는 총 `286`개다.
- L_inf mismatch는 `283/286` step에서 감소했다.
- L_inf mismatch가 증가한 case는 `case1197`, `case2869pegase`, `case9241pegase` 3개뿐이며 모두 최종 수렴했다.
- 연속 mismatch vector cosine은 `51/286` step에서 음수였다.
- 음수 cosine이 한 번 이상 나온 case는 `43/78`개다.
- 첫 Newton step cosine은 `26/78`개 case에서 음수였다.
- top-k 큰 mismatch component의 Jaccard overlap median은 `0.4286`이다.
- raw dump 크기는 약 `122 MB`다.

1차 해석:

- mismatch 크기는 거의 모든 반복에서 감소하므로, 중간 반복의 approximate solve는 감소 guard와 함께 사용할 여지가 크다.
- 반면 mismatch vector 방향은 반복 사이에 항상 유지되지 않는다.
- 따라서 “중간 반복은 방향만 대충 맞추면 된다”보다는 “중간 반복은 정확한 cuDSS solve가 아니어도 되지만, 다음 mismatch 감소를 guard해야 한다”가 더 안전한 논리다.

시각화:

```text
figures/iteration_count_median_norm.png
figures/iteration_count_median_direction.png
```

참고: 기존 합본 figure는 삭제했고, 현재 figure는 두 장만 유지한다. 두 그림 모두 제목을 제거했다. 방향 유사도 그림은 첫 전이를 열린 원, 마지막 전이를 채운 마름모로 표시해서 첫/마지막 반복에서 cosine이 상대적으로 낮아지는 패턴을 보이도록 했다.

상세 문서:

```text
exp/20260507/results/mismatch_trends/all_matpower_b1_cuda_mixed_dump_20260507/MISMATCH_DIRECTION_ANALYSIS.md
```
