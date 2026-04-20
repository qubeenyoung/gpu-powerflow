# Jacobian Assembly GPU 1 결과

## 환경

- GPU: `CUDA_VISIBLE_DEVICES=1` (`NVIDIA A10`; ncu report에서는 remap 때문에 `Device 0`으로 표시됨)
- Timing source: `results/timing_gpu1.csv`
- Nsight Compute source: `results/ncu_all_cases_gpu1_lane_occ.ncu-rep`
- Nsight Compute text: `results/ncu_all_cases_gpu1_lane_occ.txt`
- Timing command: `./build/jac_asm_bench --iters 100 --warmup 10`
- NCU metrics: `gpu__time_duration.sum`, `sm__warps_active.avg.pct_of_peak_sustained_active`, `smsp__thread_inst_executed_per_inst_executed`, `smsp__thread_inst_executed.sum`, `smsp__inst_executed.sum`

## 정의

- `edge analyze ms = common_analyze_ms + edge_map_ms`
- `vertex analyze ms = common_analyze_ms + vertex_map_ms`
- `fill ms`는 ncu replay 시간이 아니라 CUDA event 기준 100회 평균 시간.
- `occupancy % = sm__warps_active.avg.pct_of_peak_sustained_active`
- `lane util % = smsp__thread_inst_executed_per_inst_executed.pct`
- `avg lanes = smsp__thread_inst_executed_per_inst_executed.ratio`

## Elapsed Time

| Case | n_bus | Ybus nnz | J nnz | Edge analyze ms | Vertex analyze ms | Edge fill ms | Vertex fill ms | Vertex/Edge fill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 78478 | 294392 | 1143804 | 107.310 | 120.867 | 0.06189 | 0.19294 | 3.12x |
| Base_Florida_42GW | 5658 | 21384 | 83806 | 6.715 | 6.782 | 0.00522 | 0.01104 | 2.11x |
| Base_MIOHIN_76GW | 10189 | 39589 | 155152 | 12.345 | 12.420 | 0.00761 | 0.01315 | 1.73x |
| Base_Texas_66GW | 7336 | 27174 | 104763 | 8.452 | 8.506 | 0.00548 | 0.00971 | 1.77x |
| Base_West_Interconnect_121GW | 20758 | 78550 | 302486 | 24.185 | 24.626 | 0.01191 | 0.01956 | 1.64x |
| MemphisCase2026_Mar7 | 993 | 3669 | 13355 | 1.073 | 1.082 | 0.00341 | 0.01241 | 3.64x |
| Texas7k_20220923 | 6717 | 24009 | 91901 | 7.390 | 7.610 | 0.00519 | 0.01184 | 2.28x |
| case_ACTIVSg200 | 200 | 690 | 2489 | 0.188 | 0.247 | 0.00325 | 0.00683 | 2.10x |
| case_ACTIVSg2000 | 2000 | 7334 | 26345 | 2.148 | 2.175 | 0.00340 | 0.01086 | 3.20x |
| case_ACTIVSg25k | 25000 | 85220 | 318672 | 26.581 | 26.012 | 0.01241 | 0.01617 | 1.30x |
| case_ACTIVSg500 | 500 | 1668 | 6275 | 0.429 | 0.547 | 0.00330 | 0.00806 | 2.44x |
| case_ACTIVSg70k | 70000 | 236636 | 900558 | 79.645 | 84.010 | 0.04876 | 0.13234 | 2.71x |

## Nsight Compute Metrics

| Case | Edge occ % | Vertex occ % | Edge lane util % | Vertex lane util % | Edge avg lanes | Vertex avg lanes | Edge ncu us | Vertex ncu us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 81.98 | 64.55 | 89.09 | 55.34 | 28.51 | 17.71 | 76.67 | 199.36 |
| Base_Florida_42GW | 18.29 | 14.08 | 89.14 | 54.43 | 28.52 | 17.42 | 11.97 | 24.99 |
| Base_MIOHIN_76GW | 31.04 | 14.61 | 89.42 | 57.10 | 28.61 | 18.27 | 16.90 | 27.62 |
| Base_Texas_66GW | 23.04 | 14.79 | 88.85 | 54.24 | 28.43 | 17.36 | 12.64 | 21.38 |
| Base_West_Interconnect_121GW | 57.27 | 16.89 | 89.09 | 53.94 | 28.51 | 17.26 | 26.24 | 40.19 |
| MemphisCase2026_Mar7 | 15.49 | 10.69 | 88.93 | 52.19 | 28.46 | 16.70 | 8.80 | 29.95 |
| Texas7k_20220923 | 20.70 | 13.18 | 88.16 | 55.38 | 28.21 | 17.72 | 12.00 | 27.58 |
| case_ACTIVSg200 | 14.91 | 11.39 | 85.28 | 38.57 | 27.29 | 12.34 | 8.19 | 17.12 |
| case_ACTIVSg2000 | 15.84 | 12.80 | 87.30 | 43.83 | 27.94 | 14.02 | 8.45 | 23.33 |
| case_ACTIVSg25k | 61.41 | 19.07 | 87.40 | 49.38 | 27.97 | 15.80 | 27.94 | 35.94 |
| case_ACTIVSg500 | 15.36 | 13.55 | 85.77 | 37.81 | 27.45 | 12.10 | 8.45 | 19.04 |
| case_ACTIVSg70k | 81.44 | 55.92 | 87.52 | 52.12 | 28.01 | 16.68 | 61.86 | 138.05 |

## 요약

- 현재 구현에서는 모든 케이스에서 edge fill이 vertex fill보다 빠름.
- Edge lane utilization은 케이스 전반에서 대략 85-89% 수준.
- Vertex lane utilization은 대략 38-57%로 낮음. thread 하나가 row 하나를 맡고 row 길이/분기가 달라 lane 낭비가 큼.
- 큰 케이스에서 vertex occupancy도 낮음. `case_ACTIVSg70k`: edge 81.44%, vertex 55.92%.
- `Edge ncu us`, `Vertex ncu us`는 ncu metric 문맥용 단일 측정값. 커널 시간 비교는 timing table의 `fill ms`를 기준으로 보는 게 맞음.
