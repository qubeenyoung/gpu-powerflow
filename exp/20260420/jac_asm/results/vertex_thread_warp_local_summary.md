# Vertex Thread vs Warp 결과

## 환경

- GPU: local `NVIDIA GeForce RTX 3090`
- Build: `cmake -S exp/20260420/jac_asm -B exp/20260420/jac_asm/build` (`sm_86`)
- Timing command: `./exp/20260420/jac_asm/build/jac_asm_bench --mode vertex_both --iters 100 --warmup 10 --check-vertex`
- Timing source: `results/timing_vertex_thread_warp_local.csv`
- NCU source: `results/ncu_vertex_thread_warp_MemphisCase2026_Mar7_local.csv`
- NCU source: `results/ncu_vertex_thread_warp_case_ACTIVSg70k_local.csv`
- `vertex_warp_over_thread < 1`이면 warp-per-bus가 더 빠름.
- Check tolerance: `abs_tol=1e-5`, `rel_tol=1e-4`

## Timing

| Case | Thread ms | Warp ms | Warp/Thread | Check bad |
|---|---:|---:|---:|---:|
| Base_Eastern_Interconnect_515GW | 0.082298 | 0.091269 | 1.109x | 0 |
| Base_Florida_42GW | 0.009800 | 0.007926 | 0.809x | 0 |
| Base_MIOHIN_76GW | 0.012060 | 0.012032 | 0.998x | 0 |
| Base_Texas_66GW | 0.010230 | 0.009523 | 0.931x | 0 |
| Base_West_Interconnect_121GW | 0.016722 | 0.021443 | 1.282x | 0 |
| MemphisCase2026_Mar7 | 0.012790 | 0.004053 | 0.317x | 0 |
| Texas7k_20220923 | 0.013722 | 0.008694 | 0.634x | 0 |
| case_ACTIVSg200 | 0.007301 | 0.003553 | 0.487x | 0 |
| case_ACTIVSg2000 | 0.011049 | 0.004414 | 0.400x | 0 |
| case_ACTIVSg25k | 0.016260 | 0.024801 | 1.525x | 0 |
| case_ACTIVSg500 | 0.008274 | 0.003584 | 0.433x | 0 |
| case_ACTIVSg70k | 0.056605 | 0.077435 | 1.368x | 0 |

## 요약

- Warp-per-bus는 12개 중 8개 케이스에서 더 빠름.
- Geomean `warp/thread`는 `0.761x`로, 전체적으로는 warp-per-bus가 약 `1.31x` 빠른 쪽.
- 큰 케이스 중 `case_ACTIVSg25k`, `case_ACTIVSg70k`, `Base_Eastern_Interconnect_515GW`, `Base_West_Interconnect_121GW`에서는 warp-per-bus가 느림.
- 모든 케이스가 기본 check tolerance를 통과함. max abs diff는 FP32 누적 순서 차이이며, max relative diff 기준으로는 `1e-4` 안쪽.

## NCU Spot Check

| Case | Kernel | NCU time us | Occ % | Lane util % | Avg lanes | Thread count | Inst sum |
|---|---|---:|---:|---:|---:|---:|---:|
| MemphisCase2026_Mar7 | thread | 19.456 | 10.85 | 45.98 | 14.71 | 1,024 | 23,072 |
| MemphisCase2026_Mar7 | warp | 6.144 | 23.97 | 35.91 | 11.49 | 31,744 | 223,516 |
| case_ACTIVSg70k | thread | 56.960 | 49.23 | 44.11 | 14.11 | 70,144 | 1,591,310 |
| case_ACTIVSg70k | warp | 88.896 | 77.00 | 34.62 | 11.08 | 2,240,000 | 16,186,246 |

NCU 해석:

- 작은 `MemphisCase2026_Mar7`에서는 thread-per-bus가 grid 4 blocks라 병렬성이 너무 낮고, warp-per-bus는 grid 124 blocks로 늘어나 launch된 thread/inst가 많아져도 latency를 숨겨 더 빠름.
- 큰 `case_ACTIVSg70k`에서는 thread-per-bus만으로도 병렬성이 충분함. warp-per-bus는 occupancy는 높지만 thread count와 instruction count가 크게 늘고 lane util이 더 낮아서 느림.
- 결론적으로 full warp-per-bus는 작은/중간 케이스의 병렬성 부족을 보완하지만, 큰 케이스에서는 row 평균 길이가 짧아 과한 over-subscription이 됨.
