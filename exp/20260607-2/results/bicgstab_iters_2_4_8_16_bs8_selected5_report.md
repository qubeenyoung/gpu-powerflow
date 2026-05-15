# BiCGSTAB iteration count check, block size 8, selected 1K+ cases

Settings: `middle_solver=bicgstab_block_jacobi`, `preconditioner=metis_block_jacobi`, `block_size=8`, FP32 `inverse_gemv`, bootstrap cuDSS 1, polish `1e-4`, accept `0.9`, reject `1.05`, fallback immediate, warmup 1.

CSV columns in raw summary files still use legacy `gmres_*` names; here they mean BiCGSTAB middle calls/accepted steps.

## Aggregate

| iters | aggregate speedup | mean speedup | total NR | cuDSS calls | middle calls | accepted | fallback | avg middle ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.849 | 0.843 | 33 | 17 | 24 | 16 | 8 | 0.277 |
| 4 | 0.786 | 0.796 | 34 | 19 | 25 | 15 | 10 | 0.483 |
| 8 | 0.785 | 0.781 | 29 | 21 | 20 | 8 | 12 | 0.883 |
| 16 | 0.689 | 0.689 | 30 | 21 | 20 | 9 | 11 | 1.699 |

## Case comparison

| case | iters | NR | cuDSS | middle | accepted | fallback | hybrid ms | pure ms | speedup | accepted mismatch ratio | avg middle ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case6470rte | 2 | 3 | 2 | 2 | 1 | 1 | 37.93 | 35.06 | 0.924 | 0.067 | 0.300 |
| case6470rte | 4 | 3 | 3 | 1 | 0 | 1 | 37.28 | 35.36 | 0.948 | 0.000 | 0.518 |
| case6470rte | 8 | 3 | 3 | 1 | 0 | 1 | 37.49 | 35.01 | 0.934 | 0.000 | 0.935 |
| case6470rte | 16 | 3 | 3 | 1 | 0 | 1 | 39.29 | 35.00 | 0.891 | 0.000 | 1.805 |
| case2868rte | 2 | 5 | 4 | 3 | 1 | 2 | 25.42 | 22.72 | 0.894 | 0.227 | 0.259 |
| case2868rte | 4 | 5 | 5 | 3 | 0 | 3 | 27.14 | 22.68 | 0.836 | 0.000 | 0.458 |
| case2868rte | 8 | 5 | 5 | 3 | 0 | 3 | 29.03 | 22.67 | 0.781 | 0.000 | 0.835 |
| case2868rte | 16 | 5 | 5 | 3 | 0 | 3 | 32.74 | 22.82 | 0.697 | 0.000 | 1.620 |
| case9241pegase | 2 | 7 | 4 | 5 | 3 | 2 | 60.85 | 52.81 | 0.868 | 0.203 | 0.318 |
| case9241pegase | 4 | 9 | 4 | 8 | 5 | 3 | 71.90 | 52.64 | 0.732 | 0.483 | 0.553 |
| case9241pegase | 8 | 6 | 6 | 5 | 0 | 5 | 68.28 | 53.82 | 0.788 | 0.000 | 1.008 |
| case9241pegase | 16 | 7 | 6 | 5 | 1 | 4 | 75.39 | 52.73 | 0.699 | 0.045 | 1.905 |
| case2869pegase | 2 | 10 | 3 | 8 | 7 | 1 | 31.69 | 23.66 | 0.747 | 0.346 | 0.253 |
| case2869pegase | 4 | 10 | 3 | 8 | 7 | 1 | 34.18 | 23.74 | 0.695 | 0.322 | 0.444 |
| case2869pegase | 8 | 8 | 3 | 6 | 5 | 1 | 34.51 | 23.88 | 0.692 | 0.155 | 0.817 |
| case2869pegase | 16 | 8 | 3 | 6 | 5 | 1 | 41.78 | 23.69 | 0.567 | 0.124 | 1.585 |
| case2737sop | 2 | 8 | 4 | 6 | 4 | 2 | 29.00 | 22.64 | 0.781 | 0.461 | 0.255 |
| case2737sop | 4 | 7 | 4 | 5 | 3 | 2 | 29.44 | 22.66 | 0.770 | 0.318 | 0.445 |
| case2737sop | 8 | 7 | 4 | 5 | 3 | 2 | 32.27 | 22.83 | 0.707 | 0.298 | 0.820 |
| case2737sop | 16 | 7 | 4 | 5 | 3 | 2 | 38.37 | 22.64 | 0.590 | 0.328 | 1.580 |

## Notes

- `accepted mismatch ratio`는 accepted BiCGSTAB step 이후 mismatch_inf / 이전 mismatch_inf의 평균이다. 낮을수록 그 middle step 자체는 더 강하다.
- Iteration을 늘리면 일부 케이스에서 middle step 품질은 좋아지지만, accepted 수가 늘지 않거나 fallback/cuDSS 호출이 줄지 않는다.
- 평균 middle solve time은 `iters=2` 약 0.27 ms에서 `iters=16` 약 1.70 ms로 증가한다.
- 선택한 5개 케이스 전체 기준 최고 aggregate speedup은 `iters=2`이다. `iters=4/8/16`은 개선이 아니다.
- 결론: BiCGSTAB을 길게 돌리는 방향은 현재 hybrid NR 가속에 도움이 되지 않는다.
