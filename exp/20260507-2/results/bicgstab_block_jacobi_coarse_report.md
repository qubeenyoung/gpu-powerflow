# BiCGSTAB + METIS Block-Jacobi Coarse 결과
## 설정
- middle solver: `bicgstab_block_jacobi` fixed iterations 1/2/4
- preconditioner 비교: `metis_block_jacobi` vs `metis_block_jacobi_coarse`
- coarse: block당 변수 1개, `bootstrap_only`, FP32, dense coarse solve 재사용
- policy: bootstrap cuDSS 1회, polish threshold `1e-4`, accept `0.9`, reject `1.05`, fallback immediate

## case별 coarse best

| case | best coarse | NR iters | cuDSS calls | fallback | hybrid ms | pure cuDSS ms | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | BiCGSTAB(1)+coarse | 9 | 5 | 3 | 32.55 | 46.54 | 1.43 |
| case3120sp | BiCGSTAB(1)+coarse | 11 | 6 | 4 | 38.37 | 25.30 | 0.66 |
| case9241pegase | BiCGSTAB(2)+coarse | 7 | 4 | 2 | 66.86 | 52.78 | 0.79 |
| case13659pegase | BiCGSTAB(1)+coarse | 8 | 5 | 3 | 86.11 | 63.16 | 0.73 |
| case6468rte | BiCGSTAB(1)+coarse | 3 | 2 | 1 | 38.01 | 34.24 | 0.90 |

## 같은 iteration 수에서 coarse 효과

| case | iters | speedup no coarse | speedup coarse | fallback no coarse | fallback coarse | NR no coarse | NR coarse |
|---|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 1 | 1.57 | 1.43 | 3 | 3 | 9 | 9 |
| case2383wp | 2 | 1.68 | 1.34 | 3 | 3 | 7 | 10 |
| case2383wp | 4 | 1.30 | 1.27 | 3 | 3 | 11 | 9 |
| case3120sp | 1 | 0.78 | 0.66 | 4 | 4 | 8 | 11 |
| case3120sp | 2 | 0.76 | 0.58 | 4 | 4 | 8 | 12 |
| case3120sp | 4 | 0.62 | 0.60 | 4 | 4 | 10 | 9 |
| case9241pegase | 1 | 0.79 | 0.78 | 2 | 2 | 9 | 8 |
| case9241pegase | 2 | 0.84 | 0.79 | 2 | 2 | 7 | 7 |
| case9241pegase | 4 | 0.81 | 0.75 | 2 | 2 | 7 | 7 |
| case13659pegase | 1 | 0.70 | 0.73 | 4 | 3 | 10 | 8 |
| case13659pegase | 2 | 0.79 | 0.69 | 3 | 4 | 7 | 8 |
| case13659pegase | 4 | 0.77 | 0.73 | 3 | 4 | 7 | 5 |
| case6468rte | 1 | 0.92 | 0.90 | 0 | 1 | 4 | 3 |
| case6468rte | 2 | 0.97 | 0.88 | 0 | 1 | 3 | 3 |
| case6468rte | 4 | 0.91 | 0.89 | 1 | 1 | 3 | 3 |

## 첫 middle step dx 품질

| case | iters | dx ratio no coarse | dx ratio coarse | cosine no coarse | cosine coarse | mismatch ratio no coarse | mismatch ratio coarse |
|---|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 1 | 0.023 | 0.025 | 0.550 | 0.610 | 0.026 | 0.088 |
| case2383wp | 2 | 0.028 | 0.053 | 0.603 | -0.793 | 0.022 | 0.150 |
| case2383wp | 4 | 0.046 | 0.283 | 0.704 | 0.849 | 0.017 | 0.122 |
| case3120sp | 1 | 0.025 | 0.025 | 0.416 | 0.529 | 0.226 | 0.294 |
| case3120sp | 2 | 0.029 | 0.096 | 0.472 | 0.777 | 0.197 | 0.288 |
| case3120sp | 4 | 0.048 | 0.069 | 0.614 | 0.486 | 0.192 | 0.654 |
| case9241pegase | 1 | 0.291 | 0.289 | 0.254 | 0.259 | 0.284 | 0.284 |
| case9241pegase | 2 | 0.283 | 0.282 | 0.264 | 0.282 | 0.242 | 0.242 |
| case9241pegase | 4 | 0.258 | 0.258 | 0.271 | 0.298 | 0.211 | 0.210 |
| case13659pegase | 1 | 0.006 | 0.011 | -0.023 | 0.361 | 0.671 | 0.823 |
| case13659pegase | 2 | 0.009 | 0.020 | -0.003 | 0.342 | 0.467 | 0.646 |
| case13659pegase | 4 | 0.385 | 0.040 | -0.012 | 0.240 | 14.368 | 1.224 |
| case6468rte | 1 | 0.345 | 0.370 | 0.578 | 0.644 | 0.174 | 0.409 |
| case6468rte | 2 | 0.542 | 0.450 | 0.641 | 0.664 | 0.082 | 0.629 |
| case6468rte | 4 | 0.779 | 0.851 | 0.674 | 0.687 | 0.749 | 6.977 |

## 평균 middle timing

| mode | iters | middle total ms | preconditioner ms | coarse total ms | dot ms | spmv ms | update ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| block_jacobi | 1 | 0.205 | 0.043 | 0.000 | 0.115 | 0.023 | 0.021 |
| block_jacobi | 2 | 0.329 | 0.083 | 0.000 | 0.162 | 0.042 | 0.036 |
| block_jacobi | 4 | 0.560 | 0.158 | 0.000 | 0.254 | 0.077 | 0.061 |
| block_jacobi_coarse | 1 | 0.445 | 0.295 | 0.208 | 0.110 | 0.017 | 0.021 |
| block_jacobi_coarse | 2 | 0.798 | 0.567 | 0.394 | 0.159 | 0.033 | 0.035 |
| block_jacobi_coarse | 4 | 1.491 | 1.102 | 0.763 | 0.252 | 0.065 | 0.062 |

## 판단

- coarse는 BiCGSTAB preconditioner apply 안에 정상 적용되며, ILU/SpSV는 사용하지 않는다.
- coarse를 켠 첫 middle step은 일부 케이스/iteration에서 mismatch ratio를 낮췄지만, dx cosine은 음수로 떨어지는 경우가 많아 방향 품질이 안정적으로 좋아지지는 않았다.
- 평균 middle time은 coarse 때문에 대략 2배 안팎으로 늘었다. BiCGSTAB(2)는 no-coarse 약 0.33 ms에서 coarse 약 0.80 ms 수준이다.
- fallback/NR iteration은 전반적으로 줄지 않았다. case2383wp는 여전히 speedup이 있지만, 다른 hard case에서는 pure cuDSS보다 느리다.
- 결론: BiCGSTAB + coarse는 적용은 가능하지만, 현재 coarse correction은 hybrid NR middle solver로 명확한 개선을 주지 못했다.
