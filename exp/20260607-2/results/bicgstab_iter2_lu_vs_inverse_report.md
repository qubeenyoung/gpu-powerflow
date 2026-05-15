# BiCGSTAB(2) inverse_gemv vs lu_solve

## 설정
- coarse correction: off
- middle solver: BiCGSTAB fixed 2 iterations
- block size: 4, 8, 16, 32
- 비교: `inverse_gemv` vs `lu_solve`

## case별 best

| case | inverse best bs | inverse speedup | LU best bs | LU speedup | LU NR iters | LU fallback |
|---|---:|---:|---:|---:|---:|---:|
| case2383wp | 8 | 1.68 | 4 | 1.66 | 7 | 4 |
| case3120sp | 16 | 0.78 | 4 | 0.78 | 8 | 4 |
| case9241pegase | 8 | 0.86 | 8 | 0.85 | 7 | 2 |
| case13659pegase | 32 | 0.80 | 4 | 0.77 | 8 | 3 |
| case6468rte | 4 | 0.93 | 4 | 0.93 | 3 | 1 |

## 평균 timing

| block | inverse apply ms | LU apply ms | inverse middle ms | LU middle ms | inverse setup ms | LU setup ms |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.031 | 0.133 | 0.294 | 0.398 | 1.564 | 1.552 |
| 8 | 0.030 | 0.200 | 0.293 | 0.469 | 1.524 | 1.506 |
| 16 | 0.032 | 0.346 | 0.293 | 0.614 | 1.527 | 1.486 |
| 32 | 0.033 | 0.649 | 0.295 | 0.921 | 1.437 | 1.373 |

## 판단
- LU solve와 inverse_gemv는 같은 dense block LU factorization에서 출발하므로 nonlinear 경로는 대부분 거의 같다.
- 현재 `lu_solve` apply는 block당 한 thread가 serial triangular solve를 수행하는 커널이라, 작은 block에서도 inverse_gemv보다 빠르지 않았다.
- setup은 LU solve가 inverse 생성을 생략하므로 약간 유리할 수 있지만, hybrid total에서는 apply/반복/후속 fallback 차이가 더 크게 보인다.
- 이번 결과만 보면 기존 `inverse_gemv`를 유지하는 편이 낫다.
