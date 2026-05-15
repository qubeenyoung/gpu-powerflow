# BiCGSTAB(2), block size 8, all 78 MATPOWER cases

## 설정
- middle solver: `bicgstab_block_jacobi`
- fixed BiCGSTAB iterations: 2
- preconditioner: `metis_block_jacobi`
- block size: 8
- apply mode: `inverse_gemv`
- coarse correction: off
- policy: bootstrap cuDSS 1회, polish threshold `1e-4`, accept `0.9`, reject `1.05`, fallback immediate
- warmup: 1


## naming note

The CSV writer still uses legacy column names such as `gmres_calls`, `accepted_gmres_steps`, and `gmres_block_size`. In this run those columns mean BiCGSTAB middle-solver trials/accepted steps, because `middle_solver=bicgstab_block_jacobi` and `bicgstab_iters=2` for every row. The iteration log confirms this with `solver_used=bicgstab_middle`.

## 전체 요약

| metric | value |
|---|---:|
| cases | 78 |
| converged | 78 |
| aggregate speedup, sum pure / sum hybrid | 0.842 |
| median speedup | 0.887 |
| mean speedup | 0.869 |
| speedup > 1.0 cases | 5 |
| speedup >= 0.95 cases | 12 |
| speedup < 0.8 cases | 18 |
| cases with any BiCGSTAB middle trial | 68 |
| cases with accepted BiCGSTAB middle step | 49 |
| cases with zero fallback | 19 |

## size bins

| bin | cases | aggregate speedup | median speedup | wins | accepted BiCGSTAB cases |
|---|---:|---:|---:|---:|---:|
| <=100 buses | 41 | 0.875 | 0.894 | 3 | 22 |
| 101-1000 buses | 10 | 0.825 | 0.894 | 0 | 6 |
| 1001-10000 buses | 23 | 0.845 | 0.882 | 2 | 18 |
| >10000 buses | 4 | 0.829 | 0.824 | 0 | 3 |

## top speedup cases

| case | buses | speedup | hybrid ms | pure ms | NR | cuDSS | BiCGSTAB trials | accepted | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case74ds | 74 | 1.083 | 8.87 | 9.61 | 3 | 3 | 1 | 0 | 1 |
| case14 | 14 | 1.005 | 9.00 | 9.05 | 2 | 2 | 0 | 0 | 0 |
| case1197 | 1197 | 1.003 | 12.49 | 12.53 | 3 | 3 | 0 | 0 | 0 |
| case3375wp | 3374 | 1.001 | 20.10 | 20.12 | 2 | 2 | 0 | 0 | 0 |
| case39 | 39 | 1.000 | 7.64 | 7.64 | 1 | 1 | 0 | 0 | 0 |
| case1888rte | 1888 | 1.000 | 14.74 | 14.74 | 2 | 2 | 0 | 0 | 0 |
| case18nbr | 18 | 0.999 | 8.09 | 8.09 | 3 | 3 | 0 | 0 | 0 |
| case60nordic | 60 | 0.999 | 7.75 | 7.73 | 1 | 1 | 0 | 0 | 0 |
| case_ACTIVSg200 | 200 | 0.998 | 8.71 | 8.69 | 2 | 2 | 0 | 0 | 0 |
| case15nbr | 15 | 0.998 | 8.05 | 8.04 | 3 | 3 | 0 | 0 | 0 |
| case6495rte | 6495 | 0.997 | 32.00 | 31.89 | 2 | 2 | 0 | 0 | 0 |
| case6ww | 6 | 0.952 | 9.52 | 9.07 | 3 | 2 | 1 | 1 | 0 |

## worst speedup cases

| case | buses | speedup | hybrid ms | pure ms | NR | cuDSS | BiCGSTAB trials | accepted | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case18 | 18 | 0.627 | 13.31 | 8.34 | 10 | 3 | 9 | 7 | 2 |
| case_ACTIVSg500 | 500 | 0.640 | 18.88 | 12.08 | 10 | 3 | 8 | 7 | 1 |
| case59 | 59 | 0.641 | 14.19 | 9.10 | 11 | 4 | 9 | 7 | 2 |
| case2746wp | 2746 | 0.650 | 29.77 | 19.35 | 8 | 4 | 7 | 4 | 3 |
| case16ci | 16 | 0.702 | 11.37 | 7.98 | 8 | 2 | 6 | 6 | 0 |
| case_ACTIVSg2000 | 2000 | 0.716 | 27.85 | 19.94 | 9 | 3 | 7 | 6 | 1 |
| case300 | 300 | 0.719 | 14.75 | 10.61 | 9 | 3 | 7 | 6 | 1 |
| case3120sp | 3120 | 0.734 | 30.70 | 22.52 | 9 | 6 | 7 | 3 | 4 |
| case13659pegase | 13659 | 0.734 | 80.81 | 59.31 | 8 | 5 | 6 | 3 | 3 |
| case2869pegase | 2869 | 0.748 | 31.61 | 23.64 | 10 | 3 | 8 | 7 | 1 |
| case24_ieee_rts | 24 | 0.759 | 11.07 | 8.40 | 7 | 3 | 5 | 4 | 1 |
| case2737sop | 2737 | 0.770 | 26.63 | 20.51 | 8 | 4 | 6 | 4 | 2 |

## speedup > 1 cases

| case | buses | speedup | accepted BiCGSTAB | BiCGSTAB trials | fallback | note |
|---|---:|---:|---:|---:|---:|---|
| case1197 | 1197 | 1.003 | 0 | 0 | 0 | no middle BiCGSTAB |
| case14 | 14 | 1.005 | 0 | 0 | 0 | no middle BiCGSTAB |
| case3375wp | 3374 | 1.001 | 0 | 0 | 0 | no middle BiCGSTAB |
| case39 | 39 | 1.000 | 0 | 0 | 0 | no middle BiCGSTAB |
| case74ds | 74 | 1.083 | 0 | 1 | 1 | no accepted BiCGSTAB |

## 판단
- 78개 모두 수렴했다.
- 전체 합산 기준 hybrid는 pure cuDSS보다 느렸다: aggregate speedup `0.842`.
- accepted BiCGSTAB step이 실제로 있는 49개 케이스만 보면 aggregate speedup은 `0.816`이고, speedup > 1인 케이스는 `0`개다.
- speedup > 1 케이스는 대부분 middle BiCGSTAB가 쓰이지 않았거나 accept되지 않은 케이스라, hybrid middle solver가 직접 가속을 만든 증거로 보기는 어렵다.
- block size 8, BiCGSTAB(2) 조합은 안정적으로 수렴은 하지만, 전체 MATPOWER sweep에서는 cuDSS 호출 감소/NR progress가 시간을 이길 만큼 충분하지 않았다.
