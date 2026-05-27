# BiCGSTAB(2) Block Size 4/8/16/32 비교

## 설정
- coarse correction: off (`metis_block_jacobi`)
- middle solver: `bicgstab_block_jacobi`
- BiCGSTAB fixed iterations: 2
- block size: 4, 8, 16, 32
- policy: bootstrap cuDSS 1회, polish threshold `1e-4`, accept `0.9`, reject `1.05`, fallback immediate

## case별 best

| case | best block | NR iters | cuDSS calls | fallback | hybrid ms | pure cuDSS ms | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 8 | 7 | 5 | 4 | 27.69 | 46.65 | 1.68 |
| case3120sp | 16 | 8 | 6 | 4 | 32.08 | 25.08 | 0.78 |
| case9241pegase | 8 | 7 | 4 | 2 | 61.54 | 52.92 | 0.86 |
| case13659pegase | 32 | 7 | 5 | 3 | 79.09 | 63.47 | 0.80 |
| case6468rte | 4 | 3 | 2 | 1 | 37.07 | 34.41 | 0.93 |

## block size별 speedup

| case | bs4 | bs8 | bs16 | bs32 |
|---|---:|---:|---:|---:|
| case2383wp | 1.65 | 1.68 | 1.60 | 1.67 |
| case3120sp | 0.77 | 0.75 | 0.78 | 0.73 |
| case9241pegase | 0.84 | 0.86 | 0.84 | 0.74 |
| case13659pegase | 0.78 | 0.78 | 0.78 | 0.80 |
| case6468rte | 0.93 | 0.93 | 0.88 | 0.88 |

## 첫 middle step 품질

| case | bs | dx ratio | cosine | mismatch ratio | linear rel |
|---|---:|---:|---:|---:|---:|
| case2383wp | 4 | 0.011 | 0.229 | 0.048 | 0.025 |
| case2383wp | 8 | 0.013 | 0.292 | 0.042 | 0.023 |
| case2383wp | 16 | 0.017 | 0.426 | 0.027 | 0.019 |
| case2383wp | 32 | 0.031 | 0.560 | 0.020 | 0.018 |
| case3120sp | 4 | 0.008 | 0.166 | 0.194 | 0.207 |
| case3120sp | 8 | 0.040 | 0.360 | 0.398 | 0.591 |
| case3120sp | 16 | 0.021 | 0.293 | 0.115 | 0.166 |
| case3120sp | 32 | 0.051 | 0.450 | 0.202 | 0.219 |
| case9241pegase | 4 | 0.119 | 0.212 | 0.112 | 0.058 |
| case9241pegase | 8 | 0.186 | 0.257 | 0.161 | 0.042 |
| case9241pegase | 16 | 0.235 | 0.259 | 0.195 | 0.022 |
| case9241pegase | 32 | 0.229 | 0.260 | 0.194 | 0.022 |
| case13659pegase | 4 | 0.005 | 0.032 | 0.394 | 0.288 |
| case13659pegase | 8 | 0.006 | 0.029 | 0.403 | 0.297 |
| case13659pegase | 16 | 0.006 | 0.020 | 0.408 | 0.302 |
| case13659pegase | 32 | 0.008 | 0.043 | 0.197 | 0.195 |
| case6468rte | 4 | 0.130 | 0.383 | 0.353 | 0.511 |
| case6468rte | 8 | 0.370 | 0.550 | 0.473 | 0.709 |
| case6468rte | 16 | 0.148 | -0.139 | 0.371 | 0.662 |
| case6468rte | 32 | 0.271 | 0.500 | 0.862 | 0.952 |

## 평균 timing

| block | middle total ms | block-Jacobi apply ms | dot ms | spmv ms | update ms | setup ms |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.294 | 0.031 | 0.161 | 0.037 | 0.035 | 1.564 |
| 8 | 0.293 | 0.030 | 0.161 | 0.038 | 0.035 | 1.524 |
| 16 | 0.293 | 0.032 | 0.162 | 0.036 | 0.034 | 1.527 |
| 32 | 0.295 | 0.033 | 0.161 | 0.037 | 0.034 | 1.437 |

## 판단
- block size를 아주 작게 줄인다고 단조롭게 좋아지지는 않았다.
- best speedup 기준으로는 case2383wp와 case9241pegase는 block 8, case3120sp는 block 16, case13659pegase는 block 32, case6468rte는 block 4가 가장 좋았다.
- 다만 모든 hard case에서 fallback은 여전히 남아 있고, pure cuDSS보다 빠른 케이스는 case2383wp뿐이다.
- 평균 middle solve time은 block 4~32 사이에서 큰 차이가 아니며, 지배 비용은 여전히 dot/reduction 쪽이 크다.
