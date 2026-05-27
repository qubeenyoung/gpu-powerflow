# BiCGSTAB Block Size 64/32/16 비교

## 설정
- coarse correction: off (`metis_block_jacobi`)
- middle solver: `bicgstab_block_jacobi`, fixed iterations 1/2/4
- block size 비교: 64 기존 결과, 32 신규, 16 신규
- policy: bootstrap cuDSS 1회, polish threshold `1e-4`, accept `0.9`, reject `1.05`, fallback immediate

## case별 best

| case | best block | iters | NR iters | cuDSS calls | fallback | hybrid ms | pure cuDSS ms | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 64 | 2 | 7 | 5 | 3 | 27.81 | 46.83 | 1.68 |
| case3120sp | 16 | 4 | 6 | 6 | 4 | 30.70 | 25.14 | 0.82 |
| case9241pegase | 16 | 2 | 7 | 5 | 3 | 62.47 | 53.02 | 0.85 |
| case13659pegase | 16 | 4 | 6 | 5 | 3 | 76.49 | 63.25 | 0.83 |
| case6468rte | 64 | 2 | 3 | 2 | 0 | 35.35 | 34.29 | 0.97 |

## block size별 best speedup

| case | bs64 best | bs32 best | bs16 best |
|---|---:|---:|---:|
| case2383wp | 1.68 | 1.67 | 1.60 |
| case3120sp | 0.78 | 0.81 | 0.82 |
| case9241pegase | 0.84 | 0.80 | 0.85 |
| case13659pegase | 0.79 | 0.83 | 0.83 |
| case6468rte | 0.97 | 0.93 | 0.88 |

## 첫 middle step 품질: BiCGSTAB(2)

| case | bs | dx ratio | cosine | mismatch ratio | linear rel |
|---|---:|---:|---:|---:|---:|
| case2383wp | 64 | 0.028 | 0.603 | 0.022 | 0.018 |
| case2383wp | 32 | 0.031 | 0.560 | 0.020 | 0.018 |
| case2383wp | 16 | 0.017 | 0.426 | 0.027 | 0.019 |
| case3120sp | 64 | 0.029 | 0.472 | 0.197 | 0.154 |
| case3120sp | 32 | 0.051 | 0.450 | 0.202 | 0.219 |
| case3120sp | 16 | 0.021 | 0.293 | 0.115 | 0.166 |
| case9241pegase | 64 | 0.283 | 0.264 | 0.242 | 0.028 |
| case9241pegase | 32 | 0.229 | 0.260 | 0.194 | 0.022 |
| case9241pegase | 16 | 0.235 | 0.259 | 0.195 | 0.022 |
| case13659pegase | 64 | 0.009 | -0.003 | 0.467 | 0.282 |
| case13659pegase | 32 | 0.008 | 0.043 | 0.197 | 0.195 |
| case13659pegase | 16 | 0.006 | 0.020 | 0.408 | 0.302 |
| case6468rte | 64 | 0.542 | 0.641 | 0.082 | 0.164 |
| case6468rte | 32 | 0.271 | 0.500 | 0.862 | 0.952 |
| case6468rte | 16 | 0.148 | -0.139 | 0.371 | 0.662 |

## 평균 middle timing

| block | iters | middle total ms | block-Jacobi apply ms | dot ms | spmv ms | update ms | setup ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 0.205 | 0.031 | 0.115 | 0.023 | 0.021 | 1.781 |
| 64 | 2 | 0.329 | 0.058 | 0.162 | 0.042 | 0.036 | 1.756 |
| 64 | 4 | 0.560 | 0.109 | 0.254 | 0.077 | 0.061 | 1.523 |
| 32 | 1 | 0.189 | 0.018 | 0.115 | 0.020 | 0.021 | 1.515 |
| 32 | 2 | 0.297 | 0.033 | 0.162 | 0.038 | 0.035 | 1.441 |
| 32 | 4 | 0.507 | 0.063 | 0.253 | 0.071 | 0.061 | 1.328 |
| 16 | 1 | 0.186 | 0.017 | 0.114 | 0.020 | 0.021 | 1.531 |
| 16 | 2 | 0.294 | 0.032 | 0.162 | 0.036 | 0.034 | 1.533 |
| 16 | 4 | 0.512 | 0.063 | 0.259 | 0.072 | 0.060 | 1.601 |

## 판단
- block을 64에서 32/16으로 줄이면 첫 middle step의 mismatch ratio가 여러 케이스에서 좋아졌다. 특히 case2383wp, case3120sp, case13659pegase에서 효과가 보인다.
- 하지만 작은 block에서도 fallback이 완전히 줄지는 않았고, setup/solve 시간을 포함한 총 시간 개선은 제한적이다.
- case2383wp는 기존 block 64, BiCGSTAB(2)가 여전히 best이고, block 32, BiCGSTAB(2)가 거의 근접했다.
- hard case 전체로 보면 block 16/32가 dx 품질은 개선하지만, pure cuDSS보다 빠른 구조로 뒤집지는 못했다.
