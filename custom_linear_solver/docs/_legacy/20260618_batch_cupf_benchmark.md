# Batched linear solver & cuPF mixed-mode benchmark — 2026-06-18

RTX 3090 (sm_86), CUDA 12.8, **locked clocks**. cuDSS 12 (`/usr/lib/x86_64-linux-gnu/libcudss/12`). Target = the 9 power-grid cases (NR Jacobians for Part 1; the same grids' full power-flow dumps for Part 2). All FP32. Median over repeats (linear solver 5, cuPF 3); warmup discarded.

---
## Part 1 — Batched linear solver: custom vs cuDSS (FP32)

**Runners**
```
custom : custom_linear_solver_run <case> --precision fp32 --batch B --batch-only --repeat 5 --warmup 2
cuDSS  : cudss_run               <case> --precision fp32 --batch B --repeat 5   # CUDSS_CONFIG_UBATCH_SIZE
metric : per-system ms = batch median / B   (factorize_per_sys_ms, solve_per_sys_ms)
```
### FACTORIZE per-system (ms) — custom | cuDSS
| case | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|
| case118 | 0.0229 \| 0.0339 | 0.0116 \| 0.0180 | 0.0060 \| 0.0109 | 0.0031 \| 0.0074 | 0.0017 \| 0.0055 | 0.0009 \| 0.0046 |
| case1354pegase | 0.0553 \| 0.0995 | 0.0294 \| 0.0659 | 0.0170 \| 0.0497 | 0.0098 \| 0.0414 | 0.0066 \| 0.0386 | 0.0043 \| 0.0373 |
| case3120sp | 0.1022 \| 0.1475 | 0.0524 \| 0.1074 | 0.0332 \| 0.0880 | 0.0192 \| 0.0804 | 0.0117 \| 0.0758 | 0.0096 \| 0.0729 |
| case6468rte | 0.1435 \| 0.2164 | 0.0853 \| 0.1644 | 0.0564 \| 0.1412 | 0.0284 \| 0.1277 | 0.0223 \| 0.1199 | 0.0185 \| 0.1164 |
| case8387pegase | 0.2040 \| 0.3429 | 0.1122 \| 0.2471 | 0.0632 \| 0.2013 | 0.0431 \| 0.1767 | 0.0303 \| 0.1645 | 0.0243 \| 0.1578 |
| case13659pegase | 0.3385 \| 0.4347 | 0.1564 \| 0.3283 | 0.1083 \| 0.2769 | 0.0632 \| 0.2470 | 0.0486 \| 0.2329 | 0.0428 \| 0.2256 |
| case_ACTIVSg25k | 0.5176 \| 0.7398 | 0.3074 \| 0.6170 | 0.2118 \| 0.5542 | 0.1552 \| 0.5239 | 0.1201 \| 0.5079 | 0.1139 \| 0.4999 |
| case_ACTIVSg70k | 1.0374 \| 1.9042 | 0.6822 \| 1.6820 | 0.5127 \| 1.5733 | 0.4381 \| 1.5179 | 0.4159 \| 1.4892 | 0.4008 \| 1.4755 |
| case_SyntheticUSA | 1.0987 \| 2.2004 | 0.7352 \| 1.9511 | 0.5570 \| 1.8130 | 0.4744 \| 1.7509 | 0.4513 \| 1.7168 | 0.4361 \| 1.7081 |

**Ratio cuDSS/custom (FACTORIZE, >1 = custom faster)**
| case | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|
| case118 | 1.48x | 1.55x | 1.82x | 2.41x | 3.36x | 4.96x |
| case1354pegase | 1.80x | 2.24x | 2.93x | 4.24x | 5.88x | 8.64x |
| case3120sp | 1.44x | 2.05x | 2.65x | 4.20x | 6.46x | 7.60x |
| case6468rte | 1.51x | 1.93x | 2.50x | 4.50x | 5.37x | 6.29x |
| case8387pegase | 1.68x | 2.20x | 3.18x | 4.10x | 5.43x | 6.51x |
| case13659pegase | 1.28x | 2.10x | 2.56x | 3.91x | 4.79x | 5.27x |
| case_ACTIVSg25k | 1.43x | 2.01x | 2.62x | 3.38x | 4.23x | 4.39x |
| case_ACTIVSg70k | 1.84x | 2.47x | 3.07x | 3.46x | 3.58x | 3.68x |
| case_SyntheticUSA | 2.00x | 2.65x | 3.26x | 3.69x | 3.80x | 3.92x |

### SOLVE per-system (ms) — custom | cuDSS
| case | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|
| case118 | 0.0268 \| 0.0289 | 0.0133 \| 0.0147 | 0.0067 \| 0.0077 | 0.0034 \| 0.0049 | 0.0018 \| 0.0033 | 0.0010 \| 0.0025 |
| case1354pegase | 0.0546 \| 0.0618 | 0.0286 \| 0.0421 | 0.0155 \| 0.0325 | 0.0086 \| 0.0271 | 0.0053 \| 0.0246 | 0.0031 \| 0.0249 |
| case3120sp | 0.0898 \| 0.0795 | 0.0477 \| 0.0540 | 0.0281 \| 0.0413 | 0.0158 \| 0.0366 | 0.0087 \| 0.0339 | 0.0056 \| 0.0327 |
| case6468rte | 0.1296 \| 0.1049 | 0.0721 \| 0.0755 | 0.0453 \| 0.0641 | 0.0214 \| 0.0568 | 0.0135 \| 0.0532 | 0.0089 \| 0.0514 |
| case8387pegase | 0.1666 \| 0.1349 | 0.1030 \| 0.0975 | 0.0500 \| 0.0803 | 0.0301 \| 0.0698 | 0.0169 \| 0.0648 | 0.0108 \| 0.0624 |
| case13659pegase | 0.2532 \| 0.1708 | 0.1196 \| 0.1361 | 0.0699 \| 0.1129 | 0.0357 \| 0.1022 | 0.0221 \| 0.0961 | 0.0164 \| 0.0933 |
| case_ACTIVSg25k | 0.3420 \| 0.2690 | 0.1819 \| 0.2238 | 0.1061 \| 0.2002 | 0.0637 \| 0.1861 | 0.0406 \| 0.1807 | 0.0312 \| 0.1762 |
| case_ACTIVSg70k | 0.5556 \| 0.6422 | 0.2943 \| 0.5785 | 0.1731 \| 0.5431 | 0.1152 \| 0.5245 | 0.0915 \| 0.5155 | 0.0771 \| 0.5093 |
| case_SyntheticUSA | 0.5488 \| 0.7355 | 0.3000 \| 0.6624 | 0.1832 \| 0.6218 | 0.1280 \| 0.6014 | 0.0991 \| 0.5916 | 0.0878 \| 0.5858 |

**Ratio cuDSS/custom (SOLVE, >1 = custom faster)**
| case | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|
| case118 | 1.08x | 1.10x | 1.14x | 1.42x | 1.84x | 2.53x |
| case1354pegase | 1.13x | 1.47x | 2.09x | 3.14x | 4.66x | 8.06x |
| case3120sp | 0.88x | 1.13x | 1.47x | 2.32x | 3.91x | 5.81x |
| case6468rte | 0.81x | 1.05x | 1.42x | 2.65x | 3.95x | 5.79x |
| case8387pegase | 0.81x | 0.95x | 1.61x | 2.32x | 3.83x | 5.76x |
| case13659pegase | 0.67x | 1.14x | 1.61x | 2.87x | 4.36x | 5.70x |
| case_ACTIVSg25k | 0.79x | 1.23x | 1.89x | 2.92x | 4.45x | 5.65x |
| case_ACTIVSg70k | 1.16x | 1.97x | 3.14x | 4.55x | 5.63x | 6.61x |
| case_SyntheticUSA | 1.34x | 2.21x | 3.39x | 4.70x | 5.97x | 6.68x |

---
## Part 2 — cuPF mixed-mode power flow: custom vs cuDSS backend

**Build**: `WITH_CUDA=ON CUPF_ENABLE_CUSTOM_SOLVER=ON ENABLE_TIMING=ON BUILD_EVALUATORS=ON CUPF_ENABLE_CUDA_GRAPH=OFF CUPF_WITH_TORCH=OFF` (arch 86, cuDSS 12).
**Runner**: `cupf_batch_bench <dump_dir> <cudss-fp32|custom-fp32> 1,4,16,64,256 3`
**Mode**: `ComputePolicy::Mixed` — public I/O FP64, internal Jacobian + linear solve FP32. Only the linear-solver backend differs; Ybus spMM / Jacobian / voltage-update are shared.
**Metric**: total NR `solve_total_us` (whole batch); linear-solve = `factorize_us + solve_us`. All 9 cases converge in 5 NR iterations, max mismatch ~1e-8.
>  ⚠ CUDA graph is **OFF** (mutually exclusive with per-stage host timing). The graph is a custom-only feature, so these custom numbers are **conservative** (no graph-replay benefit).

### Total NR solve — ratio cuDSS/custom (>1 = custom faster)
| case | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| case118 | 1.20x | 1.27x | 1.54x | 2.49x | 4.09x |
| case1354pegase | 1.17x | 1.64x | 2.92x | 5.43x | 8.07x |
| case3120sp | 1.04x | 1.45x | 2.62x | 4.82x | 6.86x |
| case6468rte | 1.05x | 1.23x | 2.45x | 3.74x | 4.61x |
| case8387pegase | 1.19x | 1.42x | 2.45x | 3.69x | 4.02x |
| case13659pegase | 1.19x | 1.63x | 2.87x | 4.64x | 5.12x |
| case_ACTIVSg25k | 1.00x | 1.61x | 2.82x | 3.57x | 3.45x |
| case_ACTIVSg70k | 1.15x | 2.19x | 3.21x | 3.36x | 3.78x |
| case_SyntheticUSA | 1.25x | 2.28x | 3.31x | 3.48x | 3.88x |

### Linear-solve (factorize+solve) — ratio cuDSS/custom
| case | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| case118 | 1.34x | 1.40x | 1.94x | 3.69x | 8.54x |
| case1354pegase | 1.21x | 1.88x | 3.76x | 8.40x | 14.38x |
| case3120sp | 1.03x | 1.54x | 3.01x | 6.59x | 10.59x |
| case6468rte | 1.06x | 1.53x | 3.40x | 5.98x | 7.49x |
| case8387pegase | 1.23x | 1.44x | 3.21x | 5.85x | 7.34x |
| case13659pegase | 1.22x | 1.76x | 3.67x | 5.75x | 6.64x |
| case_ACTIVSg25k | 1.00x | 1.72x | 3.50x | 4.88x | 5.08x |
| case_ACTIVSg70k | 1.17x | 2.39x | 3.83x | 4.31x | 4.23x |
| case_SyntheticUSA | 1.23x | 2.50x | 3.96x | 4.41x | 4.39x |

### Absolute total NR time (ms, whole batch) — cuDSS | custom
| case | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| case118 | 0.6 \| 0.5 | 0.7 \| 0.6 | 0.9 \| 0.6 | 1.7 \| 0.7 | 5.1 \| 1.2 |
| case1354pegase | 1.5 \| 1.3 | 2.1 \| 1.3 | 5.0 \| 1.7 | 17.2 \| 3.2 | 65.3 \| 8.1 |
| case3120sp | 2.3 \| 2.2 | 3.7 \| 2.6 | 10.5 \| 4.0 | 38.7 \| 8.0 | 167.5 \| 24.4 |
| case6468rte | 1.4 \| 1.3 | 2.4 \| 1.9 | 7.1 \| 2.9 | 36.2 \| 9.7 | 185.6 \| 40.3 |
| case8387pegase | 2.8 \| 2.4 | 5.0 \| 3.5 | 13.6 \| 5.6 | 48.1 \| 13.0 | 252.9 \| 62.9 |
| case13659pegase | 5.5 \| 4.6 | 10.6 \| 6.5 | 30.8 \| 10.7 | 645.8 \| 139.1 | 2555.5 \| 499.6 |
| case_ACTIVSg25k | 6.4 \| 6.4 | 15.0 \| 9.3 | 49.4 \| 17.5 | 189.2 \| 53.0 | 773.1 \| 224.2 |
| case_ACTIVSg70k | 21.2 \| 18.4 | 58.3 \| 26.6 | 208.7 \| 65.0 | 834.4 \| 248.2 | 16045.4 \| 4250.1 |
| case_SyntheticUSA | 24.6 \| 19.7 | 67.3 \| 29.5 | 242.2 \| 73.2 | 1116.2 \| 320.4 | 18567.8 \| 4779.9 |

---
## Raw data

### Part 1a — custom batched  (`case B factor_per_sys_ms solve_per_sys_ms batch_relres`)
```
case118 2 0.0229425 0.0267595 2.64789e-07
case118 4 0.011594 0.0133022 2.64848e-07
case118 8 0.00598988 0.0067475 2.66913e-07
case118 16 0.00306256 0.00343013 2.65081e-07
case118 32 0.00165244 0.00178144 2.67811e-07
case118 64 0.000934234 0.000988391 2.74517e-07
case1354pegase 2 0.0553225 0.054641 2.7125e-06
case1354pegase 4 0.029397 0.0286255 2.53962e-06
case1354pegase 8 0.0169715 0.0155087 3.02701e-06
case1354pegase 16 0.00976875 0.00861475 2.75578e-06
case1354pegase 32 0.00655778 0.00528638 2.78836e-06
case1354pegase 64 0.00431095 0.00309027 2.81615e-06
case3120sp 2 0.102224 0.0898015 0.000122913
case3120sp 4 0.0523795 0.047746 0.000132096
case3120sp 8 0.0332004 0.0281409 0.000126531
case3120sp 16 0.0191618 0.015783 0.000140583
case3120sp 32 0.0117436 0.00869019 0.000114471
case3120sp 64 0.00959702 0.00562747 0.000127289
case6468rte 2 0.143476 0.129615 5.52365e-05
case6468rte 4 0.0852555 0.0721312 6.14274e-05
case6468rte 8 0.0564284 0.0452926 6.85587e-05
case6468rte 16 0.0283751 0.0214129 5.57454e-05
case6468rte 32 0.0223374 0.0134647 6.98862e-05
case6468rte 64 0.0185049 0.00886989 5.82073e-05
case8387pegase 2 0.204043 0.166634 4.07013e-05
case8387pegase 4 0.112203 0.103016 1.89486e-05
case8387pegase 8 0.063246 0.0500053 3.23496e-05
case8387pegase 16 0.0431456 0.0300833 2.60697e-05
case8387pegase 32 0.0303122 0.0169414 5.39649e-05
case8387pegase 64 0.0242544 0.0108354 4.036e-05
case13659pegase 2 0.338473 0.25319 0.000119394
case13659pegase 4 0.15641 0.119587 0.000127871
case13659pegase 8 0.108349 0.0699473 0.000129508
case13659pegase 16 0.0632436 0.0356724 0.000128562
case13659pegase 32 0.0485879 0.0220553 0.000133911
case13659pegase 64 0.0427789 0.0163844 0.000130372
case_ACTIVSg25k 2 0.51756 0.342029 0.000322657
case_ACTIVSg25k 4 0.30743 0.18189 0.000165238
case_ACTIVSg25k 8 0.211829 0.106131 0.000185992
case_ACTIVSg25k 16 0.155194 0.063747 0.000167626
case_ACTIVSg25k 32 0.120056 0.0406328 0.000201283
case_ACTIVSg25k 64 0.113912 0.0311722 0.000202577
case_ACTIVSg70k 2 1.03744 0.55559 0.0034035
case_ACTIVSg70k 4 0.682238 0.294323 0.00217633
case_ACTIVSg70k 8 0.512703 0.173121 0.00205514
case_ACTIVSg70k 16 0.438073 0.115182 0.00212791
case_ACTIVSg70k 32 0.415884 0.0915004 0.00169166
case_ACTIVSg70k 64 0.400819 0.0770756 0.00331421
case_SyntheticUSA 2 1.0987 0.548809 0.00171963
case_SyntheticUSA 4 0.735209 0.300047 0.00111905
case_SyntheticUSA 8 0.55698 0.183153 0.00208145
case_SyntheticUSA 16 0.474384 0.128004 0.0028012
case_SyntheticUSA 32 0.451321 0.0991186 0.00147951
case_SyntheticUSA 64 0.436053 0.0877657 0.0015696
DONE
```
### Part 1b — cuDSS batched  (same columns)
```
case118 2 0.033878 0.028934 2.87553e-07
case118 4 0.017971 0.0146775 2.62516e-07
case118 8 0.0109079 0.00767687 2.99259e-07
case118 16 0.00736938 0.00486781 3.06422e-07
case118 32 0.00554972 0.00328016 3.14482e-07
case118 64 0.00462972 0.00249934 3.18057e-07
case1354pegase 2 0.099465 0.061835 2.80424e-06
case1354pegase 4 0.0658602 0.042096 2.87272e-06
case1354pegase 8 0.0496586 0.0324554 2.96954e-06
case1354pegase 16 0.0413733 0.0270716 3.06494e-06
case1354pegase 32 0.0385568 0.0246459 3.17052e-06
case1354pegase 64 0.037252 0.0249084 3.11326e-06
case3120sp 2 0.147464 0.079453 9.9065e-05
case3120sp 4 0.107427 0.0539905 0.00011144
case3120sp 8 0.0879588 0.0412819 0.000113405
case3120sp 16 0.0804379 0.0366263 0.000111619
case3120sp 32 0.0758331 0.0339379 0.000114415
case3120sp 64 0.072932 0.0327021 0.000114936
case6468rte 2 0.216363 0.10493 5.63012e-05
case6468rte 4 0.164409 0.0754932 6.37534e-05
case6468rte 8 0.141178 0.0641483 5.76239e-05
case6468rte 16 0.127683 0.0567632 6.35496e-05
case6468rte 32 0.119933 0.0531818 6.11356e-05
case6468rte 64 0.116407 0.0513916 6.36375e-05
case8387pegase 2 0.342934 0.134876 1.77746e-05
case8387pegase 4 0.247073 0.0975215 1.88148e-05
case8387pegase 8 0.201314 0.0802645 1.81133e-05
case8387pegase 16 0.176725 0.0698219 1.91691e-05
case8387pegase 32 0.16448 0.064806 1.91055e-05
case8387pegase 64 0.157849 0.0623606 1.93124e-05
case13659pegase 2 0.434659 0.170778 0.000113348
case13659pegase 4 0.328314 0.136144 0.000114725
case13659pegase 8 0.276892 0.112892 0.000115685
case13659pegase 16 0.246974 0.102214 0.000120018
case13659pegase 32 0.232906 0.0960611 0.000118257
case13659pegase 64 0.225635 0.0933153 0.000119675
case_ACTIVSg25k 2 0.739838 0.269011 0.000131514
case_ACTIVSg25k 4 0.617044 0.223842 0.000152359
case_ACTIVSg25k 8 0.554216 0.200244 0.00013095
case_ACTIVSg25k 16 0.523907 0.186134 0.000146334
case_ACTIVSg25k 32 0.507885 0.18073 0.00014868
case_ACTIVSg25k 64 0.4999 0.176207 0.000150227
case_ACTIVSg70k 2 1.90417 0.642206 0.00123867
case_ACTIVSg70k 4 1.68202 0.578485 0.000969666
case_ACTIVSg70k 8 1.57327 0.543131 0.00122253
case_ACTIVSg70k 16 1.5179 0.524524 0.0013424
case_ACTIVSg70k 32 1.48919 0.515515 0.00134333
case_ACTIVSg70k 64 1.47548 0.509332 0.00134979
case_SyntheticUSA 2 2.20037 0.73546 0.00086114
case_SyntheticUSA 4 1.95115 0.662436 0.000891869
case_SyntheticUSA 8 1.81298 0.621763 0.000885372
case_SyntheticUSA 16 1.75088 0.601391 0.000884023
case_SyntheticUSA 32 1.71676 0.591612 0.000957153
case_SyntheticUSA 64 1.7081 0.585847 0.000972383
DONE
```
### Part 2 — cuPF mixed-mode  (`case backend B solve_total_us factorize_us solve_us`)
```
case118 cudss-fp32 1 636 200 178
case118 cudss-fp32 4 704.333 222.333 188.667
case118 cudss-fp32 16 889 356.667 241
case118 cudss-fp32 64 1707 889.333 485.667
case118 cudss-fp32 256 5105 3089.67 1508.67
case118 custom-fp32 1 531 135 146.667
case118 custom-fp32 4 553.333 136 157.667
case118 custom-fp32 16 576.667 144.333 164.333
case118 custom-fp32 64 686.333 183 189.667
case118 custom-fp32 256 1248.33 293.333 245
case1354pegase cudss-fp32 1 1530.33 689.667 479.667
case1354pegase cudss-fp32 4 2126.67 1055 675
case1354pegase cudss-fp32 16 4970 2675 1749.33
case1354pegase cudss-fp32 64 17196.3 9522.33 6380.33
case1354pegase cudss-fp32 256 65289 36524.7 24732
case1354pegase custom-fp32 1 1307.33 524.333 442
case1354pegase custom-fp32 4 1296 467.667 454.333
case1354pegase custom-fp32 16 1702.67 617.667 560.333
case1354pegase custom-fp32 64 3165.67 1094.33 799.333
case1354pegase custom-fp32 256 8094.67 2813.33 1445.67
case3120sp cudss-fp32 1 2345.33 1164.67 731.333
case3120sp cudss-fp32 4 3731.67 2143.33 1087.67
case3120sp cudss-fp32 16 10467.3 6430 2934.33
case3120sp cudss-fp32 64 38738 24900 11176.7
case3120sp cudss-fp32 256 167538 109000 48648.3
case3120sp custom-fp32 1 2246.33 1082.33 750.333
case3120sp custom-fp32 4 2582.33 1089.33 1014.67
case3120sp custom-fp32 16 4001.67 1668 1440.67
case3120sp custom-fp32 64 8035 3398 2075.67
case3120sp custom-fp32 256 24422.7 10363 4530.33
case6468rte cudss-fp32 1 1386.33 666.667 366.333
case6468rte cudss-fp32 4 2390.33 1322 616
case6468rte cudss-fp32 16 7080.67 4077.67 1818.33
case6468rte cudss-fp32 64 36206 22303 9869.33
case6468rte cudss-fp32 256 185593 116358 51190.3
case6468rte custom-fp32 1 1325 606.333 372.667
case6468rte custom-fp32 4 1936 698.667 570
case6468rte custom-fp32 16 2889.67 1022.67 710.667
case6468rte custom-fp32 64 9672 3634.67 1745
case6468rte custom-fp32 256 40253.7 16359.7 6001.67
case8387pegase cudss-fp32 1 2835.33 1629 707.333
case8387pegase cudss-fp32 4 5040.67 2978.33 1189.67
case8387pegase cudss-fp32 16 13636.7 8482 3363.67
case8387pegase cudss-fp32 64 48066 30311.3 11966.3
case8387pegase cudss-fp32 256 252880 156728 61635
case8387pegase custom-fp32 1 2376.33 1234.33 664.333
case8387pegase custom-fp32 4 3538 1600.33 1294.33
case8387pegase custom-fp32 16 5571 2220 1470
case8387pegase custom-fp32 64 13041.3 4909.67 2312
case8387pegase custom-fp32 256 62910.7 21920.3 7845.67
case13659pegase cudss-fp32 1 5461.33 3340 1380.67
case13659pegase cudss-fp32 4 10630.7 6583.67 2725.67
case13659pegase cudss-fp32 16 30831.7 19755.7 8144
case13659pegase cudss-fp32 64 645762 433731 179072
case13659pegase cudss-fp32 256 2.55554e+06 1.70994e+06 706864
case13659pegase custom-fp32 1 4589.33 2531.67 1342
case13659pegase custom-fp32 4 6537.67 2954.67 2321.33
case13659pegase custom-fp32 16 10743.3 4751 2854
case13659pegase custom-fp32 64 139110 76260.7 30240.3
case13659pegase custom-fp32 256 499609 275428 88329.7
case_ACTIVSg25k cudss-fp32 1 6435.33 3988.67 1581.33
case_ACTIVSg25k cudss-fp32 4 14969 9871.33 3591
case_ACTIVSg25k cudss-fp32 16 49423.7 33493 11899.7
case_ACTIVSg25k cudss-fp32 64 189230 127915 45004.3
case_ACTIVSg25k cudss-fp32 256 773080 505745 177194
case_ACTIVSg25k custom-fp32 1 6415.67 3468.67 2112.33
case_ACTIVSg25k custom-fp32 4 9305.33 5029 2792.67
case_ACTIVSg25k custom-fp32 16 17508.7 9312.67 3674.67
case_ACTIVSg25k custom-fp32 64 53032.3 27936.3 7531.33
case_ACTIVSg25k custom-fp32 256 224202 110201 24310.7
case_ACTIVSg70k cudss-fp32 1 21238.3 14147.3 5170.33
case_ACTIVSg70k cudss-fp32 4 58306.3 40408 13884
case_ACTIVSg70k cudss-fp32 16 208658 145574 50324
case_ACTIVSg70k cudss-fp32 64 834442 566366 195658
case_ACTIVSg70k cudss-fp32 256 1.60454e+07 1.14593e+07 3.96352e+06
case_ACTIVSg70k custom-fp32 1 18400 10825.7 5735
case_ACTIVSg70k custom-fp32 4 26599.3 15843.7 6903
case_ACTIVSg70k custom-fp32 16 65002.3 40294.7 10829.3
case_ACTIVSg70k custom-fp32 64 248204 148163 28677
case_ACTIVSg70k custom-fp32 256 4.25014e+06 3.12486e+06 517214
case_SyntheticUSA cudss-fp32 1 24597.7 16257 5741.67
case_SyntheticUSA cudss-fp32 4 67348.7 46689.3 15892.7
case_SyntheticUSA cudss-fp32 16 242165 168006 57670.3
case_SyntheticUSA cudss-fp32 64 1.11624e+06 765083 262361
case_SyntheticUSA cudss-fp32 256 1.85678e+07 1.32718e+07 4.56052e+06
case_SyntheticUSA custom-fp32 1 19678 11552 6295.33
case_SyntheticUSA custom-fp32 4 29498 17524.3 7487.67
case_SyntheticUSA custom-fp32 16 73158 44891 12136.7
case_SyntheticUSA custom-fp32 64 320351 194166 38687.7
case_SyntheticUSA custom-fp32 256 4.7799e+06 3.45739e+06 608512
DONE
```

## Notes
- Part 1 sweeps B=2..64 (the batched front-major path); B=1 single-system is measured separately.
- Part 2: a build bug was fixed mid-run — `cuPF/.../cuda_custom_solver.cpp` referenced `SolverConfig::metis_seed` / `tier_split`, both removed during the refactor; the custom backend did not compile until those two lines were dropped (uncommitted at time of writing).
- No OOM across all 54+54+90 runs.
