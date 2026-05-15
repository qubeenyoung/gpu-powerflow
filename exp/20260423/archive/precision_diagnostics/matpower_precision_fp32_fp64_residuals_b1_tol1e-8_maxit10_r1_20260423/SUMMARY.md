# cuPF Benchmark `matpower_precision_fp32_fp64_residuals_b1_tol1e-8_maxit10_r1_20260423`

## Setup

- Created UTC: 2026-04-23T03:08:21.310596+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_fp32_edge, cuda_fp64_edge
- Measurement modes: end2end
- Warmup: 0
- Repeats: 1
- Batch size: 1
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: True
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS matching: False
- cuDSS matching algorithm: DEFAULT
- cuDSS pivot epsilon: AUTO
- cuDSS LD_PRELOAD: /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case10ba | cuda_fp32_edge | False | 0.352825 | 0.255207 | 0.097618 | n/a | 10.0 |
| case10ba | cuda_fp64_edge | True | 0.198527 | 0.179158 | 0.019368 | n/a | 5.0 |
| case118 | cuda_fp32_edge | False | 0.197608 | 0.176590 | 0.021018 | n/a | 10.0 |
| case118 | cuda_fp64_edge | True | 0.205319 | 0.186041 | 0.019278 | n/a | 4.0 |
| case118zh | cuda_fp32_edge | False | 0.197825 | 0.175828 | 0.021997 | n/a | 10.0 |
| case118zh | cuda_fp64_edge | True | 0.195551 | 0.175480 | 0.020070 | n/a | 5.0 |
| case1197 | cuda_fp32_edge | False | 0.222103 | 0.186181 | 0.035921 | n/a | 10.0 |
| case1197 | cuda_fp64_edge | True | 0.203745 | 0.178403 | 0.025341 | n/a | 4.0 |
| case12da | cuda_fp32_edge | False | 0.197380 | 0.177184 | 0.020195 | n/a | 10.0 |
| case12da | cuda_fp64_edge | True | 0.209798 | 0.190826 | 0.018971 | n/a | 4.0 |
| case1354pegase | cuda_fp32_edge | False | 0.213318 | 0.176007 | 0.037311 | n/a | 10.0 |
| case1354pegase | cuda_fp64_edge | True | 0.204782 | 0.176261 | 0.028520 | n/a | 5.0 |
| case13659pegase | cuda_fp32_edge | False | 0.370490 | 0.197013 | 0.173477 | n/a | 10.0 |
| case13659pegase | cuda_fp64_edge | True | 0.361220 | 0.243628 | 0.117591 | n/a | 6.0 |
| case136ma | cuda_fp32_edge | False | 0.204869 | 0.174894 | 0.029975 | n/a | 10.0 |
| case136ma | cuda_fp64_edge | True | 0.194889 | 0.174926 | 0.019962 | n/a | 5.0 |
| case14 | cuda_fp32_edge | False | 0.196553 | 0.176521 | 0.020032 | n/a | 10.0 |
| case14 | cuda_fp64_edge | True | 0.197847 | 0.179137 | 0.018709 | n/a | 3.0 |
| case141 | cuda_fp32_edge | False | 0.197705 | 0.175668 | 0.022037 | n/a | 10.0 |
| case141 | cuda_fp64_edge | True | 0.207467 | 0.187867 | 0.019600 | n/a | 4.0 |
| case145 | cuda_fp32_edge | False | 0.195838 | 0.174148 | 0.021689 | n/a | 10.0 |
| case145 | cuda_fp64_edge | True | 0.200449 | 0.180815 | 0.019633 | n/a | 4.0 |
| case15da | cuda_fp32_edge | False | 0.193385 | 0.173333 | 0.020051 | n/a | 10.0 |
| case15da | cuda_fp64_edge | True | 0.193028 | 0.174078 | 0.018950 | n/a | 4.0 |
| case15nbr | cuda_fp32_edge | False | 0.194281 | 0.174226 | 0.020054 | n/a | 10.0 |
| case15nbr | cuda_fp64_edge | True | 0.219373 | 0.200555 | 0.018817 | n/a | 4.0 |
| case16am | cuda_fp32_edge | False | 0.195019 | 0.175078 | 0.019941 | n/a | 10.0 |
| case16am | cuda_fp64_edge | True | 0.196714 | 0.177589 | 0.019124 | n/a | 6.0 |
| case16ci | cuda_fp32_edge | False | 0.194509 | 0.174672 | 0.019836 | n/a | 10.0 |
| case16ci | cuda_fp64_edge | True | 0.193421 | 0.174690 | 0.018730 | n/a | 4.0 |
| case17me | cuda_fp32_edge | False | 0.198733 | 0.178619 | 0.020113 | n/a | 10.0 |
| case17me | cuda_fp64_edge | True | 0.211546 | 0.192522 | 0.019024 | n/a | 5.0 |
| case18 | cuda_fp32_edge | False | 0.194999 | 0.174891 | 0.020107 | n/a | 10.0 |
| case18 | cuda_fp64_edge | True | 0.197694 | 0.178681 | 0.019013 | n/a | 5.0 |
| case1888rte | cuda_fp32_edge | False | 0.223256 | 0.179401 | 0.043855 | n/a | 10.0 |
| case1888rte | cuda_fp64_edge | True | 0.205080 | 0.179018 | 0.026062 | n/a | 3.0 |
| case18nbr | cuda_fp32_edge | False | 0.196959 | 0.177271 | 0.019687 | n/a | 10.0 |
| case18nbr | cuda_fp64_edge | True | 0.193699 | 0.174812 | 0.018887 | n/a | 4.0 |
| case1951rte | cuda_fp32_edge | False | 0.222581 | 0.178076 | 0.044505 | n/a | 10.0 |
| case1951rte | cuda_fp64_edge | True | 0.209674 | 0.180623 | 0.029050 | n/a | 4.0 |
| case22 | cuda_fp32_edge | False | 0.196233 | 0.175989 | 0.020243 | n/a | 10.0 |
| case22 | cuda_fp64_edge | True | 0.198173 | 0.179230 | 0.018942 | n/a | 4.0 |
| case2383wp | cuda_fp32_edge | False | 0.231209 | 0.181651 | 0.049558 | n/a | 10.0 |
| case2383wp | cuda_fp64_edge | True | 0.222543 | 0.181273 | 0.041270 | n/a | 7.0 |
| case24_ieee_rts | cuda_fp32_edge | False | 0.201902 | 0.181659 | 0.020242 | n/a | 10.0 |
| case24_ieee_rts | cuda_fp64_edge | True | 0.194160 | 0.175297 | 0.018863 | n/a | 5.0 |
| case2736sp | cuda_fp32_edge | False | 0.235169 | 0.180440 | 0.054729 | n/a | 10.0 |
| case2736sp | cuda_fp64_edge | True | 0.213856 | 0.176941 | 0.036915 | n/a | 5.0 |
| case2737sop | cuda_fp32_edge | False | 0.232609 | 0.177187 | 0.055421 | n/a | 10.0 |
| case2737sop | cuda_fp64_edge | True | 0.223160 | 0.181761 | 0.041399 | n/a | 6.0 |
| case2746wop | cuda_fp32_edge | False | 0.243165 | 0.188478 | 0.054687 | n/a | 10.0 |
| case2746wop | cuda_fp64_edge | True | 0.213280 | 0.176143 | 0.037137 | n/a | 5.0 |
| case2746wp | cuda_fp32_edge | False | 0.236120 | 0.181796 | 0.054323 | n/a | 10.0 |
| case2746wp | cuda_fp64_edge | True | 0.219246 | 0.182277 | 0.036969 | n/a | 5.0 |
| case2848rte | cuda_fp32_edge | False | 0.240646 | 0.185031 | 0.055614 | n/a | 10.0 |
| case2848rte | cuda_fp64_edge | True | 0.210640 | 0.180843 | 0.029796 | n/a | 3.0 |
| case2868rte | cuda_fp32_edge | False | 0.236264 | 0.180396 | 0.055868 | n/a | 10.0 |
| case2868rte | cuda_fp64_edge | True | 0.227161 | 0.177072 | 0.050088 | n/a | 6.0 |
| case2869pegase | cuda_fp32_edge | False | 0.238031 | 0.183263 | 0.054768 | n/a | 10.0 |
| case2869pegase | cuda_fp64_edge | True | 0.225049 | 0.177262 | 0.047786 | n/a | 7.0 |
| case28da | cuda_fp32_edge | False | 0.197152 | 0.176775 | 0.020377 | n/a | 10.0 |
| case28da | cuda_fp64_edge | True | 0.196784 | 0.177593 | 0.019190 | n/a | 4.0 |
| case30 | cuda_fp32_edge | False | 0.195528 | 0.175075 | 0.020453 | n/a | 10.0 |
| case30 | cuda_fp64_edge | True | 0.194425 | 0.175399 | 0.019026 | n/a | 4.0 |
| case300 | cuda_fp32_edge | False | 0.200404 | 0.176286 | 0.024117 | n/a | 10.0 |
| case300 | cuda_fp64_edge | True | 0.200643 | 0.178965 | 0.021677 | n/a | 6.0 |
| case3012wp | cuda_fp32_edge | False | 0.236726 | 0.178523 | 0.058203 | n/a | 10.0 |
| case3012wp | cuda_fp64_edge | True | 0.218596 | 0.184099 | 0.034496 | n/a | 4.0 |
| case30Q | cuda_fp32_edge | False | 0.195061 | 0.174789 | 0.020272 | n/a | 10.0 |
| case30Q | cuda_fp64_edge | True | 0.209588 | 0.190739 | 0.018849 | n/a | 4.0 |
| case30pwl | cuda_fp32_edge | False | 0.206214 | 0.185738 | 0.020476 | n/a | 10.0 |
| case30pwl | cuda_fp64_edge | True | 0.196693 | 0.178049 | 0.018644 | n/a | 4.0 |
| case3120sp | cuda_fp32_edge | False | 0.237521 | 0.177495 | 0.060026 | n/a | 10.0 |
| case3120sp | cuda_fp64_edge | True | 0.229546 | 0.181756 | 0.047789 | n/a | 7.0 |
| case3375wp | cuda_fp32_edge | False | 0.245463 | 0.183011 | 0.062452 | n/a | 10.0 |
| case3375wp | cuda_fp64_edge | True | 0.214492 | 0.181246 | 0.033245 | n/a | 3.0 |
| case33bw | cuda_fp32_edge | False | 0.196061 | 0.175599 | 0.020461 | n/a | 10.0 |
| case33bw | cuda_fp64_edge | True | 0.197332 | 0.178368 | 0.018964 | n/a | 4.0 |
| case33mg | cuda_fp32_edge | False | 0.195196 | 0.174822 | 0.020374 | n/a | 10.0 |
| case33mg | cuda_fp64_edge | True | 0.197202 | 0.177989 | 0.019213 | n/a | 5.0 |
| case34sa | cuda_fp32_edge | False | 0.196172 | 0.175600 | 0.020571 | n/a | 10.0 |
| case34sa | cuda_fp64_edge | True | 0.196236 | 0.177146 | 0.019090 | n/a | 4.0 |
| case38si | cuda_fp32_edge | False | 0.198789 | 0.178340 | 0.020449 | n/a | 10.0 |
| case38si | cuda_fp64_edge | True | 0.193742 | 0.174536 | 0.019206 | n/a | 5.0 |
| case39 | cuda_fp32_edge | False | 0.195327 | 0.174767 | 0.020559 | n/a | 10.0 |
| case39 | cuda_fp64_edge | True | 0.201800 | 0.183372 | 0.018428 | n/a | 2.0 |
| case4_dist | cuda_fp32_edge | False | 0.201433 | 0.181942 | 0.019491 | n/a | 10.0 |
| case4_dist | cuda_fp64_edge | True | 0.201835 | 0.183476 | 0.018358 | n/a | 4.0 |
| case4gs | cuda_fp32_edge | False | 0.195323 | 0.175565 | 0.019758 | n/a | 10.0 |
| case4gs | cuda_fp64_edge | True | 0.197638 | 0.178743 | 0.018895 | n/a | 4.0 |
| case5 | cuda_fp32_edge | False | 0.199505 | 0.179730 | 0.019774 | n/a | 10.0 |
| case5 | cuda_fp64_edge | True | 0.198818 | 0.180037 | 0.018780 | n/a | 4.0 |
| case51ga | cuda_fp32_edge | False | 0.207258 | 0.176578 | 0.030680 | n/a | 10.0 |
| case51ga | cuda_fp64_edge | True | 0.195559 | 0.176122 | 0.019436 | n/a | 5.0 |
| case51he | cuda_fp32_edge | False | 0.206379 | 0.185635 | 0.020743 | n/a | 10.0 |
| case51he | cuda_fp64_edge | True | 0.198271 | 0.179197 | 0.019074 | n/a | 4.0 |
| case533mt_hi | cuda_fp32_edge | False | 0.207585 | 0.180510 | 0.027075 | n/a | 10.0 |
| case533mt_hi | cuda_fp64_edge | True | 0.198351 | 0.176706 | 0.021644 | n/a | 4.0 |
| case533mt_lo | cuda_fp32_edge | False | 0.232924 | 0.205822 | 0.027102 | n/a | 10.0 |
| case533mt_lo | cuda_fp64_edge | True | 0.202349 | 0.180645 | 0.021703 | n/a | 4.0 |
| case57 | cuda_fp32_edge | False | 0.199312 | 0.178316 | 0.020996 | n/a | 10.0 |
| case57 | cuda_fp64_edge | True | 0.196027 | 0.176802 | 0.019225 | n/a | 4.0 |
| case59 | cuda_fp32_edge | False | 0.201397 | 0.180624 | 0.020773 | n/a | 10.0 |
| case59 | cuda_fp64_edge | True | 0.195895 | 0.176094 | 0.019800 | n/a | 6.0 |
| case60nordic | cuda_fp32_edge | False | 0.196791 | 0.175912 | 0.020879 | n/a | 10.0 |
| case60nordic | cuda_fp64_edge | True | 0.193928 | 0.175389 | 0.018539 | n/a | 2.0 |
| case6468rte | cuda_fp32_edge | False | 0.298474 | 0.187588 | 0.110885 | n/a | 10.0 |
| case6468rte | cuda_fp64_edge | True | 0.242840 | 0.189419 | 0.053420 | n/a | 4.0 |
| case6470rte | cuda_fp32_edge | False | 0.289191 | 0.187227 | 0.101964 | n/a | 10.0 |
| case6470rte | cuda_fp64_edge | True | 0.247228 | 0.193836 | 0.053392 | n/a | 4.0 |
| case6495rte | cuda_fp32_edge | False | 0.325670 | 0.222911 | 0.102759 | n/a | 10.0 |
| case6495rte | cuda_fp64_edge | True | 0.238638 | 0.194273 | 0.044365 | n/a | 3.0 |
| case6515rte | cuda_fp32_edge | False | 0.291873 | 0.189239 | 0.102634 | n/a | 10.0 |
| case6515rte | cuda_fp64_edge | True | 0.239757 | 0.186288 | 0.053469 | n/a | 4.0 |
| case69 | cuda_fp32_edge | False | 0.196813 | 0.175814 | 0.020999 | n/a | 10.0 |
| case69 | cuda_fp64_edge | True | 0.203301 | 0.183656 | 0.019645 | n/a | 5.0 |
| case6ww | cuda_fp32_edge | False | 0.203946 | 0.184115 | 0.019830 | n/a | 10.0 |
| case6ww | cuda_fp64_edge | True | 0.194226 | 0.175455 | 0.018770 | n/a | 4.0 |
| case70da | cuda_fp32_edge | False | 0.195834 | 0.175339 | 0.020494 | n/a | 10.0 |
| case70da | cuda_fp64_edge | True | 0.194735 | 0.175658 | 0.019076 | n/a | 5.0 |
| case74ds | cuda_fp32_edge | False | 0.197321 | 0.176265 | 0.021055 | n/a | 10.0 |
| case74ds | cuda_fp64_edge | True | 0.198344 | 0.179149 | 0.019194 | n/a | 4.0 |
| case8387pegase | cuda_fp32_edge | False | 0.313289 | 0.193970 | 0.119319 | n/a | 10.0 |
| case8387pegase | cuda_fp64_edge | True | 0.253262 | 0.192846 | 0.060415 | n/a | 4.0 |
| case85 | cuda_fp32_edge | False | 0.196636 | 0.175591 | 0.021045 | n/a | 10.0 |
| case85 | cuda_fp64_edge | True | 0.216577 | 0.188603 | 0.027973 | n/a | 5.0 |
| case89pegase | cuda_fp32_edge | False | 0.197318 | 0.175463 | 0.021855 | n/a | 10.0 |
| case89pegase | cuda_fp64_edge | True | 0.221559 | 0.200676 | 0.020882 | n/a | 6.0 |
| case9 | cuda_fp32_edge | False | 0.198432 | 0.178396 | 0.020036 | n/a | 10.0 |
| case9 | cuda_fp64_edge | True | 0.210016 | 0.190947 | 0.019068 | n/a | 5.0 |
| case9241pegase | cuda_fp32_edge | False | 0.322950 | 0.190298 | 0.132652 | n/a | 10.0 |
| case9241pegase | cuda_fp64_edge | True | 0.298185 | 0.194633 | 0.103551 | n/a | 7.0 |
| case94pi | cuda_fp32_edge | False | 0.197749 | 0.176419 | 0.021329 | n/a | 10.0 |
| case94pi | cuda_fp64_edge | True | 0.194852 | 0.175163 | 0.019688 | n/a | 5.0 |
| case9Q | cuda_fp32_edge | False | 0.201312 | 0.181333 | 0.019979 | n/a | 10.0 |
| case9Q | cuda_fp64_edge | True | 0.203965 | 0.185016 | 0.018949 | n/a | 5.0 |
| case9target | cuda_fp32_edge | False | 0.194619 | 0.174991 | 0.019627 | n/a | 10.0 |
| case9target | cuda_fp64_edge | True | 0.198589 | 0.179269 | 0.019320 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_fp32_edge | False | 0.340596 | 0.193426 | 0.147170 | n/a | 10.0 |
| case_ACTIVSg10k | cuda_fp64_edge | True | 0.274118 | 0.192917 | 0.081201 | n/a | 5.0 |
| case_ACTIVSg200 | cuda_fp32_edge | False | 0.259904 | 0.236904 | 0.022999 | n/a | 10.0 |
| case_ACTIVSg200 | cuda_fp64_edge | True | 0.195478 | 0.176022 | 0.019455 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_fp32_edge | False | 0.232037 | 0.185748 | 0.046289 | n/a | 10.0 |
| case_ACTIVSg2000 | cuda_fp64_edge | True | 0.207021 | 0.176531 | 0.030490 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_fp32_edge | False | 0.541524 | 0.221762 | 0.319761 | n/a | 10.0 |
| case_ACTIVSg25k | cuda_fp64_edge | True | 0.400062 | 0.222925 | 0.177136 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_fp32_edge | False | 0.205038 | 0.178420 | 0.026618 | n/a | 10.0 |
| case_ACTIVSg500 | cuda_fp64_edge | True | 0.198057 | 0.176467 | 0.021589 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_fp32_edge | False | 1.171944 | 0.295511 | 0.876432 | n/a | 10.0 |
| case_ACTIVSg70k | cuda_fp64_edge | True | 0.973046 | 0.321936 | 0.651109 | n/a | 7.0 |
| case_RTS_GMLC | cuda_fp32_edge | False | 0.208817 | 0.187968 | 0.020849 | n/a | 10.0 |
| case_RTS_GMLC | cuda_fp64_edge | True | 0.200152 | 0.180708 | 0.019444 | n/a | 5.0 |
| case_SyntheticUSA | cuda_fp32_edge | False | 1.348911 | 0.322565 | 1.026345 | n/a | 10.0 |
| case_SyntheticUSA | cuda_fp64_edge | True | 1.076017 | 0.324552 | 0.751464 | n/a | 7.0 |
| case_ieee30 | cuda_fp32_edge | False | 0.196754 | 0.176329 | 0.020425 | n/a | 10.0 |
| case_ieee30 | cuda_fp64_edge | True | 0.196119 | 0.177357 | 0.018761 | n/a | 3.0 |

## Files

- `manifest.json`: run configuration and environment
- `summary.csv`: one row per measured run across all measurement modes
- `aggregates.csv`: grouped statistics by mode/case/profile
- `summary_<mode>.csv` and `aggregates_<mode>.csv`: mode-specific views
- `raw/<mode>/`: per-run timing payload

## Nsight Hints

Use the operators benchmark binary directly for profiling. Prefer `--warmup 1` to remove one-time CUDA setup from measured repeats.

```bash
nsys profile --trace=cuda,nvtx -o cupf_nsys \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto-dump/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_fp32_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto-dump/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_fp32_edge --warmup 1 --repeats 1
```
