# cuPF Benchmark `matpower_precision_fp32_fp64_end2end_b1_tol1e-8_maxit10_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T03:06:50.452678+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_fp32_edge, cuda_fp64_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
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
| case10ba | cuda_fp32_edge | False | 0.010646 | 0.009607 | 0.001038 | n/a | 10.0 |
| case10ba | cuda_fp64_edge | True | 0.010111 | 0.009585 | 0.000525 | n/a | 5.0 |
| case118 | cuda_fp32_edge | False | 0.011482 | 0.010017 | 0.001464 | n/a | 10.0 |
| case118 | cuda_fp64_edge | True | 0.010626 | 0.010021 | 0.000605 | n/a | 4.0 |
| case118zh | cuda_fp32_edge | False | 0.011293 | 0.009915 | 0.001377 | n/a | 10.0 |
| case118zh | cuda_fp64_edge | True | 0.010674 | 0.009941 | 0.000733 | n/a | 5.0 |
| case1197 | cuda_fp32_edge | False | 0.014897 | 0.012616 | 0.002281 | n/a | 10.0 |
| case1197 | cuda_fp64_edge | True | 0.013556 | 0.012572 | 0.000983 | n/a | 4.0 |
| case12da | cuda_fp32_edge | False | 0.010693 | 0.009597 | 0.001095 | n/a | 10.0 |
| case12da | cuda_fp64_edge | True | 0.010013 | 0.009575 | 0.000438 | n/a | 4.0 |
| case1354pegase | cuda_fp32_edge | False | 0.016672 | 0.013384 | 0.003289 | n/a | 10.0 |
| case1354pegase | cuda_fp64_edge | True | 0.016523 | 0.014625 | 0.001898 | n/a | 5.0 |
| case13659pegase | cuda_fp32_edge | False | 0.048627 | 0.038013 | 0.010613 | n/a | 10.0 |
| case13659pegase | cuda_fp64_edge | True | 0.044437 | 0.035381 | 0.009056 | n/a | 6.0 |
| case136ma | cuda_fp32_edge | False | 0.011210 | 0.009920 | 0.001290 | n/a | 10.0 |
| case136ma | cuda_fp64_edge | True | 0.010606 | 0.009925 | 0.000680 | n/a | 5.0 |
| case14 | cuda_fp32_edge | False | 0.010655 | 0.009614 | 0.001041 | n/a | 10.0 |
| case14 | cuda_fp64_edge | True | 0.009912 | 0.009612 | 0.000300 | n/a | 3.0 |
| case141 | cuda_fp32_edge | False | 0.011559 | 0.010023 | 0.001536 | n/a | 10.0 |
| case141 | cuda_fp64_edge | True | 0.010653 | 0.010033 | 0.000620 | n/a | 4.0 |
| case145 | cuda_fp32_edge | False | 0.011964 | 0.010236 | 0.001728 | n/a | 10.0 |
| case145 | cuda_fp64_edge | True | 0.010994 | 0.010242 | 0.000751 | n/a | 4.0 |
| case15da | cuda_fp32_edge | False | 0.010704 | 0.009594 | 0.001110 | n/a | 10.0 |
| case15da | cuda_fp64_edge | True | 0.016049 | 0.015601 | 0.000447 | n/a | 4.0 |
| case15nbr | cuda_fp32_edge | False | 0.010722 | 0.009610 | 0.001112 | n/a | 10.0 |
| case15nbr | cuda_fp64_edge | True | 0.010034 | 0.009597 | 0.000437 | n/a | 4.0 |
| case16am | cuda_fp32_edge | False | 0.010603 | 0.009579 | 0.001023 | n/a | 10.0 |
| case16am | cuda_fp64_edge | True | 0.010231 | 0.009593 | 0.000638 | n/a | 6.0 |
| case16ci | cuda_fp32_edge | False | 0.010527 | 0.009568 | 0.000959 | n/a | 10.0 |
| case16ci | cuda_fp64_edge | True | 0.013231 | 0.012842 | 0.000388 | n/a | 4.0 |
| case17me | cuda_fp32_edge | False | 0.010712 | 0.009602 | 0.001110 | n/a | 10.0 |
| case17me | cuda_fp64_edge | True | 0.010174 | 0.009609 | 0.000564 | n/a | 5.0 |
| case18 | cuda_fp32_edge | False | 0.010791 | 0.009651 | 0.001140 | n/a | 10.0 |
| case18 | cuda_fp64_edge | True | 0.010209 | 0.009620 | 0.000589 | n/a | 5.0 |
| case1888rte | cuda_fp32_edge | False | 0.017346 | 0.013881 | 0.003464 | n/a | 10.0 |
| case1888rte | cuda_fp64_edge | True | 0.015069 | 0.013919 | 0.001149 | n/a | 3.0 |
| case18nbr | cuda_fp32_edge | False | 0.010816 | 0.009629 | 0.001187 | n/a | 10.0 |
| case18nbr | cuda_fp64_edge | True | 0.010106 | 0.009642 | 0.000463 | n/a | 4.0 |
| case1951rte | cuda_fp32_edge | False | 0.017980 | 0.014101 | 0.003879 | n/a | 10.0 |
| case1951rte | cuda_fp64_edge | True | 0.015973 | 0.014165 | 0.001808 | n/a | 4.0 |
| case22 | cuda_fp32_edge | False | 0.010767 | 0.009631 | 0.001136 | n/a | 10.0 |
| case22 | cuda_fp64_edge | True | 0.012027 | 0.011565 | 0.000461 | n/a | 4.0 |
| case2383wp | cuda_fp32_edge | False | 0.018771 | 0.014811 | 0.003960 | n/a | 10.0 |
| case2383wp | cuda_fp64_edge | True | 0.018262 | 0.014799 | 0.003463 | n/a | 7.0 |
| case24_ieee_rts | cuda_fp32_edge | False | 0.010807 | 0.009683 | 0.001123 | n/a | 10.0 |
| case24_ieee_rts | cuda_fp64_edge | True | 0.010282 | 0.009694 | 0.000587 | n/a | 5.0 |
| case2736sp | cuda_fp32_edge | False | 0.020175 | 0.015665 | 0.004510 | n/a | 10.0 |
| case2736sp | cuda_fp64_edge | True | 0.018398 | 0.015657 | 0.002741 | n/a | 5.0 |
| case2737sop | cuda_fp32_edge | False | 0.019940 | 0.015631 | 0.004309 | n/a | 10.0 |
| case2737sop | cuda_fp64_edge | True | 0.018894 | 0.015668 | 0.003225 | n/a | 6.0 |
| case2746wop | cuda_fp32_edge | False | 0.020336 | 0.015719 | 0.004617 | n/a | 10.0 |
| case2746wop | cuda_fp64_edge | True | 0.018633 | 0.015755 | 0.002877 | n/a | 5.0 |
| case2746wp | cuda_fp32_edge | False | 0.019581 | 0.015273 | 0.004308 | n/a | 10.0 |
| case2746wp | cuda_fp64_edge | True | 0.018034 | 0.015409 | 0.002624 | n/a | 5.0 |
| case2848rte | cuda_fp32_edge | False | 0.025915 | 0.021845 | 0.004069 | n/a | 10.0 |
| case2848rte | cuda_fp64_edge | True | 0.017075 | 0.015682 | 0.001393 | n/a | 3.0 |
| case2868rte | cuda_fp32_edge | False | 0.020104 | 0.015779 | 0.004325 | n/a | 10.0 |
| case2868rte | cuda_fp64_edge | True | 0.019034 | 0.015773 | 0.003261 | n/a | 6.0 |
| case2869pegase | cuda_fp32_edge | False | 0.021340 | 0.017338 | 0.004001 | n/a | 10.0 |
| case2869pegase | cuda_fp64_edge | True | 0.019190 | 0.015771 | 0.003419 | n/a | 7.0 |
| case28da | cuda_fp32_edge | False | 0.010867 | 0.009661 | 0.001206 | n/a | 10.0 |
| case28da | cuda_fp64_edge | True | 0.015625 | 0.015125 | 0.000499 | n/a | 4.0 |
| case30 | cuda_fp32_edge | False | 0.010958 | 0.009732 | 0.001226 | n/a | 10.0 |
| case30 | cuda_fp64_edge | True | 0.010212 | 0.009709 | 0.000503 | n/a | 4.0 |
| case300 | cuda_fp32_edge | False | 0.012752 | 0.010634 | 0.002117 | n/a | 10.0 |
| case300 | cuda_fp64_edge | True | 0.012088 | 0.010619 | 0.001468 | n/a | 6.0 |
| case3012wp | cuda_fp32_edge | False | 0.025913 | 0.021423 | 0.004490 | n/a | 10.0 |
| case3012wp | cuda_fp64_edge | True | 0.017965 | 0.015834 | 0.002130 | n/a | 4.0 |
| case30Q | cuda_fp32_edge | False | 0.010970 | 0.009736 | 0.001234 | n/a | 10.0 |
| case30Q | cuda_fp64_edge | True | 0.010216 | 0.009713 | 0.000502 | n/a | 4.0 |
| case30pwl | cuda_fp32_edge | False | 0.010939 | 0.009723 | 0.001216 | n/a | 10.0 |
| case30pwl | cuda_fp64_edge | True | 0.010210 | 0.009709 | 0.000500 | n/a | 4.0 |
| case3120sp | cuda_fp32_edge | False | 0.026374 | 0.021696 | 0.004677 | n/a | 10.0 |
| case3120sp | cuda_fp64_edge | True | 0.026135 | 0.021932 | 0.004202 | n/a | 7.0 |
| case3375wp | cuda_fp32_edge | False | 0.022036 | 0.017234 | 0.004802 | n/a | 10.0 |
| case3375wp | cuda_fp64_edge | True | 0.018325 | 0.016642 | 0.001684 | n/a | 3.0 |
| case33bw | cuda_fp32_edge | False | 0.013011 | 0.011823 | 0.001187 | n/a | 10.0 |
| case33bw | cuda_fp64_edge | True | 0.010138 | 0.009671 | 0.000466 | n/a | 4.0 |
| case33mg | cuda_fp32_edge | False | 0.010871 | 0.009681 | 0.001189 | n/a | 10.0 |
| case33mg | cuda_fp64_edge | True | 0.010270 | 0.009673 | 0.000596 | n/a | 5.0 |
| case34sa | cuda_fp32_edge | False | 0.010896 | 0.009691 | 0.001205 | n/a | 10.0 |
| case34sa | cuda_fp64_edge | True | 0.024789 | 0.024285 | 0.000503 | n/a | 4.0 |
| case38si | cuda_fp32_edge | False | 0.010938 | 0.009698 | 0.001239 | n/a | 10.0 |
| case38si | cuda_fp64_edge | True | 0.010343 | 0.009713 | 0.000630 | n/a | 5.0 |
| case39 | cuda_fp32_edge | False | 0.011004 | 0.009724 | 0.001279 | n/a | 10.0 |
| case39 | cuda_fp64_edge | True | 0.009948 | 0.009728 | 0.000220 | n/a | 2.0 |
| case4_dist | cuda_fp32_edge | False | 0.010190 | 0.009517 | 0.000673 | n/a | 10.0 |
| case4_dist | cuda_fp64_edge | True | 0.009800 | 0.009515 | 0.000285 | n/a | 4.0 |
| case4gs | cuda_fp32_edge | False | 0.010408 | 0.009534 | 0.000874 | n/a | 10.0 |
| case4gs | cuda_fp64_edge | True | 0.009891 | 0.009551 | 0.000340 | n/a | 4.0 |
| case5 | cuda_fp32_edge | False | 0.010402 | 0.009541 | 0.000861 | n/a | 10.0 |
| case5 | cuda_fp64_edge | True | 0.009869 | 0.009529 | 0.000339 | n/a | 4.0 |
| case51ga | cuda_fp32_edge | False | 0.011035 | 0.009756 | 0.001278 | n/a | 10.0 |
| case51ga | cuda_fp64_edge | True | 0.013398 | 0.012746 | 0.000651 | n/a | 5.0 |
| case51he | cuda_fp32_edge | False | 0.010993 | 0.009744 | 0.001247 | n/a | 10.0 |
| case51he | cuda_fp64_edge | True | 0.010281 | 0.009765 | 0.000515 | n/a | 4.0 |
| case533mt_hi | cuda_fp32_edge | False | 0.012845 | 0.011117 | 0.001728 | n/a | 10.0 |
| case533mt_hi | cuda_fp64_edge | True | 0.019272 | 0.018517 | 0.000755 | n/a | 4.0 |
| case533mt_lo | cuda_fp32_edge | False | 0.013111 | 0.011388 | 0.001722 | n/a | 10.0 |
| case533mt_lo | cuda_fp64_edge | True | 0.011858 | 0.011111 | 0.000747 | n/a | 4.0 |
| case57 | cuda_fp32_edge | False | 0.011391 | 0.009867 | 0.001523 | n/a | 10.0 |
| case57 | cuda_fp64_edge | True | 0.019404 | 0.018786 | 0.000618 | n/a | 4.0 |
| case59 | cuda_fp32_edge | False | 0.011269 | 0.009817 | 0.001451 | n/a | 10.0 |
| case59 | cuda_fp64_edge | True | 0.010794 | 0.009821 | 0.000972 | n/a | 6.0 |
| case60nordic | cuda_fp32_edge | False | 0.011269 | 0.009835 | 0.001434 | n/a | 10.0 |
| case60nordic | cuda_fp64_edge | True | 0.013924 | 0.013683 | 0.000241 | n/a | 2.0 |
| case6468rte | cuda_fp32_edge | False | 0.029250 | 0.023183 | 0.006066 | n/a | 10.0 |
| case6468rte | cuda_fp64_edge | True | 0.025761 | 0.022644 | 0.003116 | n/a | 4.0 |
| case6470rte | cuda_fp32_edge | False | 0.035516 | 0.029442 | 0.006074 | n/a | 10.0 |
| case6470rte | cuda_fp64_edge | True | 0.025937 | 0.022870 | 0.003067 | n/a | 4.0 |
| case6495rte | cuda_fp32_edge | False | 0.027961 | 0.022037 | 0.005924 | n/a | 10.0 |
| case6495rte | cuda_fp64_edge | True | 0.024127 | 0.021972 | 0.002155 | n/a | 3.0 |
| case6515rte | cuda_fp32_edge | False | 0.030181 | 0.023691 | 0.006490 | n/a | 10.0 |
| case6515rte | cuda_fp64_edge | True | 0.025465 | 0.022152 | 0.003313 | n/a | 4.0 |
| case69 | cuda_fp32_edge | False | 0.011145 | 0.009776 | 0.001368 | n/a | 10.0 |
| case69 | cuda_fp64_edge | True | 0.010526 | 0.009797 | 0.000728 | n/a | 5.0 |
| case6ww | cuda_fp32_edge | False | 0.010483 | 0.009552 | 0.000931 | n/a | 10.0 |
| case6ww | cuda_fp64_edge | True | 0.009917 | 0.009546 | 0.000370 | n/a | 4.0 |
| case70da | cuda_fp32_edge | False | 0.014345 | 0.013145 | 0.001199 | n/a | 10.0 |
| case70da | cuda_fp64_edge | True | 0.010361 | 0.009727 | 0.000633 | n/a | 5.0 |
| case74ds | cuda_fp32_edge | False | 0.011107 | 0.009750 | 0.001356 | n/a | 10.0 |
| case74ds | cuda_fp64_edge | True | 0.010313 | 0.009769 | 0.000543 | n/a | 4.0 |
| case8387pegase | cuda_fp32_edge | False | 0.036405 | 0.027725 | 0.008679 | n/a | 10.0 |
| case8387pegase | cuda_fp64_edge | True | 0.032069 | 0.027720 | 0.004349 | n/a | 4.0 |
| case85 | cuda_fp32_edge | False | 0.011320 | 0.009889 | 0.001431 | n/a | 10.0 |
| case85 | cuda_fp64_edge | True | 0.014424 | 0.013688 | 0.000735 | n/a | 5.0 |
| case89pegase | cuda_fp32_edge | False | 0.012283 | 0.010195 | 0.002088 | n/a | 10.0 |
| case89pegase | cuda_fp64_edge | True | 0.011669 | 0.010200 | 0.001468 | n/a | 6.0 |
| case9 | cuda_fp32_edge | False | 0.010559 | 0.009567 | 0.000991 | n/a | 10.0 |
| case9 | cuda_fp64_edge | True | 0.014775 | 0.014278 | 0.000496 | n/a | 5.0 |
| case9241pegase | cuda_fp32_edge | False | 0.038454 | 0.029489 | 0.008965 | n/a | 10.0 |
| case9241pegase | cuda_fp64_edge | True | 0.038760 | 0.030105 | 0.008655 | n/a | 7.0 |
| case94pi | cuda_fp32_edge | False | 0.011251 | 0.009867 | 0.001384 | n/a | 10.0 |
| case94pi | cuda_fp64_edge | True | 0.010618 | 0.009878 | 0.000740 | n/a | 5.0 |
| case9Q | cuda_fp32_edge | False | 0.010608 | 0.009616 | 0.000992 | n/a | 10.0 |
| case9Q | cuda_fp64_edge | True | 0.010924 | 0.010422 | 0.000502 | n/a | 5.0 |
| case9target | cuda_fp32_edge | False | 0.010571 | 0.009588 | 0.000983 | n/a | 10.0 |
| case9target | cuda_fp64_edge | True | 0.010290 | 0.009584 | 0.000706 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_fp32_edge | False | 0.038155 | 0.030048 | 0.008106 | n/a | 10.0 |
| case_ACTIVSg10k | cuda_fp64_edge | True | 0.039287 | 0.034034 | 0.005252 | n/a | 5.0 |
| case_ACTIVSg200 | cuda_fp32_edge | False | 0.012178 | 0.010322 | 0.001855 | n/a | 10.0 |
| case_ACTIVSg200 | cuda_fp64_edge | True | 0.010857 | 0.010315 | 0.000541 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_fp32_edge | False | 0.020987 | 0.015456 | 0.005530 | n/a | 10.0 |
| case_ACTIVSg2000 | cuda_fp64_edge | True | 0.017530 | 0.014735 | 0.002794 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_fp32_edge | False | 0.073322 | 0.056140 | 0.017182 | n/a | 10.0 |
| case_ACTIVSg25k | cuda_fp64_edge | True | 0.068269 | 0.056408 | 0.011861 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_fp32_edge | False | 0.013359 | 0.011144 | 0.002214 | n/a | 10.0 |
| case_ACTIVSg500 | cuda_fp64_edge | True | 0.012141 | 0.011147 | 0.000993 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_fp32_edge | False | 0.165957 | 0.126856 | 0.039101 | n/a | 10.0 |
| case_ACTIVSg70k | cuda_fp64_edge | True | 0.163391 | 0.122994 | 0.040397 | n/a | 7.0 |
| case_RTS_GMLC | cuda_fp32_edge | False | 0.012190 | 0.010734 | 0.001455 | n/a | 10.0 |
| case_RTS_GMLC | cuda_fp64_edge | True | 0.010672 | 0.009906 | 0.000765 | n/a | 5.0 |
| case_SyntheticUSA | cuda_fp32_edge | False | 0.210664 | 0.167625 | 0.043038 | n/a | 10.0 |
| case_SyntheticUSA | cuda_fp64_edge | True | 0.206985 | 0.163697 | 0.043288 | n/a | 7.0 |
| case_ieee30 | cuda_fp32_edge | False | 0.011000 | 0.009718 | 0.001280 | n/a | 10.0 |
| case_ieee30 | cuda_fp64_edge | True | 0.010955 | 0.010589 | 0.000365 | n/a | 3.0 |

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
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_fp32_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_fp32_edge --warmup 1 --repeats 1
```
