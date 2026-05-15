# cuPF Benchmark `matpower_precision_fp32_mixed_fp64_convergence_b1_tol1e-8_maxit10_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T03:32:49.209133+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_fp32_edge, cuda_edge, cuda_fp64_edge
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
| case10ba | cuda_fp32_edge | False | 0.010644 | 0.009606 | 0.001038 | n/a | 10.0 |
| case10ba | cuda_edge | True | 0.010378 | 0.009889 | 0.000489 | n/a | 5.0 |
| case10ba | cuda_fp64_edge | True | 0.010109 | 0.009583 | 0.000525 | n/a | 5.0 |
| case118 | cuda_fp32_edge | False | 0.015036 | 0.013571 | 0.001465 | n/a | 10.0 |
| case118 | cuda_edge | True | 0.010552 | 0.010022 | 0.000530 | n/a | 4.0 |
| case118 | cuda_fp64_edge | True | 0.010630 | 0.010026 | 0.000603 | n/a | 4.0 |
| case118zh | cuda_fp32_edge | False | 0.011287 | 0.009901 | 0.001385 | n/a | 10.0 |
| case118zh | cuda_edge | True | 0.011572 | 0.010935 | 0.000637 | n/a | 5.0 |
| case118zh | cuda_fp64_edge | True | 0.017354 | 0.016621 | 0.000732 | n/a | 5.0 |
| case1197 | cuda_fp32_edge | False | 0.018785 | 0.016498 | 0.002287 | n/a | 10.0 |
| case1197 | cuda_edge | True | 0.013468 | 0.012608 | 0.000859 | n/a | 4.0 |
| case1197 | cuda_fp64_edge | True | 0.013575 | 0.012591 | 0.000984 | n/a | 4.0 |
| case12da | cuda_fp32_edge | False | 0.010688 | 0.009596 | 0.001092 | n/a | 10.0 |
| case12da | cuda_edge | True | 0.010003 | 0.009597 | 0.000406 | n/a | 4.0 |
| case12da | cuda_fp64_edge | True | 0.010038 | 0.009600 | 0.000438 | n/a | 4.0 |
| case1354pegase | cuda_fp32_edge | False | 0.021662 | 0.018364 | 0.003298 | n/a | 10.0 |
| case1354pegase | cuda_edge | True | 0.014904 | 0.013406 | 0.001497 | n/a | 5.0 |
| case1354pegase | cuda_fp64_edge | True | 0.015381 | 0.013479 | 0.001901 | n/a | 5.0 |
| case13659pegase | cuda_fp32_edge | False | 0.047875 | 0.037253 | 0.010622 | n/a | 10.0 |
| case13659pegase | cuda_edge | True | 0.042273 | 0.035066 | 0.007207 | n/a | 7.0 |
| case13659pegase | cuda_fp64_edge | True | 0.044688 | 0.035624 | 0.009064 | n/a | 6.0 |
| case136ma | cuda_fp32_edge | False | 0.011216 | 0.009928 | 0.001288 | n/a | 10.0 |
| case136ma | cuda_edge | True | 0.010521 | 0.009928 | 0.000593 | n/a | 5.0 |
| case136ma | cuda_fp64_edge | True | 0.011367 | 0.010693 | 0.000674 | n/a | 5.0 |
| case14 | cuda_fp32_edge | False | 0.010666 | 0.009619 | 0.001046 | n/a | 10.0 |
| case14 | cuda_edge | True | 0.014093 | 0.013803 | 0.000289 | n/a | 3.0 |
| case14 | cuda_fp64_edge | True | 0.009932 | 0.009630 | 0.000301 | n/a | 3.0 |
| case141 | cuda_fp32_edge | False | 0.012349 | 0.010808 | 0.001541 | n/a | 10.0 |
| case141 | cuda_edge | True | 0.010868 | 0.010024 | 0.000844 | n/a | 6.0 |
| case141 | cuda_fp64_edge | True | 0.018399 | 0.017771 | 0.000627 | n/a | 4.0 |
| case145 | cuda_fp32_edge | False | 0.011970 | 0.010233 | 0.001736 | n/a | 10.0 |
| case145 | cuda_edge | True | 0.011710 | 0.011095 | 0.000615 | n/a | 4.0 |
| case145 | cuda_fp64_edge | True | 0.010997 | 0.010238 | 0.000759 | n/a | 4.0 |
| case15da | cuda_fp32_edge | False | 0.010720 | 0.009612 | 0.001108 | n/a | 10.0 |
| case15da | cuda_edge | True | 0.010020 | 0.009609 | 0.000411 | n/a | 4.0 |
| case15da | cuda_fp64_edge | True | 0.010903 | 0.010461 | 0.000442 | n/a | 4.0 |
| case15nbr | cuda_fp32_edge | False | 0.010717 | 0.009608 | 0.001109 | n/a | 10.0 |
| case15nbr | cuda_edge | True | 0.010015 | 0.009602 | 0.000413 | n/a | 4.0 |
| case15nbr | cuda_fp64_edge | True | 0.010059 | 0.009620 | 0.000439 | n/a | 4.0 |
| case16am | cuda_fp32_edge | False | 0.012313 | 0.011285 | 0.001027 | n/a | 10.0 |
| case16am | cuda_edge | True | 0.013235 | 0.012852 | 0.000382 | n/a | 4.0 |
| case16am | cuda_fp64_edge | True | 0.010221 | 0.009580 | 0.000641 | n/a | 6.0 |
| case16ci | cuda_fp32_edge | False | 0.010542 | 0.009582 | 0.000960 | n/a | 10.0 |
| case16ci | cuda_edge | True | 0.009952 | 0.009585 | 0.000366 | n/a | 4.0 |
| case16ci | cuda_fp64_edge | True | 0.009981 | 0.009589 | 0.000391 | n/a | 4.0 |
| case17me | cuda_fp32_edge | False | 0.010727 | 0.009608 | 0.001119 | n/a | 10.0 |
| case17me | cuda_edge | True | 0.015478 | 0.014959 | 0.000518 | n/a | 5.0 |
| case17me | cuda_fp64_edge | True | 0.010174 | 0.009606 | 0.000567 | n/a | 5.0 |
| case18 | cuda_fp32_edge | False | 0.010773 | 0.009632 | 0.001140 | n/a | 10.0 |
| case18 | cuda_edge | True | 0.010170 | 0.009631 | 0.000539 | n/a | 5.0 |
| case18 | cuda_fp64_edge | True | 0.010223 | 0.009631 | 0.000591 | n/a | 5.0 |
| case1888rte | cuda_fp32_edge | False | 0.017397 | 0.013915 | 0.003482 | n/a | 10.0 |
| case1888rte | cuda_edge | True | 0.020718 | 0.019781 | 0.000937 | n/a | 3.0 |
| case1888rte | cuda_fp64_edge | True | 0.015812 | 0.014661 | 0.001150 | n/a | 3.0 |
| case18nbr | cuda_fp32_edge | False | 0.016030 | 0.014835 | 0.001195 | n/a | 10.0 |
| case18nbr | cuda_edge | True | 0.010100 | 0.009662 | 0.000437 | n/a | 4.0 |
| case18nbr | cuda_fp64_edge | True | 0.011593 | 0.011126 | 0.000467 | n/a | 4.0 |
| case1951rte | cuda_fp32_edge | False | 0.017992 | 0.014098 | 0.003892 | n/a | 10.0 |
| case1951rte | cuda_edge | True | 0.015448 | 0.014054 | 0.001393 | n/a | 4.0 |
| case1951rte | cuda_fp64_edge | True | 0.015987 | 0.014181 | 0.001806 | n/a | 4.0 |
| case22 | cuda_fp32_edge | False | 0.010791 | 0.009635 | 0.001155 | n/a | 10.0 |
| case22 | cuda_edge | True | 0.010064 | 0.009633 | 0.000430 | n/a | 4.0 |
| case22 | cuda_fp64_edge | True | 0.010102 | 0.009640 | 0.000462 | n/a | 4.0 |
| case2383wp | cuda_fp32_edge | False | 0.018752 | 0.014780 | 0.003972 | n/a | 10.0 |
| case2383wp | cuda_edge | True | 0.017374 | 0.014778 | 0.002596 | n/a | 7.0 |
| case2383wp | cuda_fp64_edge | True | 0.018289 | 0.014821 | 0.003469 | n/a | 7.0 |
| case24_ieee_rts | cuda_fp32_edge | False | 0.011609 | 0.010482 | 0.001127 | n/a | 10.0 |
| case24_ieee_rts | cuda_edge | True | 0.013436 | 0.012907 | 0.000528 | n/a | 5.0 |
| case24_ieee_rts | cuda_fp64_edge | True | 0.010309 | 0.009721 | 0.000588 | n/a | 5.0 |
| case2736sp | cuda_fp32_edge | False | 0.020204 | 0.015677 | 0.004526 | n/a | 10.0 |
| case2736sp | cuda_edge | True | 0.017830 | 0.015742 | 0.002087 | n/a | 5.0 |
| case2736sp | cuda_fp64_edge | True | 0.019892 | 0.017144 | 0.002747 | n/a | 5.0 |
| case2737sop | cuda_fp32_edge | False | 0.019930 | 0.015619 | 0.004310 | n/a | 10.0 |
| case2737sop | cuda_edge | True | 0.023510 | 0.021082 | 0.002427 | n/a | 6.0 |
| case2737sop | cuda_fp64_edge | True | 0.024444 | 0.021216 | 0.003227 | n/a | 6.0 |
| case2746wop | cuda_fp32_edge | False | 0.020345 | 0.015717 | 0.004627 | n/a | 10.0 |
| case2746wop | cuda_edge | True | 0.018018 | 0.015466 | 0.002551 | n/a | 6.0 |
| case2746wop | cuda_fp64_edge | True | 0.021023 | 0.018133 | 0.002890 | n/a | 5.0 |
| case2746wp | cuda_fp32_edge | False | 0.019677 | 0.015355 | 0.004321 | n/a | 10.0 |
| case2746wp | cuda_edge | True | 0.024061 | 0.021659 | 0.002401 | n/a | 6.0 |
| case2746wp | cuda_fp64_edge | True | 0.018079 | 0.015454 | 0.002625 | n/a | 5.0 |
| case2848rte | cuda_fp32_edge | False | 0.019617 | 0.015546 | 0.004070 | n/a | 10.0 |
| case2848rte | cuda_edge | True | 0.022220 | 0.020940 | 0.001279 | n/a | 3.4 |
| case2848rte | cuda_fp64_edge | True | 0.016990 | 0.015595 | 0.001395 | n/a | 3.0 |
| case2868rte | cuda_fp32_edge | False | 0.020164 | 0.015828 | 0.004336 | n/a | 10.0 |
| case2868rte | cuda_edge | True | 0.018229 | 0.015783 | 0.002446 | n/a | 6.0 |
| case2868rte | cuda_fp64_edge | True | 0.018810 | 0.015541 | 0.003269 | n/a | 6.0 |
| case2869pegase | cuda_fp32_edge | False | 0.019783 | 0.015776 | 0.004006 | n/a | 10.0 |
| case2869pegase | cuda_edge | True | 0.018437 | 0.015764 | 0.002672 | n/a | 7.0 |
| case2869pegase | cuda_fp64_edge | True | 0.025030 | 0.021604 | 0.003425 | n/a | 7.0 |
| case28da | cuda_fp32_edge | False | 0.013183 | 0.011981 | 0.001201 | n/a | 10.0 |
| case28da | cuda_edge | True | 0.010155 | 0.009675 | 0.000480 | n/a | 4.4 |
| case28da | cuda_fp64_edge | True | 0.010990 | 0.010498 | 0.000492 | n/a | 4.0 |
| case30 | cuda_fp32_edge | False | 0.016454 | 0.015227 | 0.001227 | n/a | 10.0 |
| case30 | cuda_edge | True | 0.012742 | 0.012290 | 0.000452 | n/a | 4.0 |
| case30 | cuda_fp64_edge | True | 0.010221 | 0.009718 | 0.000503 | n/a | 4.0 |
| case300 | cuda_fp32_edge | False | 0.012735 | 0.010614 | 0.002121 | n/a | 10.0 |
| case300 | cuda_edge | True | 0.011813 | 0.010616 | 0.001197 | n/a | 6.0 |
| case300 | cuda_fp64_edge | True | 0.015816 | 0.014341 | 0.001474 | n/a | 6.0 |
| case3012wp | cuda_fp32_edge | False | 0.020432 | 0.015942 | 0.004489 | n/a | 10.0 |
| case3012wp | cuda_edge | True | 0.022997 | 0.021376 | 0.001620 | n/a | 4.0 |
| case3012wp | cuda_fp64_edge | True | 0.017988 | 0.015854 | 0.002134 | n/a | 4.0 |
| case30Q | cuda_fp32_edge | False | 0.010950 | 0.009729 | 0.001220 | n/a | 10.0 |
| case30Q | cuda_edge | True | 0.010184 | 0.009733 | 0.000450 | n/a | 4.0 |
| case30Q | cuda_fp64_edge | True | 0.010229 | 0.009723 | 0.000505 | n/a | 4.0 |
| case30pwl | cuda_fp32_edge | False | 0.010943 | 0.009711 | 0.001232 | n/a | 10.0 |
| case30pwl | cuda_edge | True | 0.010186 | 0.009733 | 0.000452 | n/a | 4.0 |
| case30pwl | cuda_fp64_edge | True | 0.010220 | 0.009715 | 0.000504 | n/a | 4.0 |
| case3120sp | cuda_fp32_edge | False | 0.020969 | 0.016279 | 0.004690 | n/a | 10.0 |
| case3120sp | cuda_edge | True | 0.020117 | 0.017056 | 0.003060 | n/a | 7.0 |
| case3120sp | cuda_fp64_edge | True | 0.021166 | 0.016961 | 0.004205 | n/a | 7.0 |
| case3375wp | cuda_fp32_edge | False | 0.022190 | 0.017375 | 0.004815 | n/a | 10.0 |
| case3375wp | cuda_edge | True | 0.017646 | 0.016305 | 0.001341 | n/a | 3.0 |
| case3375wp | cuda_fp64_edge | True | 0.018364 | 0.016682 | 0.001682 | n/a | 3.0 |
| case33bw | cuda_fp32_edge | False | 0.010895 | 0.009697 | 0.001197 | n/a | 10.0 |
| case33bw | cuda_edge | True | 0.010130 | 0.009693 | 0.000436 | n/a | 4.0 |
| case33bw | cuda_fp64_edge | True | 0.010150 | 0.009684 | 0.000466 | n/a | 4.0 |
| case33mg | cuda_fp32_edge | False | 0.010874 | 0.009679 | 0.001195 | n/a | 10.0 |
| case33mg | cuda_edge | True | 0.010238 | 0.009685 | 0.000553 | n/a | 5.0 |
| case33mg | cuda_fp64_edge | True | 0.010269 | 0.009670 | 0.000598 | n/a | 5.0 |
| case34sa | cuda_fp32_edge | False | 0.017870 | 0.016642 | 0.001227 | n/a | 10.0 |
| case34sa | cuda_edge | True | 0.010105 | 0.009670 | 0.000434 | n/a | 4.0 |
| case34sa | cuda_fp64_edge | True | 0.012243 | 0.011756 | 0.000486 | n/a | 4.0 |
| case38si | cuda_fp32_edge | False | 0.010959 | 0.009705 | 0.001254 | n/a | 10.0 |
| case38si | cuda_edge | True | 0.010285 | 0.009709 | 0.000576 | n/a | 5.0 |
| case38si | cuda_fp64_edge | True | 0.010321 | 0.009695 | 0.000625 | n/a | 5.0 |
| case39 | cuda_fp32_edge | False | 0.011001 | 0.009724 | 0.001276 | n/a | 10.0 |
| case39 | cuda_edge | True | 0.009934 | 0.009721 | 0.000213 | n/a | 2.0 |
| case39 | cuda_fp64_edge | True | 0.009951 | 0.009731 | 0.000219 | n/a | 2.0 |
| case4_dist | cuda_fp32_edge | False | 0.010186 | 0.009516 | 0.000669 | n/a | 10.0 |
| case4_dist | cuda_edge | True | 0.010083 | 0.009808 | 0.000275 | n/a | 4.0 |
| case4_dist | cuda_fp64_edge | True | 0.010086 | 0.009801 | 0.000285 | n/a | 4.0 |
| case4gs | cuda_fp32_edge | False | 0.010387 | 0.009518 | 0.000869 | n/a | 10.0 |
| case4gs | cuda_edge | True | 0.011665 | 0.011328 | 0.000337 | n/a | 4.0 |
| case4gs | cuda_fp64_edge | True | 0.009876 | 0.009535 | 0.000340 | n/a | 4.0 |
| case5 | cuda_fp32_edge | False | 0.010535 | 0.009670 | 0.000865 | n/a | 10.0 |
| case5 | cuda_edge | True | 0.009876 | 0.009540 | 0.000335 | n/a | 4.0 |
| case5 | cuda_fp64_edge | True | 0.010197 | 0.009856 | 0.000340 | n/a | 4.0 |
| case51ga | cuda_fp32_edge | False | 0.011048 | 0.009770 | 0.001277 | n/a | 10.0 |
| case51ga | cuda_edge | True | 0.010347 | 0.009764 | 0.000583 | n/a | 5.0 |
| case51ga | cuda_fp64_edge | True | 0.010448 | 0.009791 | 0.000656 | n/a | 5.0 |
| case51he | cuda_fp32_edge | False | 0.011007 | 0.009744 | 0.001263 | n/a | 10.0 |
| case51he | cuda_edge | True | 0.010234 | 0.009780 | 0.000454 | n/a | 4.0 |
| case51he | cuda_fp64_edge | True | 0.010277 | 0.009759 | 0.000517 | n/a | 4.0 |
| case533mt_hi | cuda_fp32_edge | False | 0.012859 | 0.011130 | 0.001728 | n/a | 10.0 |
| case533mt_hi | cuda_edge | True | 0.011793 | 0.011116 | 0.000677 | n/a | 4.0 |
| case533mt_hi | cuda_fp64_edge | True | 0.011860 | 0.011111 | 0.000748 | n/a | 4.0 |
| case533mt_lo | cuda_fp32_edge | False | 0.013630 | 0.011891 | 0.001738 | n/a | 10.0 |
| case533mt_lo | cuda_edge | True | 0.012584 | 0.011908 | 0.000675 | n/a | 4.0 |
| case533mt_lo | cuda_fp64_edge | True | 0.011843 | 0.011096 | 0.000746 | n/a | 4.0 |
| case57 | cuda_fp32_edge | False | 0.011412 | 0.009889 | 0.001522 | n/a | 10.0 |
| case57 | cuda_edge | True | 0.010424 | 0.009890 | 0.000534 | n/a | 4.0 |
| case57 | cuda_fp64_edge | True | 0.010500 | 0.009883 | 0.000616 | n/a | 4.0 |
| case59 | cuda_fp32_edge | False | 0.011290 | 0.009835 | 0.001454 | n/a | 10.0 |
| case59 | cuda_edge | True | 0.017443 | 0.016629 | 0.000813 | n/a | 6.0 |
| case59 | cuda_fp64_edge | True | 0.010813 | 0.009838 | 0.000974 | n/a | 6.0 |
| case60nordic | cuda_fp32_edge | False | 0.011339 | 0.009907 | 0.001431 | n/a | 10.0 |
| case60nordic | cuda_edge | True | 0.010066 | 0.009843 | 0.000222 | n/a | 2.0 |
| case60nordic | cuda_fp64_edge | True | 0.013026 | 0.012785 | 0.000240 | n/a | 2.0 |
| case6468rte | cuda_fp32_edge | False | 0.028691 | 0.022607 | 0.006083 | n/a | 10.0 |
| case6468rte | cuda_edge | True | 0.024909 | 0.022571 | 0.002337 | n/a | 4.0 |
| case6468rte | cuda_fp64_edge | True | 0.030818 | 0.027698 | 0.003120 | n/a | 4.0 |
| case6470rte | cuda_fp32_edge | False | 0.029010 | 0.022910 | 0.006099 | n/a | 10.0 |
| case6470rte | cuda_edge | True | 0.025072 | 0.022767 | 0.002304 | n/a | 4.0 |
| case6470rte | cuda_fp64_edge | True | 0.025718 | 0.022648 | 0.003069 | n/a | 4.0 |
| case6495rte | cuda_fp32_edge | False | 0.028089 | 0.022155 | 0.005934 | n/a | 10.0 |
| case6495rte | cuda_edge | True | 0.026491 | 0.024796 | 0.001694 | n/a | 3.0 |
| case6495rte | cuda_fp64_edge | True | 0.024208 | 0.022043 | 0.002165 | n/a | 3.0 |
| case6515rte | cuda_fp32_edge | False | 0.028760 | 0.022269 | 0.006490 | n/a | 10.0 |
| case6515rte | cuda_edge | True | 0.024553 | 0.022093 | 0.002460 | n/a | 4.0 |
| case6515rte | cuda_fp64_edge | True | 0.025479 | 0.022162 | 0.003317 | n/a | 4.0 |
| case69 | cuda_fp32_edge | False | 0.011168 | 0.009793 | 0.001374 | n/a | 10.0 |
| case69 | cuda_edge | True | 0.013321 | 0.012691 | 0.000630 | n/a | 5.0 |
| case69 | cuda_fp64_edge | True | 0.018049 | 0.017321 | 0.000727 | n/a | 5.0 |
| case6ww | cuda_fp32_edge | False | 0.010503 | 0.009567 | 0.000935 | n/a | 10.0 |
| case6ww | cuda_edge | True | 0.009896 | 0.009541 | 0.000354 | n/a | 4.0 |
| case6ww | cuda_fp64_edge | True | 0.010609 | 0.010239 | 0.000369 | n/a | 4.0 |
| case70da | cuda_fp32_edge | False | 0.010967 | 0.009764 | 0.001203 | n/a | 10.0 |
| case70da | cuda_edge | True | 0.010303 | 0.009746 | 0.000557 | n/a | 5.0 |
| case70da | cuda_fp64_edge | True | 0.010425 | 0.009786 | 0.000638 | n/a | 5.0 |
| case74ds | cuda_fp32_edge | False | 0.011142 | 0.009779 | 0.001362 | n/a | 10.0 |
| case74ds | cuda_edge | True | 0.010315 | 0.009827 | 0.000489 | n/a | 4.0 |
| case74ds | cuda_fp64_edge | True | 0.010317 | 0.009774 | 0.000543 | n/a | 4.0 |
| case8387pegase | cuda_fp32_edge | False | 0.036743 | 0.028048 | 0.008695 | n/a | 10.0 |
| case8387pegase | cuda_edge | True | 0.030704 | 0.027482 | 0.003221 | n/a | 4.0 |
| case8387pegase | cuda_fp64_edge | True | 0.032121 | 0.027771 | 0.004350 | n/a | 4.0 |
| case85 | cuda_fp32_edge | False | 0.011328 | 0.009887 | 0.001440 | n/a | 10.0 |
| case85 | cuda_edge | True | 0.010545 | 0.009902 | 0.000643 | n/a | 5.0 |
| case85 | cuda_fp64_edge | True | 0.014385 | 0.013651 | 0.000734 | n/a | 5.0 |
| case89pegase | cuda_fp32_edge | False | 0.012371 | 0.010265 | 0.002105 | n/a | 10.0 |
| case89pegase | cuda_edge | True | 0.011338 | 0.010219 | 0.001119 | n/a | 6.0 |
| case89pegase | cuda_fp64_edge | True | 0.011741 | 0.010273 | 0.001468 | n/a | 6.0 |
| case9 | cuda_fp32_edge | False | 0.014694 | 0.013704 | 0.000989 | n/a | 10.0 |
| case9 | cuda_edge | True | 0.010062 | 0.009586 | 0.000476 | n/a | 5.0 |
| case9 | cuda_fp64_edge | True | 0.010086 | 0.009588 | 0.000497 | n/a | 5.0 |
| case9241pegase | cuda_fp32_edge | False | 0.038828 | 0.029833 | 0.008994 | n/a | 10.0 |
| case9241pegase | cuda_edge | True | 0.035453 | 0.029555 | 0.005897 | n/a | 7.0 |
| case9241pegase | cuda_fp64_edge | True | 0.037812 | 0.029167 | 0.008645 | n/a | 7.0 |
| case94pi | cuda_fp32_edge | False | 0.014849 | 0.013456 | 0.001392 | n/a | 10.0 |
| case94pi | cuda_edge | True | 0.010526 | 0.009880 | 0.000645 | n/a | 5.0 |
| case94pi | cuda_fp64_edge | True | 0.011456 | 0.010719 | 0.000737 | n/a | 5.0 |
| case9Q | cuda_fp32_edge | False | 0.010579 | 0.009573 | 0.001005 | n/a | 10.0 |
| case9Q | cuda_edge | True | 0.011638 | 0.011161 | 0.000476 | n/a | 5.0 |
| case9Q | cuda_fp64_edge | True | 0.010079 | 0.009581 | 0.000499 | n/a | 5.0 |
| case9target | cuda_fp32_edge | False | 0.010571 | 0.009565 | 0.001005 | n/a | 10.0 |
| case9target | cuda_edge | True | 0.010274 | 0.009605 | 0.000669 | n/a | 7.0 |
| case9target | cuda_fp64_edge | True | 0.010288 | 0.009576 | 0.000711 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_fp32_edge | False | 0.037191 | 0.029064 | 0.008127 | n/a | 10.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.033455 | 0.028923 | 0.004532 | n/a | 5.8 |
| case_ACTIVSg10k | cuda_fp64_edge | True | 0.033575 | 0.028310 | 0.005264 | n/a | 5.0 |
| case_ACTIVSg200 | cuda_fp32_edge | False | 0.016346 | 0.014481 | 0.001865 | n/a | 10.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.010779 | 0.010322 | 0.000457 | n/a | 3.0 |
| case_ACTIVSg200 | cuda_fp64_edge | True | 0.010878 | 0.010332 | 0.000545 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_fp32_edge | False | 0.020243 | 0.014706 | 0.005536 | n/a | 10.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.019697 | 0.017798 | 0.001898 | n/a | 4.0 |
| case_ACTIVSg2000 | cuda_fp64_edge | True | 0.017570 | 0.014771 | 0.002798 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_fp32_edge | False | 0.077715 | 0.060528 | 0.017187 | n/a | 10.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.068512 | 0.060358 | 0.008154 | n/a | 5.0 |
| case_ACTIVSg25k | cuda_fp64_edge | True | 0.068025 | 0.056166 | 0.011859 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_fp32_edge | False | 0.013358 | 0.011143 | 0.002214 | n/a | 10.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.011940 | 0.011125 | 0.000815 | n/a | 4.0 |
| case_ACTIVSg500 | cuda_fp64_edge | True | 0.012900 | 0.011905 | 0.000994 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_fp32_edge | False | 0.161250 | 0.122172 | 0.039078 | n/a | 10.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.149968 | 0.123056 | 0.026911 | n/a | 7.0 |
| case_ACTIVSg70k | cuda_fp64_edge | True | 0.166576 | 0.126118 | 0.040458 | n/a | 7.0 |
| case_RTS_GMLC | cuda_fp32_edge | False | 0.011320 | 0.009872 | 0.001448 | n/a | 10.0 |
| case_RTS_GMLC | cuda_edge | True | 0.011399 | 0.010733 | 0.000665 | n/a | 5.0 |
| case_RTS_GMLC | cuda_fp64_edge | True | 0.013948 | 0.013184 | 0.000764 | n/a | 5.0 |
| case_SyntheticUSA | cuda_fp32_edge | False | 0.211971 | 0.168950 | 0.043020 | n/a | 10.0 |
| case_SyntheticUSA | cuda_edge | True | 0.195613 | 0.163290 | 0.032322 | n/a | 7.6 |
| case_SyntheticUSA | cuda_fp64_edge | True | 0.205853 | 0.162553 | 0.043299 | n/a | 7.0 |
| case_ieee30 | cuda_fp32_edge | False | 0.011011 | 0.009734 | 0.001276 | n/a | 10.0 |
| case_ieee30 | cuda_edge | True | 0.010108 | 0.009767 | 0.000340 | n/a | 3.0 |
| case_ieee30 | cuda_fp64_edge | True | 0.010089 | 0.009723 | 0.000365 | n/a | 3.0 |

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
