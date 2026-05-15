# cuPF Benchmark `matpower_precision_fp32_mixed_fp64_fixed10updates_b1_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T03:30:07.153133+00:00
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
| case10ba | cuda_fp32_edge | False | 0.010616 | 0.009582 | 0.001034 | n/a | 10.0 |
| case10ba | cuda_edge | False | 0.010669 | 0.009568 | 0.001100 | n/a | 10.0 |
| case10ba | cuda_fp64_edge | False | 0.010780 | 0.009590 | 0.001190 | n/a | 10.0 |
| case118 | cuda_fp32_edge | False | 0.011460 | 0.010001 | 0.001459 | n/a | 10.0 |
| case118 | cuda_edge | False | 0.011573 | 0.010026 | 0.001546 | n/a | 10.0 |
| case118 | cuda_fp64_edge | False | 0.011840 | 0.010010 | 0.001829 | n/a | 10.0 |
| case118zh | cuda_fp32_edge | False | 0.011276 | 0.009899 | 0.001377 | n/a | 10.0 |
| case118zh | cuda_edge | False | 0.013433 | 0.011983 | 0.001449 | n/a | 10.0 |
| case118zh | cuda_fp64_edge | False | 0.012431 | 0.010735 | 0.001696 | n/a | 10.0 |
| case1197 | cuda_fp32_edge | False | 0.015634 | 0.013347 | 0.002287 | n/a | 10.0 |
| case1197 | cuda_edge | False | 0.015778 | 0.013391 | 0.002387 | n/a | 10.0 |
| case1197 | cuda_fp64_edge | False | 0.015462 | 0.012600 | 0.002861 | n/a | 10.0 |
| case12da | cuda_fp32_edge | False | 0.010670 | 0.009573 | 0.001096 | n/a | 10.0 |
| case12da | cuda_edge | False | 0.010741 | 0.009600 | 0.001141 | n/a | 10.0 |
| case12da | cuda_fp64_edge | False | 0.010899 | 0.009630 | 0.001268 | n/a | 10.0 |
| case1354pegase | cuda_fp32_edge | False | 0.016716 | 0.013423 | 0.003292 | n/a | 10.0 |
| case1354pegase | cuda_edge | False | 0.016820 | 0.013415 | 0.003405 | n/a | 10.0 |
| case1354pegase | cuda_fp64_edge | False | 0.017921 | 0.013460 | 0.004460 | n/a | 10.0 |
| case13659pegase | cuda_fp32_edge | False | 0.047378 | 0.036794 | 0.010584 | n/a | 10.0 |
| case13659pegase | cuda_edge | False | 0.046392 | 0.035133 | 0.011258 | n/a | 10.0 |
| case13659pegase | cuda_fp64_edge | False | 0.054444 | 0.037174 | 0.017270 | n/a | 10.0 |
| case136ma | cuda_fp32_edge | False | 0.011210 | 0.009916 | 0.001293 | n/a | 10.0 |
| case136ma | cuda_edge | False | 0.011231 | 0.009898 | 0.001333 | n/a | 10.0 |
| case136ma | cuda_fp64_edge | False | 0.011484 | 0.009906 | 0.001577 | n/a | 10.0 |
| case14 | cuda_fp32_edge | False | 0.010654 | 0.009610 | 0.001044 | n/a | 10.0 |
| case14 | cuda_edge | False | 0.010679 | 0.009600 | 0.001079 | n/a | 10.0 |
| case14 | cuda_fp64_edge | False | 0.012781 | 0.011574 | 0.001206 | n/a | 10.0 |
| case141 | cuda_fp32_edge | False | 0.011552 | 0.010014 | 0.001537 | n/a | 10.0 |
| case141 | cuda_edge | False | 0.015270 | 0.013689 | 0.001580 | n/a | 10.0 |
| case141 | cuda_fp64_edge | False | 0.011916 | 0.010033 | 0.001883 | n/a | 10.0 |
| case145 | cuda_fp32_edge | False | 0.016098 | 0.014379 | 0.001718 | n/a | 10.0 |
| case145 | cuda_edge | False | 0.012039 | 0.010226 | 0.001812 | n/a | 10.0 |
| case145 | cuda_fp64_edge | False | 0.012561 | 0.010232 | 0.002329 | n/a | 10.0 |
| case15da | cuda_fp32_edge | False | 0.010727 | 0.009613 | 0.001114 | n/a | 10.0 |
| case15da | cuda_edge | False | 0.012760 | 0.011612 | 0.001147 | n/a | 10.0 |
| case15da | cuda_fp64_edge | False | 0.010886 | 0.009605 | 0.001281 | n/a | 10.0 |
| case15nbr | cuda_fp32_edge | False | 0.010698 | 0.009589 | 0.001109 | n/a | 10.0 |
| case15nbr | cuda_edge | False | 0.010748 | 0.009608 | 0.001140 | n/a | 10.0 |
| case15nbr | cuda_fp64_edge | False | 0.010887 | 0.009613 | 0.001274 | n/a | 10.0 |
| case16am | cuda_fp32_edge | False | 0.010606 | 0.009579 | 0.001026 | n/a | 10.0 |
| case16am | cuda_edge | False | 0.015657 | 0.014595 | 0.001061 | n/a | 10.0 |
| case16am | cuda_fp64_edge | False | 0.010763 | 0.009572 | 0.001190 | n/a | 10.0 |
| case16ci | cuda_fp32_edge | False | 0.011344 | 0.010382 | 0.000962 | n/a | 10.0 |
| case16ci | cuda_edge | False | 0.010584 | 0.009572 | 0.001011 | n/a | 10.0 |
| case16ci | cuda_fp64_edge | False | 0.011517 | 0.010412 | 0.001104 | n/a | 10.0 |
| case17me | cuda_fp32_edge | False | 0.010721 | 0.009607 | 0.001113 | n/a | 10.0 |
| case17me | cuda_edge | False | 0.010788 | 0.009642 | 0.001147 | n/a | 10.0 |
| case17me | cuda_fp64_edge | False | 0.010913 | 0.009619 | 0.001294 | n/a | 10.0 |
| case18 | cuda_fp32_edge | False | 0.010759 | 0.009624 | 0.001134 | n/a | 10.0 |
| case18 | cuda_edge | False | 0.010827 | 0.009626 | 0.001200 | n/a | 10.0 |
| case18 | cuda_fp64_edge | False | 0.010991 | 0.009647 | 0.001343 | n/a | 10.0 |
| case1888rte | cuda_fp32_edge | False | 0.017324 | 0.013858 | 0.003465 | n/a | 10.0 |
| case1888rte | cuda_edge | False | 0.017477 | 0.013867 | 0.003609 | n/a | 10.0 |
| case1888rte | cuda_fp64_edge | False | 0.018795 | 0.013925 | 0.004869 | n/a | 10.0 |
| case18nbr | cuda_fp32_edge | False | 0.010824 | 0.009635 | 0.001189 | n/a | 10.0 |
| case18nbr | cuda_edge | False | 0.010855 | 0.009631 | 0.001223 | n/a | 10.0 |
| case18nbr | cuda_fp64_edge | False | 0.010976 | 0.009615 | 0.001360 | n/a | 10.0 |
| case1951rte | cuda_fp32_edge | False | 0.017990 | 0.014107 | 0.003883 | n/a | 10.0 |
| case1951rte | cuda_edge | False | 0.022412 | 0.018403 | 0.004008 | n/a | 10.0 |
| case1951rte | cuda_fp64_edge | False | 0.019695 | 0.014177 | 0.005517 | n/a | 10.0 |
| case22 | cuda_fp32_edge | False | 0.017224 | 0.016086 | 0.001138 | n/a | 10.0 |
| case22 | cuda_edge | False | 0.010838 | 0.009628 | 0.001209 | n/a | 10.0 |
| case22 | cuda_fp64_edge | False | 0.010966 | 0.009630 | 0.001336 | n/a | 10.0 |
| case2383wp | cuda_fp32_edge | False | 0.020200 | 0.016233 | 0.003966 | n/a | 10.0 |
| case2383wp | cuda_edge | False | 0.018691 | 0.014585 | 0.004106 | n/a | 10.0 |
| case2383wp | cuda_fp64_edge | False | 0.024308 | 0.018702 | 0.005606 | n/a | 10.0 |
| case24_ieee_rts | cuda_fp32_edge | False | 0.010802 | 0.009676 | 0.001126 | n/a | 10.0 |
| case24_ieee_rts | cuda_edge | False | 0.010861 | 0.009686 | 0.001175 | n/a | 10.0 |
| case24_ieee_rts | cuda_fp64_edge | False | 0.011009 | 0.009672 | 0.001337 | n/a | 10.0 |
| case2736sp | cuda_fp32_edge | False | 0.025737 | 0.021225 | 0.004512 | n/a | 10.0 |
| case2736sp | cuda_edge | False | 0.025753 | 0.021041 | 0.004711 | n/a | 10.0 |
| case2736sp | cuda_fp64_edge | False | 0.028035 | 0.021556 | 0.006478 | n/a | 10.0 |
| case2737sop | cuda_fp32_edge | False | 0.019791 | 0.015480 | 0.004311 | n/a | 10.0 |
| case2737sop | cuda_edge | False | 0.020162 | 0.015666 | 0.004496 | n/a | 10.0 |
| case2737sop | cuda_fp64_edge | False | 0.028550 | 0.022356 | 0.006194 | n/a | 10.0 |
| case2746wop | cuda_fp32_edge | False | 0.027207 | 0.022591 | 0.004615 | n/a | 10.0 |
| case2746wop | cuda_edge | False | 0.020450 | 0.015672 | 0.004777 | n/a | 10.0 |
| case2746wop | cuda_fp64_edge | False | 0.027101 | 0.020270 | 0.006831 | n/a | 10.0 |
| case2746wp | cuda_fp32_edge | False | 0.025296 | 0.020979 | 0.004317 | n/a | 10.0 |
| case2746wp | cuda_edge | False | 0.019733 | 0.015270 | 0.004462 | n/a | 10.0 |
| case2746wp | cuda_fp64_edge | False | 0.021612 | 0.015423 | 0.006187 | n/a | 10.0 |
| case2848rte | cuda_fp32_edge | False | 0.021069 | 0.016985 | 0.004084 | n/a | 10.0 |
| case2848rte | cuda_edge | False | 0.019773 | 0.015513 | 0.004259 | n/a | 10.0 |
| case2848rte | cuda_fp64_edge | False | 0.021411 | 0.015531 | 0.005879 | n/a | 10.0 |
| case2868rte | cuda_fp32_edge | False | 0.020125 | 0.015794 | 0.004330 | n/a | 10.0 |
| case2868rte | cuda_edge | False | 0.020337 | 0.015803 | 0.004533 | n/a | 10.0 |
| case2868rte | cuda_fp64_edge | False | 0.022058 | 0.015817 | 0.006240 | n/a | 10.0 |
| case2869pegase | cuda_fp32_edge | False | 0.019587 | 0.015595 | 0.003991 | n/a | 10.0 |
| case2869pegase | cuda_edge | False | 0.019950 | 0.015762 | 0.004188 | n/a | 10.0 |
| case2869pegase | cuda_fp64_edge | False | 0.021265 | 0.015776 | 0.005488 | n/a | 10.0 |
| case28da | cuda_fp32_edge | False | 0.011655 | 0.010454 | 0.001201 | n/a | 10.0 |
| case28da | cuda_edge | False | 0.012969 | 0.011741 | 0.001228 | n/a | 10.0 |
| case28da | cuda_fp64_edge | False | 0.011280 | 0.009840 | 0.001440 | n/a | 10.0 |
| case30 | cuda_fp32_edge | False | 0.011740 | 0.010513 | 0.001226 | n/a | 10.0 |
| case30 | cuda_edge | False | 0.011010 | 0.009719 | 0.001291 | n/a | 10.0 |
| case30 | cuda_fp64_edge | False | 0.011194 | 0.009709 | 0.001485 | n/a | 10.0 |
| case300 | cuda_fp32_edge | False | 0.012740 | 0.010620 | 0.002119 | n/a | 10.0 |
| case300 | cuda_edge | False | 0.012814 | 0.010611 | 0.002203 | n/a | 10.0 |
| case300 | cuda_fp64_edge | False | 0.018119 | 0.015336 | 0.002782 | n/a | 10.0 |
| case3012wp | cuda_fp32_edge | False | 0.023015 | 0.018528 | 0.004487 | n/a | 10.0 |
| case3012wp | cuda_edge | False | 0.020580 | 0.015929 | 0.004651 | n/a | 10.0 |
| case3012wp | cuda_fp64_edge | False | 0.023103 | 0.016598 | 0.006504 | n/a | 10.0 |
| case30Q | cuda_fp32_edge | False | 0.010922 | 0.009703 | 0.001218 | n/a | 10.0 |
| case30Q | cuda_edge | False | 0.011019 | 0.009727 | 0.001292 | n/a | 10.0 |
| case30Q | cuda_fp64_edge | False | 0.011203 | 0.009712 | 0.001490 | n/a | 10.0 |
| case30pwl | cuda_fp32_edge | False | 0.010927 | 0.009710 | 0.001216 | n/a | 10.0 |
| case30pwl | cuda_edge | False | 0.016757 | 0.015464 | 0.001292 | n/a | 10.0 |
| case30pwl | cuda_fp64_edge | False | 0.011200 | 0.009714 | 0.001485 | n/a | 10.0 |
| case3120sp | cuda_fp32_edge | False | 0.020981 | 0.016302 | 0.004679 | n/a | 10.0 |
| case3120sp | cuda_edge | False | 0.021109 | 0.016251 | 0.004857 | n/a | 10.0 |
| case3120sp | cuda_fp64_edge | False | 0.023660 | 0.016838 | 0.006821 | n/a | 10.0 |
| case3375wp | cuda_fp32_edge | False | 0.021431 | 0.016634 | 0.004796 | n/a | 10.0 |
| case3375wp | cuda_edge | False | 0.021519 | 0.016465 | 0.005053 | n/a | 10.0 |
| case3375wp | cuda_fp64_edge | False | 0.023526 | 0.016467 | 0.007059 | n/a | 10.0 |
| case33bw | cuda_fp32_edge | False | 0.010865 | 0.009672 | 0.001193 | n/a | 10.0 |
| case33bw | cuda_edge | False | 0.010908 | 0.009683 | 0.001225 | n/a | 10.0 |
| case33bw | cuda_fp64_edge | False | 0.011062 | 0.009681 | 0.001380 | n/a | 10.0 |
| case33mg | cuda_fp32_edge | False | 0.010879 | 0.009683 | 0.001195 | n/a | 10.0 |
| case33mg | cuda_edge | False | 0.010904 | 0.009671 | 0.001233 | n/a | 10.0 |
| case33mg | cuda_fp64_edge | False | 0.011035 | 0.009659 | 0.001376 | n/a | 10.0 |
| case34sa | cuda_fp32_edge | False | 0.010864 | 0.009661 | 0.001202 | n/a | 10.0 |
| case34sa | cuda_edge | False | 0.015841 | 0.014606 | 0.001235 | n/a | 10.0 |
| case34sa | cuda_fp64_edge | False | 0.011108 | 0.009666 | 0.001442 | n/a | 10.0 |
| case38si | cuda_fp32_edge | False | 0.010949 | 0.009714 | 0.001235 | n/a | 10.0 |
| case38si | cuda_edge | False | 0.011006 | 0.009714 | 0.001291 | n/a | 10.0 |
| case38si | cuda_fp64_edge | False | 0.020570 | 0.019117 | 0.001452 | n/a | 10.0 |
| case39 | cuda_fp32_edge | False | 0.011011 | 0.009730 | 0.001281 | n/a | 10.0 |
| case39 | cuda_edge | False | 0.011047 | 0.009734 | 0.001312 | n/a | 10.0 |
| case39 | cuda_fp64_edge | False | 0.011273 | 0.009728 | 0.001544 | n/a | 10.0 |
| case4_dist | cuda_fp32_edge | False | 0.010900 | 0.010212 | 0.000687 | n/a | 10.0 |
| case4_dist | cuda_edge | False | 0.010219 | 0.009497 | 0.000721 | n/a | 10.0 |
| case4_dist | cuda_fp64_edge | False | 0.010266 | 0.009516 | 0.000749 | n/a | 10.0 |
| case4gs | cuda_fp32_edge | False | 0.010395 | 0.009523 | 0.000871 | n/a | 10.0 |
| case4gs | cuda_edge | False | 0.010435 | 0.009535 | 0.000900 | n/a | 10.0 |
| case4gs | cuda_fp64_edge | False | 0.010864 | 0.009906 | 0.000957 | n/a | 10.0 |
| case5 | cuda_fp32_edge | False | 0.010740 | 0.009872 | 0.000867 | n/a | 10.0 |
| case5 | cuda_edge | False | 0.010775 | 0.009867 | 0.000908 | n/a | 10.0 |
| case5 | cuda_fp64_edge | False | 0.010495 | 0.009549 | 0.000946 | n/a | 10.0 |
| case51ga | cuda_fp32_edge | False | 0.013995 | 0.012721 | 0.001273 | n/a | 10.0 |
| case51ga | cuda_edge | False | 0.011077 | 0.009765 | 0.001312 | n/a | 10.0 |
| case51ga | cuda_fp64_edge | False | 0.012067 | 0.010555 | 0.001512 | n/a | 10.0 |
| case51he | cuda_fp32_edge | False | 0.011001 | 0.009756 | 0.001245 | n/a | 10.0 |
| case51he | cuda_edge | False | 0.015952 | 0.014643 | 0.001309 | n/a | 10.0 |
| case51he | cuda_fp64_edge | False | 0.011290 | 0.009756 | 0.001534 | n/a | 10.0 |
| case533mt_hi | cuda_fp32_edge | False | 0.012830 | 0.011104 | 0.001725 | n/a | 10.0 |
| case533mt_hi | cuda_edge | False | 0.012931 | 0.011111 | 0.001819 | n/a | 10.0 |
| case533mt_hi | cuda_fp64_edge | False | 0.020649 | 0.018531 | 0.002117 | n/a | 10.0 |
| case533mt_lo | cuda_fp32_edge | False | 0.012833 | 0.011105 | 0.001727 | n/a | 10.0 |
| case533mt_lo | cuda_edge | False | 0.012915 | 0.011099 | 0.001815 | n/a | 10.0 |
| case533mt_lo | cuda_fp64_edge | False | 0.013240 | 0.011121 | 0.002118 | n/a | 10.0 |
| case57 | cuda_fp32_edge | False | 0.019270 | 0.017708 | 0.001561 | n/a | 10.0 |
| case57 | cuda_edge | False | 0.011443 | 0.009880 | 0.001563 | n/a | 10.0 |
| case57 | cuda_fp64_edge | False | 0.014428 | 0.012561 | 0.001867 | n/a | 10.0 |
| case59 | cuda_fp32_edge | False | 0.012077 | 0.010627 | 0.001449 | n/a | 10.0 |
| case59 | cuda_edge | False | 0.017205 | 0.015669 | 0.001536 | n/a | 10.0 |
| case59 | cuda_fp64_edge | False | 0.011727 | 0.009867 | 0.001860 | n/a | 10.0 |
| case60nordic | cuda_fp32_edge | False | 0.011279 | 0.009844 | 0.001434 | n/a | 10.0 |
| case60nordic | cuda_edge | False | 0.011302 | 0.009822 | 0.001479 | n/a | 10.0 |
| case60nordic | cuda_fp64_edge | False | 0.011595 | 0.009846 | 0.001749 | n/a | 10.0 |
| case6468rte | cuda_fp32_edge | False | 0.028862 | 0.022770 | 0.006092 | n/a | 10.0 |
| case6468rte | cuda_edge | False | 0.028911 | 0.022541 | 0.006370 | n/a | 10.0 |
| case6468rte | cuda_fp64_edge | False | 0.034445 | 0.025195 | 0.009249 | n/a | 10.0 |
| case6470rte | cuda_fp32_edge | False | 0.029021 | 0.022932 | 0.006089 | n/a | 10.0 |
| case6470rte | cuda_edge | False | 0.029011 | 0.022655 | 0.006356 | n/a | 10.0 |
| case6470rte | cuda_fp64_edge | False | 0.032459 | 0.023241 | 0.009218 | n/a | 10.0 |
| case6495rte | cuda_fp32_edge | False | 0.028066 | 0.022141 | 0.005924 | n/a | 10.0 |
| case6495rte | cuda_edge | False | 0.029457 | 0.023241 | 0.006216 | n/a | 10.0 |
| case6495rte | cuda_fp64_edge | False | 0.032441 | 0.023398 | 0.009043 | n/a | 10.0 |
| case6515rte | cuda_fp32_edge | False | 0.030067 | 0.023568 | 0.006499 | n/a | 10.0 |
| case6515rte | cuda_edge | False | 0.030470 | 0.023655 | 0.006815 | n/a | 10.0 |
| case6515rte | cuda_fp64_edge | False | 0.032193 | 0.022210 | 0.009982 | n/a | 10.0 |
| case69 | cuda_fp32_edge | False | 0.011143 | 0.009774 | 0.001369 | n/a | 10.0 |
| case69 | cuda_edge | False | 0.016982 | 0.015545 | 0.001436 | n/a | 10.0 |
| case69 | cuda_fp64_edge | False | 0.011494 | 0.009801 | 0.001693 | n/a | 10.0 |
| case6ww | cuda_fp32_edge | False | 0.010490 | 0.009562 | 0.000929 | n/a | 10.0 |
| case6ww | cuda_edge | False | 0.010523 | 0.009548 | 0.000974 | n/a | 10.0 |
| case6ww | cuda_fp64_edge | False | 0.010951 | 0.009911 | 0.001039 | n/a | 10.0 |
| case70da | cuda_fp32_edge | False | 0.010950 | 0.009748 | 0.001201 | n/a | 10.0 |
| case70da | cuda_edge | False | 0.011006 | 0.009754 | 0.001252 | n/a | 10.0 |
| case70da | cuda_fp64_edge | False | 0.011234 | 0.009759 | 0.001475 | n/a | 10.0 |
| case74ds | cuda_fp32_edge | False | 0.011130 | 0.009769 | 0.001361 | n/a | 10.0 |
| case74ds | cuda_edge | False | 0.011151 | 0.009753 | 0.001398 | n/a | 10.0 |
| case74ds | cuda_fp64_edge | False | 0.011396 | 0.009776 | 0.001620 | n/a | 10.0 |
| case8387pegase | cuda_fp32_edge | False | 0.036347 | 0.027633 | 0.008713 | n/a | 10.0 |
| case8387pegase | cuda_edge | False | 0.036722 | 0.027665 | 0.009056 | n/a | 10.0 |
| case8387pegase | cuda_fp64_edge | False | 0.041578 | 0.028178 | 0.013398 | n/a | 10.0 |
| case85 | cuda_fp32_edge | False | 0.011333 | 0.009892 | 0.001440 | n/a | 10.0 |
| case85 | cuda_edge | False | 0.011351 | 0.009881 | 0.001469 | n/a | 10.0 |
| case85 | cuda_fp64_edge | False | 0.011611 | 0.009886 | 0.001725 | n/a | 10.0 |
| case89pegase | cuda_fp32_edge | False | 0.012300 | 0.010202 | 0.002097 | n/a | 10.0 |
| case89pegase | cuda_edge | False | 0.012348 | 0.010204 | 0.002144 | n/a | 10.0 |
| case89pegase | cuda_fp64_edge | False | 0.013043 | 0.010200 | 0.002843 | n/a | 10.0 |
| case9 | cuda_fp32_edge | False | 0.010622 | 0.009638 | 0.000984 | n/a | 10.0 |
| case9 | cuda_edge | False | 0.011427 | 0.010383 | 0.001044 | n/a | 10.0 |
| case9 | cuda_fp64_edge | False | 0.011846 | 0.010726 | 0.001119 | n/a | 10.0 |
| case9241pegase | cuda_fp32_edge | False | 0.038911 | 0.029919 | 0.008991 | n/a | 10.0 |
| case9241pegase | cuda_edge | False | 0.039004 | 0.029674 | 0.009329 | n/a | 10.0 |
| case9241pegase | cuda_fp64_edge | False | 0.043400 | 0.029428 | 0.013971 | n/a | 10.0 |
| case94pi | cuda_fp32_edge | False | 0.011300 | 0.009888 | 0.001411 | n/a | 10.0 |
| case94pi | cuda_edge | False | 0.011377 | 0.009898 | 0.001479 | n/a | 10.0 |
| case94pi | cuda_fp64_edge | False | 0.019190 | 0.017465 | 0.001725 | n/a | 10.0 |
| case9Q | cuda_fp32_edge | False | 0.010563 | 0.009573 | 0.000990 | n/a | 10.0 |
| case9Q | cuda_edge | False | 0.010629 | 0.009579 | 0.001049 | n/a | 10.0 |
| case9Q | cuda_fp64_edge | False | 0.010709 | 0.009588 | 0.001120 | n/a | 10.0 |
| case9target | cuda_fp32_edge | False | 0.010588 | 0.009596 | 0.000991 | n/a | 10.0 |
| case9target | cuda_edge | False | 0.010639 | 0.009589 | 0.001051 | n/a | 10.0 |
| case9target | cuda_fp64_edge | False | 0.011857 | 0.010737 | 0.001120 | n/a | 10.0 |
| case_ACTIVSg10k | cuda_fp32_edge | False | 0.037297 | 0.029162 | 0.008134 | n/a | 10.0 |
| case_ACTIVSg10k | cuda_edge | False | 0.037524 | 0.028950 | 0.008573 | n/a | 10.0 |
| case_ACTIVSg10k | cuda_fp64_edge | False | 0.042109 | 0.029873 | 0.012235 | n/a | 10.0 |
| case_ACTIVSg200 | cuda_fp32_edge | False | 0.018865 | 0.017007 | 0.001857 | n/a | 10.0 |
| case_ACTIVSg200 | cuda_edge | False | 0.012234 | 0.010329 | 0.001905 | n/a | 10.0 |
| case_ACTIVSg200 | cuda_fp64_edge | False | 0.012751 | 0.010331 | 0.002420 | n/a | 10.0 |
| case_ACTIVSg2000 | cuda_fp32_edge | False | 0.020273 | 0.014721 | 0.005552 | n/a | 10.0 |
| case_ACTIVSg2000 | cuda_edge | False | 0.020410 | 0.014729 | 0.005681 | n/a | 10.0 |
| case_ACTIVSg2000 | cuda_fp64_edge | False | 0.023559 | 0.014746 | 0.008812 | n/a | 10.0 |
| case_ACTIVSg25k | cuda_fp32_edge | False | 0.077649 | 0.060465 | 0.017184 | n/a | 10.0 |
| case_ACTIVSg25k | cuda_edge | False | 0.074771 | 0.056731 | 0.018040 | n/a | 10.0 |
| case_ACTIVSg25k | cuda_fp64_edge | False | 0.084963 | 0.056916 | 0.028047 | n/a | 10.0 |
| case_ACTIVSg500 | cuda_fp32_edge | False | 0.013330 | 0.011121 | 0.002209 | n/a | 10.0 |
| case_ACTIVSg500 | cuda_edge | False | 0.013437 | 0.011138 | 0.002298 | n/a | 10.0 |
| case_ACTIVSg500 | cuda_fp64_edge | False | 0.014070 | 0.011142 | 0.002927 | n/a | 10.0 |
| case_ACTIVSg70k | cuda_fp32_edge | False | 0.165010 | 0.125900 | 0.039110 | n/a | 10.0 |
| case_ACTIVSg70k | cuda_edge | False | 0.165551 | 0.123764 | 0.041787 | n/a | 10.0 |
| case_ACTIVSg70k | cuda_fp64_edge | False | 0.186821 | 0.121576 | 0.065244 | n/a | 10.0 |
| case_RTS_GMLC | cuda_fp32_edge | False | 0.011338 | 0.009885 | 0.001453 | n/a | 10.0 |
| case_RTS_GMLC | cuda_edge | False | 0.011394 | 0.009875 | 0.001518 | n/a | 10.0 |
| case_RTS_GMLC | cuda_fp64_edge | False | 0.011664 | 0.009886 | 0.001778 | n/a | 10.0 |
| case_SyntheticUSA | cuda_fp32_edge | False | 0.209908 | 0.166775 | 0.043133 | n/a | 10.0 |
| case_SyntheticUSA | cuda_edge | False | 0.212515 | 0.166383 | 0.046131 | n/a | 10.0 |
| case_SyntheticUSA | cuda_fp64_edge | False | 0.232091 | 0.162634 | 0.069456 | n/a | 10.0 |
| case_ieee30 | cuda_fp32_edge | False | 0.011040 | 0.009768 | 0.001271 | n/a | 10.0 |
| case_ieee30 | cuda_edge | False | 0.011040 | 0.009734 | 0.001306 | n/a | 10.0 |
| case_ieee30 | cuda_fp64_edge | False | 0.011260 | 0.009733 | 0.001527 | n/a | 10.0 |

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
