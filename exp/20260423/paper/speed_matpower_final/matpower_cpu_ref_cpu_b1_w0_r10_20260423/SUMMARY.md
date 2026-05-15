# cuPF Benchmark `matpower_cpu_ref_cpu_b1_w0_r10_20260423`

## Setup

- Created UTC: 2026-04-23T01:31:33.127879+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cpp_naive, cpp
- Measurement modes: end2end
- Warmup: 0
- Repeats: 10
- Batch size: 1
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS matching: False
- cuDSS matching algorithm: DEFAULT
- cuDSS pivot epsilon: AUTO
- cuDSS LD_PRELOAD: 

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpp | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case10ba | cpp_naive | True | 0.000236 | 0.000000 | 0.000235 | 0.14x | 5.0 |
| case10ba | cpp | True | 0.000032 | 0.000011 | 0.000021 | 1.00x | 5.0 |
| case118 | cpp_naive | True | 0.001001 | 0.000002 | 0.000998 | 0.25x | 4.0 |
| case118 | cpp | True | 0.000254 | 0.000105 | 0.000148 | 1.00x | 4.0 |
| case118zh | cpp_naive | True | 0.001332 | 0.000001 | 0.001330 | 0.21x | 5.0 |
| case118zh | cpp | True | 0.000277 | 0.000098 | 0.000179 | 1.00x | 5.0 |
| case1197 | cpp_naive | True | 0.008108 | 0.000015 | 0.008093 | 0.60x | 4.0 |
| case1197 | cpp | True | 0.004891 | 0.003318 | 0.001573 | 1.00x | 4.0 |
| case12da | cpp_naive | True | 0.000180 | 0.000001 | 0.000179 | 0.18x | 4.0 |
| case12da | cpp | True | 0.000032 | 0.000013 | 0.000019 | 1.00x | 4.0 |
| case1354pegase | cpp_naive | True | 0.014729 | 0.000014 | 0.014714 | 0.40x | 5.0 |
| case1354pegase | cpp | True | 0.005943 | 0.001845 | 0.004098 | 1.00x | 5.0 |
| case13659pegase | cpp_naive | True | 0.179400 | 0.000223 | 0.179177 | 0.35x | 6.0 |
| case13659pegase | cpp | True | 0.063167 | 0.016056 | 0.047111 | 1.00x | 6.0 |
| case136ma | cpp_naive | True | 0.001293 | 0.000002 | 0.001291 | 0.25x | 5.0 |
| case136ma | cpp | True | 0.000327 | 0.000117 | 0.000209 | 1.00x | 5.0 |
| case14 | cpp_naive | True | 0.000161 | 0.000001 | 0.000160 | 0.21x | 3.0 |
| case14 | cpp | True | 0.000034 | 0.000016 | 0.000017 | 1.00x | 3.0 |
| case141 | cpp_naive | True | 0.001117 | 0.000003 | 0.001114 | 0.27x | 4.0 |
| case141 | cpp | True | 0.000301 | 0.000120 | 0.000179 | 1.00x | 4.0 |
| case145 | cpp_naive | True | 0.001673 | 0.000003 | 0.001669 | 0.32x | 4.0 |
| case145 | cpp | True | 0.000533 | 0.000212 | 0.000320 | 1.00x | 4.0 |
| case15da | cpp_naive | True | 0.000247 | 0.000001 | 0.000246 | 0.16x | 4.0 |
| case15da | cpp | True | 0.000039 | 0.000016 | 0.000022 | 1.00x | 4.0 |
| case15nbr | cpp_naive | True | 0.000522 | 0.000000 | 0.000521 | 0.07x | 4.0 |
| case15nbr | cpp | True | 0.000039 | 0.000016 | 0.000023 | 1.00x | 4.0 |
| case16am | cpp_naive | False | 0.003670 | 0.000001 | 0.003669 | 0.08x | 50.0 |
| case16am | cpp | False | 0.000307 | 0.000016 | 0.000291 | 1.00x | 50.0 |
| case16ci | cpp_naive | True | 0.000197 | 0.000001 | 0.000196 | 0.19x | 4.0 |
| case16ci | cpp | True | 0.000038 | 0.000015 | 0.000023 | 1.00x | 4.0 |
| case17me | cpp_naive | True | 0.000362 | 0.000000 | 0.000362 | 0.14x | 5.0 |
| case17me | cpp | True | 0.000049 | 0.000018 | 0.000031 | 1.00x | 5.0 |
| case18 | cpp_naive | True | 0.000395 | 0.000000 | 0.000395 | 0.13x | 5.0 |
| case18 | cpp | True | 0.000052 | 0.000019 | 0.000033 | 1.00x | 5.0 |
| case1888rte | cpp_naive | True | 0.010935 | 0.000028 | 0.010907 | 0.52x | 3.0 |
| case1888rte | cpp | True | 0.005709 | 0.002663 | 0.003045 | 1.00x | 3.0 |
| case18nbr | cpp_naive | True | 0.000281 | 0.000000 | 0.000281 | 0.16x | 4.0 |
| case18nbr | cpp | True | 0.000045 | 0.000019 | 0.000026 | 1.00x | 4.0 |
| case1951rte | cpp_naive | True | 0.015422 | 0.000026 | 0.015395 | 0.44x | 4.0 |
| case1951rte | cpp | True | 0.006815 | 0.002582 | 0.004232 | 1.00x | 4.0 |
| case22 | cpp_naive | True | 0.000318 | 0.000001 | 0.000317 | 0.17x | 4.0 |
| case22 | cpp | True | 0.000054 | 0.000022 | 0.000032 | 1.00x | 4.0 |
| case2383wp | cpp_naive | True | 0.037056 | 0.000034 | 0.037021 | 0.35x | 7.0 |
| case2383wp | cpp | True | 0.012874 | 0.002892 | 0.009981 | 1.00x | 7.0 |
| case24_ieee_rts | cpp_naive | True | 0.000421 | 0.000001 | 0.000419 | 0.16x | 5.0 |
| case24_ieee_rts | cpp | True | 0.000069 | 0.000024 | 0.000044 | 1.00x | 5.0 |
| case2736sp | cpp_naive | True | 0.030133 | 0.000040 | 0.030093 | 0.36x | 5.0 |
| case2736sp | cpp | True | 0.010809 | 0.003110 | 0.007698 | 1.00x | 5.0 |
| case2737sop | cpp_naive | True | 0.036864 | 0.000037 | 0.036826 | 0.36x | 6.0 |
| case2737sop | cpp | True | 0.013195 | 0.003284 | 0.009910 | 1.00x | 6.0 |
| case2746wop | cpp_naive | True | 0.029565 | 0.000046 | 0.029518 | 0.37x | 5.0 |
| case2746wop | cpp | True | 0.010950 | 0.003262 | 0.007688 | 1.00x | 5.0 |
| case2746wp | cpp_naive | True | 0.028828 | 0.000036 | 0.028791 | 0.38x | 5.0 |
| case2746wp | cpp | True | 0.010839 | 0.003259 | 0.007580 | 1.00x | 5.0 |
| case2848rte | cpp_naive | True | 0.015784 | 0.000038 | 0.015746 | 0.50x | 3.0 |
| case2848rte | cpp | True | 0.007855 | 0.003623 | 0.004231 | 1.00x | 3.0 |
| case2868rte | cpp_naive | True | 0.035963 | 0.000038 | 0.035925 | 0.35x | 6.0 |
| case2868rte | cpp | True | 0.012681 | 0.003427 | 0.009254 | 1.00x | 6.0 |
| case2869pegase | cpp_naive | True | 0.045573 | 0.000042 | 0.045530 | 0.35x | 7.0 |
| case2869pegase | cpp | True | 0.015841 | 0.003591 | 0.012249 | 1.00x | 7.0 |
| case28da | cpp_naive | True | 0.000380 | 0.000000 | 0.000380 | 0.17x | 4.0 |
| case28da | cpp | True | 0.000065 | 0.000027 | 0.000037 | 1.00x | 4.0 |
| case30 | cpp_naive | True | 0.000402 | 0.000000 | 0.000401 | 0.20x | 4.0 |
| case30 | cpp | True | 0.000080 | 0.000032 | 0.000048 | 1.00x | 4.0 |
| case300 | cpp_naive | True | 0.004997 | 0.000004 | 0.004992 | 0.28x | 6.0 |
| case300 | cpp | True | 0.001402 | 0.000397 | 0.001004 | 1.00x | 6.0 |
| case3012wp | cpp_naive | True | 0.024883 | 0.000039 | 0.024844 | 0.41x | 4.0 |
| case3012wp | cpp | True | 0.010196 | 0.003539 | 0.006656 | 1.00x | 4.0 |
| case30Q | cpp_naive | True | 0.000414 | 0.000001 | 0.000412 | 0.19x | 4.0 |
| case30Q | cpp | True | 0.000080 | 0.000032 | 0.000047 | 1.00x | 4.0 |
| case30pwl | cpp_naive | True | 0.000411 | 0.000000 | 0.000410 | 0.20x | 4.0 |
| case30pwl | cpp | True | 0.000081 | 0.000032 | 0.000047 | 1.00x | 4.0 |
| case3120sp | cpp_naive | True | 0.048067 | 0.000042 | 0.048025 | 0.34x | 7.0 |
| case3120sp | cpp | True | 0.016209 | 0.003536 | 0.012673 | 1.00x | 7.0 |
| case3375wp | cpp_naive | True | 0.018841 | 0.000045 | 0.018796 | 0.47x | 3.0 |
| case3375wp | cpp | True | 0.008917 | 0.003733 | 0.005184 | 1.00x | 3.0 |
| case33bw | cpp_naive | True | 0.000415 | 0.000000 | 0.000414 | 0.18x | 4.0 |
| case33bw | cpp | True | 0.000074 | 0.000031 | 0.000043 | 1.00x | 4.0 |
| case33mg | cpp_naive | True | 0.000551 | 0.000000 | 0.000550 | 0.16x | 5.0 |
| case33mg | cpp | True | 0.000087 | 0.000031 | 0.000055 | 1.00x | 5.0 |
| case34sa | cpp_naive | True | 0.000424 | 0.000001 | 0.000422 | 0.18x | 4.0 |
| case34sa | cpp | True | 0.000078 | 0.000032 | 0.000045 | 1.00x | 4.0 |
| case38si | cpp_naive | True | 0.000641 | 0.000001 | 0.000639 | 0.15x | 5.0 |
| case38si | cpp | True | 0.000099 | 0.000035 | 0.000064 | 1.00x | 5.0 |
| case39 | cpp_naive | True | 0.000173 | 0.000001 | 0.000172 | 0.38x | 2.0 |
| case39 | cpp | True | 0.000066 | 0.000037 | 0.000028 | 1.00x | 2.0 |
| case4_dist | cpp_naive | True | 0.000045 | 0.000000 | 0.000044 | 0.34x | 4.0 |
| case4_dist | cpp | True | 0.000015 | 0.000006 | 0.000009 | 1.00x | 4.0 |
| case4gs | cpp_naive | True | 0.000045 | 0.000000 | 0.000045 | 0.35x | 4.0 |
| case4gs | cpp | True | 0.000016 | 0.000006 | 0.000009 | 1.00x | 4.0 |
| case5 | cpp_naive | True | 0.000046 | 0.000000 | 0.000046 | 0.35x | 4.0 |
| case5 | cpp | True | 0.000016 | 0.000006 | 0.000009 | 1.00x | 4.0 |
| case51ga | cpp_naive | True | 0.001267 | 0.000000 | 0.001267 | 0.10x | 5.0 |
| case51ga | cpp | True | 0.000124 | 0.000044 | 0.000080 | 1.00x | 5.0 |
| case51he | cpp_naive | True | 0.000548 | 0.000001 | 0.000546 | 0.20x | 4.0 |
| case51he | cpp | True | 0.000108 | 0.000045 | 0.000063 | 1.00x | 4.0 |
| case533mt_hi | cpp_naive | True | 0.004229 | 0.000007 | 0.004221 | 0.30x | 4.0 |
| case533mt_hi | cpp | True | 0.001269 | 0.000602 | 0.000666 | 1.00x | 4.0 |
| case533mt_lo | cpp_naive | True | 0.004240 | 0.000007 | 0.004233 | 0.30x | 4.0 |
| case533mt_lo | cpp | True | 0.001263 | 0.000597 | 0.000665 | 1.00x | 4.0 |
| case57 | cpp_naive | True | 0.000604 | 0.000001 | 0.000603 | 0.26x | 4.0 |
| case57 | cpp | True | 0.000154 | 0.000058 | 0.000096 | 1.00x | 4.0 |
| case59 | cpp_naive | True | 0.000849 | 0.000002 | 0.000847 | 0.21x | 6.0 |
| case59 | cpp | True | 0.000179 | 0.000054 | 0.000125 | 1.00x | 6.0 |
| case60nordic | cpp_naive | True | 0.000213 | 0.000001 | 0.000212 | 0.42x | 2.0 |
| case60nordic | cpp | True | 0.000090 | 0.000052 | 0.000037 | 1.00x | 2.0 |
| case6468rte | cpp_naive | True | 0.052795 | 0.000094 | 0.052700 | 0.40x | 4.0 |
| case6468rte | cpp | True | 0.021100 | 0.006918 | 0.014181 | 1.00x | 4.0 |
| case6470rte | cpp_naive | True | 0.052646 | 0.000088 | 0.052558 | 0.40x | 4.0 |
| case6470rte | cpp | True | 0.021308 | 0.007270 | 0.014037 | 1.00x | 4.0 |
| case6495rte | cpp_naive | True | 0.036619 | 0.000088 | 0.036532 | 0.48x | 3.0 |
| case6495rte | cpp | True | 0.017459 | 0.007466 | 0.009993 | 1.00x | 3.0 |
| case6515rte | cpp_naive | True | 0.053873 | 0.000091 | 0.053781 | 0.39x | 4.0 |
| case6515rte | cpp | True | 0.021023 | 0.007003 | 0.014019 | 1.00x | 4.0 |
| case69 | cpp_naive | True | 0.000900 | 0.000002 | 0.000898 | 0.19x | 5.0 |
| case69 | cpp | True | 0.000168 | 0.000059 | 0.000108 | 1.00x | 5.0 |
| case6ww | cpp_naive | True | 0.000056 | 0.000000 | 0.000056 | 0.37x | 4.0 |
| case6ww | cpp | True | 0.000021 | 0.000008 | 0.000012 | 1.00x | 4.0 |
| case70da | cpp_naive | True | 0.000777 | 0.000001 | 0.000776 | 0.22x | 5.0 |
| case70da | cpp | True | 0.000171 | 0.000061 | 0.000110 | 1.00x | 5.0 |
| case74ds | cpp_naive | True | 0.000601 | 0.000001 | 0.000599 | 0.25x | 4.0 |
| case74ds | cpp | True | 0.000150 | 0.000061 | 0.000089 | 1.00x | 4.0 |
| case8387pegase | cpp_naive | True | 0.072928 | 0.000134 | 0.072793 | 0.40x | 4.0 |
| case8387pegase | cpp | True | 0.029408 | 0.010170 | 0.019237 | 1.00x | 4.0 |
| case85 | cpp_naive | True | 0.000999 | 0.000001 | 0.000997 | 0.48x | 5.0 |
| case85 | cpp | True | 0.000476 | 0.000340 | 0.000135 | 1.00x | 5.0 |
| case89pegase | cpp_naive | True | 0.002084 | 0.000002 | 0.002082 | 0.25x | 6.0 |
| case89pegase | cpp | True | 0.000516 | 0.000145 | 0.000370 | 1.00x | 6.0 |
| case9 | cpp_naive | True | 0.000094 | 0.000000 | 0.000094 | 0.33x | 5.0 |
| case9 | cpp | True | 0.000031 | 0.000010 | 0.000021 | 1.00x | 5.0 |
| case9241pegase | cpp_naive | True | 0.166245 | 0.000147 | 0.166098 | 0.34x | 7.0 |
| case9241pegase | cpp | True | 0.055738 | 0.012187 | 0.043551 | 1.00x | 7.0 |
| case94pi | cpp_naive | True | 0.001063 | 0.000002 | 0.001060 | 0.21x | 5.0 |
| case94pi | cpp | True | 0.000227 | 0.000081 | 0.000146 | 1.00x | 5.0 |
| case9Q | cpp_naive | True | 0.000094 | 0.000001 | 0.000093 | 0.33x | 5.0 |
| case9Q | cpp | True | 0.000031 | 0.000010 | 0.000020 | 1.00x | 5.0 |
| case9target | cpp_naive | True | 0.000132 | 0.000001 | 0.000131 | 0.30x | 7.0 |
| case9target | cpp | True | 0.000039 | 0.000011 | 0.000028 | 1.00x | 7.0 |
| case_ACTIVSg10k | cpp_naive | True | 0.109928 | 0.000143 | 0.109785 | 0.38x | 5.0 |
| case_ACTIVSg10k | cpp | True | 0.041332 | 0.010491 | 0.030840 | 1.00x | 5.0 |
| case_ACTIVSg200 | cpp_naive | True | 0.001386 | 0.000003 | 0.001383 | 0.38x | 3.0 |
| case_ACTIVSg200 | cpp | True | 0.000534 | 0.000231 | 0.000302 | 1.00x | 3.0 |
| case_ACTIVSg2000 | cpp_naive | True | 0.021368 | 0.000028 | 0.021338 | 0.51x | 4.0 |
| case_ACTIVSg2000 | cpp | True | 0.010975 | 0.002502 | 0.008472 | 1.00x | 4.0 |
| case_ACTIVSg25k | cpp_naive | True | 0.405768 | 0.000432 | 0.405335 | 0.29x | 5.0 |
| case_ACTIVSg25k | cpp | True | 0.118570 | 0.026579 | 0.091991 | 1.00x | 5.0 |
| case_ACTIVSg500 | cpp_naive | True | 0.004925 | 0.000007 | 0.004918 | 0.34x | 4.0 |
| case_ACTIVSg500 | cpp | True | 0.001698 | 0.000618 | 0.001080 | 1.00x | 4.0 |
| case_ACTIVSg70k | cpp_naive | True | 1.971964 | 0.001173 | 1.970791 | 0.32x | 7.0 |
| case_ACTIVSg70k | cpp | True | 0.638737 | 0.076910 | 0.561826 | 1.00x | 7.0 |
| case_RTS_GMLC | cpp_naive | True | 0.000900 | 0.000002 | 0.000898 | 0.22x | 5.0 |
| case_RTS_GMLC | cpp | True | 0.000195 | 0.000063 | 0.000131 | 1.00x | 5.0 |
| case_SyntheticUSA | cpp_naive | True | 2.194378 | 0.001404 | 2.192974 | 0.33x | 7.0 |
| case_SyntheticUSA | cpp | True | 0.722740 | 0.090688 | 0.632051 | 1.00x | 7.0 |
| case_ieee30 | cpp_naive | True | 0.000313 | 0.000000 | 0.000312 | 0.22x | 3.0 |
| case_ieee30 | cpp | True | 0.000069 | 0.000032 | 0.000037 | 1.00x | 3.0 |

## Files

- `manifest.json`: run configuration and environment
- `summary.csv`: one row per measured run across all measurement modes
- `aggregates.csv`: grouped statistics by mode/case/profile
- `summary_<mode>.csv` and `aggregates_<mode>.csv`: mode-specific views
- `raw/<mode>/`: per-run timing payload

## Nsight Hints

This run did not include CUDA profiles. Build a CUDA benchmark run before using Nsight, for example:

```bash
python3 /workspace/gpu-powerflow-master/cuPF/benchmarks/run_benchmarks.py \
  --dataset-root /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps \
  --cases case10ba \
  --profiles cuda_edge \
  --with-cuda --warmup 1 --repeats 1
```
