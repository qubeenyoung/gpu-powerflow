# cuPF Benchmark `matpower_cuda_mixed_edge_operators_r10`

## Setup

- Created UTC: 2026-04-28T10:08:46.204963+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: operators
- Warmup: 1
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

## operators aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case10ba | cuda_edge | True | 0.010252 | 0.009624 | 0.000628 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.017823 | 0.017197 | 0.000626 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.013980 | 0.013215 | 0.000765 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.013621 | 0.012651 | 0.000969 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.010114 | 0.009610 | 0.000503 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.027424 | 0.025746 | 0.001677 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.047201 | 0.039763 | 0.007437 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.010656 | 0.009927 | 0.000728 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.010028 | 0.009662 | 0.000365 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.011071 | 0.010065 | 0.001005 | n/a | 6.0 |
| case145 | cuda_edge | True | 0.010965 | 0.010252 | 0.000713 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.010129 | 0.009623 | 0.000506 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.010146 | 0.009637 | 0.000508 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.010074 | 0.009592 | 0.000481 | n/a | 4.0 |
| case16ci | cuda_edge | True | 0.010068 | 0.009598 | 0.000469 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.010290 | 0.009644 | 0.000646 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.010403 | 0.009737 | 0.000666 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.022371 | 0.021357 | 0.001013 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.010192 | 0.009659 | 0.000533 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.015702 | 0.014207 | 0.001494 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.010231 | 0.009693 | 0.000538 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.017623 | 0.014834 | 0.002788 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.010373 | 0.009713 | 0.000659 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.018020 | 0.015799 | 0.002221 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.018414 | 0.015804 | 0.002609 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.024353 | 0.021642 | 0.002711 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.017938 | 0.015380 | 0.002558 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.016987 | 0.015578 | 0.001408 | n/a | 3.5 |
| case2868rte | cuda_edge | True | 0.023729 | 0.021119 | 0.002609 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.018679 | 0.015814 | 0.002865 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.010275 | 0.009681 | 0.000594 | n/a | 4.4 |
| case30 | cuda_edge | True | 0.011084 | 0.010534 | 0.000549 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.014163 | 0.012808 | 0.001355 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.019258 | 0.017531 | 0.001727 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.010284 | 0.009735 | 0.000549 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.010301 | 0.009749 | 0.000552 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.021360 | 0.018119 | 0.003241 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.021667 | 0.020254 | 0.001413 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.010218 | 0.009685 | 0.000532 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.010369 | 0.009696 | 0.000673 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.012741 | 0.012202 | 0.000538 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.016962 | 0.016207 | 0.000754 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.009980 | 0.009730 | 0.000250 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.010179 | 0.009799 | 0.000379 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.009978 | 0.009542 | 0.000435 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.009972 | 0.009539 | 0.000432 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.011834 | 0.011121 | 0.000712 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.010303 | 0.009749 | 0.000553 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.011913 | 0.011137 | 0.000776 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.011900 | 0.011128 | 0.000771 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.016195 | 0.015561 | 0.000633 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.010804 | 0.009838 | 0.000965 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.010096 | 0.009831 | 0.000264 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.025061 | 0.022617 | 0.002443 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.025752 | 0.023331 | 0.002420 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.023829 | 0.022061 | 0.001767 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.024411 | 0.021860 | 0.002551 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.016915 | 0.016149 | 0.000767 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.010354 | 0.009899 | 0.000454 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.010427 | 0.009735 | 0.000692 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.010353 | 0.009767 | 0.000586 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.036575 | 0.033245 | 0.003329 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.010685 | 0.009909 | 0.000775 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.011486 | 0.010202 | 0.001283 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.010192 | 0.009591 | 0.000601 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.038497 | 0.032404 | 0.006092 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.010649 | 0.009880 | 0.000768 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.013945 | 0.013342 | 0.000603 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.011588 | 0.010728 | 0.000859 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.033523 | 0.028918 | 0.004604 | n/a | 5.7 |
| case_ACTIVSg200 | cuda_edge | True | 0.011009 | 0.010478 | 0.000530 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.016791 | 0.014791 | 0.001999 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.067118 | 0.058833 | 0.008285 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.016018 | 0.015090 | 0.000928 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.150877 | 0.123761 | 0.027116 | n/a | 7.0 |
| case_RTS_GMLC | cuda_edge | True | 0.010698 | 0.009903 | 0.000794 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 0.201793 | 0.168423 | 0.033370 | n/a | 7.8 |
| case_ieee30 | cuda_edge | True | 0.010138 | 0.009735 | 0.000402 | n/a | 3.0 |

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
  /workspace/gpu-powerflow/exp/20260428/build/operators-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow/exp/20260428/build/operators-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
