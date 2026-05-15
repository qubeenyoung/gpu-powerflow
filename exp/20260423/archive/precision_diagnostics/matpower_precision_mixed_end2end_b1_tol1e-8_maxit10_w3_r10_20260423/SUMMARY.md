# cuPF Benchmark `matpower_precision_mixed_end2end_b1_tol1e-8_maxit10_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T03:19:35.300701+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
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
| case10ba | cuda_edge | True | 0.011206 | 0.010717 | 0.000487 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.010564 | 0.010036 | 0.000528 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.011361 | 0.010728 | 0.000633 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.013681 | 0.012823 | 0.000857 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.010005 | 0.009599 | 0.000406 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.014925 | 0.013423 | 0.001501 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.042285 | 0.035077 | 0.007208 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.011483 | 0.010890 | 0.000593 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.009935 | 0.009641 | 0.000294 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.011945 | 0.011099 | 0.000845 | n/a | 6.0 |
| case145 | cuda_edge | True | 0.011944 | 0.011331 | 0.000612 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.015239 | 0.014826 | 0.000412 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.010024 | 0.009614 | 0.000409 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.009977 | 0.009593 | 0.000383 | n/a | 4.0 |
| case16ci | cuda_edge | True | 0.009950 | 0.009586 | 0.000363 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.010126 | 0.009607 | 0.000517 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.010158 | 0.009621 | 0.000537 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.014772 | 0.013842 | 0.000929 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.010074 | 0.009640 | 0.000434 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.015561 | 0.014171 | 0.001389 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.010063 | 0.009637 | 0.000425 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.017390 | 0.014797 | 0.002592 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.010202 | 0.009678 | 0.000523 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.017837 | 0.015753 | 0.002083 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.018486 | 0.016059 | 0.002426 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.018957 | 0.016410 | 0.002547 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.017708 | 0.015313 | 0.002395 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.016789 | 0.015352 | 0.001436 | n/a | 3.8 |
| case2868rte | cuda_edge | True | 0.018251 | 0.015810 | 0.002440 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.018423 | 0.015756 | 0.002667 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.010170 | 0.009676 | 0.000494 | n/a | 4.5 |
| case30 | cuda_edge | True | 0.010170 | 0.009723 | 0.000446 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.011810 | 0.010616 | 0.001193 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.017530 | 0.015914 | 0.001616 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.010170 | 0.009725 | 0.000445 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.010970 | 0.010521 | 0.000448 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.019282 | 0.016239 | 0.003042 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.023613 | 0.022277 | 0.001335 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.010113 | 0.009679 | 0.000433 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.014478 | 0.013930 | 0.000548 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.010114 | 0.009679 | 0.000435 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.010287 | 0.009713 | 0.000573 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.009921 | 0.009709 | 0.000211 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.009780 | 0.009498 | 0.000281 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.009858 | 0.009523 | 0.000335 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.009841 | 0.009508 | 0.000333 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.010335 | 0.009754 | 0.000581 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.017361 | 0.016900 | 0.000460 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.011765 | 0.011091 | 0.000674 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.011758 | 0.011085 | 0.000673 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.010408 | 0.009876 | 0.000531 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.019013 | 0.018204 | 0.000808 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.010049 | 0.009825 | 0.000223 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.024884 | 0.022554 | 0.002329 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.025115 | 0.022810 | 0.002305 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.023778 | 0.022074 | 0.001704 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.024498 | 0.022059 | 0.002439 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.010380 | 0.009791 | 0.000589 | n/a | 4.7 |
| case6ww | cuda_edge | True | 0.009887 | 0.009532 | 0.000355 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.010284 | 0.009729 | 0.000554 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.010242 | 0.009755 | 0.000487 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.030705 | 0.027487 | 0.003218 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.010523 | 0.009878 | 0.000645 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.014869 | 0.013747 | 0.001122 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.010041 | 0.009569 | 0.000471 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.041431 | 0.035521 | 0.005909 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.010509 | 0.009865 | 0.000644 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.010055 | 0.009583 | 0.000472 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.010229 | 0.009559 | 0.000669 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.032855 | 0.028248 | 0.004606 | n/a | 5.9 |
| case_ACTIVSg200 | cuda_edge | True | 0.010781 | 0.010326 | 0.000455 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.046701 | 0.044780 | 0.001921 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.064583 | 0.056386 | 0.008197 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.011937 | 0.011121 | 0.000816 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.148534 | 0.121649 | 0.026884 | n/a | 7.0 |
| case_RTS_GMLC | cuda_edge | True | 0.010630 | 0.009972 | 0.000657 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 0.198869 | 0.166896 | 0.031973 | n/a | 7.5 |
| case_ieee30 | cuda_edge | True | 0.010051 | 0.009715 | 0.000336 | n/a | 3.0 |

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
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu-cudss-mt-auto/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
