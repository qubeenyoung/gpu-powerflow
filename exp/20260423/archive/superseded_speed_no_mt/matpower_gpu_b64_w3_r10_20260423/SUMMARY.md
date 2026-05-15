# cuPF Benchmark `matpower_gpu_b64_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T01:38:08.119004+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 64
- cuDSS reordering algorithm: DEFAULT
- cuDSS MT mode: False
- cuDSS host threads: AUTO
- cuDSS ND_NLEVELS: AUTO
- cuDSS matching: False
- cuDSS matching algorithm: DEFAULT
- cuDSS pivot epsilon: AUTO
- cuDSS LD_PRELOAD: 

## end2end aggregate timing

| case | profile | success | elapsed mean (s) | analyze mean (s) | solve mean (s) | speedup vs cpu_fp64_edge | iterations mean |
|---|---|---:|---:|---:|---:|---:|---:|
| case10ba | cuda_edge | True | 0.018451 | 0.009537 | 0.008914 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.020869 | 0.010085 | 0.010784 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.020822 | 0.009917 | 0.010904 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.040182 | 0.015477 | 0.024705 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.018380 | 0.009527 | 0.008853 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.052121 | 0.016961 | 0.035160 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.314250 | 0.058984 | 0.255265 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.020831 | 0.010011 | 0.010819 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.018331 | 0.009574 | 0.008757 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.022201 | 0.010124 | 0.012076 | n/a | 6.1 |
| case145 | cuda_edge | True | 0.021661 | 0.010361 | 0.011300 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.018433 | 0.009553 | 0.008880 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.018464 | 0.009583 | 0.008881 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.018475 | 0.009549 | 0.008925 | n/a | 5.0 |
| case16ci | cuda_edge | True | 0.018336 | 0.009542 | 0.008793 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.018600 | 0.009566 | 0.009034 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.018680 | 0.009587 | 0.009092 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.049495 | 0.017677 | 0.031817 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.018541 | 0.009582 | 0.008958 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.055213 | 0.017511 | 0.037702 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.018740 | 0.009683 | 0.009057 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.082064 | 0.018951 | 0.063112 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.018867 | 0.009643 | 0.009222 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.078712 | 0.020537 | 0.058174 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.085515 | 0.020616 | 0.064898 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.085173 | 0.020784 | 0.064388 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.083973 | 0.020243 | 0.063729 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.070705 | 0.020595 | 0.050109 | n/a | 4.0 |
| case2868rte | cuda_edge | True | 0.083416 | 0.020792 | 0.062623 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.093718 | 0.021470 | 0.072248 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.018947 | 0.009646 | 0.009301 | n/a | 5.0 |
| case30 | cuda_edge | True | 0.018895 | 0.009686 | 0.009208 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.028390 | 0.011495 | 0.016894 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.075094 | 0.021223 | 0.053870 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.018877 | 0.009680 | 0.009196 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.018893 | 0.009694 | 0.009199 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.099711 | 0.021743 | 0.077968 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.074288 | 0.022766 | 0.051522 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.018802 | 0.009627 | 0.009175 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.018997 | 0.009633 | 0.009364 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.018870 | 0.009658 | 0.009211 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.019176 | 0.009663 | 0.009512 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.018596 | 0.009705 | 0.008890 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.017952 | 0.009444 | 0.008507 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.018046 | 0.009464 | 0.008581 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.018059 | 0.009472 | 0.008586 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.019550 | 0.009733 | 0.009817 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.019164 | 0.009735 | 0.009429 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.029295 | 0.013237 | 0.016058 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.029343 | 0.013268 | 0.016074 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.019855 | 0.009869 | 0.009986 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.020170 | 0.009820 | 0.010350 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.019003 | 0.009837 | 0.009166 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.133611 | 0.034755 | 0.098855 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.133865 | 0.034849 | 0.099016 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.118527 | 0.033696 | 0.084831 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.132936 | 0.033800 | 0.099136 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.019995 | 0.009792 | 0.010202 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.018142 | 0.009469 | 0.008672 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.019576 | 0.009746 | 0.009830 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.019635 | 0.009764 | 0.009871 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.168829 | 0.043659 | 0.125170 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.020311 | 0.009919 | 0.010391 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.022084 | 0.010203 | 0.011881 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.018377 | 0.009514 | 0.008863 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.237847 | 0.047736 | 0.190110 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.020455 | 0.009904 | 0.010550 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.018371 | 0.009509 | 0.008861 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.018622 | 0.009530 | 0.009092 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.239660 | 0.049699 | 0.189961 | n/a | 6.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.022050 | 0.010468 | 0.011581 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.070065 | 0.021883 | 0.048181 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.530136 | 0.110178 | 0.419958 | n/a | 5.2 |
| case_ACTIVSg500 | cuda_edge | True | 0.030178 | 0.012902 | 0.017275 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 1.730627 | 0.271519 | 1.459107 | n/a | 7.5 |
| case_RTS_GMLC | cuda_edge | True | 0.020442 | 0.009910 | 0.010531 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 2.180645 | 0.354024 | 1.826621 | n/a | 8.0 |
| case_ieee30 | cuda_edge | True | 0.018702 | 0.009679 | 0.009023 | n/a | 3.0 |

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
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```

```bash
ncu --set full -o cupf_ncu \
  /workspace/gpu-powerflow-master/cuPF/build/bench-end2end-superlu/benchmarks/cupf_case_benchmark \
  --case-dir /workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps/case10ba \
  --profile cuda_mixed_edge --warmup 1 --repeats 1
```
