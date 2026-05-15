# cuPF Benchmark `matpower_gpu_mt_b16_w3_r10_20260423`

## Setup

- Created UTC: 2026-04-23T02:08:57.311745+00:00
- Dataset root: `/workspace/gpu-powerflow/datasets/matpower8.1/cupf_all_dumps`
- Cases: case10ba, case118, case118zh, case1197, case12da, case1354pegase, case13659pegase, case136ma, case14, case141, case145, case15da, case15nbr, case16am, case16ci, case17me, case18, case1888rte, case18nbr, case1951rte, case22, case2383wp, case24_ieee_rts, case2736sp, case2737sop, case2746wop, case2746wp, case2848rte, case2868rte, case2869pegase, case28da, case30, case300, case3012wp, case30Q, case30pwl, case3120sp, case3375wp, case33bw, case33mg, case34sa, case38si, case39, case4_dist, case4gs, case5, case51ga, case51he, case533mt_hi, case533mt_lo, case57, case59, case60nordic, case6468rte, case6470rte, case6495rte, case6515rte, case69, case6ww, case70da, case74ds, case8387pegase, case85, case89pegase, case9, case9241pegase, case94pi, case9Q, case9target, case_ACTIVSg10k, case_ACTIVSg200, case_ACTIVSg2000, case_ACTIVSg25k, case_ACTIVSg500, case_ACTIVSg70k, case_RTS_GMLC, case_SyntheticUSA, case_ieee30
- Profiles: cuda_edge
- Measurement modes: end2end
- Warmup: 3
- Repeats: 10
- Batch size: 16
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
| case10ba | cuda_edge | True | 0.018461 | 0.009590 | 0.008870 | n/a | 5.0 |
| case118 | cuda_edge | True | 0.029645 | 0.013666 | 0.015978 | n/a | 4.0 |
| case118zh | cuda_edge | True | 0.019641 | 0.009940 | 0.009700 | n/a | 5.0 |
| case1197 | cuda_edge | True | 0.027676 | 0.012948 | 0.014727 | n/a | 4.0 |
| case12da | cuda_edge | True | 0.024617 | 0.013805 | 0.010811 | n/a | 4.0 |
| case1354pegase | cuda_edge | True | 0.031264 | 0.013418 | 0.017846 | n/a | 5.0 |
| case13659pegase | cuda_edge | True | 0.124324 | 0.040186 | 0.084137 | n/a | 7.0 |
| case136ma | cuda_edge | True | 0.019570 | 0.009928 | 0.009642 | n/a | 5.0 |
| case14 | cuda_edge | True | 0.024489 | 0.014783 | 0.009706 | n/a | 3.0 |
| case141 | cuda_edge | True | 0.020230 | 0.010035 | 0.010195 | n/a | 6.2 |
| case145 | cuda_edge | True | 0.020207 | 0.010251 | 0.009956 | n/a | 4.0 |
| case15da | cuda_edge | True | 0.020909 | 0.011341 | 0.009568 | n/a | 4.0 |
| case15nbr | cuda_edge | True | 0.018444 | 0.009626 | 0.008817 | n/a | 4.0 |
| case16am | cuda_edge | True | 0.018484 | 0.009636 | 0.008848 | n/a | 4.8 |
| case16ci | cuda_edge | True | 0.018335 | 0.009597 | 0.008737 | n/a | 4.0 |
| case17me | cuda_edge | True | 0.024901 | 0.012578 | 0.012322 | n/a | 5.0 |
| case18 | cuda_edge | True | 0.018693 | 0.009660 | 0.009032 | n/a | 5.0 |
| case1888rte | cuda_edge | True | 0.035891 | 0.013886 | 0.022006 | n/a | 3.0 |
| case18nbr | cuda_edge | True | 0.023996 | 0.013101 | 0.010894 | n/a | 4.0 |
| case1951rte | cuda_edge | True | 0.032612 | 0.014181 | 0.018430 | n/a | 4.0 |
| case22 | cuda_edge | True | 0.018573 | 0.009644 | 0.008929 | n/a | 4.0 |
| case2383wp | cuda_edge | True | 0.045647 | 0.015312 | 0.030334 | n/a | 7.0 |
| case24_ieee_rts | cuda_edge | True | 0.018778 | 0.009696 | 0.009081 | n/a | 5.0 |
| case2736sp | cuda_edge | True | 0.040008 | 0.016278 | 0.023730 | n/a | 5.0 |
| case2737sop | cuda_edge | True | 0.041172 | 0.015705 | 0.025466 | n/a | 6.0 |
| case2746wop | cuda_edge | True | 0.046680 | 0.015647 | 0.031033 | n/a | 6.0 |
| case2746wp | cuda_edge | True | 0.040230 | 0.015363 | 0.024866 | n/a | 6.0 |
| case2848rte | cuda_edge | True | 0.042860 | 0.021787 | 0.021073 | n/a | 4.0 |
| case2868rte | cuda_edge | True | 0.047112 | 0.016643 | 0.030469 | n/a | 6.0 |
| case2869pegase | cuda_edge | True | 0.043598 | 0.015801 | 0.027796 | n/a | 7.0 |
| case28da | cuda_edge | True | 0.018783 | 0.009696 | 0.009086 | n/a | 4.8 |
| case30 | cuda_edge | True | 0.018778 | 0.009728 | 0.009049 | n/a | 4.0 |
| case300 | cuda_edge | True | 0.022551 | 0.010665 | 0.011886 | n/a | 6.0 |
| case3012wp | cuda_edge | True | 0.038143 | 0.015944 | 0.022198 | n/a | 4.0 |
| case30Q | cuda_edge | True | 0.018772 | 0.009725 | 0.009046 | n/a | 4.0 |
| case30pwl | cuda_edge | True | 0.018788 | 0.009735 | 0.009052 | n/a | 4.0 |
| case3120sp | cuda_edge | True | 0.050942 | 0.016803 | 0.034138 | n/a | 7.0 |
| case3375wp | cuda_edge | True | 0.038988 | 0.017352 | 0.021636 | n/a | 3.0 |
| case33bw | cuda_edge | True | 0.018686 | 0.009686 | 0.009000 | n/a | 4.0 |
| case33mg | cuda_edge | True | 0.025620 | 0.012023 | 0.013596 | n/a | 5.0 |
| case34sa | cuda_edge | True | 0.018700 | 0.009688 | 0.009011 | n/a | 4.0 |
| case38si | cuda_edge | True | 0.021581 | 0.009711 | 0.011869 | n/a | 5.0 |
| case39 | cuda_edge | True | 0.018543 | 0.009749 | 0.008794 | n/a | 2.0 |
| case4_dist | cuda_edge | True | 0.018437 | 0.009520 | 0.008916 | n/a | 4.0 |
| case4gs | cuda_edge | True | 0.018163 | 0.009532 | 0.008630 | n/a | 4.0 |
| case5 | cuda_edge | True | 0.018177 | 0.009541 | 0.008635 | n/a | 4.0 |
| case51ga | cuda_edge | True | 0.030276 | 0.014144 | 0.016131 | n/a | 5.0 |
| case51he | cuda_edge | True | 0.018871 | 0.009757 | 0.009113 | n/a | 4.0 |
| case533mt_hi | cuda_edge | True | 0.024408 | 0.012690 | 0.011718 | n/a | 4.0 |
| case533mt_lo | cuda_edge | True | 0.031101 | 0.016667 | 0.014433 | n/a | 4.0 |
| case57 | cuda_edge | True | 0.019283 | 0.009896 | 0.009387 | n/a | 4.0 |
| case59 | cuda_edge | True | 0.019512 | 0.009854 | 0.009658 | n/a | 6.0 |
| case60nordic | cuda_edge | True | 0.018782 | 0.009841 | 0.008941 | n/a | 2.0 |
| case6468rte | cuda_edge | True | 0.059094 | 0.022619 | 0.036474 | n/a | 4.0 |
| case6470rte | cuda_edge | True | 0.064842 | 0.023323 | 0.041519 | n/a | 4.0 |
| case6495rte | cuda_edge | True | 0.057651 | 0.026849 | 0.030802 | n/a | 3.0 |
| case6515rte | cuda_edge | True | 0.061557 | 0.021895 | 0.039662 | n/a | 4.0 |
| case69 | cuda_edge | True | 0.019259 | 0.009820 | 0.009438 | n/a | 5.0 |
| case6ww | cuda_edge | True | 0.018574 | 0.009543 | 0.009030 | n/a | 4.0 |
| case70da | cuda_edge | True | 0.019015 | 0.009751 | 0.009263 | n/a | 5.0 |
| case74ds | cuda_edge | True | 0.021728 | 0.009791 | 0.011937 | n/a | 4.0 |
| case8387pegase | cuda_edge | True | 0.077222 | 0.028806 | 0.048415 | n/a | 4.0 |
| case85 | cuda_edge | True | 0.021632 | 0.009904 | 0.011728 | n/a | 5.0 |
| case89pegase | cuda_edge | True | 0.020655 | 0.010218 | 0.010437 | n/a | 6.0 |
| case9 | cuda_edge | True | 0.024109 | 0.012439 | 0.011670 | n/a | 5.0 |
| case9241pegase | cuda_edge | True | 0.091882 | 0.029871 | 0.062011 | n/a | 7.0 |
| case94pi | cuda_edge | True | 0.019480 | 0.009914 | 0.009566 | n/a | 5.0 |
| case9Q | cuda_edge | True | 0.019176 | 0.009587 | 0.009588 | n/a | 5.0 |
| case9target | cuda_edge | True | 0.018611 | 0.009581 | 0.009030 | n/a | 7.0 |
| case_ACTIVSg10k | cuda_edge | True | 0.093167 | 0.028503 | 0.064664 | n/a | 6.0 |
| case_ACTIVSg200 | cuda_edge | True | 0.020301 | 0.010347 | 0.009953 | n/a | 3.0 |
| case_ACTIVSg2000 | cuda_edge | True | 0.036602 | 0.015084 | 0.021518 | n/a | 4.0 |
| case_ACTIVSg25k | cuda_edge | True | 0.193237 | 0.062696 | 0.130540 | n/a | 5.0 |
| case_ACTIVSg500 | cuda_edge | True | 0.030365 | 0.017612 | 0.012752 | n/a | 4.0 |
| case_ACTIVSg70k | cuda_edge | True | 0.530346 | 0.132534 | 0.397812 | n/a | 7.0 |
| case_RTS_GMLC | cuda_edge | True | 0.019484 | 0.009900 | 0.009583 | n/a | 5.0 |
| case_SyntheticUSA | cuda_edge | True | 0.701266 | 0.162287 | 0.538979 | n/a | 8.0 |
| case_ieee30 | cuda_edge | True | 0.018672 | 0.009743 | 0.008929 | n/a | 3.0 |

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
